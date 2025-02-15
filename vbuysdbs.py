from flask import Flask, request, render_template, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from torch import cuda
import cv2
import numpy as np
from celery import Celery
import uuid
import shutil

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Celery for asynchronous tasks
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Load models at startup (cached for subsequent use)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
gpt_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
gpt_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Mock UI element detection (replace with a real model)
def detect_ui_elements(image):
    detected_elements = [
        {"type": "button", "color": "blue", "size": "50x20"},
        {"type": "text input", "color": "white", "size": "200x40"},
        {"type": "dropdown", "color": "gray", "size": "150x30"}
    ]
    return detected_elements

# Refine UI description with GPT
def refine_with_gpt(description):
    inputs = gpt_tokenizer(description, return_tensors="pt")
    output = gpt_model.generate(**inputs, max_length=512)
    refined_text = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    return refined_text

# Generate UI description
def generate_ui_description(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    description = blip_processor.decode(out[0], skip_special_tokens=True)
    return description

# Refine UI description with detected elements
def refine_ui_description(description, image_path):
    image = cv2.imread(image_path)
    detected_elements = detect_ui_elements(image)
    detailed_description = f"{description}. The UI contains {len(detected_elements)} elements: "
    for element in detected_elements:
        detailed_description += f"{element['type']} (Color: {element['color']}, Size: {element['size']}), "
    refined_description = refine_with_gpt(detailed_description)
    return refined_description.strip(', ')

# Generate code from description
def generate_code(description, user_description):
    return f"""
    <!-- Code generated from UI description -->
    <!-- AI Description: {description} -->
    <!-- User Description: {user_description} -->
    <div class="container">
        <button style="background-color: blue; width: 50px; height: 20px;">Button</button>
        <input type="text" style="background-color: white; width: 200px; height: 40px;" placeholder="Enter text">
        <select style="background-color: gray; width: 150px; height: 30px;">
            <option>Option 1</option>
            <option>Option 2</option>
        </select>
    </div>
    """

# Celery task for asynchronous processing
@celery.task
def process_image_async(filepath, user_description, task_id):
    try:
        ai_description = generate_ui_description(filepath)
        refined_description = refine_ui_description(ai_description, filepath)
        generated_code = generate_code(refined_description, user_description)
        
        # Save generated code
        generated_filename = f"{task_id}.html"
        generated_filepath = os.path.join(GENERATED_FOLDER, generated_filename)
        with open(generated_filepath, 'w') as f:
            f.write(generated_code)
        
        # Optionally, remove uploaded image after processing
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return {
            'status': 'success',
            'generated_code': generated_code,
            'ai_description': refined_description,
            'generated_filename': generated_filename
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    user_description = request.form.get('description', 'No description provided.')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Process image asynchronously
    process_image_async.apply_async(args=[filepath, user_description, task_id])
    
    return jsonify({
        'message': 'File is being processed',
        'task_id': task_id
    })

@app.route('/status/<task_id>')
def check_status(task_id):
    task = process_image_async.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        return jsonify(task.result)
    elif task.state == 'FAILURE':
        return jsonify({'status': 'error', 'message': 'Processing failed'}), 500
    else:
        return jsonify({'status': 'pending'})

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(GENERATED_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
