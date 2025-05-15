from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os
import uuid
import json
from werkzeug.utils import secure_filename
from PIL import Image
from datetime import datetime
import google.generativeai as genai

# === Gemini API Setup ===
genai.configure(api_key="AIzaSyB0z0o_sKnN3ZHMFE0bWByyDTZY58z0B0o")  # 🔐 Replace with your actual API key
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# === Flask App Setup ===
app = Flask(__name__)
UPLOAD_FOLDER = 'static'
JSON_LOG_FILE = 'chatbot_logs.json'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Model Details ===
class_names = ['bad_quality', 'empty_background', 'good_quality']
IMAGE_SIZE = 255

# === Custom Loss Function ===
def custom_loss_function():
    return tf.keras.losses.SparseCategoricalCrossentropy(reduction='mean')

# === Load Trained Model ===
model = tf.keras.models.load_model(
    r'C:\Users\KDS\PycharmProjects\model.h5',
    custom_objects={'SparseCategoricalCrossentropy': custom_loss_function}
)

# === Image Prediction ===
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# === Allowed Extensions ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# === Read/Write JSON logs ===
def append_to_json_file(entry, filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(entry)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            predicted_class, confidence = predict(img)

            return render_template('index.html',
                                   image_path=filepath,
                                   actual_label=predicted_class,
                                   predicted_label=predicted_class,
                                   confidence=confidence)

    return render_template('index.html', message='Upload an image')

@app.route('/describe', methods=['POST'])
def describe():
    image_path = request.form['image_path']
    try:
        image = Image.open(image_path)

        # Gemini response
        response = vision_model.generate_content([
            "Describe the crop issue in this image:",
            image
        ])
        ai_text = response.text

        # Split into readable lines
        description_lines = [line.strip() for line in ai_text.strip().split('\n') if line.strip()]

        # Log entry
        log_entry = {
            "image_path": image_path,
            "description": description_lines,
            "timestamp": datetime.now().isoformat()
        }
        append_to_json_file(log_entry, JSON_LOG_FILE)

    except Exception as e:
        description_lines = [f"Error processing image: {str(e)}"]

    return render_template('index.html',
                           image_path=image_path,
                           ai_description="\n".join(description_lines))

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)


