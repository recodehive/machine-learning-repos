import os
import io
import base64
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image

def load_model_and_labels():
    model_path = r"C:\Users\Chimni\Desktop\New folder\Notebooks\Models\Brain_Tumor.h5"
    model = tf.keras.models.load_model(model_path)
    class_labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
    return model, class_labels

def preprocess_image(img):
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

app = Flask(__name__)
model, class_labels = load_model_and_labels()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = request.files['image']
        img = Image.open(image.stream)
        img = preprocess_image(img)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        img_base64 = image_to_base64(img)
        return render_template('result.html', description=predicted_label, image_data=img_base64)
    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
