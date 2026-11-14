import os
import io
import base64
from flask import Flask, render_template, request
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except ImportError:
    tf = None

CLASS_LABELS = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']


def load_model_and_labels():
    if tf is None:
        return None, CLASS_LABELS
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, "best_model.h5")
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model, CLASS_LABELS
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            return None, CLASS_LABELS
    return None, CLASS_LABELS


def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


app = Flask(__name__, static_folder='styles')
model, class_labels = load_model_and_labels()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files or not request.files['image'].filename:
            return render_template('error.html', error_message="No image file was uploaded.")

        image_file = request.files['image']
        raw_img = Image.open(image_file.stream).convert('RGB')

        global model
        if model is None:
            model, _ = load_model_and_labels()

        if model is None:
            return render_template(
                'error.html',
                error_message="Model file 'best_model.h5' was not found in the Web App directory. Please ensure the trained model is placed in the project directory."
            )

        processed_img = preprocess_image(raw_img)
        predictions = model.predict(processed_img)
        predicted_class = int(np.argmax(predictions))
        predicted_label = class_labels[predicted_class]
        img_base64 = image_to_base64(raw_img)

        return render_template('result.html', description=predicted_label, image_data=img_base64)
    except Exception as e:
        return render_template('error.html', error_message=str(e))


if __name__ == '__main__':
    app.run(debug=True)
