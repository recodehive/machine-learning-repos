from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import base64

app = Flask(__name__)
model = load_model('model.h5')
img_width, img_height = 150, 150
class_labels = ['male', 'female']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                roi_color = frame[y:y + h, x:x + w]
                resized_frame = cv2.resize(roi_color, (img_width, img_height))
                normalized_frame = resized_frame / 255.0
                reshaped_frame = np.reshape(normalized_frame, (1, img_width, img_height, 3))
                predictions = model.predict(reshaped_frame)
                male_confidence = predictions[0][0] * 100
                female_confidence = 100 - male_confidence
                if male_confidence >= 50:
                    label = f"male: {male_confidence:.2f}%"
                    color = (0, 255, 0)
                else:
                    label = f"female: {female_confidence:.2f}%"
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            img = cv2.imread(filepath)
            return predict_image(img)
    return render_template('upload.html')

def predict_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi_color = img[y:y + h, x:x + w]
        resized_frame = cv2.resize(roi_color, (img_width, img_height))
        normalized_frame = resized_frame / 255.0
        reshaped_frame = np.reshape(normalized_frame, (1, img_width, img_height, 3))
        predictions = model.predict(reshaped_frame)
        male_confidence = predictions[0][0] * 100
        female_confidence = 100 - male_confidence
        if male_confidence >= 50:
            gender_prediction = f"Male ( {male_confidence:.2f}% ) "
            container_class = "male-container"
        else:
            gender_prediction = f"Female ( {female_confidence:.2f}% ) "
            container_class = "female-container"

        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_str = buffer.tobytes()
        img_data = 'data:image/jpeg;base64,' + base64.b64encode(img_str).decode('utf-8')
        return render_template('result.html', img_data=img_data, gender_prediction=gender_prediction, container_class=container_class)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
