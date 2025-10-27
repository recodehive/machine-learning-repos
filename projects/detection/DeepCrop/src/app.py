from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = Flask(__name__)
CORS(app)  # allow React frontend to talk to Flask backend

# ---- config ----
MODEL_PATH = "../models/model.keras"           # your model file
IMG_SIZE = (128, 128)                # must match your training size
UPLOAD_DIR = "../data/temp"                  # temporary save for incoming files
os.makedirs(UPLOAD_DIR, exist_ok=True)

# These are the 3 classes we want to serve in the app UI
SERVE_LABELS = ["Healthy", "Early Blight", "Late Blight"]

# ---- load model once ----
model = load_model(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]

    # save with random name to avoid collisions on repeated tests
    ext = os.path.splitext(f.filename)[1].lower()
    fname = f"{uuid.uuid4().hex}{ext if ext else '.jpg'}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    f.save(fpath)

    # --- preprocess exactly like training ---
    img = image.load_img(fpath, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0) / 255.0

    # --- raw prediction (model outputs 10 classes in your case) ---
    preds = model.predict(arr)

    # ---- IMPORTANT: keep only the first 3 outputs (Healthy, Early, Late)
    # If your model’s class order differs, adjust the slice or indices here.
    trimmed = preds[0][:3].astype(np.float64)

    # renormalize the 3 scores so they sum to 1
    total = np.sum(trimmed)
    if total > 0:
        trimmed /= total

    idx = int(np.argmax(trimmed))
    conf = float(trimmed[idx])

    # logging for debugging in console
    print("Raw preds:", preds)
    print("Trimmed (3-class) preds:", trimmed)
    print("Predicted:", SERVE_LABELS[idx], "Confidence:", conf)

    return jsonify({
        "label": SERVE_LABELS[idx],
        "confidence": round(conf * 100, 2)
    })


@app.route("/", methods=["GET"])
def health():
    # simple health check
    return jsonify({"ok": True, "model": MODEL_PATH})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
