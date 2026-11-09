import os
import cv2
import numpy as np
from PIL import ImageTk, Image
import tkinter as tk
from tkinter import filedialog

try:
    import keras
except ImportError:
    from tensorflow import keras

# Safely resolve model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mask_detection.keras")
model = keras.models.load_model(MODEL_PATH)

# Initialize the GUI
root = tk.Tk()
root.title("Mask Detection")
root.geometry("500x400")


# Function to classify the image
def classify_image(image):
    # Ensure 3-channel RGB representation
    image = image.convert('RGB')
    image = image.resize((128, 128))
    image = np.array(image, dtype='float32') / 255.0
    input_image_reshaped = np.reshape(image, [1, 128, 128, 3])

    # Make predictions
    predictions = model.predict(input_image_reshaped)
    input_pred_label = np.argmax(predictions)

    if input_pred_label == 1:
        result = "With Mask"
    else:
        result = "Without Mask"

    # Update the result label text
    result_label.configure(text="Prediction: " + result)


# Function to handle file upload
def upload_image():
    file_path = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Select Image File",
        filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
    )
    if file_path:
        # Display the uploaded image in the GUI
        image = Image.open(file_path)
        display_img = image.copy()
        display_img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(display_img)
        photo_label.configure(image=photo)
        photo_label.image = photo

        classify_image(image)


# Function to handle capturing photo from webcam
def capture_photo():
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    video_capture.release()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        classify_image(image)

        # Display the captured photo in the GUI
        display_img = image.copy()
        display_img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(display_img)
        photo_label.configure(image=photo)
        photo_label.image = photo


# Create the GUI components
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=10)

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

capture_button = tk.Button(root, text="Capture Photo", command=capture_photo)
capture_button.pack(pady=10)

photo_label = tk.Label(root)
photo_label.pack()

# Run the GUI main loop
if __name__ == "__main__":
    root.mainloop()
