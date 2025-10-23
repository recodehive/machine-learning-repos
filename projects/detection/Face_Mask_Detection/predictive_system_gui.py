import cv2
import numpy as np
import keras
import tk as tk
from PIL import ImageTk, Image
import tkinter as tk
from tkinter import filedialog
import os

# Load the face mask detector model
model =keras.models.load_model("mask_detection.keras")

# Initialize the GUI
root = tk.Tk()
root.title("Mask Detection")
root.geometry("500x400")

# Function to classify the image
def classify_image(image):
    # Preprocess the image
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255.0
    input_image_reshaped = np.reshape(image, [1,128, 128, 3])
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
    file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File",
                                               filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
    if file_path:
        # Display the uploaded image in the GUI
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        photo_label.configure(image=photo)
        photo_label.image = photo

        classify_image(image)

# Function to handle capturing photo from webcam
def capture_photo():
    video_capture = cv2.VideoCapture(0)
    _, frame = video_capture.read()
    video_capture.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    classify_image(image)

    # Display the captured photo in the GUI
    image.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(image)
    photo_label.configure(image=photo)
    photo_label.image = photo

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=10)

# Create the GUI components
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

capture_button = tk.Button(root, text="Capture Photo", command=capture_photo)
capture_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

photo_label = tk.Label(root)
photo_label.pack()

# Run the GUI main loop
root.mainloop()
