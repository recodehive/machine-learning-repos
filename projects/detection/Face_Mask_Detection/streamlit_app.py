import os
import cv2
import numpy as np
import streamlit as st
import PIL.Image

try:
    import keras
except ImportError:
    from tensorflow import keras

# Safely resolve model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mask_detection.keras')
model = keras.models.load_model(MODEL_PATH)


def main():
    st.title("Mask Detection App")
    st.write("Please choose an option")

    option = st.selectbox("Select an option", ("Upload Image", "Capture Photo"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read and preprocess the image (ensuring 3 RGB channels)
            image = PIL.Image.open(uploaded_file).convert('RGB')
            image_resized = image.resize((128, 128))
            image_array = np.array(image_resized, dtype='float32') / 255.0
            input_image_reshaped = np.reshape(image_array, [1, 128, 128, 3])

            # Make predictions
            predictions = model.predict(input_image_reshaped)
            input_pred_label = np.argmax(predictions)

            # Display the result
            if input_pred_label == 1:
                st.markdown("<h3 style='text-align: center; color: green;'>Prediction: With Mask</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='text-align: center; color: red;'>Prediction: Without Mask</h3>", unsafe_allow_html=True)

            # Display the uploaded image
            st.write("")
            st.write("**Uploaded Image**")
            st.image(image, use_container_width=True)

    elif option == "Capture Photo":
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        video_capture.release()

        if ret and frame is not None:
            # Convert captured frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to model expected shape (128, 128)
            image = cv2.resize(frame, (128, 128))
            image = image.astype('float32') / 255.0
            input_image_reshaped = np.reshape(image, [1, 128, 128, 3])

            # Make predictions
            predictions = model.predict(input_image_reshaped)
            input_pred_label = np.argmax(predictions)

            # Display the result
            if input_pred_label == 1:
                st.markdown("<h3 style='text-align: center; color: green;'>Prediction: With Mask</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='text-align: center; color: red;'>Prediction: Without Mask</h3>", unsafe_allow_html=True)

            # Display the captured photo
            st.write("")
            st.write("**Captured Photo**")
            st.image(frame, use_container_width=True)
        else:
            st.warning("Webcam not detected or unavailable. Please upload an image instead.")


if __name__ == '__main__':
    main()
