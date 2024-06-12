import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from fastdtw import fastdtw
# Load your Siamese model
# siamese_model = tf.keras.models.load_model('siamese_net.h5', compile=False)  # Use compile=False
class SessionState:
    def __init__(self):
        self.reset_form_open = False

# Create a SessionState object
session_state = SessionState()
def render_dashboard(session_state):
    st.success('Successfully logged in')
    st.title("Dashboard")
    st.write("Welcome to the dashboard!")

    st.title("Signature Verification using Dynamic Time Wrapping")

    path1 = st.file_uploader("Signature 1", type=["png", "jpg"])
    path2 = st.file_uploader("Signature 2", type=["png", "jpg"])
    submitted = st.button(label="Submit")
    if path1 is not None and path2 is not None and submitted:
        # Load and preprocess the uploaded images
        img1 = preprocess_image(path1)
        img2 = preprocess_image(path2)

        # Perform signature verification using your Siamese model
        similarity_score = verify_signature(img1, img2)

        if similarity_score < 110000000:
            st.write(f"Forged Signatures, Similarity Score: {similarity_score:.2f}")
        else:
            st.write(f"Original Signatures, Similarity Score: {similarity_score:.2f}")

    logout_button = st.button("Logout")
    if logout_button:
        session_state.is_authenticated = False
        st.experimental_rerun()
    reset_password_button = st.button("Reset Password")
    if reset_password_button:
    # Display a form to reset the password
        session_state.reset_form_open = True
        if session_state.reset_form_open:
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            reset_button = st.button("Reset")

            if new_password and confirm_password and reset_button:
                if new_password == confirm_password:
                    # Update the user's password in the database
                    st.success("Password reset successful!")
                    session_state.reset_form_open = False
                else:
                    st.warning("Passwords do not match. Please make sure they match.")



def preprocess_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert("L")  # Convert to grayscale

    # Resize to match the input size (adjust as needed)
    img = img.resize((551, 1117))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Convert the image to sequences of points (coordinates)
    points = np.argwhere(img_array > 128)  # Adjust the threshold as needed

    return points





# def preprocess_image(image_path):
#     # Load and preprocess the image
#     img = Image.open(image_path).convert("L")  # Convert to grayscale
#     img = img.resize((551, 1117))  # Resize to match the input size

#     # Convert the image to a NumPy array
#     img = np.array(img)

#     # Normalize the image to [0, 1] if needed
#     img = img / 255.0

#     return img




def verify_signature(image1, image2):
    # Perform signature verification using DTW
    distance, _ = fastdtw(image1, image2)
    return distance






# def verify_signature(image1, image2):
#     # Perform signature verification using structural similarity
#     similarity_score = ssim(image1, image2, data_range=1)
#     return similarity_score

# Usage example
if __name__ == "__main__":
    session_state = st.session_state
    render_dashboard(session_state)
