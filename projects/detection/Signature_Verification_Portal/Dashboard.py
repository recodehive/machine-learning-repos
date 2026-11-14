import streamlit as st
from PIL import Image
import os
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None


class SessionState:
    def __init__(self):
        self.reset_form_open = False


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
        img1 = preprocess_image(path1)
        img2 = preprocess_image(path2)

        similarity_score = verify_signature(img1, img2)

        if similarity_score < 110000000:
            st.write(f"Forged Signatures, Similarity Score: {similarity_score:.2f}")
        else:
            st.write(f"Original Signatures, Similarity Score: {similarity_score:.2f}")

    logout_button = st.button("Logout")
    if logout_button:
        session_state.is_authenticated = False
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

    reset_password_button = st.button("Reset Password")
    if reset_password_button:
        session_state.reset_form_open = True
        if session_state.reset_form_open:
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            reset_button = st.button("Reset")

            if new_password and confirm_password and reset_button:
                if new_password == confirm_password:
                    st.success("Password reset successful!")
                    session_state.reset_form_open = False
                else:
                    st.warning("Passwords do not match. Please make sure they match.")


def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((551, 1117))
    img_array = np.array(img)
    points = np.argwhere(img_array > 128)
    return points


def verify_signature(image1, image2):
    if fastdtw is not None:
        distance, _ = fastdtw(image1, image2)
        return distance
    min_len = min(len(image1), len(image2))
    if min_len == 0:
        return float('inf')
    return float(np.sum(np.abs(image1[:min_len] - image2[:min_len])))


if __name__ == "__main__":
    session_state = st.session_state
    render_dashboard(session_state)
