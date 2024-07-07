import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

# Function to convert user input to model input
def convert_user_input(gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain):
    gender = 1 if gender.lower() == 'male' else 2
    smoking = 1 if smoking.lower() == 'yes' else 0
    yellow_fingers = 1 if yellow_fingers.lower() == 'yes' else 0
    anxiety = 1 if anxiety.lower() == 'yes' else 0
    peer_pressure = 1 if peer_pressure.lower() == 'yes' else 0
    chronic_disease = 1 if chronic_disease.lower() == 'yes' else 0
    fatigue = 1 if fatigue.lower() == 'yes' else 0
    allergy = 1 if allergy.lower() == 'yes' else 0
    wheezing = 1 if wheezing.lower() == 'yes' else 0
    alcohol_consuming = 1 if alcohol_consuming.lower() == 'yes' else 0
    coughing = 1 if coughing.lower() == 'yes' else 0
    shortness_of_breath = 1 if shortness_of_breath.lower() == 'yes' else 0
    swallowing_difficulty = 1 if swallowing_difficulty.lower() == 'yes' else 0
    chest_pain = 1 if chest_pain.lower() == 'yes' else 0

    return [gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]

# Function to predict lung cancer based on user input
def predict_lung_cancer(user_input):
    prediction = model.predict([user_input])
    return "You have Lung Cancer" if prediction[0] == 1 else "You don't have Lung Cancer"

# Streamlit app
def main():
    st.set_page_config(page_title="Lung Cancer Prediction App", page_icon="🩺", layout="centered")

    st.markdown(
        """
        <style>
        body {
            background-color: #1e1e1e;
        }
        .main:before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 200%; /* Increase horizontal coverage */
            height: 100%;
            background-image: url('https://images.rawpixel.com/image_400/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjkyOWJhdGNoMi1rdWwtMDIteC5qcGc.jpg'); /* Replace with your image URL */
            background-size: auto 100%; /* Adjust to maintain aspect ratio and cover vertically */
            background-repeat: repeat-x; /* Repeat horizontally */
            opacity: 0.2; /* Adjust opacity for less faint appearance */
            z-index: -1; /* Send it to the back */
        }
        .main > * {
            position: relative;
            z-index: 2;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #f39c12;
            text-align: center;
        }
        .subheader {
            font-size: 20px;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 20px; /* Added spacing below */
        }
        .input_label {
            font-size: 16px;
            color: #ecf0f1;
        }
        .button {
            background-color: #e67e22;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
        }
        .result {
            font-size: 20px;
            font-weight: bold;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .result.red {
            background-color: #e74c3c;
            color: white;
        }
        .result.green {
            background-color: #27ae60;
            color: white;
        }
        .stTextInput, .stSelectbox, .stButton {
            background-color: #3c3c3c !important;
            border: 1px solid #555 !important;
            color: #ecf0f1 !important;
            padding: 10px !important;
            border-radius: 5px !important;
        }
        .stSlider {
            background-color: #3c3c3c !important;
            color: #ecf0f1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Lung Cancer Prediction App</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Enter your details to predict the risk of lung cancer</div>", unsafe_allow_html=True)

    # User input
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    age = st.slider("Age", 0, 100, 25)
    smoking = st.selectbox("Do you smoke?", ["Yes", "No"], index=1)
    yellow_fingers = st.selectbox("Do you have yellow fingers?", ["Yes", "No"], index=1)
    anxiety = st.selectbox("Do you have anxiety?", ["Yes", "No"], index=1)
    peer_pressure = st.selectbox("Do you experience peer pressure?", ["Yes", "No"], index=1)
    chronic_disease = st.selectbox("Do you have a chronic disease?", ["Yes", "No"], index=1)
    fatigue = st.selectbox("Do you experience fatigue?", ["Yes", "No"], index=1)
    allergy = st.selectbox("Do you have allergies?", ["Yes", "No"], index=1)
    wheezing = st.selectbox("Do you wheeze?", ["Yes", "No"], index=1)
    alcohol_consuming = st.selectbox("Do you consume alcohol?", ["Yes", "No"], index=1)
    coughing = st.selectbox("Do you cough?", ["Yes", "No"], index=1)
    shortness_of_breath = st.selectbox("Do you have shortness of breath?", ["Yes", "No"], index=1)
    swallowing_difficulty = st.selectbox("Do you have difficulty swallowing?", ["Yes", "No"], index=1)
    chest_pain = st.selectbox("Do you have chest pain?", ["Yes", "No"], index=1)

    # Convert input
    user_input = convert_user_input(
        gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain
    )

    # Predict button
    if st.button("Predict", key='predict_button'):
        result = predict_lung_cancer(user_input)
        if "don't" in result:
            st.markdown(f"<div class='result green'>{result}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result red'>{result}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
