import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Safely resolve model and encoder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_xgb.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')

# Load the pre-trained model and label encoders
model_xgb = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)


# Function to preprocess user input
def preprocess_input(data):
    # Apply fitted label encoding to categorical columns
    cat_cols = ['Gender', 'Occupation', 'BMI Category']
    for col in cat_cols:
        if col in label_encoders:
            data[col] = label_encoders[col].transform(data[col])

    # Split the 'Blood Pressure' column safely into two numeric columns
    bp_parts = str(data['Blood Pressure'].iloc[0]).split('/')
    if len(bp_parts) == 2:
        data['Systolic BP'] = [pd.to_numeric(bp_parts[0].strip(), errors='coerce')]
        data['Diastolic BP'] = [pd.to_numeric(bp_parts[1].strip(), errors='coerce')]
    else:
        data['Systolic BP'] = [120]
        data['Diastolic BP'] = [80]

    # Drop the original 'Blood Pressure' column
    data = data.drop('Blood Pressure', axis=1)

    return data


# UI elements and CSS styling
st.set_page_config(page_title="Sleep Disorder Prediction", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        background-image: url('https://www.transparenttextures.com/patterns/asfalt-light.png');
        background-size: cover;
    }
    .stButton button {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        color: black;
        background-color: #00c853;
        border: none;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    h1 {
        text-align: center;
        font-family: Arial, sans-serif;
        color: red;
    }
    h3 {
        font-family: Arial, sans-serif;
        color: skyblue;
        font-size: 17px;
        text-align: left;
    }
    .container {
        border-radius: 15px;
        padding: 10px 20px;
        margin: 10px auto;
        width: 60%;
        max-width: 800px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>Sleep Disorder Prediction</h1>", unsafe_allow_html=True)

# Create input form
with st.form("user_input_form"):
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    st.markdown("<h3>Enter your information:</h3>", unsafe_allow_html=True)
    gender = st.selectbox("Gender:", ['Male', 'Female'])
    age = st.slider("Age:", 0, 100, 25)
    occupation = st.selectbox("Occupation:",
                              ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 'Engineer',
                               'Accountant', 'Scientist', 'Lawyer', 'Salesperson', 'Manager'])
    sleep_duration = st.slider("Sleep Duration (hours):", 1, 12, 8)
    quality_of_sleep = st.slider("Quality of Sleep (1-10):", 1, 10, 5)
    physical_activity = st.slider("Physical Activity Level (1-100):", 0, 100, 30)
    stress_level = st.slider("Stress Level (1-10):", 1, 10, 5)
    bmi_category = st.selectbox("BMI Category:", ['Underweight', 'Normal Weight', 'Overweight', 'Obese'])
    blood_pressure = st.text_input("Blood Pressure (Systolic/Diastolic) (ex: 120/80) :", "120/80")
    heart_rate = st.slider("Heart Rate (bpm):", 20, 120, 70)
    daily_steps = st.slider("Daily Steps:", 100, 20000, 5000)

    submit_button = st.form_submit_button(label="Predict")

    st.markdown("</div>", unsafe_allow_html=True)

if submit_button:
    user_data = {
        'Gender': gender,
        'Age': age,
        'Occupation': occupation,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'BMI Category': bmi_category,
        'Blood Pressure': blood_pressure,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps
    }

    user_df = pd.DataFrame([user_data])
    user_df = preprocess_input(user_df)

    # Predict sleep disorder class
    predicted_class = model_xgb.predict(user_df)[0]

    # Map numerical prediction back to original classes using fitted target encoder
    if 'Sleep Disorder' in label_encoders:
        predicted_class_label = label_encoders['Sleep Disorder'].inverse_transform([predicted_class])[0]
    else:
        sleep_disorder_map = {0: 'Insomnia', 1: 'No Disorder', 2: 'Sleep Apnea'}
        predicted_class_label = sleep_disorder_map.get(predicted_class, 'No Disorder')

    # Display the result in different colored containers
    container_style = """
        <div class='container' style='background-color: {}; color: black; width: 680px; height: 60px; display: flex; justify-content: center; align-items: flex-start;'>
            <h2 style='text-align: center; margin-top: -20px;'>{}</h2>
        </div>
    """

    if predicted_class_label == 'No Disorder':
        st.markdown(container_style.format('green', 'You are Healthy'), unsafe_allow_html=True)
    elif predicted_class_label == 'Sleep Apnea':
        st.markdown(container_style.format('grey', 'You are suffering from sleep apnea'), unsafe_allow_html=True)
    else:
        st.markdown(container_style.format('red', 'You are suffering from insomnia'), unsafe_allow_html=True)
