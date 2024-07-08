import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the pre-trained model
model_xgb = joblib.load('model_xgb.pkl')

# Load label encoders for categorical variables
label_encoder = LabelEncoder()

# Function to preprocess user input
def preprocess_input(data):
    # Apply label encoding to categorical columns
    cat_cols = ['Gender', 'Occupation', 'BMI Category']
    for col in cat_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # Split the 'Blood Pressure' column into two columns
    data[['Systolic BP', 'Diastolic BP']] = data['Blood Pressure'].str.split('/', expand=True)

    # Convert the new columns to numeric type
    data[['Systolic BP', 'Diastolic BP']] = data[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)

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
    daily_steps = st.slider("Daily Steps:", 100 ,20000, 5000)

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
    predicted_class = model_xgb.predict(user_df)

    # Map numerical prediction back to original classes
    sleep_disorder_map = {0: 'No Disorder', 1: 'Sleep Apnea', 2: 'Insomnia'}
    predicted_class_label = sleep_disorder_map[predicted_class[0]]

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
