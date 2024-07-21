import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

# Load the model
pickle_in = open('model.pkl', 'rb')
classifier = pickle.load(pickle_in)

# Prediction function
def prediction(AQI, PM10, PM2_5, NO2, SO2, O3, Temperature, Humidity,
               WindSpeed, RespiratoryCases, CardiovascularCases,
               HospitalAdmissions):
    prediction = classifier.predict(
        [[float(AQI), float(PM10), float(PM2_5), float(NO2), float(SO2), float(O3), float(Temperature), float(Humidity),
          float(WindSpeed), float(RespiratoryCases), float(CardiovascularCases), float(HospitalAdmissions)]])
    return prediction

# Main function to define the webpage
def main():
    # Set the title of the app
    st.title("Air Quality Analysis and Health Impact Score Prediction")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
        }
        h1 {
            color: #004080;
            text-align: center;
        }
        h2 {
            color: #FF5733;
            text-align: center;
        }
        .stButton button {
            background-color: #004080;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stTextInput input {
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
        .stTextInput input::placeholder {
            color: grey;
        }
        .result {
            font-size: 24px;
            color: #004080;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        .result-less {
            background-color: #d4edda;
            color: #155724;
        }
        .result-average {
            background-color: #fff3cd;
            color: #856404;
        }
        .result-severe {
            background-color: #f8d7da;
            color: #721c24;
        }
        .stTextInput label {
            color: #004080;
            font-weight: bold;
        }
        .description {
            font-size: 22px;
            color: #004080;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Introduction about Air Quality and Health Impact
    st.markdown("""
        <div class="description">
            <h2>Introduction</h2>
            <p>Air quality significantly affects human health and the environment. Poor air quality can lead to respiratory and cardiovascular diseases,
            increased hospital admissions, and overall lower quality of life. Monitoring air quality and understanding its impact on health is crucial
            for implementing measures to improve the environment and public health.</p>
        </div>
        """, unsafe_allow_html=True)

    # Input fields in two columns
    col1, col2 = st.columns(2)

    with col1:
        AQI = st.text_input("Air Quality Index", placeholder="Enter AQI Value")
        PM10 = st.text_input("Particulate Matter 10", placeholder="Enter PM10 Value")
        NO2 = st.text_input("NO2 Score", placeholder="Enter NO2 Score")
        O3 = st.text_input("O3", placeholder="Enter O3 Score")
        Humidity = st.text_input("Humidity", placeholder="Enter Humidity")
        CardiovascularCases = st.text_input("Cardiovascular Cases per month (Default=10)", placeholder="Enter Cardiovascular Cases per month", value="10")

    with col2:
        PM2_5 = st.text_input("Particulate Matter 2.5", placeholder="Enter PM2.5 Value")
        SO2 = st.text_input("SO2", placeholder="Enter SO2 Score")
        Temperature = st.text_input("Temperature", placeholder="Enter Temperature")
        WindSpeed = st.text_input("Wind Speed", placeholder="Enter Wind Speed")
        RespiratoryCases = st.text_input("Respiratory Cases per month (Default=10)", placeholder="Enter Respiratory Cases per month", value="10")
        HospitalAdmissions = st.text_input("Hospital Admissions per month (Default=10)", placeholder="Enter Hospital Admissions per month", value="10")

    result = ""

    # Predict button
    if st.button("Predict"):
        result = prediction(AQI, PM10, PM2_5, NO2, SO2, O3, Temperature, Humidity,
                            WindSpeed, RespiratoryCases, CardiovascularCases, HospitalAdmissions)[0]
        
        # Determine result box class and message
        if result < 35:
            result_class = 'result-less'
            result_message = 'Less Impacting'
        elif 36 <= result <= 45:
            result_class = 'result-average'
            result_message = 'Average Impacting'
        else:
            result_class = 'result-severe'
            result_message = 'Severely Impacting'

        st.markdown(f'<div class="result-box {result_class}">The output is {result}<br>{result_message}</div>', unsafe_allow_html=True)

        # Summary of the result
        st.markdown("""
            <div class="description">
                <h2>Result Summary</h2>
                <p>Based on the input data, the predicted health impact score indicates the level of impact due to air quality.</p>
                <ul>
                <li> A score less than 35 suggests a less impacting air quality on health.</li> 
                <li> A score between 35 and 45 indicates an average impact.</li>
                <li> while a score higher than 45 suggests a severe impact on health.</li>
                </ul> 
                <p>It is essential to take necessary measures to mitigate air pollution and improve air quality
                to reduce health risks.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
