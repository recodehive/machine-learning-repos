import os
import streamlit as st
import pandas as pd
import joblib

# Safely resolve model and encoder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'linear_regression_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')

# Load the trained model and label encoders
model_lr = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)


# Function to predict insurance charges
def predict_insurance_charges(age, sex, bmi, children, smoker, region):
    # Transform categorical variables using label encoders
    sex_encoded = label_encoders['sex'].transform([sex])[0]
    smoker_encoded = label_encoders['smoker'].transform([smoker])[0]
    region_encoded = label_encoders['region'].transform([region])[0]

    # Prepare input data as DataFrame
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_encoded],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_encoded],
        'region': [region_encoded]
    })

    # Make prediction using the trained Linear Regression model
    predicted_charge = model_lr.predict(input_data)[0]

    return max(0.0, float(predicted_charge))


# Streamlit app
def main():
    st.title('Health Insurance Price Prediction')
    st.markdown('Enter the following details to predict insurance charges:')

    # Input fields with realistic defaults
    age = st.number_input('Age', min_value=0, max_value=100, value=25, step=1)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    children = st.number_input('Number of Children', min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox('Smoker', ['yes', 'no'])
    region = st.selectbox('Region of India', ['northeast', 'northwest', 'southeast', 'southwest'])

    if st.button('Predict'):
        # Call prediction function
        predicted_charge = predict_insurance_charges(age, sex, bmi, children, smoker, region)

        # Display prediction result in a green container with bold text
        st.markdown(
            f'<div style="background-color:#00FF00; padding:10px; border-radius:10px;"><h2 style="color:black; text-align:center;">Predicted Insurance Charge: <b>{predicted_charge:.2f}Rs</b></h2></div>',
            unsafe_allow_html=True)


if __name__ == '__main__':
    main()
