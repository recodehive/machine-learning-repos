import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained Naive Bayes model
model_nb = joblib.load('naive_bayes_model.pkl')

# Load the pre-trained CNN model
model_cnn = load_model('hair_no_hair_classifier.h5')

# Define unique values for encoding
unique_values = {
    'Genetics': ['Yes', 'No'],
    'Hormonal Changes': ['No', 'Yes'],
    'Medical Conditions': ['No medical problem',
                            'Eczema',
                            'Dermatosis',
                            'Ringworm',
                            'Psoriasis',
                            'Alopecia Areata ',
                            'Scalp Infection',
                            'Seborrheic Dermatitis',
                            'Dermatitis',
                            'Thyroid Problems',
                            'Androgenetic Alopecia'],
    'Medications & Treatments': ['No Medications & Treatments',
                                 'Antibiotics',
                                 'Antifungal Cream',
                                 'Accutane',
                                 'Chemotherapy',
                                 'Steroids',
                                 'Rogaine',
                                 'Blood Pressure Medication',
                                 'Immunomodulators',
                                 'Antidepressants ',
                                 'Heart Medication '],
    'Nutritional Deficiencies ': ['Magnesium deficiency',
                                  'Protein deficiency',
                                  'Biotin Deficiency ',
                                  'Iron deficiency',
                                  'Selenium deficiency',
                                  'Omega-3 fatty acids',
                                  'Zinc Deficiency',
                                  'Vitamin A Deficiency',
                                  'Vitamin D Deficiency',
                                  'No Nutritional Deficiencies',
                                  'Vitamin E deficiency'],
    'Stress': ['Moderate', 'High', 'Low'],
    'Poor Hair Care Habits ': ['Yes', 'No'],
    'Environmental Factors': ['Yes', 'No'],
    'Smoking': ['No', 'Yes'],
    'Weight Loss ': ['No', 'Yes']
}

# Initialize LabelEncoders
label_encoders = {col: LabelEncoder().fit(values) for col, values in unique_values.items()}

# Function to preprocess user input
def preprocess_input(data):
    # Apply label encoding to categorical columns
    for col in label_encoders:
        data[col] = label_encoders[col].transform(data[col])

    return data

# Function to predict hair loss
def predict_hair_loss(user_input):
    # Encode and preprocess the user input
    encoded_input = preprocess_input(pd.DataFrame([user_input]))

    # Predict using the loaded model
    prediction = model_nb.predict(encoded_input)

    return prediction[0]

# Function to preprocess and predict using the CNN model
def predict_image(image):
    img = Image.open(image)
    img = img.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model_cnn.predict(img_array)
    return 'Hair' if prediction[0][0] < 0.5 else 'No Hair'

# Streamlit app
st.set_page_config(page_title="Hair Loss Prediction", layout="centered")

# Custom CSS for enhanced styling
st.markdown("""
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
        color: skyblue;
        font-weight: bold;
    }
    .stSelectbox, .stSlider, .stTextInput {
        margin-bottom: 20px;
    }
    .stButton button:hover {
        background-color: #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Hair Loss Prediction</h1>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Menu")
options = st.sidebar.radio("Choose an option", ["Input-Based Prediction", "Image-Based Prediction"])

if options == "Input-Based Prediction":
    # Create input form
    with st.form("user_input_form"):
        st.markdown("<h3>Enter your information:</h3>", unsafe_allow_html=True)
        genetics = st.selectbox("Do you have family history of Baldness ? ", unique_values['Genetics'])
        hormonal_changes = st.selectbox("Do you experience any harmonal changes ? ", unique_values['Hormonal Changes'])
        medical_conditions = st.selectbox("Select the medical condition from which you are suffering currently (No medical condition if none) : ", unique_values['Medical Conditions'])
        medications_treatments = st.selectbox("Select Your ongoing medical treatments or surgeries (No medical Treatments if none) :", unique_values['Medications & Treatments'])
        nutritional_deficiencies = st.selectbox("Which nutritional deficiency do you have ? (No nutritional deficiencies if None) ", unique_values['Nutritional Deficiencies '])
        stress = st.selectbox("Enter the stress level :", unique_values['Stress'])
        age = st.slider("Enter your Age:", 0, 150, 30)
        poor_hair_care_habits = st.selectbox("Do you have Poor Hair Care Habits ? ", unique_values['Poor Hair Care Habits '])
        environmental_factors = st.selectbox("Did you expose to any Environmental Factors that cause hair loss ? ", unique_values['Environmental Factors'])
        smoking = st.selectbox("Do you smoke ? :", unique_values['Smoking'])
        weight_loss = st.selectbox("Did you experience significant Weight Loss ? ", unique_values['Weight Loss '])

        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        user_data = {
            'Genetics': genetics,
            'Hormonal Changes': hormonal_changes,
            'Medical Conditions': medical_conditions,
            'Medications & Treatments': medications_treatments,
            'Nutritional Deficiencies ': nutritional_deficiencies,
            'Stress': stress,
            'Age': age,
            'Poor Hair Care Habits ': poor_hair_care_habits,
            'Environmental Factors': environmental_factors,
            'Smoking': smoking,
            'Weight Loss ': weight_loss
        }

        # Predict hair loss
        predicted_class = predict_hair_loss(user_data)

        # Map numerical prediction back to original classes
        hair_loss_map = {0: 'Not Bald', 1: 'Bald'}
        predicted_class_label = hair_loss_map[predicted_class]

        # Display the result
        if predicted_class_label == 'Not Bald':
            st.success('You are Not Bald. You do not have a hair loss problem.')
        else:
            st.error('You are Bald. You have a hair loss problem.')

elif options == "Image-Based Prediction":
    st.markdown("<h3>Upload an image to predict hair loss:</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        label = predict_image(uploaded_file)

        if label == 'Hair':
            st.success('You are not suffering from baldness.')
        else:
            st.error('You are bald.')
