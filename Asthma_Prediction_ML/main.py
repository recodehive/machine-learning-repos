import streamlit as st
import numpy as np
from joblib import load

# Load the trained RandomForestClassifier model
model = load('rf_asthma_model_prediction.pkl')

# Function to preprocess user inputs
def preprocess_inputs(tiredness, dry_cough, difficulty_breathing, sore_throat, pains, nasal_congestion, runny_nose,
                      age_group, gender):
    # Convert inputs to integers (Yes=1, No=0)
    symptoms = [1 if symptom == 'Yes' else 0 for symptom in [tiredness, dry_cough, difficulty_breathing,
                                                             sore_throat, pains, nasal_congestion, runny_nose]]

    # Convert age_group to one-hot encoded format
    age_0_9 = 1 if age_group == '0s-9' else 0
    age_10_19 = 1 if age_group == '10-19' else 0
    age_20_24 = 1 if age_group == '20-24' else 0
    age_25_59 = 1 if age_group == '25-59' else 0
    age_60_plus = 1 if age_group == '60+' else 0

    # Convert gender to one-hot encoded format
    gender_female = 1 if gender == 'Female' else 0
    gender_male = 1 if gender == 'Male' else 0

    # Combine all inputs into a numpy array
    inputs = np.array(
        symptoms + [age_0_9, age_10_19, age_20_24, age_25_59, age_60_plus, gender_female, gender_male]).reshape(1, -1)

    return inputs

# Function to predict asthma severity
def predict_asthma(inputs):
    prediction = model.predict(inputs)[0]
    if prediction == 1:
        return "Mild Asthma"
    elif prediction == 2:
        return "Moderate Asthma"
    elif prediction == 3:
        return "No Asthma"
    else:
        return "Unknown"

# Streamlit app
def main():
    st.title('Asthma Prediction App')
    st.write('Enter your symptoms and demographics to predict asthma:')

    # User inputs
    tiredness = st.radio('Are you suffering from tiredness?', ('Yes', 'No'))
    dry_cough = st.radio('Do you have dry cough?', ('Yes', 'No'))
    difficulty_breathing = st.radio('Do you experience difficulty in breathing?', ('Yes', 'No'))
    sore_throat = st.radio('Do you have sore throat?', ('Yes', 'No'))
    pains = st.radio('Do you have pains?', ('Yes', 'No'))
    nasal_congestion = st.radio('Do you have nasal congestion?', ('Yes', 'No'))
    runny_nose = st.radio('Do you have runny nose?', ('Yes', 'No'))

    age_group = st.selectbox('Select your age group:', ('0-9', '10-19', '20-24', '25-59', '60+'))
    gender = st.selectbox('Select your gender:', ('Female', 'Male'))

    if st.button('Predict'):
        # Preprocess inputs
        inputs = preprocess_inputs(tiredness, dry_cough, difficulty_breathing, sore_throat, pains, nasal_congestion,
                                   runny_nose, age_group, gender)

        # Predict asthma
        prediction = predict_asthma(inputs)

        # Display prediction result
        st.subheader('Prediction:')
        if prediction == "Mild Asthma":
            st.error("You are suffering from Mild Asthma")
        elif prediction == "Moderate Asthma":
            st.error("You are suffering from Moderate Asthma")
        elif prediction == "No Asthma":
            st.success("You are not suffering from Asthma")
        else:
            st.warning("Prediction could not be made. Please check your inputs.")

if __name__ == '__main__':
    main()
