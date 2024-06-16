import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Streamlit app title
st.title('Bank customer churn prediction app')

# Load the model
def load_model():
    try:
        model = joblib.load('../Model/model.pkl')
        return model
    except Exception as e:
        st.write(f"Error loading the model: {e}")
        return None

model = load_model()

if model:
    st.write("Model loaded successfully.")
else:
    st.write("Failed to load the model.")

# Initialize scalers
minmax_scaler = MinMaxScaler()
Standard_Scaler=StandardScaler()
# Load training data
training_data = pd.read_csv('../Dataset/Customer.csv') 

# Fit the scaler on training data
minmax_scaler.fit(training_data[['Tenure', 'NumOfProducts']])
Standard_Scaler.fit(training_data[['CreditScore', 'Balance'  ]])

def preprocess_input(data):
    # Copy the input data to avoid modifying the original data
    processed_data = data.copy()
    CreditScore_scaled = Standard_Scaler.transform([[processed_data['CreditScore'], 0]])[0][0]
    processed_data['CreditScore'] = CreditScore_scaled
    # Convert Gender to numerical value
    processed_data['Gender'] = 1 if processed_data['Gender'] == 'Male' else 0
    # MinMax scaling
    tenure_scaled = minmax_scaler.transform([[processed_data['Tenure'], 0]])[0][0]
    processed_data['Tenure'] = tenure_scaled
    Balance_scaled = Standard_Scaler.transform([[processed_data['Balance'], 0]])[0][0]
    processed_data['Balance'] = Balance_scaled
    num_of_products_scaled = minmax_scaler.transform([[processed_data['NumOfProducts'], 0]])[0][0]
    processed_data['NumOfProducts'] = num_of_products_scaled
        
    return processed_data


def make_prediction(model, data):
    # Preprocess input data
    processed_data = preprocess_input(data)
    
    # Assuming 'data' is a dictionary containing the input features
    input_data = pd.DataFrame(processed_data, index=[0])
    print("Input Data:", input_data)  # Debug statement
    prediction = model.predict(input_data)
    print("Prediction:", prediction)  # Debug statement
    return prediction

def main():
    st.title("Credit Approval Prediction")

    # Example input data
    new_data = {
        'CreditScore': st.slider('Credit Score', 300, 850, 600),
        'Gender': st.radio('Gender', ['Male', 'Female']),
        'Age': st.slider('Age', 18, 100, 40),
        'Tenure': st.slider('Tenure', 0, 10, 3),
        'Balance': st.number_input('Account Balance', value=50000),
        'NumOfProducts': st.slider('Number of Products', 1, 4, 2),
        'HasCrCard': st.selectbox('Has Credit Card', [0, 1]),
        'IsActiveMember': st.selectbox('Is Active Member', [0, 1]),
        'Geography_Germany': st.selectbox('Geography (Germany)', [0, 1]),
        'Geography_Spain': st.selectbox('Geography (Spain)', [0, 1])
    }

    # Convert Gender to numerical value
    new_data['Gender'] = 1 if new_data['Gender'] == 'Male' else 0

    # Make prediction on new data
    prediction = make_prediction(model, new_data)
    st.write("Predicted Class:", prediction)
    if prediction==1:
        st.write("The customer is likely to leave the bank.")
    else:
        st.write("The customer is likely to stay with the bank.")

if __name__ == "__main__":
    main()
