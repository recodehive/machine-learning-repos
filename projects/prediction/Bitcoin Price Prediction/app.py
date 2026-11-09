import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Safely resolve model and image paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'bitcoin_model.pkl')
IMAGE_PATH = os.path.join(BASE_DIR, 'price.png')

# Load pre-trained Lasso regression model
model = joblib.load(MODEL_PATH)

r = st.sidebar.radio("Navigation Menu", ["Home", "Bitcoin Price"])

if r == "Home":
    st.write("""
    # Bitcoin Price Predictive 
    """)
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH)
    st.subheader("Bitcoin Price & Marketcap Prediction")
    st.write(
        "Bitcoin is a widely used decentralized cryptocurrency for the digital market. "
        "Transactions are simple, secure, and recorded on the blockchain. "
        "Bitcoin price is variable and it is widely used, making it essential to analyze and predict "
        "its valuation for informed investment decisions."
    )

if r == "Bitcoin Price":
    st.header("Know the Valuation of Bitcoin")
    high = st.number_input("Highest Price of Bitcoin ($)", min_value=0.0, value=50000.0, step=100.0)
    low = st.number_input("Lowest Price of Bitcoin ($)", min_value=0.0, value=48000.0, step=100.0)
    open_price = st.number_input("Opening Price of Bitcoin ($)", min_value=0.0, value=49000.0, step=100.0)
    close_price = st.number_input("Closing Price of Bitcoin ($)", min_value=0.0, value=49500.0, step=100.0)
    volume = st.number_input("Trading Volume of Bitcoin", min_value=0.0, value=30000000000.0, step=1000000.0)

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            'High': high,
            'Low': low,
            'Open': open_price,
            'Close': close_price,
            'Volume': volume
        }])
        ypred = model.predict(input_data)[0]
        st.success(f"Predicted Bitcoin Market Capitalization: ${abs(float(ypred)):,.2f}")
