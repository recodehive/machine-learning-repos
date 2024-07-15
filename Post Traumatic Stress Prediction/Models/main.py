import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Define paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/PTSD_Detection_Model.h5"
model = load_model(model_path)

# Preprocessing function
def preprocess_text(text):
    # Tokenize text
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=100)
    return padded_sequences

# Streamlit app
st.title('🧠 PTSD Detection')
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    # Read the text file
    text_data = uploaded_file.read().decode("utf-8")
    text_data = [text_data]

    # Preprocess the text data
    with st.spinner('Processing text...'):
        preprocessed_text = preprocess_text(text_data)
        predicted_probabilities = model.predict(preprocessed_text)
        predicted_class_index = np.argmax(predicted_probabilities)
        
        # Map class index to PTSD status
        class_mapping = {0: "No PTSD", 1: "PTSD"}
        predicted_class = class_mapping[int(predicted_class_index)]

    st.write(f"Predicted PTSD status: {predicted_class}")
