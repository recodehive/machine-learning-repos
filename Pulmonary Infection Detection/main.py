import numpy as np 
import streamlit as st
import os
import librosa
from tensorflow.keras.models import load_model


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/Respiratory_Disease_Classifier.h5"
model = load_model(model_path)

def preprocess_audio(audio_file):
    # Load audio file
    audio_data, sample_rate = librosa.load(audio_file, sr=22050, mono=True)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    mfccs_processed = np.expand_dims(mfccs_processed, axis=0)

    return mfccs_processed


st.title('💨 Respiratory Disease Classifier')
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])


if uploaded_file is not None:
    # Process the audio file
    with st.spinner('Processing audio...'):
        # Load the audio file
        audio_data, _ = st.read(uploaded_file)
        mfccs_processed = preprocess_audio(uploaded_file)
        predicted_probabilities = model.predict(mfccs_processed)
        predicted_class_index = np.argmax(predicted_probabilities)
        # Map class index to disease name
        class_mapping = {0: "Normal", 1: "Asthma", 2: "Bronchiectasis", 3: "Bronchiolitis", 4: "COPD", 5: "Pneumonia", 6: "URTI", 7: "LRTI"}
        predicted_class = class_mapping[int(predicted_class_index)]

    st.write(f"Predicted respiratory disease: {predicted_class}")
