from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np

def create_model(embedding_matrix, max_sequence_length, max_words):
    model = Sequential()
    model.add(Embedding(max_words, embedding_matrix.shape[1], input_length=max_sequence_length, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)))  
    model.add(Dropout(0.5))  # Increased dropout rate
    model.add(LSTM(128, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)))  
    model.add(Dropout(0.5))  # Increased dropout rate
    model.add(Dense(64, activation="relu"))  
    model.add(BatchNormalization())
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model

def classify_text(input_text, model, tokenizer, max_sequence_length):
    # Tokenize and pad the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length)

    # Predict the class probability
    prediction = model.predict(input_padded)

    # Determine the predicted class label
    predicted_label = "AI-Generated 😔" if prediction[0] >= 0.5 else "Human-Generated"

    return predicted_label
