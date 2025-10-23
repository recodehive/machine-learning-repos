from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index

def read_text_from_uploaded_file(file):
    try:
        text = file.read().decode('utf-8')
        return text
    except Exception as e:
        print(f"Error reading uploaded file: {e}")
        return None

glove_file_path = "glove.6B.100d.txt"  # Adjust the path based on your downloaded file
glove_embeddings = load_glove_embeddings(glove_file_path)

dataset = pd.read_csv("Training_Essay_Data.csv")

X_train, X_test, y_train, y_test = train_test_split(
    dataset['text'], dataset['generated'].astype(int), test_size=0.2, random_state=42
)

max_words = 10000  # Choose an appropriate value
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)



max_sequence_length = 100  # Choose an appropriate value
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

word_index = tokenizer.word_index
embedding_dim = 100  # Use the same dimension as your GloVe file
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_sequence_length, weights=[embedding_matrix], trainable=False))
model.add(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)))  
model.add(Dropout(0.5))  # Increased dropout rate
model.add(LSTM(128, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)))  
model.add(Dropout(0.5))  # Increased dropout rate
model.add(Dense(64, activation="relu"))  
model.add(BatchNormalization())
model.add(Dense(1, activation="sigmoid"))

# # Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # Load the trained weights
# model.load_weights('model_updated_weight.h5')

def classify_text(input_text, model, tokenizer, max_sequence_length):
    # Tokenize and pad the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length)

    # Predict the class probability
    prediction = model.predict(input_padded)

    # Determine the predicted class label
    predicted_label = "AI-generated" if prediction[0] >= 0.5 else "Human-generated"

    return predicted_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Check if an uploaded file exists
    uploaded_file = request.files.get('file')

    if uploaded_file:
        input_text = read_text_from_uploaded_file(uploaded_file)
        if input_text is None:
            return jsonify({'result': None})
    else:
        data = request.get_json()
        input_text = data.get('text', '')

    result = classify_text(input_text, model, tokenizer, max_sequence_length)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)