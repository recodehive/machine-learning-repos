import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Load and preprocess data
def load_data(data_dir):
    texts, labels = [], []
    for label, category in enumerate(['without_ptsd', 'with_ptsd']):
        category_dir = os.path.join(data_dir, category)
        for filename in os.listdir(category_dir):
            with open(os.path.join(category_dir, filename), 'r') as file:
                texts.append(file.read())
                labels.append(label)
    return texts, np.array(labels)

data_dir = 'dataset'
texts, labels = load_data(data_dir)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=100)

# Split data into training and test sets
split_idx = int(0.75 * len(padded_sequences))
x_train, x_test = padded_sequences[:split_idx], padded_sequences[split_idx:]
y_train, y_test = labels[:split_idx], labels[split_idx:]

# Build the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Save the model
model.save(model_path)