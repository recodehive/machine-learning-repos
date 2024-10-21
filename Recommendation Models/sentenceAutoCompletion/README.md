## Sentence Auto-Completion

This project implements a sentence auto-completion model using a deep learning approach, specifically leveraging LSTM (Long Short-Term Memory) networks from the TensorFlow/Keras library. The goal is to predict the next word in a sequence of text, providing automatic sentence completion functionality.

### Project Structure
```
├── SentenceAutoCompletion.ipynb  # Jupyter notebook containing the entire implementation
├── README.md                     # Project overview and instructions
└── holmes.txt                    # Input text file used for training the model
```

### Model Overview

The project builds a sentence auto-completion model with the following components:
- **LSTM-based model**: Uses a recurrent neural network (RNN) with LSTM layers to predict the next word in a sequence of text.
- **Tokenizer and Padding**: Text data is tokenized, and sequences are padded to ensure uniform input size for the neural network.
- **Bidirectional LSTM**: A bidirectional LSTM is used to capture both past and future context in text sequences.
  
The training text is taken from *Project Gutenberg* and is preprocessed to remove special characters, emojis, and extra spaces.

### Setup and Dependencies

To set up this project, you need to install the following libraries:

```bash
pip install tensorflow nltk pandas
```

### Data Preprocessing

Before training, the data undergoes several preprocessing steps:
- **Loading the dataset**: The text data is read from the `holmes.txt` file.
- **Cleaning the text**: Special characters, emojis, and excessive whitespace are removed.
- **Tokenization**: The text is tokenized into sequences of words, and these sequences are then transformed into numerical format.
- **Padding sequences**: To ensure consistent input size, sequences are padded.

### Model Training

The model is trained on the cleaned and tokenized dataset using the following process:
1. **Embedding layer**: Converts words into dense vectors of fixed size.
2. **LSTM layers**: A bidirectional LSTM processes the input text sequence.
3. **Dense layers**: The final layers output predictions for the next word in the sequence.

Training uses the Adam optimizer, and the loss function is `categorical_crossentropy`.

### Usage

To run the model:
1. Clone the repository or download the Jupyter notebook.
2. Download or prepare a dataset and save it as `holmes.txt` (or any other text file).
3. Run the notebook to preprocess the text, build the model, and train it.
4. After training, use the model to predict the next word given a sequence of words.



