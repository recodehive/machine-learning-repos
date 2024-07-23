
# Sentiment Analysis using LSTM

This project demonstrates a Sentiment Analysis model using Long Short-Term Memory (LSTM) networks. The dataset consists of text data with corresponding sentiment labels, and the model is built using TensorFlow and Keras.
## Table of contact

* Project Overview
* Dataset
* Preprocessing
* Model Architecture
* Usage
* Results
## Projects Overview
This project aims to classify text data into different sentiment categories using an LSTM-based neural network. The primary steps include data preprocessing, model building, training, and evaluation.


## Dataset
The dataset used for this project contains text data labeled with sentiment categories:

* Positive (1)
* Negative (-1)
* Neutral (0)
## Preprocessing
- Text Tokenization: Convert text into sequences of tokens.
- Padding Sequences: Pad the token sequences to ensure uniform input length.
- One-Hot Encoding: Convert labels to one-hot encoded format for training.
## Model Architecture
The model is built using a sequential Keras model with the following layers:  
- Embedding Layer: For word embeddings. 
- LSTM Layers: Two LSTM layers for sequence learning.
- Dropout Layer: For regularization. Dense Layers: For classification with regularization.
## Usage
```bash
git clone https://github.com/yourusername/sentiment-analysis-lstm.git
```

```bash
cd sentiment-analysis-lstm
```
```bash
pip install -r requirements.txt
```
```bash
jupyter notebook SentimentAnalysis.ipynb
```
## Results
The model's performance is evaluated using precision, recall, and F1 score metrics. Detailed results and visualizations can be found in the Jupyter Notebook.