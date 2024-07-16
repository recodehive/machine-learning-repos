import pandas as pd
import numpy as np
import re
import nltk
import sklearn
print(sklearn.__version__)
from sklearn.metrics import ConfusionMatrixDisplay
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from numpy import array
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import asarray
from numpy import zeros

def extract_zip():
  from zipfile import ZipFile 
  file_name = "/content/drive/My Drive/Sarcasm.csv.zip"
  with ZipFile(file_name, 'r') as zip1:

    # printing all the contents of the zip file 
    zip1.printdir() 
    
    # extracting all the files 
    print('Extracting all the files now...') 
    zip1.extractall() 
    print('Done!') 

def preprocess_text(sentence):
    contractions = {
      "ain't": "am not",
      "aren't": "are not",
      "can't": "cannot",
      "can't've": "cannot have",
      "'cause": "because",
      "could've": "could have",
      "couldn't": "could not",
      "couldn't've": "could not have",
      "didn't": "did not",
      "doesn't": "does not",
      "don't": "do not",
      "hadn't": "had not",
      "hadn't've": "had not have",
      "hasn't": "has not",
      "haven't": "have not",
      "he'd": "he would",
      "he'd've": "he would have",
      "he'll": "he will",
      "he'll've": "he will have",
      "he's": "he has / he is",
      "how'd": "how did",
      "how'd'y": "how do you",
      "how'll": "how will",
      "how's": "how is",
      "i'd": "I would",
      "i'd've": "I would have",
      "i'll": "I will",
      "i'll've": "I will have",
      "i'm": "I am",
      "i've": "I have",
      "isn't": "is not",
      "it'd": "it would",
      "it'd've": "it would have",
      "it'll": "it will",
      "it'll've": "it will have",
      "it's": "it is",
      "let's": "let us",
      "ma'am": "madam",
      "mayn't": "may not",
      "might've": "might have",
      "mightn't": "might not",
      "mightn't've": "might not have",
      "must've": "must have",
      "mustn't": "must not",
      "mustn't've": "must not have",
      "needn't": "need not",
      "needn't've": "need not have",
      "o'clock": "of the clock",
      "oughtn't": "ought not",
      "oughtn't've": "ought not have",
      "shan't": "shall not",
      "sha'n't": "shall not",
      "shan't've": "shall not have",
      "she'd": "she would",
      "she'd've": "she would have",
      "she'll": "she will",
      "she'll've": "she will have",
      "she's": "she is",
      "should've": "should have",
      "shouldn't": "should not",
      "shouldn't've": "should not have",
      "so've": "so have",
      "so's": "so is",
      "that'd": "that would",
      "that'd've": "that would have",
      "that's": "that is",
      "there'd": "there would",
      "there'd've": "there would have",
      "there's": "there is",
      "they'd": "they would",
      "they'd've": "they would have",
      "they'll": "they will",
      "they'll've": "they will have",
      "they're": "they are",
      "they've": "they have",
      "to've": "to have",
      "wasn't": "was not",
      "we'd": "we would",
      "we'd've": "we would have",
      "we'll": "we will",
      "we'll've": "we will have",
      "we're": "we are",
      "we've": "we have",
      "weren't": "were not",
      "what'll": "what will",
      "what'll've": "what will have",
      "what're": "what are",
      "what's": "what is",
      "what've": "what have",
      "when's": "when is",
      "when've": "when have",
      "where'd": "where did",
      "where's": "where is",
      "where've": "where have",
      "who'll": "who will",
      "who'll've": "who will have",
      "who's": "who is",
      "who've": "who have",
      "why's": "why is",
      "why've": "why have",
      "will've": "will have",
      "won't": "will not",
      "won't've": "will not have",
      "would've": "would have",
      "wouldn't": "would not",
      "wouldn't've": "would not have",
      "y'all": "you all",
      "y'all'd": "you all would",
      "y'all'd've": "you all would have",
      "y'all're": "you all are",
      "y'all've": "you all have",
      "you'd": "you would",
      "you'd've": "you would have",
      "you'll": "you will",
      "you'll've": "you will have",
      "you're": "you are",
      "you've": "you have"
      }
    for word in sentence.split():
      if word.lower() in contractions:
          sentence = sentence.replace(word, contractions[word.lower()])
    sentence = sentence.lower()
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
   # sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

def create_embedding():
	embeddings_dictionary = dict()
	glove_file = open('/content/drive/My Drive/glove.6B.100d.txt', encoding="utf8")

	for line in glove_file:
    		records = line.split()
    		word = records[0]
    		vector_dimensions = asarray(records[1:], dtype='float32')
    		embeddings_dictionary [word] = vector_dimensions
	glove_file.close()
	embedding_matrix = zeros((vocab_size, 100))
	for word, index in tokenizer.word_index.items():
    		embedding_vector = embeddings_dictionary.get(word)
    		if embedding_vector is not None:
        		embedding_matrix[index] = embedding_vector
	return embedding_matrix


def Crisp_class(yhat_probs):
  yhat_classes = []
  i = 0
  for x in yhat_probs:
    if x < 0.5:
      yhat_classes.append(0)
    else:
      yhat_classes.append(1)
  yhat_classes = np.asarray(yhat_classes)
  return yhat_classes



