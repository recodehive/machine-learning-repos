from functions import *
import pandas as pd
import numpy as np
import re
import nltk
from numpy import array,asarray,zeros
from numpy import array
import gensim
from sklearn import linear_model
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score,precision_score,f1_score,plot_confusion_matrix,recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten,GlobalMaxPooling1D,Dense,Dropout,Activation
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
#Reading data from csv
extract_zip()
train_data1 = pd.read_csv("/content/train-balanced-sarcasm.csv")     #sentiment null code
train_data1.isnull().values.any()
train_data1.shape
print(train_data1.head())

test_data1 = pd.read_csv("/content/drive/My Drive/Sarcasm_Detection/finalAmazonDataset.csv")
test_data1.isnull().values.any()
test_data1.shape
print(test_data1.head())
#data preprocessing
new = []
for parent_comment, comment in zip(train_data1['parent_comment'],train_data1['comment']):
  new.append(str(parent_comment)+str(comment))

ser = pd.Series(new)
train_data1['comment'] = ser
train_data1.info()
train_data = train_data1[['label', 'comment','parent_comment']].dropna()
train_data
embedding_dim = 16
max_length = 100
training_size = int(round(train_data['label'].count(), -1) * 0.8)
print(training_size)
X = []
sentences = list(train_data['comment'])
for sen in sentences:
    X.append(preprocess_text(sen))

ama_comm = []
sentences = list(test_data1['review'])
for sen in sentences:
  ama_comm.append(preprocess_text(sen))

comment = train_data['comment']
labels = train_data['label']
X_train = X[0:training_size]
y_train = labels[0:training_size]
X_test = X[training_size:]
y_test = labels[training_size:]
ama_comment=test_data1['review']
ama_label = test_data1['label']
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
ama_comm = tokenizer.texts_to_sequences(ama_comm)
ama_comm = pad_sequences(ama_comm, padding='post', maxlen=maxlen)
embeddings_dictionary = dict()
glove_file = open('/content/drive/My Drive/glove.6B.100d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()
#creating embedding matrix
embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode = 'min', verbose = 1, patience=5)
mc = tf.keras.callbacks.ModelCheckpoint('/content/drive/My Drive/Sarcasm_bestModel/SeqNN_best.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1,validation_split=0.1,callbacks=[callback, mc])
score = model.evaluate(X_test, y_test, verbose=1)
score = model.evaluate(X_test, y_test, verbose=1) 
score
score = model.evaluate(ama_comm, ama_label, verbose=1) 
score
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
#Plotting Confusion Matrix
results = confusion_matrix(y_test, yhat_classes, labels = [0,1])
print(results)
print("\n")
# Plotting confusion matrix
ax= plt.subplot()
sns.heatmap(results, annot=True, ax = ax, fmt = 'd'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix of Sequential NN on SARC dataset'); 
ax.xaxis.set_ticklabels(['Non - Sarcastic', 'Sarcastic']); ax.yaxis.set_ticklabels(['Non - Sarcastic', 'Sarcastic']);
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)

yhat_probs = model.predict(ama_comm, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(ama_comm, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
#Plotting Confusion Matrix
results = confusion_matrix(ama_label, yhat_classes) 
print(results)
print("\n")
# Plotting confusion matrix
ax= plt.subplot()
sns.heatmap(results, annot=True, ax = ax, fmt = 'd'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix of Sequential NN on Amazon dataset'); 
ax.xaxis.set_ticklabels(['Non - Sarcastic', 'Sarcastic']); ax.yaxis.set_ticklabels(['Non - Sarcastic', 'Sarcastic']);

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ama_label, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(ama_label, yhat_classes,labels=np.unique(yhat_classes))
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ama_label, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ama_label, yhat_classes)
print('F1 score: %f' % f1)