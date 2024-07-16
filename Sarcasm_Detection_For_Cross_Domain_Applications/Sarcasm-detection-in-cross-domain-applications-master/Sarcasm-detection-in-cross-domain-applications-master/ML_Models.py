
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import re
import nltk
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from numpy import array
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from numba.typed import List
from sklearn.metrics import classification_report
from functions import *

extract_zip()
#Reading data from csv files
train_data1 = pd.read_csv("/content/train-balanced-sarcasm.csv")     
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
training_size = int(round(train_data['label'].count(), -1) * 0.8)
print(training_size)   
print('total size of dataset',train_data['label'].count())
X = []
sentences = list(train_data['comment'])
for sen in sentences:
    sen_processed = preprocess_text(sen)
    X.append(sen_processed)
comment = train_data['comment']
labels = train_data['label']
X_train = X[0:training_size]
y_train = labels[0:training_size]
X_test = X[training_size:]
y_test = labels[training_size:]

ama_comment=test_data1['review']
ama_label = test_data1['label']

#tokenization 
tokenizer = Tokenizer(num_words=10000)

tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

ama_comm = []
sentences = list(test_data1['review'])
for sen in sentences:
  ama_comm.append(preprocess_text(sen))

ama_comm = tokenizer.texts_to_sequences(ama_comm)
ama_comm = pad_sequences(ama_comm, padding='post', maxlen=maxlen)

#SVM

clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)
print("On Sarc")
prediction_linear = clf.predict(X_test)
report = classification_report(y_test, prediction_linear, output_dict=True)
print(report)
print("On Amazon")
prediction_linear = clf.predict(ama_comm)
report = classification_report(ama_label, prediction_linear, output_dict=True)
print(report)

Y_pred = clf.predict(X_test)
acc_svc = round(clf.score(X_test, y_test) * 100, 2)
print(acc_svc)
#for amazon
Y_pred = clf.predict(ama_comm)
acc_svc = round(clf.score(ama_comm, ama_label) * 100, 2)
print(acc_svc)

#Naive Bayes

#Create a Gaussian Classifier
NBmodel = GaussianNB()
NBmodel.fit(X_train,y_train)
#testing on SARC
Y_pred = NBmodel.predict(X_test)
acc_nb = round(NBmodel.score(X_test, y_test) * 100, 2)
print(acc_nb)
#testing on Amazon
Y_pred = NBmodel.predict(ama_comm)
acc_nb = round(NBmodel.score(ama_comm, ama_label) * 100, 2)
print(acc_nb)

print("On Sarc")
prediction_linear = NBmodel.predict(X_test)
report = classification_report(y_test, prediction_linear, output_dict=True)
print(report)
print("On Amazon")
prediction_linear = NBmodel.predict(ama_comm)
report = classification_report(ama_label, prediction_linear, output_dict=True)
print(report)

#DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
acc_decision_tree

Y_pred = decision_tree.predict(ama_comm)
acc_decision_tree = round(decision_tree.score(ama_comm, ama_label) * 100, 2)
acc_decision_tree

print("On Sarc")
prediction_linear = decision_tree.predict(X_test)
report = classification_report(y_test, prediction_linear, output_dict=True)
print(report)
print("On Amazon")
prediction_linear = decision_tree.predict(ama_comm)
report = classification_report(ama_label, prediction_linear, output_dict=True)
print(report)

#LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_test, y_test) * 100, 2)
acc_log

Y_pred = logreg.predict(ama_comm)
acc_log = round(logreg.score(ama_comm, ama_label) * 100, 2)
acc_log

print("On Sarc")
prediction_linear = logreg.predict(X_test)
report = classification_report(y_test, prediction_linear, output_dict=True)
print(report)
print("On Amazon")
prediction_linear = logreg.predict(ama_comm)
report = classification_report(ama_label, prediction_linear, output_dict=True)
print(report)

#RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)  #measuring acc on test data
acc_random_forest

Y_pred = random_forest.predict(ama_comm)
acc_random_forest = round(random_forest.score(ama_comm, ama_label) * 100, 2)  #measuring acc on test data
acc_random_forest
print("On Sarc")
prediction_linear = random_forest.predict(X_test)
report = classification_report(y_test, prediction_linear, output_dict=True)
print(report)
print("On Amazon")
prediction_linear = random_forest.predict(ama_comm)
report = classification_report(ama_label, prediction_linear, output_dict=True)
print(report)