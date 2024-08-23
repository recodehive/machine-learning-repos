import pandas as pd
import string
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle

data = pd.read_csv("Stress.csv") #reading csv file
data.drop(['post_id','sentence_range'],axis=1,inplace=True) #droping these columns because these are less required columns

data['text_length'] = data['text'].apply(len)
'''dividing timestamp into day hr min sec month and year'''
data['date'] = data['social_timestamp'].apply(lambda time: datetime.fromtimestamp(time))
data['month'] = data['date'].apply(lambda date: date.month)
data['day'] = data['date'].apply(lambda date: date.day)
data['week_day'] = data['date'].apply(lambda date: date.day_name)
data['hour'] = data['date'].apply(lambda date: date.hour)
data['minumte'] = data['date'].apply(lambda date: date.minute)
data['sec'] = data['date'].apply(lambda date: date.second)

data.drop(['date','week_day'],axis=1,inplace=True) #droping date and week_day columns
data.drop(['social_timestamp'],axis=1,inplace=True)


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

mess_data = data[['label', 'text_length', 'text']]
mess_data['label_name'] = data['label'].map({0: 'Not Stress', 1: 'Stress'})

# Train the model
model = MultinomialNB()
mess_transformer = CountVectorizer().fit(mess_data['text'])
messages_bow = mess_transformer.transform(mess_data['text'])
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
model.fit(messages_tfidf, mess_data['label_name'])

# Pickle the model and transformers
with open('Stress_Prediction_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('CountVectorizer.pkl', 'wb') as cv_file:
    pickle.dump(mess_transformer, cv_file)

with open('TfidfTransformer.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf_transformer, tfidf_file)
