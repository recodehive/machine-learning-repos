'''
Preprocessing strategy : 
1. Tokenization
2. Lowering and stem
3. Punctuation Removal
4. Bag Of Words
'''
import enum
import numpy as np
import nltk

#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer=PorterStemmer()
def tokenize(sentence):
        return nltk.word_tokenize(sentence)   #Tokenization happens here. for e,g, I am girl tokenized to I, am , girl


def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,all_words):
    '''
    tokenized_Sentence=['I','am','saylee']
    words=['I','am','saylee','hello','code']
    bag=[1,1,1,0,0]

    '''
    tokenized_sentence=[stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words),dtype=np.float32)
    for index,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[index]=1.0


    return bag

tokenized_Sentence=["I","am","Saylee","you"]
words=["hi","hello","am","I","Saylee","thank","cool"]
bag=bag_of_words(tokenized_Sentence,words)
print(bag)