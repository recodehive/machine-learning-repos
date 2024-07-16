# Sarcasm detection on Cross Domain Applications 
  This project proposes the accuracy and efficiency of ML and NN models trained on one dataset and tested on other dataset. SARC dataset is used for taining and amazon     review dataset is used for testing the models. This enables Sarcasm detection on Cross Domain applications.
## System Requirements
     1. Python 2.7
     2. Python package Gensim, NLTK, Keras,Matplotlib, Numpy, SkLearn, Pandas, Re, Tensorflow 2.0
## Running the code

**1. Get the Pre-requisites**
  - Get the pre-trained GloVe file - [GLoVe.6d.100B.txt](https://nlp.stanford.edu/projects/glove/).
  - Get SARC dataset zip file - [SARC dataset](https://www.kaggle.com/danofer/sarcasm) and store it in the folder named `dataset` within the project repository.
  
**2. Training and Evaluation**

  - **To run the ML models -**
   
        Run python ML_Models.py
        
  - **To run the NN models with Word2Vec embedding -**
   
        Run python word2Vec/Model_name.py
        
        Example to run LSTM model - run python Word2Vec/LSTM.py
      
  - **To run the NN models with GloVe embedding -**
   
        Run python GloVe/Model_name.py
        
        Example to run LSTM model - run python GloVe/LSTM.py

