# Ecommerce product categorization

## Goal
This project implements an automated product categorization system for e-commerce platforms using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The system analyzes product descriptions, titles, and metadata to automatically assign products to the most relevant categories. The solution handles large datasets, and continuously improves product discoverability and categorization consistency.

## Introduction
Product categorization is the task of classifying products as belonging to one or more categories from a given taxonomy.It helps customers navigate an ecommerce store with ease. It deals with organizing our ecommerce products into categories and tags that give us a system to get customers to the exact product they are looking for quicker. This includes creating categories, tags, attributes and more to create a hierarchy for similar products. 

## Dataset
The dataset used in this project is sourced from Kaggle(https://www.kaggle.com/datasets/sumedhdataaspirant/e-commerce-text-dataset) . It consists of >50000 records for 4 categories - "Electronics", "Household", "Books" and "Clothing & Accessories", which cover almost 80% of any E-commerce website.

![image](https://github.com/user-attachments/assets/5647eacf-2a1b-40f7-b887-7283216ee25d)


## Methodology
Basic NLP steps for categorizing the E-commerce dataset include:-

**1. Importing Libraries**

 - Libraries such as NumPy, Pandas, Matplotlib are imported for data manipulation and visualization , NLTK for nlp processing, sklearn for model building and performance metrics.
   
**2. Data preprocessing**
   
 - **Tokenization:** Tokenization is the process of splitting text into smaller units, typically words or phrases.Tokenizes product titles and descriptions.
 - **Stopword Removal:** Removes common stopwords that do not provide categorization value.
 - **Stemming:** Involves reducing words to their root form. It removes suffixes like "-ing", "-ed", and "-ly", simplifying words to their base form.
 - **Lemmatization:** Similar to stemming but more sophisticated. Instead of just chopping off word endings, it transforms words into their dictionary base form (or lemma) based on their context.
 - **Vectorization:** Once text is preprocessed (tokenized, lowercased, and lemmatized), it’s transformed into numerical vectors that can be fed into a machine learning model. Techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or Word2Vec are used to convert textual data into a format that a model can understand.
 - **Removing Special Characters :** Before or during the vectorization process, unnecessary characters like punctuation marks, symbols, and numbers (unless relevant to the product, like in technical specifications) are removed from the text.

**3. Model Overview**

**a. Multinomial Naive Bayes (MultinomialNB)**

Multinomial Naive Bayes is a popular algorithm for text classification tasks. It’s based on Bayes' Theorem.
- How it works: MultinomialNB assumes that features (words) are conditionally independent given the class and calculates the probability of a product belonging to a specific category.

**b. Support Vector Machine (SVM)**

Support Vector Machine (SVM) is a supervised learning algorithm used for classification tasks. It aims to find the best hyperplane that separates different classes in the feature space.
 - How it works: SVM tries to maximize the margin between different classes by finding the hyperplane that best separates the data points. In the case of text, the features are usually word embeddings or TF-IDF vectors.

**c. Random Forest Classifier**

Random Forest is an ensemble learning method that builds multiple decision trees during training and outputs the mode of the classes for classification. It is a bagging method that helps improve accuracy and reduce overfitting.

 - How it works: Random Forest creates multiple decision trees based on different subsets of the training data and features. The final prediction is made by averaging the results from all the trees (voting for classification).

**d. Logistic Regression**

Logistic Regression is a linear classification algorithm that models the probability of a product belonging to a particular category. 
- How it works: Logistic Regression calculates the probability of a class based on the input features using a logistic (sigmoid) function. It finds the best-fitting hyperplane between categories.

**4. Model training**
Before training, the dataset is split into two parts:
 - Training Set: Used to train the model (typically 70-80% of the data).
 - Test Set: Used to evaluate the model’s performance on unseen data (typically 20-30%).

**5. Model Evaluation**
Once trained, the model is evaluated on the test set to ensure it generalizes well. Key evaluation metrics used include:

 - Accuracy: Percentage of correct predictions.
 - Precision: Fraction of relevant products correctly classified.
 - Recall: Fraction of relevant products retrieved.
 - F1 Score: Harmonic mean of precision and recall, useful for imbalanced datasets.
 - Confusion Matrix: Provides insight into the number of true positives, true negatives, false positives, and false negatives.

## Results
Accuracy of various models on test data is compared below. Out of all models SVM performs the best, closely followed by Logistic Regression and Random forest. :

MultinomialNB - 92%

SVM - 96%

RandomForestClassifier - 93.058%

LogisticRegression - 95%

