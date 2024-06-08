# Sentiment Analysis of IMDB Movie Reviews

## Problem Statement:

The goal of this project is to predict the sentiment (positive or negative) of IMDB movie reviews using different classification models.

## Dataset:

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Steps:

1. Import necessary libraries.
2. Import the training dataset.
3. Exploratory data analysis.
4. Count sentiment occurrences.
5. Split the training dataset.
6. Text normalization:
    - Remove HTML tags and noise text.
    - Remove special characters.
    - Text stemming.
    - Remove stopwords.
7. Normalize train reviews.
8. Normalize test reviews.
9. **Bags of Words Model**:
    - Term Frequency-Inverse Document Frequency (TFIDF) model:
        - Convert text documents to a matrix of TFIDF features.
    - Label the sentiment text.
    - Split the sentiment data.
    - Model the dataset:
        - Build logistic regression model for both bag of words and TFIDF features.
            - Logistic regression model performance on the test dataset:
                - Accuracy of the model.
                - Print the classification report.
                - Confusion matrix.
            - Stochastic gradient descent or Linear support vector machines for bag of words and TFIDF features:
                - Model performance on test data.
                - Accuracy of the model.
                - Print the classification report.
                - Plot the confusion matrix.
            - Multinomial Naive Bayes for bag of words and TFIDF features:
                - Model performance on test data.
                - Accuracy of the model.
                - Print the classification report.
                - Plot the confusion matrix.
    - **Word Clouds**:
        - Word cloud for positive review words.
        - Word cloud for negative review words.

## Evaluation:

- Confusion matrix:
    ```
    [[3736 1271]
     [1219 3774]]
    [[3729 1278]
     [1213 3780]]
    ```
- Classification Report:
    ```
                    precision    recall  f1-score   support

        Positive       0.75      0.76      0.75      4993
        Negative       0.75      0.75      0.75      5007

        accuracy                           0.75     10000
       macro avg       0.75      0.75      0.75     10000
    weighted avg       0.75      0.75      0.75     10000

                  precision    recall  f1-score   support

        Positive       0.75      0.76      0.75      4993
        Negative       0.75      0.74      0.75      5007

        accuracy                           0.75     10000
       macro avg       0.75      0.75      0.75     10000
    weighted avg       0.75      0.75      0.75     10000
    ```
- Multinomial Naive Bayes (MNB) Scores:
    - MNB Bag of Words Score: 0.751
    - MNB TFIDF Score: 0.7509
