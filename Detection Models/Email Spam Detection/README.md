# Email Spam Detection

This project aims to detect unwanted emails (spam) from a user's inbox. It uses a combination of **Natural Language Processing (NLP)** techniques and machine learning algorithms to analyze the content, metadata, and patterns of emails, distinguishing between spam and legitimate messages.

## Goal

The goal is to identify and filter out unsolicited emails (spam) from a user's inbox, ensuring only legitimate and important emails are delivered. This reduces clutter and protects users from phishing attacks, malware, and other malicious activities.

## Methodology

Using a mix of **Exploratory Data Analysis (EDA)** and **NLP**, the data is analyzed to identify patterns and correlations. Key steps include:
- Data cleaning
- Text preprocessing through NLP
- Feature engineering and insightful visualizations

## Data Preprocessing

Steps involved:
1. Stop words removal
2. Lemmatization/Stemming (NLP)
3. Vectorization using **TF-IDF** (NLP)
4. Tokenization for breaking down email content

## NLP Techniques Used

- **Tokenization**: Breaking down email texts into words
- **Stemming/Lemmatization**: Reducing words to their base form
- **TF-IDF**: Transforming text into numerical values based on importance
- **Bag of Words**: Converting text into features for model building

## Models Utilized

1. **Logistic Regression**
2. **Random Forest Regressor**
3. **Multinomial Naive Bayes**

## Libraries Used

1. **numpy**: Numerical operations
2. **pandas**: Data manipulation and analysis
3. **nltk**: NLP toolkit for text processing
4. **seaborn**: Statistical visualizations
5. **matplotlib**: Data visualization
6. **sklearn**: Machine learning algorithms

## Results

1. **Logistic Regression**: 98% accuracy
2. **Multinomial Naive Bayes**: 97% accuracy

## Enhanced Features

Additional NLP-based features include:
- Word frequency analysis using **nltk** and **collections.Counter**
- Character, word, and sentence count analysis
- Heatmap and distribution plots for word frequency and relationships

## Conclusion

Through rigorous analysis and the use of advanced **NLP** techniques, the **Multinomial Naive Bayes** model yielded the highest predictive accuracy for email spam detection.
