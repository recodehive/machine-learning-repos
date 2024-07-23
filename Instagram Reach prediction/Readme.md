
# Instagram Reach Analysis

## Overview

This project analyzes Instagram reach by examining the relationship between the number of followers, the time since a post was made, the hashtags used, and the number of likes received. The dataset includes information such as username, caption, number of followers, hashtags, hours since posted, and likes.

## Dataset

The dataset contains 100 entries with the following columns:
- `Username`: Instagram username of the account.
- `Caption`: Text caption of the Instagram post.
- `Followers`: Number of followers the account has.
- `Hashtags`: Hashtags used in the post.
- `Hours Since Posted`: Time since the post was made (in hours).
- `Likes`: Number of likes the post received.

## Data Preprocessing

1. **Loading Data**: The data is loaded from a CSV file.
2. **Cleaning Data**: Dropped unnecessary columns (`Unnamed: 0`, `S.No`) and renamed columns for consistency.
3. **Handling Missing Values**: Dropped rows with missing values in the `Caption` column.
4. **Converting Data Types**: Converted the `Hours Since Posted` column from string to integer.

## Exploratory Data Analysis

1. **Correlation Matrix**: Displayed a heatmap of the correlation matrix for numerical features.
2. **Regression Plots**: Created regression plots to visualize the relationship between:
   - Followers and Likes
   - Hours Since Posted and Likes
3. **Word Clouds**: Generated word clouds for the `Caption` and `Hashtags` columns to visualize the most common words and hashtags.

## Feature Engineering

1. **One-Hot Encoding**: Applied one-hot encoding to categorical columns (`Username`, `Caption`, `Hashtags`).
2. **Feature Scaling**: Standardized the features using `StandardScaler`.

## Model Training and Evaluation

Three regression models were trained and evaluated:

1. **Random Forest Regressor**
   - `R^2` Score: 0.9215

2. **Gradient Boosting Regressor**
   - `R^2` Score: 0.9633

3. **Linear Regression**
   - `R^2` Score: 1.0

## Key Insights

- Posts related to topics like artificial intelligence, machine learning, big data, and trading tend to have better reach on Instagram.
- While more followers generally lead to more likes, posts that have been up longer also tend to get more reach.
- A strategic use of hashtags tailored to the followers can significantly enhance the reach of Instagram posts.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- wordcloud

## How to Run

1. Clone the repository or download the dataset.
2. Install the required libraries using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn wordcloud
   ```
3. Run the Jupyter notebook or Python script containing the analysis.

## Conclusion

This project demonstrates how different factors such as the number of followers, the time since a post was made, and the hashtags used can influence the reach of Instagram posts. The analysis highlights the importance of timely posting and the clever use of hashtags to maximize engagement on Instagram.
