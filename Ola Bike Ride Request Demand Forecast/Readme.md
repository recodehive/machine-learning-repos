
# **Ola Bike Ride Request Demand Forecast**


## **GOAL**

The aim of this project is to forecast the demand for Ola Bike Rental Services. Using a dataset containing two years of data on various features influencing customer ride request demand, the project predicts the future demand trends.

## **WHAT HAVE I DONE**

1. **Loading Datasets**
2. **Handling Null Values**
3. **Changing Date and Time into Datetime Format**
4. **Mapping Key Values to Dictionary**
5. **Performing Exploratory Data Analysis (EDA)**
   - Distribution of the target variable 'count'
   - Visualizing demand using multiple variables
   - Visualizing continuous variables using histograms
   - Analyzing the correlation matrix of continuous features
   - Analyzing the correlation matrix of all independent features
6. **Data Preprocessing**
   - One-hot encoding categorical features
   - Dropping features with low correlation
   - Visualization of the correlation matrix of the preprocessed data
7. **Data Splitting**
8. **Model Training and Evaluation**
   - **Linear Regression**: Accuracy - 86.55%
   - **Decision Tree**: Accuracy - 99.03%
   - **Hypertuned KNN**: Accuracy - 99.32%
   - **Hypertuned Random Forest**: Accuracy - 99.995%
   - **Hypertuned XGBoost**: Accuracy - 99.97%
9. **Saving and Loading Models**
   - Saved Hypertuned Random Forest Regressor and XGBoost Regressor models
   - Loaded and used these models to get predictions from the test set
   - Saved the predictions in a dataset

## **MODELS USED**

- **Linear Regression**: A machine learning algorithm for regression tasks, modeling the relationship between variables to forecast values.
- **XGBoost**: eXtreme Gradient Boost algorithm, an ensemble learning technique that improves accuracy by passing underfitted data of weak learners to strong learners.
- **Decision Tree**: An algorithm that creates a tree-like model of decisions based on feature values.
- **K-Nearest Neighbors (KNN)**: An algorithm that classifies new cases based on the similarity to existing cases.
- **Random Forest**: An ensemble learning algorithm using bagging technique to train multiple predictors on sampled instances for higher accuracy.
- **GridSearchCV**: A hyperparameter optimization technique to enhance model performance by finding the best hyperparameter values.

## **LIBRARIES NEEDED**

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `datetime`
- `calendar`
- `pickle`
- `xgboost`

## **CONCLUSION**

This project involved a comprehensive analysis and visualization of the training dataset using various Exploratory Data Analysis (EDA) techniques. After evaluating different regression models, it was found that most models achieved an accuracy above 99%, with the Hypertuned Random Forest Regressor achieving the highest accuracy of 99.995%. The analysis and model comparison demonstrate the effectiveness of advanced ensemble methods for predicting bike rental demand.
