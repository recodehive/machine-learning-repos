# Bank Customer Churn Prediction

## Goal 🎯
The main goal of this project is to predict whether a bank customer will churn, i.e., leave the bank, using various machine learning models. By accurately predicting customer churn, the bank can implement targeted retention strategies to improve customer satisfaction and reduce attrition rates.

## Model(s) used for the Web App 🧮
The backend of the web application utilizes multiple machine learning algorithms to predict customer churn. The models used and their hyperparameter tuning results are as follows:

### Logistic Regression:
- **Best Parameters**: `{'solver': 'liblinear', 'penalty': 'l1', 'C': 10}`
- **Best Score**: `0.757`
- **Train Accuracy**: `0.757`
- **Test Accuracy**: `0.754`

### Random Forest:
- **Best Parameters**: `{'n_estimators': 200, 'min_samples_split': 2, 'max_depth': None}`
- **Best Score**: `0.892`
- **Train Accuracy**: `0.999`
- **Test Accuracy**: `0.906`

### Decision Tree:
- **Best Parameters**: `{'min_samples_split': 5, 'max_depth': 20}`
- **Best Score**: `0.853`
- **Train Accuracy**: `0.949`
- **Test Accuracy**: `0.866`

### Gradient Boosting:
- **Best Parameters**: `{'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.2}`
- **Best Score**: `0.908`
- **Train Accuracy**: `0.918`
- **Test Accuracy**: `0.908`

### AdaBoost:
- **Best Parameters**: `{'n_estimators': 200, 'learning_rate': 1}`
- **Best Score**: `0.879`
- **Train Accuracy**: `0.880`
- **Test Accuracy**: `0.879`

## Video Demonstration 🎥
https://github.com/NIKITA320495/ML-Crate/assets/115877450/61bd650a-3844-45dd-97e2-131f27064bfc




## Signature ✒️
Developed by Nikita Babbar 
- GitHub: https://github.com/NIKITA320495  
- LinkedIn: https://www.linkedin.com/in/nikita-babbar-b0291026a/

