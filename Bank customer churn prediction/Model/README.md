# Machine Learning Model for Bank Customer Churn Prediction

## Overview

This machine learning model predicts customer churn in a bank based on customer demographics, financial status, and banking behavior. It is trained on a dataset containing customer information and churn labels. By analyzing these features, the model identifies patterns indicating whether a customer is likely to churn, enabling proactive retention strategies.

## Preprocessing

- **Feature Scaling**: MinMax scaling for 'Tenure' and 'NumOfProducts', Standard scaling for 'CreditScore', 'Age', and 'Balance'.
- **One-Hot Encoding**: Encode categorical feature 'Gender' to numerical format.

## Libraries Used

- scikit-learn: Machine learning library for model training and evaluation.
- pandas: Data manipulation library for preprocessing and data handling.
- numpy: Numerical computing library for mathematical operations.
- matplotlib, seaborn, plotly: Visualization libraries for data exploration and analysis.


## Model Details
### Hyperparameter Tuning Results

- **Logistic Regression**:
  - Best Parameters: {'solver': 'liblinear', 'penalty': 'l1', 'C': 10}
  - Best Score: 0.757
  - Train Accuracy: 0.757
  - Test Accuracy: 0.754

- **Random Forest**:
  - Best Parameters: {'n_estimators': 200, 'min_samples_split': 2, 'max_depth': None}
  - Best Score: 0.892
  - Train Accuracy: 0.999
  - Test Accuracy: 0.906

- **Decision Tree**:
  - Best Parameters: {'min_samples_split': 5, 'max_depth': 20}
  - Best Score: 0.853
  - Train Accuracy: 0.949
  - Test Accuracy: 0.866

- **Gradient Boosting**:
  - Best Parameters: {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.2}
  - Best Score: 0.908
  - Train Accuracy: 0.918
  - Test Accuracy: 0.908

- **AdaBoost**:
  - Best Parameters: {'n_estimators': 200, 'learning_rate': 1}
  - Best Score: 0.879
  - Train Accuracy: 0.880
  - Test Accuracy: 0.879
## Model Evaluation

### Accuracy Score

The accuracy score is a metric used to measure the overall performance of a classification model. It represents the proportion of correctly predicted instances out of the total instances. In the context of this bank customer churn prediction model, the accuracy score indicates the percentage of correctly predicted churned and non-churned customers.

### Area Under the ROC Curve (AUC-ROC)

The AUC-ROC (Area Under the Receiver Operating Characteristic Curve) is a metric used to evaluate the performance of binary classification models. It represents the area under the curve plotted by the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity) for different threshold values. A higher AUC-ROC value indicates better discrimination capability of the model in distinguishing between positive and negative classes.

### Determining the Best Model

To determine the best model among the trained classifiers, we consider both the accuracy score and the AUC-ROC value. 

1. **Accuracy Score**: We look for the model with the highest accuracy score on the test dataset. A higher accuracy score indicates that the model predicts the correct class labels more accurately.

2. **AUC-ROC**: We also consider the AUC-ROC value. A higher AUC-ROC value indicates better overall performance in terms of both sensitivity and specificity. It helps to assess how well the model is able to distinguish between positive and negative instances.

By comparing the accuracy scores and AUC-ROC values of different models, we can determine which model performs best for the task of predicting bank customer churn. Typically, we choose the model with the highest accuracy score and AUC-ROC value as the best-performing model for deployment.

