# Bank Customer Churn Prediction Model

## Overview
This repository contains code for a machine learning model that predicts customer churn for a bank. Customer churn refers to the phenomenon where customers stop doing business with a company. In the context of a bank, it means customers closing their accounts or ceasing to use banking services.
The model is built using [Python](https://www.python.org/) and popular machine learning libraries such as [scikit-learn](https://scikit-learn.org/). It incorporates classification algorithms including Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, and AdaBoost. The model is then integrated with Streamlit for interactive visualization and deployment. Leveraging historical customer data, it predicts the likelihood of a customer churning in the future.

## App-Interface 
![interface](images/interface.png)
## Dataset

The model is trained on a dataset [Customer.csv](Dataset/Customer.csv) containing historical information about bank customers. The dataset includes various features such as customer demographics, transaction history, account balances, etc. Each entry in the dataset is labeled with whether the customer churned or not.

### Features

1. **id**: A unique identifier for each record.
2. **CustomerId**: A unique identifier for each customer.
3. **Surname**: The surname of the customer.
4. **CreditScore**: The credit score of the customer, which indicates their creditworthiness.
5. **Geography**: The geographical location of the customer.
6. **Gender**: The gender of the customer.
7. **Age**: The age of the customer.
8. **Tenure**: The number of years the customer has been with the bank.
9. **Balance**: The account balance of the customer.
10. **NumOfProducts**: The number of bank products the customer uses.
11. **HasCrCard**: Whether the customer has a credit card (1 if yes, 0 if no).
12. **IsActiveMember**: Whether the customer is an active member of the bank (1 if yes, 0 if no).
13. **EstimatedSalary**: The estimated salary of the customer.
14. **Exited**: Whether the customer has churned (1 if yes, 0 if no).

This dataset can be used to train machine learning models to predict customer churn based on the provided features. The goal is to build a model that can accurately identify customers who are likely to churn, allowing the bank to take proactive measures to retain them.

## Visualizations 

Here are some visualizations generated from the dataset:

### Churn Rate Pie Chart
![Churn Rate Pie Chart](images/piechart.png)
*Description: This pie chart shows the distribution of churn rates among customers.*

### Exited Customers vs Geography
![Exited Customers vs Geography](images/exited-geography.png)
*Description: This visualization depicts the relationship between customer churn and geographical location.*

### Exited vs Gender
![Exited vs Gender](images/Exited-gender.png)
*Description: This plot illustrates the correlation between customer churn and gender.*

## Data Preprocessing

Before training the machine learning models, the dataset undergoes preprocessing to prepare it for analysis. Here's an overview of the preprocessing steps applied:

### Label Encoding and One-Hot Encoding

Categorical variables like 'Gender' are encoded using Label Encoding, converting them into numerical values. Additionally, to handle categorical variables with more than two categories, One-Hot Encoding is employed. This technique creates binary columns for each category, improving the model's ability to interpret categorical data.

### Feature Scaling

Feature scaling is crucial for machine learning algorithms, especially those based on distance calculations or gradient descent optimization. Two common scaling techniques are applied:

#### Min-Max Scaling

Min-Max scaling rescales features to a range between 0 and 1, preserving the distribution's shape while bringing all features to the same scale. This scaling is applied to features like 'Tenure' and 'NumOfProducts'.

#### Standard Scaling (Z-score Normalization)

Standard scaling transforms features to have a mean of 0 and a standard deviation of 1. It's particularly useful when features have different units or scales. Features such as 'CreditScore', 'Age', and 'Balance' are standardized using this technique.

### Final Preprocessed Dataset

After preprocessing, the dataset is ready for model training. Categorical variables are encoded, and numerical features are scaled to ensure all features contribute equally to model training. The preprocessed dataset is now suitable for feeding into machine learning algorithms for churn prediction.


## Model

The model employs several algorithms commonly used in binary classification tasks. Here's a brief overview of the algorithms utilized and their rationale:

### Logistic Regression

Logistic Regression is a simple yet powerful algorithm for binary classification. It's chosen for its interpretability, ease of implementation, and ability to provide probabilities for predictions. Despite its simplicity, Logistic Regression can capture linear relationships between features and the target variable.

### Random Forest

Random Forest is an ensemble learning technique that constructs multiple decision trees during training and outputs the mode of the classes (classification) or the mean prediction (regression) of the individual trees. It's known for its robustness to overfitting, handling of high-dimensional datasets, and feature importance estimation.

### Decision Tree

Decision Tree is a non-parametric supervised learning method used for classification and regression. It works by recursively partitioning the input space into regions based on feature values, with the goal of maximizing the homogeneity of the target variable within each region. Decision Trees are easy to interpret and visualize, making them valuable for understanding feature importance.

### Gradient Boosting

Gradient Boosting is an ensemble learning technique that builds models sequentially, with each new model correcting errors made by the previous ones. It combines weak learners (usually decision trees) to create a strong learner. Gradient Boosting is effective in capturing complex relationships in the data and has been successful in various machine learning competitions.

### AdaBoost

AdaBoost (Adaptive Boosting) is another ensemble learning method that combines multiple weak learners to form a strong classifier. It assigns higher weights to misclassified data points in each iteration, allowing subsequent weak learners to focus on the difficult examples. AdaBoost is particularly useful when dealing with imbalanced datasets and noisy data.

These algorithms are chosen for their complementary strengths in handling different aspects of the dataset and their proven effectiveness in classification tasks. By employing a diverse set of algorithms, we aim to build a robust model capable of accurately predicting customer churn and providing valuable insights for retention strategies.


## Usage

To use the model, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Run the app.py file in the terminal - for running app.py, open the command prompt and navigate to this directory, then run the command: python -m streamlit run app.py
4. Once the app is opened in your web browser, you can adjust various parameters to view the results.


## Evaluation
The model's performance is evaluated by Accuracy score and then Auc curves are plotted to check for the overfitting.In gradient boosting algorithm the train accuracy is nearly 0.91 and test acucracy is near 0.90 so it is the best model we can consider for further use as the data is not overfitting here.

### AUC Curve for Logistic Regression Algorithm
![AUC Curve for Logistic Regression Algorithm](images/output-lr.png)
*   Description: This graph represents the Area Under the Curve (AUC) for the Logistic Regression algorithm.*

### AUC Curve for Decision Tree Algorithm
![AUC Curve for Decision Tree Algorithm](images/output-dt.png)
*   Description: This chart displays the Area Under the Curve (AUC) for the Decision Tree algorithm.*

### AUC Curve for Gradient Boosting Algorithm
![AUC Curve for Gradient Boosting Algorithm](images/output-gb.png)
*   Description: This graph demonstrates the Area Under the Curve (AUC) for the Gradient Boosting algorithm.*
## Contributors
- [Nikita](https://github.com/NIKITA320495) - Contributor


