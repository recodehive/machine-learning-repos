# Dataset: [Customer.csv](Dataset/Customer.csv)

## Overview

This dataset contains information on bank customers and their characteristics, with 100001 rows and 14 columns. Each row represents a unique customer record, while the columns provide various features such as customer demographics, financial status, and banking behavior. With its diverse range of features, this dataset is suitable for predictive modeling tasks, particularly in predicting customer churn. It offers valuable insights into customer behavior and allows businesses to implement targeted strategies for customer retention and satisfaction.

### Features

- **id**: A unique identifier for each record.
- **CustomerId**: A unique identifier for each customer.
- **Surname**: The surname of the customer.
- **CreditScore**: The credit score of the customer, indicating their creditworthiness.
- **Geography**: The geographical location of the customer.
- **Gender**: The gender of the customer.
- **Age**: The age of the customer.
- **Tenure**: The number of years the customer has been with the bank.
- **Balance**: The account balance of the customer.
- **NumOfProducts**: The number of bank products the customer uses.
- **HasCrCard**: Whether the customer has a credit card (1 if yes, 0 if no).
- **IsActiveMember**: Whether the customer is an active member of the bank (1 if yes, 0 if no).
- **EstimatedSalary**: The estimated salary of the customer.


### Dataset Details

- **Total Instances**: 100001
- **Features**: 13
- **Target Variable**: Exited (Whether the customer has churned: 1 if yes, 0 if no)

## Preprocessing

The dataset has undergone preprocessing to prepare it for machine learning tasks. The following steps were applied:

- **Feature Scaling**: Two scaling techniques were used:
  - **MinMax Scaling**: Applied to columns 'Tenure' and 'NumOfProducts' using `MinMaxScaler`.
  - **Standard Scaling**: Applied to columns 'CreditScore', 'Age', and 'Balance' using `StandardScaler`.
- **One-Hot Encoding**: The categorical column 'Gender' was one-hot encoded to convert it into numerical format, enabling machine learning algorithms to process it effectively.

These preprocessing steps ensure that all features are on the same scale and in a suitable format for training machine learning models. The dataset is now ready for further analysis and model building.

## Usage

This dataset can be used for various purposes, including:

- **Training Machine Learning Models**: Use the dataset to train machine learning models for predicting the target variable (e.g., customer churn). By leveraging the provided features, you can build predictive models to identify patterns and trends in customer behavior.
- **Exploratory Data Analysis (EDA)**: Perform EDA to gain insights into specific aspects of the data, such as customer demographics, financial behavior, and churn patterns. Visualization techniques can help uncover relationships and correlations between different features, providing valuable insights for decision-making.
- **Research Projects**: The dataset can serve as a valuable resource for research projects related to banking, customer behavior analysis, and predictive modeling. Researchers can explore various hypotheses and conduct experiments to validate findings in the context of customer churn prediction and proactive retention strategies.
