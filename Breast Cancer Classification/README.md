# Breast Cancer Classification using Logistic Regression

## Project Overview

Welcome to the Breast Cancer Classification project! This project is a part of the GirlScript Summer of Code program. The objective of this project is to build a machine learning model using Logistic Regression to classify whether a given tumor is malignant or benign based on various features.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [License](#license)

## Introduction

Breast cancer is one of the most common cancers among women worldwide. Early detection and treatment are crucial for improving survival rates. Machine learning can aid in the early detection of breast cancer by classifying tumors as malignant or benign based on various features.

In this project, we will:
1. Load and preprocess the breast cancer dataset.
2. Train a Logistic Regression model on the dataset.
3. Evaluate the performance of the model.
4. Provide usage instructions for making predictions on new data.

## Installation

To get started with this project, you'll need to have Python installed on your machine. You can install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Data Set. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

- **Features:** 30 numeric features representing characteristics of the cell nuclei.
- **Target:** Binary classification (malignant or benign).


## Data Preprocessing

Data preprocessing steps include:
1. Loading the dataset.
2. Handling missing values (if any).
3. Encoding categorical variables (if any).
4. Scaling the features.

## Model Training

We will use Logistic Regression for the classification task. Logistic Regression is a simple yet powerful algorithm for binary classification problems. The steps for training the model are as follows:
1. Split the dataset into training and testing sets.
2. Initialize the Logistic Regression model.
3. Train the model on the training data.
4. Tune hyperparameters if necessary.

## Model Evaluation

To evaluate the performance of the model, we will use metrics such as:
- Accuracy
- Precision
- Recall
- F1-Score

Confusion matrix and ROC-AUC curve will also be plotted for a comprehensive evaluation.

## Usage

To use the trained model for making predictions on new data, follow these steps:

1. Load the pre-trained model from the saved file.
2. Prepare the input data in the same format as the training data.
3. Use the model to make predictions.

Example code:

```python
import pickle
import numpy as np

# Load the pre-trained model
with open('breast_cancer_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sample input data (replace with actual data)
input_data = np.array([[...]])

# Make predictions
predictions = model.predict(input_data)
print(predictions)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
