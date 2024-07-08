# Rock vs Mine Classification Using Logistic Regression

This project demonstrates a machine learning classification technique using Logistic Regression to predict whether a given sample is a rock or a mine. The model achieves an accuracy of 85%.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model](#model)
- [Results](#results)

## Introduction
This project aims to classify sonar signals as either rocks or mines. Using a Logistic Regression model, we have developed a highly accurate classifier that achieves an accuracy of 85%.

## Dataset
The dataset used in this project consists of sonar signals that have been labeled as either rocks or mines. Each sample is represented by a set of numerical features extracted from the sonar signal.

- **Source**: [UCI Machine Learning Repository: Sonar, Mines vs. Rocks Data Set](https://www.kaggle.com/datasets/mayurdalvi/sonar-mine-dataset)
- **Attributes**: 60 numerical features representing the sonar signal.
- **Classes**: Rock (R) and Mine (M).

## Installation
To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Recode-Hive/machine-learning-repos.git
cd rock-vs-mine-classification
pip install -r requirements.txt
```

# Model

## Logistic Regression for Binary Classification

In this project, we employ Logistic Regression—a robust statistical model primarily used for binary classification tasks. Unlike linear regression, which predicts continuous values, Logistic Regression predicts the probability of an instance belonging to a particular class, typically encoded as 0 or 1.

### Mathematical Intuition

Logistic Regression operates by applying a logistic (or sigmoid) function to a linear combination of input features. Let's denote:
- \( x_i \) as the feature vector for the \( i \)-th sample,
- \( \theta \) as the parameter vector,
- \( \theta^T x_i \) as the dot product of \( \theta \) and \( x_i \).

The logistic function \( \sigma(z) \) transforms the linear combination \( \theta^T x_i \) into a probability \( \hat{y}_i \) that the sample \( x_i \) belongs to class 1:

\[ \hat{y}_i = \sigma(\theta^T x_i) = \frac{1}{1 + e^{-\theta^T x_i}} \]

Where:
- \( e \) is the base of the natural logarithm.

### Training

During training, the model's parameters \( \theta \) are optimized to maximize the likelihood of correctly predicting the observed labels in the training data. This optimization typically involves minimizing the logistic loss function:

\[ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] \]

Where:
- \( m \) is the number of samples,
- \( y_i \) is the true label of the \( i \)-th sample.

### Implementation

In this project, we implement Logistic Regression using the `LogisticRegression` class from the `scikit-learn` library. This library provides an efficient and user-friendly interface for training and deploying Logistic Regression models. The model is trained on labeled data, and predictions are made based on the learned parameters \( \theta \).

This approach allows us to achieve high accuracy in classifying sonar signals as rocks or mines based on their acoustic features.

---

This markdown file provides a detailed overview of Logistic Regression as applied in this project, covering its mathematical foundation, training process, and implementation using the `scikit-learn` library.

# Results

## Performance Metrics

The Logistic Regression model was evaluated on the test dataset, and the following performance metrics were obtained:

- **Accuracy**: 85%
- **Recall**: High

These metrics indicate the model's effectiveness in distinguishing between rocks and mines based on sonar signals.

### Confusion Matrix

The confusion matrix provides a detailed breakdown of the model's performance:

|                | Predicted Rock | Predicted Mine |
|----------------|----------------|----------------|
| **Actual Rock**| True Positives (TP) | False Negatives (FN) |
| **Actual Mine**| False Positives (FP) | True Negatives (TN) |

### Focus on False Negatives

A critical aspect of this project was to minimize false negatives (FN), ensuring that the model does not miss detecting a mine when it is actually present. This focus helps in preventing potentially fatal errors in real-world applications where missing a mine detection could have severe consequences.

