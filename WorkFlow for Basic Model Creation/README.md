# Workflow for basic Machine Learning Model

Creating and saving models in machine learning involves several steps. Here's an overview of the general process followed :

## 1. Define the Problem and Collect Data

1.	Problem Definition: Clearly define the problem you're trying to solve (e.g., classification, regression, clustering).
2.	Data Collection: Gather the data you need. This could be from a CSV file, database, or an API.

## 2. Data Preprocessing

1.	Data Cleaning: Handle missing values, remove duplicates, and correct errors.
2.	Feature Engineering: Create new features, normalize/standardize data, and encode categorical variables.
3.	Train-Test Split: Split your data into training and testing sets.

## 3. Model Selection and Training

1.	Choose a Model: Select an appropriate machine learning model based on the problem.
2.	Train the Model: Fit the model to your training data.
3.	Evaluate the Model: Assess the model’s performance using metrics like accuracy, precision, recall, F1-score, etc.

## 4. Model Tuning

1.	Hyperparameter Tuning: Optimize the model’s hyperparameters using techniques like grid search or random search.

## 5. Model Saving

1.	Save the Model: Save the trained model to disk so that it can be loaded and used later without retraining.


(Note: These are just the basic steps, implementation of these highly depends on the dataset)

## Contents

1.  Model training using Scikit-learn.
2.  Model training using TensorFlow.