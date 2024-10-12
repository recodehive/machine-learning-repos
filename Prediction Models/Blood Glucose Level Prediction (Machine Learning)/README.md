# Blood Glucose Level Prediction

## Overview

This project aims to predict blood glucose levels using machine learning techniques, specifically focusing on the relationship between glucose levels, insulin doses, and carbohydrate intake over time. The dataset includes timestamps, glucose levels, insulin doses, and carbohydrate intake.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Training and Evaluation](#model-training-and-evaluation)


## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Dataset

The dataset used for this project is `blood_glucose_data.csv`, which contains the following columns:

- `timestamp`: The date and time of the recorded glucose level.
- `glucose_level`: The level of glucose in mg/dL.
- `insulin_dose`: The dose of insulin administered in units.
- `carb_intake`: The amount of carbohydrate intake in grams.

## Installation

To run this project, make sure you have the following libraries installed. You can install them using pip:

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Model Training and Evaluation
The model is trained using a linear regression approach with the following features:

- Hour of the day.
- Day of the week.
- Insulin dose.
- Carbohydrate intake.
  
### Model Evaluation Metrics
- Mean Absolute Error (MAE): 15.43
- Root Mean Squared Error (RMSE): 19.20
Results
The results of the model training can be visualized to compare actual glucose levels with predicted values. 
