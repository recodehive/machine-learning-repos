# Retail Price Optimization Machine Learning Model

This repository contains a machine learning model for retail price optimization. The model predicts the optimal selling price of a product based on various factors such as historical sales data, competitor prices, product features, customer ratings, and more. The model is designed to help retailers maximize revenue by dynamically adjusting prices in real-time.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Requirements](#requirements)

- [Results](#results)

## Project Overview

In highly competitive retail markets, setting the right price for products is crucial to maximizing profits while remaining competitive. This model leverages machine learning to provide dynamic, real-time price recommendations based on historical data and product characteristics.

### Key Features:
- Automatically optimizes product prices based on past sales, competitor prices, product ratings, and stock levels.
- Supports real-time price adjustments to maximize revenue and profit margins.
- Includes a regression-based approach using models like **Random Forest** and **XGBoost** for prediction.

## Dataset

The dataset contains the following key features:
- **Cost_Price**: The retailer's cost for the product.
- **Historical_Sales**: Number of units sold in the past.
- **Discount_Offered**: Discounts applied to the product.
- **Competitor_Price**: The price offered by competitors for similar products.
- **Rating**: Customer ratings of the product.
- **Stock_Available**: Number of units currently in stock.
- **Category**: Product categories, such as Electronics, Fashion, Home Decor, etc.

Each product is represented by a unique combination of these features, and the target variable is the **Selling_Price**.

## Model

The model used in this project is a **RandomForestRegressor**, but other regression models (like **XGBoost** or **Linear Regression**) can also be used.

### Key Steps:
1. **Data Preprocessing**: Handle missing values, convert categorical variables to dummy/one-hot encoding, and normalize features.
2. **Model Training**: Train the model on historical sales and product data.
3. **Prediction**: Use the model to predict optimal prices for new products.
4. **Evaluation**: Evaluate model performance using RMSE, MAE, and R² score.

## Requirements

To run this project, you need the following libraries:

```bash
- Python 3.7+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- joblib
```
Install the dependencies using:

```bash
pip install -r requirements.txt

```
### Results
The model is evaluated using the following metrics:

- **Root Mean Squared Error (RMSE):** Measures the average magnitude of errors in predictions.
- **Mean Absolute Error (MAE):** Measures the average of the absolute errors.
- **R² Score:** Measures the proportion of variance in the dependent variable that the independent variables explain.

You can also visualize the results using various plots (predicted vs actual prices, error distribution).
