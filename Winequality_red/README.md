
## Wine Quality Detector

## Goal
The primary goal of this project is to develop a predictive model to determine the quality of wine based on various physicochemical properties. By leveraging different machine learning algorithms, we aim to find the most accurate model for predicting wine quality.
## Methodology

1. Data Collection: The dataset is sourced from the UCI Machine Learning Repository and contains chemical properties of red wine samples along with quality ratings.

2. Data Preprocessing: Handle missing values, standardize features, and split the data into training and testing sets.

3. Model Training: Train multiple machine learning models to predict wine quality.

4. Model Evaluation: Evaluate the performance of each model using appropriate regression metrics.
## Models Utilized

1. Support Vector Regression (SVR)
2. Random Forest Regressor
3. Gradient Boosting Machine (XGBoost)

## Libraries Used

1. numpy: For efficient numerical operations
2. pandas: For data manipulation and analysis
3. xgboost: For Extreme Gradient Boosting
5. Sklearn: For implementing machine learning algorithms
## Results

1. SVR:
    -Mean Absolute Error (MAE): 0.4535910872530087
    -Mean Squared Error (MSE): 0.35172233375700596
    -R-squared (R²): 0.46179161408990865
2. Random Forest Regressor:
    -Mean Absolute Error (MAE): 0.42384374999999996
    -Mean Squared Error (MSE): 0.30341593749999995
    -R-squared (R²): 0.5357104559243264
3. XGBoost Regressor:
    -Mean Absolute Error (MAE): 0.4191360227763653
    -Mean Squared Error (MSE): 0.3603729762415421
    -R-squared (R²): 0.4485543303526067
## Conclusion

Based on the evaluation metrics, the model with the highest R-squared value and lowest MAE and MSE is considered the best performing model for predicting wine quality. Typically, models like Random Forest and XGBoost perform well on complex datasets due to their ability to capture non-linear relationships. The exact metrics will determine the best model in this case.
