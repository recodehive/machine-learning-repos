
# Diamond Price Detection

## Overview

This project involves building a predictive model to estimate the price of diamonds based on various features. The model is trained using multiple machine learning regressor algorithms and Keras deep learning model, with the LightGBM Regressor providing the best performance, achieving a 97.91% accuracy.
## Methodology

Utilizing a combination of EDA techniques and machine learning algorithms and deep learning, we have meticulously analyzed data to discern patterns and correlations associated with the diamonds features and price . Key steps include data cleaning, feature engineering, and insightful visualization to extract meaningful insights.
## Data Preprocessing 

Data preprocessing steps include:
1. Handling Missing Values: Impute missing values if any.
2. Encoding Categorical Features: Use one-hot encoding for ordinal and nominal features.
3. Scaling: Standardize numerical features to have zero mean and unit variance.


## Models Utilized

1. Linear Regression
2. Random Forest Regressor
3. Decision Tree Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor
6. LightGBM Regressor (best performer)
7. Keras Deep Learning Model
## Libraries Used

1. numpy: For efficient numerical operations
2. pandas: For data manipulation and analysis
3. seaborn: For visually appealing statistical graphics
4. matplotlib: For comprehensive data visualization
5. Sklearn: For implementing machine learning algorithms
6. TensorFlow and Keras
## Results

1. Linear Regression : 91.86%
2. Random Forest Regressor : 95.87%
3. Decision Tree Regressor : 95.00%
4. Gradient Boosting Regressor : 97.84%
5. XGBoost Regressor : 97.76%
6. LightGBM Regressor : 97.91%
7. Keras Deep Learning Model :  Test loss: 0.05000348016619682


## Conclusion
Through rigorous analysis and experimentation, it has been determined that LightGBM Regressor model exhibit the highest predictive accuracy for Diamond Price Prediction.