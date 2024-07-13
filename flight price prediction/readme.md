# Flight price prediction using machine learning

n this project, I developed a flight price prediction system using machine learning techniques. The dataset consisted of various flight details including airline, source, destination, total stops, departure and arrival times, and flight duration. I began by preprocessing the data, converting 24-hour times to 12-hour format with AM/PM labels and formatting flight durations into hours and minutes. Categorical variables were encoded appropriately for model training.

I trained multiple regression models including Linear Regression, Random Forest, Gradient Boosting Regressor, and XGBoost to predict flight prices. Each model was evaluated using metrics such as R^2 score, mean squared error (MSE), root mean squared error (RMSE), and mean absolute percentage error (MAPE). Visualizations, including scatter plots comparing actual vs. predicted prices, provided insights into model performance.

The Gradient Boosting Regressor demonstrated the highest accuracy, achieving an R^2 score of 0.50. Real-time predictions were implemented using this model, allowing users to input flight details and receive instant price estimates. Overall, the project integrated data preprocessing, model training, evaluation, and real-time prediction, showcasing the application of machine learning in predicting flight prices accurately.

## Models applied

- GBM
- Multiple linear regression
- XGBOOST
- Random Forest