# Tesla Stock Price Prediction Using Facebook Prophet 🤖

This project utilizes the Facebook Prophet algorithm to predict Tesla's stock price for the next 30 days. By leveraging historical data from Yahoo Finance, the model can generate forecasts that help in analyzing and understanding stock price trends, which can be beneficial for investors, analysts, and financial enthusiasts.

## Goal 🎯

The aim of this project is to develop a robust stock price prediction model using the Facebook Prophet algorithm that can accurately forecast Tesla's stock price for the next 30 days.

## Data Set 📊

The historical stock price data for Tesla is sourced from Yahoo Finance. This dataset includes key financial metrics such as Open, High, Low, Close prices, and Volume.

## Methodology 🔎

The project follows a structured approach to train and evaluate the stock price prediction model. Key steps include:

1.  **Data Collection**: Collected historical stock price data for Tesla from Yahoo Finance.
2.  **Data Visualization**: Visualized the historical stock price data using Plotly Express to identify trends and patterns.
3.  **Data Preparation**: Prepared the data to fit the input requirements of the Facebook Prophet model.
4.  **Model Training**: Trained the Facebook Prophet model on the historical stock price data.
5.  **Forecasting**: Used the trained model to forecast Tesla's stock price for the next 30 days.
6.  **Evaluation**: Compared the forecasted data with actual stock prices from Google Finance and evaluated the model's performance using metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

## Model Utilized 🚀

-   **Facebook Prophet**: A powerful time series forecasting algorithm developed by Facebook, designed for handling time series data with daily observations that display seasonality, trends, and holiday effects.

## Libraries Used 📝

1.  **pandas**: For data manipulation and analysis.
2.  **numpy**: For numerical computations.
3.  **matplotlib**: For data visualization.
4.  **seaborn**: For enhanced data visualization.
5.  **plotly.express**: For interactive visualizations.
6.  **yfinance**: For fetching historical stock price data from Yahoo Finance.
7.  **fbprophet**: For time series forecasting.
8.  **Google Finance**: For validating and comparing forecasted data with actual stock prices.

## Results 📢

The Facebook Prophet model provided reliable predictions for Tesla's stock price. The evaluation metrics used to assess the model's performance included:

-   **Mean Absolute Error (MAE)** and **Root Mean Square Error (RMSE)** are calculated and evaluated.

These results demonstrate the model's ability to effectively forecast stock prices with reasonable accuracy.

## 📌Conclusion

The Facebook Prophet-based stock price prediction model proved to be effective in forecasting Tesla's stock price for the next 30 days. The combination of Prophet's robust time series forecasting capabilities and the comprehensive historical data from Yahoo Finance makes this model a valuable tool for financial analysis and decision-making.
