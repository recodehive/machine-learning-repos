# Traffic Prediction Model

This project aims to explore and model traffic data from four junctions in a city to predict traffic patterns and help reduce congestion. By analyzing the number of vehicles at different times, the project can assist city planners in making data-driven decisions to improve infrastructure and traffic management.

## Project Overview

Urban areas often face traffic congestion due to increasing populations and outdated infrastructure. Using historical traffic data from sensors at four key junctions, this project builds predictive models that forecast the number of vehicles at any given time. These predictions can be used to manage traffic more effectively and alleviate congestion in critical areas.

### Dataset

The dataset consists of hourly vehicle counts at four junctions in the city. Some junctions provide sparse data due to sensor limitations. The data is stored in a CSV file within the `data/` directory.

### Features of the Dataset
- **DateTime**: The date and time of data collection.
- **Junctions**: The ID of the junction where the data was collected.
- **Vehicles**: The number of vehicles recorded at the junction at that specific time.
- **ID**: Unique identifier for each record.

### Use Case
Accurate traffic prediction can help city authorities and urban planners optimize traffic flow, adjust traffic signals, and develop infrastructure that accommodates growing populations. This project serves as a starting point for building such a system.

## Key Features

### Data Preprocessing
- **Datetime conversion**: The `DateTime` feature is split into components like `Hour`, `Day`, `Month`, and `Weekday` to capture time-specific patterns.
- **Handling missing data**: Missing and sparse data from some junctions are addressed through imputation or removal.
- **Feature engineering**: Created features like rush-hour indicators and weekend flags to improve model accuracy.

### Model Building
- **Linear Regression**: A simple baseline model.
- **Random Forest, XGBoost**: More advanced models to capture non-linear traffic patterns.
- **Time Series Models (SARIMA, Prophet)**: Predict traffic trends over time.
- **Deep Learning (LSTM)**: Capture long-term dependencies for sequence modeling.

### Evaluation Metrics
- **Mean Absolute Percentage Error (MAPE)**: Used to measure the accuracy of traffic predictions.
- **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**: To understand error magnitudes.

## Results

- **Initial findings**: Traffic shows clear patterns during peak hours (morning and evening rush hours) and fluctuates on weekends.
- **Model performance**: Random Forest and XGBoost models outperformed simpler models, capturing complex patterns in traffic flow. Time series models like SARIMA showed promising results for time-dependent predictions.
- **Future improvements**: The model can be further enhanced by incorporating external data like weather, holidays, or accidents to improve predictions.

## Conclusion

This project provides a robust foundation for predicting urban traffic patterns, helping cities optimize their infrastructure to reduce congestion. Future work includes deploying the model into a real-time system that assists traffic management teams.
