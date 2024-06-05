import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the holidays data
holidays_data = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv')

# Convert the 'date' column to datetime format
holidays_data['date'] = pd.to_datetime(holidays_data['date'])

# Generate a date range from the start to the end of the holidays data
date_range = pd.date_range(start=holidays_data['date'].min(), end=holidays_data['date'].max(), freq='D')

# Create a DataFrame with the date range
time_series_data = pd.DataFrame(date_range, columns=['date'])

# Count the number of holidays on each date
holiday_counts = holidays_data['date'].value_counts().sort_index()

# Merge the holiday counts with the date range DataFrame, filling missing dates with 0
time_series_data = time_series_data.merge(holiday_counts.rename('count'), left_on='date', right_index=True, how='left').fillna(0)

# Set the 'date' column as the index
time_series_data.set_index('date', inplace=True)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
plt.plot(time_series_data['count'], label='Holiday Count')
plt.title('Holiday Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.show()

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(time_series_data['count'], ax=axes[0])
plot_pacf(time_series_data['count'], ax=axes[1])
plt.show()

# Fit an ARIMA model
model = ARIMA(time_series_data['count'], order=(1, 1, 1))
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Diagnostic plots
model_fit.plot_diagnostics(figsize=(12, 8))
plt.show()

# Forecasting
forecast_steps = 30
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(time_series_data['count'], label='Observed')
plt.plot(forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Holiday Count')
plt.legend()
plt.show()

# Model evaluation
# Creating a synthetic future for evaluation purposes
future_data = time_series_data['count'].append(pd.Series([0] * forecast_steps, index=pd.date_range(start=time_series_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)))
future_actual = future_data[-forecast_steps:]
mae = mean_absolute_error(future_actual, forecast.predicted_mean)
mse = mean_squared_error(future_actual, forecast.predicted_mean)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
