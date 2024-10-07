
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import GridSearchCV
import os.path
import pickle

# Function to fetch historical Ethereum price data from CoinGecko API
def fetch_ethereum_data():
    # Make a request to the CoinGecko API to fetch historical Ethereum price data
    response = requests.get('https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=365')
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON data
        ethereum_data = response.json()
        
        # Extract relevant data from the response
        prices = ethereum_data['prices']
        ethereum_df = pd.DataFrame(prices, columns=['Timestamp', 'Price'])
        
        # Convert timestamp to datetime
        ethereum_df['Timestamp'] = pd.to_datetime(ethereum_df['Timestamp'], unit='ms')
        
        return ethereum_df
    else:
        # If the request failed, return None
        print("Error in API")
        return None

# Function to train the model
def train_model(X, y):
    # Feature scaling using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scale the features
    
    # Hyperparameter tuning for Random Forest
    rf_param_grid = {'n_estimators': [100, 200, 300],
                     'max_depth': [None, 10, 20, 30],
                     'min_samples_split': [2, 5, 10]}
    rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=5)
    rf_grid_search.fit(X_scaled, y)
    best_rf_model = rf_grid_search.best_estimator_

    # Hyperparameter tuning for Extra Trees
    et_param_grid = {'n_estimators': [100, 200, 300],
                     'max_depth': [None, 10, 20, 30],
                     'min_samples_split': [2, 5, 10]}
    et_grid_search = GridSearchCV(ExtraTreesRegressor(), et_param_grid, cv=5)
    et_grid_search.fit(X_scaled, y)
    best_et_model = et_grid_search.best_estimator_
    
    return best_rf_model, best_et_model, scaler  # Return scaler for future scaling

# Function to check if the model needs to be retrained
def check_if_retrain_needed(ethereum_data):
    # Check if the model file exists
    if not os.path.exists("trained_model.pkl"):
        return True
    
    # Check the last modification time of the model file
    last_modified_time = datetime.fromtimestamp(os.path.getmtime("trained_model.pkl"))
    
    # Check if the last training time is more than a day ago
    if datetime.now() - last_modified_time > timedelta(days=1):
        return True
    else:
        return False

# Load historical Ethereum price data from the CoinGecko API
ethereum_data = fetch_ethereum_data()

if ethereum_data is not None and check_if_retrain_needed(ethereum_data):
    
    ethereum_data['Year'] = ethereum_data['Timestamp'].dt.year
    ethereum_data['Month'] = ethereum_data['Timestamp'].dt.month
    ethereum_data['Day'] = ethereum_data['Timestamp'].dt.day
    ethereum_data['Weekday'] = ethereum_data['Timestamp'].dt.weekday
        
    # Split data into features (X) and target variable (y)
    X = ethereum_data[['Year', 'Month', 'Day', 'Weekday']]  
    y = ethereum_data['Price']  
    
    # Train the model
    best_rf_model, best_et_model, scaler = train_model(X, y)
    
    # Save the trained model and scaler
    with open("trained_model.pkl", "wb") as f:
        pickle.dump((best_rf_model, best_et_model, scaler), f)
        
elif ethereum_data is not None and not check_if_retrain_needed(ethereum_data):
    print("Model is already up to date, no need to retrain.")
    
else:
    print("Failed to fetch Ethereum data from the CoinGecko API.")

# Load the pre-trained model and scaler
if os.path.exists("trained_model.pkl"):
    with open("trained_model.pkl", "rb") as f:
        best_rf_model, best_et_model, scaler = pickle.load(f)

# Random Forest
def predict_prices_for_future_random_forest(date):
    # Preprocess input date provided by the user
    date = pd.to_datetime(date)
    year = date.year
    month = date.month
    day = date.day
    weekday = date.weekday()

    # Feature scaling for prediction using the stored scaler
    scaled_features_for_date = scaler.transform([[year, month, day, weekday]])

    # Make prediction using the best Random Forest model
    predicted_price = best_rf_model.predict(scaled_features_for_date)[0]

    # Return tuple of predicted price and default value for predicted high and low
    return predicted_price, predicted_price + 100, predicted_price - 200
    
    #Extra Trees

def predict_prices_for_future_extra_trees(date):
    
    date = pd.to_datetime(date)
    year = date.year
    month = date.month
    day = date.day
    weekday = date.weekday()

    
    scaled_features_for_date = scaler.transform([[year, month, day, weekday]])

   
    predicted_price = best_et_model.predict(scaled_features_for_date)[0]

    return predicted_price, predicted_price + 100, predicted_price - 200
