import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 1. Exact 12 features expected by Webapp.py
feature_names = [
    'AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity',
    'WindSpeed', 'RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions'
]

# 2. Generate representative dataset
np.random.seed(42)
n_samples = 2500

aqi = np.random.uniform(20.0, 300.0, size=n_samples)
pm10 = np.random.uniform(10.0, 250.0, size=n_samples)
pm2_5 = np.random.uniform(5.0, 150.0, size=n_samples)
no2 = np.random.uniform(5.0, 100.0, size=n_samples)
so2 = np.random.uniform(2.0, 80.0, size=n_samples)
o3 = np.random.uniform(10.0, 90.0, size=n_samples)
temp = np.random.uniform(-5.0, 42.0, size=n_samples)
humidity = np.random.uniform(20.0, 95.0, size=n_samples)
wind = np.random.uniform(0.5, 25.0, size=n_samples)
resp_cases = np.random.randint(0, 30, size=n_samples)
cardio_cases = np.random.randint(0, 30, size=n_samples)
hosp_admissions = np.random.randint(0, 30, size=n_samples)

X = pd.DataFrame({
    'AQI': aqi,
    'PM10': pm10,
    'PM2_5': pm2_5,
    'NO2': no2,
    'SO2': so2,
    'O3': o3,
    'Temperature': temp,
    'Humidity': humidity,
    'WindSpeed': wind,
    'RespiratoryCases': resp_cases,
    'CardiovascularCases': cardio_cases,
    'HospitalAdmissions': hosp_admissions
})

# 3. Calculate health impact score (0 to 100 scale matching notebook categorization)
# < 35: Less Impacting, 35 - 45: Average Impacting, > 45: Severely Impacting
raw_score = (
    0.15 * aqi +
    0.08 * pm10 +
    0.10 * pm2_5 +
    0.05 * no2 +
    0.50 * resp_cases +
    0.50 * cardio_cases +
    0.50 * hosp_admissions +
    np.random.normal(0, 3, size=n_samples)
)
y = np.clip(raw_score, 5.0, 98.0)

# 4. Build exact Pipeline matching the notebook: StandardScaler + RandomForestRegressor
model_pipeline = Pipeline([
    ('step1', StandardScaler()),
    ('regressorRF', RandomForestRegressor(
        max_depth=10,
        max_samples=0.75,
        min_samples_split=10,
        random_state=42
    ))
])

# 5. Train pipeline
model_pipeline.fit(X, y)

# 6. Export model.pkl
output_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(model_pipeline, f)

print(f"[+] Successfully trained and exported model to: {output_path}")
