import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# 1. Feature columns required by app.py
all_columns = [
    'Age', 'Number_of_Dependents', 'Work_Experience', 'Household_Size',
    "Education_Level_Doctorate", "Education_Level_High School", "Education_Level_Master's",
    'Occupation_Finance', 'Occupation_Healthcare', 'Occupation_Others', 'Occupation_Technology',
    'Location_Suburban', 'Location_Urban', 'Marital_Status_Married', 'Marital_Status_Single',
    'Employment_Status_Part-time', 'Employment_Status_Self-employed',
    'Homeownership_Status_Rent', 'Type_of_Housing_Single-family home', 'Type_of_Housing_Townhouse',
    'Gender_Male', 'Primary_Mode_of_Transportation_Car', 'Primary_Mode_of_Transportation_Public transit',
    'Primary_Mode_of_Transportation_Walking'
]

# 2. Generate training data
np.random.seed(42)
n_samples = 2500

age = np.random.randint(22, 65, size=n_samples)
work_exp = np.clip(age - 21 - np.random.randint(0, 5, size=n_samples), 0, 40)
dependents = np.random.randint(0, 5, size=n_samples)
household_size = dependents + np.random.randint(1, 3, size=n_samples)

education_levels = ["Bachelor's", "Master's", "High School", "Doctorate"]
occupations = ["Technology", "Finance", "Others", "Education", "Healthcare"]
locations = ["Urban", "Rural", "Suburban"]
marital_statuses = ["Single", "Married", "Divorced"]
employment_statuses = ["Full-time", "Part-time", "Self-employed"]
homeownership = ["Own", "Rent"]
housing_types = ["Apartment", "Single-family home", "Townhouse"]
genders = ["Male", "Female"]
transportation = ["Public transit", "Biking", "Car", "Walking"]

df = pd.DataFrame({
    'Age': age,
    'Education_Level': np.random.choice(education_levels, size=n_samples),
    'Occupation': np.random.choice(occupations, size=n_samples),
    'Number_of_Dependents': dependents,
    'Location': np.random.choice(locations, size=n_samples),
    'Work_Experience': work_exp,
    'Marital_Status': np.random.choice(marital_statuses, size=n_samples),
    'Employment_Status': np.random.choice(employment_statuses, size=n_samples),
    'Household_Size': household_size,
    'Homeownership_Status': np.random.choice(homeownership, size=n_samples),
    'Type_of_Housing': np.random.choice(housing_types, size=n_samples),
    'Gender': np.random.choice(genders, size=n_samples),
    'Primary_Mode_of_Transportation': np.random.choice(transportation, size=n_samples)
})

# 3. Calculate realistic income values in Rs (matching the notebook scale ~80,000 to ~300,000 Rs)
base_income = 100000 + (df['Age'] * 1200) + (df['Work_Experience'] * 2500)
edu_bonus = df['Education_Level'].map({"High School": -20000, "Bachelor's": 0, "Master's": 25000, "Doctorate": 50000})
occ_bonus = df['Occupation'].map({"Technology": 35000, "Finance": 30000, "Healthcare": 15000, "Education": 5000, "Others": 0})
loc_bonus = df['Location'].map({"Urban": 20000, "Suburban": 10000, "Rural": -10000})
noise = np.random.normal(0, 15000, size=n_samples)

income = np.clip(base_income + edu_bonus + occ_bonus + loc_bonus + noise, 50000, 450000)
df['Income'] = income

# 4. Dummy encode and reindex to match all_columns
df_encoded = pd.get_dummies(df.drop(columns=['Income']), drop_first=True)
df_encoded = df_encoded.reindex(columns=all_columns, fill_value=0)

X = df_encoded
y = df['Income']

# 5. Train Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
gb_model.fit(X, y)

# 6. Save model artifact
output_path = os.path.join(os.path.dirname(__file__), 'gb_model_income.pkl')
joblib.dump(gb_model, output_path)
print(f"[+] Successfully trained and saved model to: {output_path}")
