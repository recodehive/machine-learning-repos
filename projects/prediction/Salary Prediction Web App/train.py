import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# 1. Categories defined in the Streamlit app
countries = [
    "United States of America",
    "Germany",
    "United Kingdom of Great Britain and Northern Ireland",
    "Canada",
    "France",
    "India",
    "Netherlands",
    "Australia",
    "Spain",
    "Sweden",
    "Brazil",
    "Italy",
    "Poland",
    "Switzerland",
    "Denmark",
    "Norway",
    "Israel",
]

education_levels = [
    "Bachelor’s degree",
    "Master’s degree",
    "Less than a Bachelors",
    "Post grad",
]

# Base salary scale per country (approximate annual USD)
base_salary = {
    "United States of America": 125000,
    "Germany": 85000,
    "United Kingdom of Great Britain and Northern Ireland": 90000,
    "Canada": 95000,
    "France": 75000,
    "India": 25000,
    "Netherlands": 85000,
    "Australia": 100000,
    "Spain": 55000,
    "Sweden": 70000,
    "Brazil": 35000,
    "Italy": 55000,
    "Poland": 48000,
    "Switzerland": 135000,
    "Denmark": 90000,
    "Norway": 88000,
    "Israel": 110000,
}

edu_multiplier = {
    "Less than a Bachelors": 0.85,
    "Bachelor’s degree": 1.0,
    "Master’s degree": 1.15,
    "Post grad": 1.25,
}

# 2. Generate training data
np.random.seed(42)
n_samples = 2500

sample_countries = np.random.choice(countries, size=n_samples)
sample_ed = np.random.choice(education_levels, size=n_samples)
sample_exp = np.random.uniform(0.5, 30.0, size=n_samples)

salaries = []
for c, ed, exp in zip(sample_countries, sample_ed, sample_exp):
    base = base_salary[c] * edu_multiplier[ed]
    exp_boost = exp * 3500
    noise = np.random.normal(0, 8000)
    salary = max(15000, min(250000, base + exp_boost + noise))
    salaries.append(salary)

salaries = np.array(salaries)

# 3. Fit Encoders
le_country = LabelEncoder()
le_country.fit(countries)
country_encoded = le_country.transform(sample_countries)

le_education = LabelEncoder()
le_education.fit(education_levels)
education_encoded = le_education.transform(sample_ed)

X = np.column_stack([country_encoded, education_encoded, sample_exp])
y = salaries

# 4. Train Model
regressor = DecisionTreeRegressor(max_depth=6, random_state=42)
regressor.fit(X, y)

# 5. Save model artifact
output_dir = os.path.dirname(__file__)
model_file = os.path.join(output_dir, "saved_steps.pkl")

with open(model_file, "wb") as f:
    pickle.dump({"model": regressor, "le_country": le_country, "le_education": le_education}, f)

print(f"[+] Successfully exported model artifact to: {model_file}")

# 6. Save a sample survey CSV for the explore page
survey_file = os.path.join(output_dir, "survey_results_public.csv")
survey_df = pd.DataFrame({
    "Country": sample_countries,
    "EdLevel": sample_ed,
    "YearsCodePro": sample_exp.round().astype(int),
    "Employment": "Employed, full-time",
    "ConvertedCompYearly": salaries.round(2),
})
survey_df.to_csv(survey_file, index=False)
print(f"[+] Successfully created sample dataset for explore page: {survey_file}")
