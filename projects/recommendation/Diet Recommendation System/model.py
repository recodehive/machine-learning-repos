# Install necessary libraries
# !pip install pandas numpy scikit-learn fpdf

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from fpdf import FPDF

# Sample Data Loading (replace this with a real dataset)
# Assuming we have a dataset with columns: Age, Gender, Weight, Height, Activity_Level, Goal, Calories, Dietary_Type
# For demonstration, we'll create a synthetic dataset

data = pd.DataFrame({
    'Age': np.random.randint(18, 60, 100),
    'Gender': np.random.choice(['Male', 'Female'], 100),
    'Weight': np.random.randint(50, 100, 100),
    'Height': np.random.randint(150, 200, 100),
    'Activity_Level': np.random.choice(['low', 'medium', 'high'], 100),
    'Goal': np.random.choice(['maintain', 'lose', 'gain'], 100),
    'Calories': np.random.randint(1500, 3000, 100),
    'Dietary_Type': np.random.choice(['balanced', 'high-protein', 'low-carb'], 100)
})

# Convert categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Activity_Level', 'Goal'], drop_first=True)

# Split data
X = data.drop(['Calories', 'Dietary_Type'], axis=1)
y_calories = data['Calories']
y_diet_type = data['Dietary_Type']

X_train, X_test, y_train_calories, y_test_calories = train_test_split(X, y_calories, test_size=0.2, random_state=42)
X_train, X_test, y_train_diet, y_test_diet = train_test_split(X, y_diet_type, test_size=0.2, random_state=42)

# Train the models
calorie_model = RandomForestRegressor(n_estimators=100, random_state=42)
diet_type_model = RandomForestClassifier(n_estimators=100, random_state=42)

calorie_model.fit(X_train, y_train_calories)
diet_type_model.fit(X_train, y_train_diet)

# Function to get recommendations
def get_diet_recommendation(age, gender, weight, height, activity_level, goal):
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age], 'Weight': [weight], 'Height': [height],
        'Gender_Male': [1 if gender.lower() == 'male' else 0],
        'Activity_Level_medium': [1 if activity_level == 'medium' else 0],
        'Activity_Level_high': [1 if activity_level == 'high' else 0],
        'Goal_lose': [1 if goal == 'lose' else 0],
        'Goal_gain': [1 if goal == 'gain' else 0]
    })

    # Predict calories and diet type
    calories_needed = calorie_model.predict(input_data)[0]
    diet_type = diet_type_model.predict(input_data)[0]

    # Basic meal recommendations based on diet type
    diet_plan = {
        'balanced': {
            'Breakfast': f'Oatmeal with fruits - {calories_needed * 0.3:.0f} cal',
            'Lunch': f'Grilled chicken with mixed veggies - {calories_needed * 0.4:.0f} cal',
            'Dinner': f'Salad with chickpeas - {calories_needed * 0.3:.0f} cal'
        },
        'high-protein': {
            'Breakfast': f'Egg whites with avocado - {calories_needed * 0.3:.0f} cal',
            'Lunch': f'Salmon with quinoa - {calories_needed * 0.4:.0f} cal',
            'Dinner': f'Chicken breast with broccoli - {calories_needed * 0.3:.0f} cal'
        },
        'low-carb': {
            'Breakfast': f'Smoothie with protein powder - {calories_needed * 0.3:.0f} cal',
            'Lunch': f'Tofu stir-fry - {calories_needed * 0.4:.0f} cal',
            'Dinner': f'Caesar salad with grilled meat - {calories_needed * 0.3:.0f} cal'
        }
    }

    return {
        'Calories Needed': round(calories_needed),
        'Dietary Type': diet_type,
        'Diet Plan': diet_plan[diet_type]
    }

# Example usage
age = 25
gender = 'Male'
weight = 70
height = 175
activity_level = 'medium'
goal = 'maintain'

recommendation = get_diet_recommendation(age, gender, weight, height, activity_level, goal)
print("\nYour Diet Recommendation:")
print(f"Calories Needed: {recommendation['Calories Needed']}")
print(f"Dietary Type: {recommendation['Dietary Type']}")
print("Meal Plan:")
for meal, details in recommendation['Diet Plan'].items():
    print(f"{meal}: {details}")
