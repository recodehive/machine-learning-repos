from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained Gradient Boosting Regressor model
gb_model_loaded = joblib.load('gb_model_income.pkl')

# Define all possible columns based on the training data features
all_columns = ['Age', 'Number_of_Dependents', 'Work_Experience', 'Household_Size',
               "Education_Level_Doctorate", "Education_Level_High School", "Education_Level_Master's",
               'Occupation_Finance', 'Occupation_Healthcare', 'Occupation_Others', 'Occupation_Technology',
               'Location_Suburban', 'Location_Urban', 'Marital_Status_Married', 'Marital_Status_Single',
               'Employment_Status_Part-time', 'Employment_Status_Self-employed',
               'Homeownership_Status_Rent', 'Type_of_Housing_Single-family home', 'Type_of_Housing_Townhouse',
               'Gender_Male', 'Primary_Mode_of_Transportation_Car', 'Primary_Mode_of_Transportation_Public transit',
               'Primary_Mode_of_Transportation_Walking']


# Route to render the home page with form
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Receive input values from the form
    user_input = request.json

    # Convert input to DataFrame
    user_df = pd.DataFrame(user_input, index=[0])

    # Encode categorical variables to dummy variables
    user_df_encoded = pd.get_dummies(user_df, drop_first=True)

    # Reindex columns to match the model's expected input
    user_df_encoded = user_df_encoded.reindex(columns=all_columns, fill_value=0)

    # Predict using loaded Gradient Boosting Regressor
    predicted_income = gb_model_loaded.predict(user_df_encoded)

    # Return predicted income as JSON response
    return jsonify({'predicted_income': predicted_income[0]})


if __name__ == '__main__':
    app.run(debug=True)
