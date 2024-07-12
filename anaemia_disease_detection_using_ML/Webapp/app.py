import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model_rf = load('model_rf.joblib')


# Function to predict anemia
def predict_anemia(sex, red_pixel, green_pixel, blue_pixel, Hb):
    # Create a DataFrame with user input
    user_input = {
        'Sex': sex,
        '%Red Pixel': red_pixel,
        '%Green pixel': green_pixel,
        '%Blue pixel': blue_pixel,
        'Hb': Hb
    }
    user_df = pd.DataFrame([user_input])

    # Label encode 'Sex' column
    label_encoder = LabelEncoder()
    user_df['Sex'] = label_encoder.fit_transform(user_df['Sex'])

    # Ensure the columns are in the same order as the model expects
    X_user = user_df[model_rf.feature_names_in_]  # Assuming model_rf has 'feature_names_in_' attribute

    # Make predictions
    prediction = model_rf.predict(X_user)[0]

    # Return prediction
    return prediction


# Main Streamlit application
def main():
    # Page title
    st.title('Anemia Prediction App')
    sex = st.selectbox('Gender', ['Male', 'Female'])
    red_pixel = st.number_input('%Red Pixel', min_value=0.0, max_value=100.0, step=0.1, format="%.4f")
    green_pixel = st.number_input('%Green Pixel', min_value=0.0, max_value=100.0, step=0.1, format="%.4f")
    blue_pixel = st.number_input('%Blue Pixel', min_value=0.0, max_value=100.0, step=0.1, format="%.4f")
    Hb = st.number_input('Hb', min_value=0.0, step=0.1, format="%.1f")

    # Predict anemia
    if st.button('Predict'):
        prediction = predict_anemia(sex, red_pixel, green_pixel, blue_pixel, Hb)
        if prediction == 1:
            st.error("Prediction: You are suffering from anemia.")
        else:
            st.success("Prediction: You are not suffering from anemia.")


if __name__ == '__main__':
    main()
