import os
import numpy as np
import pickle
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_steps.pkl")


def load_model():
    with open(MODEL_PATH, "rb") as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### We need some information to predict the salary""")

    countries = (
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
    )

    education = (
        "Bachelor’s degree",
        "Master’s degree",
        "Less than a Bachelors",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education_selected = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience", 0, 15, 3)

    if st.button("Calculate Salary"):
        country_encoded = le_country.transform([country])[0]
        education_encoded = le_education.transform([education_selected])[0]
        input_features = np.array([[country_encoded, education_encoded, experience]]).astype(float)

        estimated_salary = regressor.predict(input_features)
        st.subheader(f"The estimated salary is ${estimated_salary[0]:,.2f}")


if __name__ == "__main__":
    show_predict_page()
