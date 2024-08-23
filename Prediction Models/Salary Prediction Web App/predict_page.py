import numpy as np 
import pickle
import streamlit as st

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data=pickle.load(file)
    return data

data = load_model()

regressor=data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    # Country selection
    st.write("""### We need some information to predict the salary""")
   
    countries=(
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
        "Israel"
    )

    education = (
        "Bachelor’s degree",
        "Master’s degree",
        "Less than a Bachelors",
        "Post grad"

    )

    country = st.selectbox("Country",countries)
    education = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience",0,15,3)

    
    if st.button("Calculate Salary"):
        #if we clicked on the button
        country_encoded = le_country.transform([country])[0]
        education_encoded = le_education.transform([education])[0]
        input_features = np.array([[country_encoded, education_encoded, experience]]).astype(float)

        # Predict salary
        estimated_salary = regressor.predict(input_features)

        st.subheader(f"The estimated salary is ${estimated_salary[0]:,.2f}")

if __name__ == "__main__":
    show_predict_page()
        # X=np.array([[country,education,experience]])
        # X[:, 0] = le_country.transform(X[:,0])
        # X[:, 1] = le_education.transform(X[:,1])
        # X = X.astype(float)

        # salary=regressor.predict(X)
        # st.subheader(f"The estimated salary is ${salary[0]:.2f}") 


