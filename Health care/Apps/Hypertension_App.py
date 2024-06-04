import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_toggle import st_toggle_switch
from Classifier_Models import Classifier_model_builder_hypertension as cmb
import json
from streamlit_lottie import st_lottie

def app():
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

    lottie_coding = load_lottiefile("res/Yoga_Padmasana.json")

    st.title("Hypertension Blood Pressure Detector")
    st.info("This app predicts whether a person have any hypertension blood pressure or not")

    st.sidebar.header('User Input Features')
    # st.sidebar.markdown("""
    # [Import input CSV file](https://github.com/ChakraDeep8/Heart-Disease-Detector/tree/master/res)""")

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def patient_details():
            age = st.sidebar.slider('Age', 0, 98)
            sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
            chest_pain_type = st.sidebar.selectbox('Chest Pain Type',
                                                   ['Asymptomatic', 'Typical Angina', 'Atypical Angina', 'Non-anginal'])
            resting_bp = st.sidebar.slider('Resting Blood Pressure', 94, 200)
            serum_cholesterol = st.sidebar.slider('Serum Cholesterol', 126, 564)
            fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar',
                                              ['Yes', 'No'])  # if the patient's fasting blood sugar > 120 mg/dl
            resting_ecg = st.sidebar.selectbox('Resting ECG',
                                               ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
            max_hr = st.sidebar.slider('Max Heart Rate', 71, 202)
            exercise_angina = st.sidebar.selectbox('Exercise-Induced Angina', ['Yes', 'No'])
            oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2)
            st_slope = st.sidebar.selectbox('ST Slope', ['Upsloping', 'Flat', 'Downsloping'])
            major_vessels = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4)
            thalassemia = st.sidebar.slider('Thalassemia', 0, 3)

            data = {'age': age,
                    'sex': sex,
                    'cp': chest_pain_type,
                    'trestbps': resting_bp,
                    'chol': serum_cholesterol,
                    'fbs': fasting_bs,
                    'restecg': resting_ecg,
                    'thalach': max_hr,
                    'exang': exercise_angina,
                    'oldpeak': oldpeak,
                    'slope': st_slope,
                    'ca': major_vessels,
                    'thal': thalassemia, }

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = patient_details()

    hypertension_disease_raw = pd.read_csv('res/dataset/hypertension_data.csv')
    hypertension = hypertension_disease_raw.drop(columns=['target'])
    df = pd.concat([input_df, hypertension], axis=0)

    # Encoding of ordinal features
    encode = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]
    df = df[:1]  # Selects only the first row (the user input data)
    df.loc[:, ~df.columns.duplicated()]

    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        df = df.loc[:, ~df.columns.duplicated()]
        st.write(df)

    # Load the classification models
    load_clf_NB = pickle.load(open('res/pickle/hypertension_disease_classifier_NB.pkl', 'rb'))
    load_clf_KNN = pickle.load(open('res/pickle/hypertension_disease_classifier_KNN.pkl', 'rb'))
    load_clf_DT = pickle.load(open('res/pickle/hypertension_disease_classifier_DT.pkl', 'rb'))
    load_clf_LR = pickle.load(open('res/pickle/hypertension_disease_classifier_LR.pkl', 'rb'))
    load_clf_RF = pickle.load(open('res/pickle/hypertension_disease_classifier_RF.pkl', 'rb'))

    # Apply models to make predictions
    prediction_NB = load_clf_NB.predict(df)
    prediction_proba_NB = load_clf_NB.predict_proba(df)
    prediction_KNN = load_clf_KNN.predict(df)
    prediction_proba_KNN = load_clf_KNN.predict_proba(df)
    prediction_DT = load_clf_DT.predict(df)
    prediction_proba_DT = load_clf_DT.predict_proba(df)
    prediction_LR = load_clf_LR.predict(df)
    prediction_proba_LR = load_clf_LR.predict_proba(df)
    prediction_RF = load_clf_RF.predict(df)
    prediction_proba_RF = load_clf_RF.predict_proba(df)

    def NB():
        st.subheader('Naive Bayes Prediction')
        NB_prediction = np.array([0, 1])
        if NB_prediction[prediction_NB] == 1:
            st.write("<p style='font-size:20px;color: orange'><b>You have hypertension.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine 👍</b></p>", unsafe_allow_html=True)
        enabled = st_toggle_switch("See detailed prediction")
        if enabled:
            st.subheader('Naive Bayes Prediction Probability')
            st.write(prediction_proba_NB)
            col1, col2 = st.columns(2)
            with col1:
                st.text('Why Classifier Report',
                        help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
            with col2:
                st.text('How to read',
                        help="By looking at the cells where the true and predicted labels intersect, you can see the counts of correct and incorrect predictions. This helps evaluate the model's performance in distinguishing between 'No Disease' and 'Disease' categories.")

            cmb.plt_NB()

    def KNN():
        st.subheader('K-Nearest Neighbour Prediction')
        knn_prediction = np.array([0, 1])
        if knn_prediction[prediction_KNN] == 1:
            st.write("<p style='font-size:20px;color: orange'><b>You have hypertension.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine 👍</b></p>", unsafe_allow_html=True)
        enabled = st_toggle_switch("See detailed prediction")
        if enabled:
            st.subheader('KNN Prediction Probability')
            st.write(prediction_proba_KNN)
            col1, col2 = st.columns(2)
            with col1:
                st.text('Why Classifier Report',
                        help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
            with col2:
                st.text('How to read',
                        help="By looking at the cells where the true and predicted labels intersect, you can see the counts of correct and incorrect predictions. This helps evaluate the model's performance in distinguishing between 'No Disease' and 'Disease' categories.")

            cmb.plt_KNN()

    def DT():
        st.subheader('Decision Tree Prediction')
        DT_prediction = np.array([0, 1])
        if DT_prediction[prediction_DT] == 1:
            st.write("<p style='font-size:20px; color: orange'><b>You have hypertension.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine 👍</b></p>", unsafe_allow_html=True)
        enabled = st_toggle_switch("See detailed prediction")
        if enabled:
            st.subheader('Decision Tree Prediction Probability')
            st.write(prediction_proba_DT)
            col1, col2 = st.columns(2)
            with col1:
                st.text('Why Classifier Report',
                        help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
            with col2:
                st.text('How to read',
                        help="By looking at the cells where the true and predicted labels intersect, you can see the counts of correct and incorrect predictions. This helps evaluate the model's performance in distinguishing between 'No Disease' and 'Disease' categories.")

            cmb.plt_DT()

    def LR():
        st.subheader('Logistic Regression Prediction')
        LR_prediction = np.array([0, 1])
        if LR_prediction[prediction_LR] == 1:
            st.write("<p style='font-size:20px; color: orange'><b>You have hypertension.<b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine 👍</b></p>", unsafe_allow_html=True)
        enabled = st_toggle_switch("See detailed prediction")
        if enabled:
            st.subheader('Logistic Regression Probability')
            st.write(prediction_proba_LR)
            col1, col2 = st.columns(2)
            with col1:
                st.text('Why Classifier Report',
                        help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
            with col2:
                st.text('How to read',
                        help="By looking at the cells where the true and predicted labels intersect, you can see the counts of correct and incorrect predictions. This helps evaluate the model's performance in distinguishing between 'No Disease' and 'Disease' categories.")

            cmb.plt_LR()

    def RF():
        st.subheader('Random Forest Prediction')
        RF_prediction = np.array([0, 1])
        if RF_prediction[prediction_RF] == 1:
            st.write("<p style='font-size:20px; color: orange'><b>You have hypertension.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine 👍</b></p>", unsafe_allow_html=True)
        enabled = st_toggle_switch("See detailed prediction")
        if enabled:
            st.subheader('Random Forest Probability')
            st.write(prediction_proba_RF)
            col1, col2 = st.columns(2)
            with col1:
                st.text('Why Classifier Report',
                        help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
            with col2:
                st.text('How to read',
                        help="By looking at the cells where the true and predicted labels intersect, you can see the counts of correct and incorrect predictions. This helps evaluate the model's performance in distinguishing between 'No Disease' and 'Disease' categories.")
            cmb.plt_RF()

    def predict_best_algorithm():
        if cmb.best_model == 'Naive Bayes':
            NB()
        elif cmb.best_model == 'K-Nearest Neighbors (KNN)':
            KNN()
        elif cmb.best_model == 'Decision Tree':
            DT()
        elif cmb.best_model == 'Logistic Regression':
            LR()
        elif cmb.best_model == 'Random Forest':
            RF()
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine 👍</b></p>", unsafe_allow_html=True)

    st.markdown("👈 Provide your input data in the sidebar")
    # Displays the user input features
    with st.expander("Prediction Results",expanded=False):
        # Display the input dataframe
        st.write("Your input values are shown below:")
        st.dataframe(input_df)
        # Call the predict_best_algorithm() function
        st.text('Here, The best algorithm is selected among all algorithm', help='It is based on classifier report')
        predict_best_algorithm()

        # Tips, Diagnosis, Treatment, and Recommendations.
        st.subheader("👨‍⚕️ Expert Insights on Disease")
        tab1, tab2, tab3 = st.tabs(["Tips", "Exercises", "Diet"])
        with tab1:
            st.subheader("Tips for Managing Hypertension:")
            management_tips = [
                "Don't skip breakfast.",
                "Cut down on caffeine.",
                "Stop smoking.",
                "Avoid being overweight, which forces the heart to work harder to pump blood around your body.",
                "Men and women are advised not to drink regularly."
            ]
            for tip in management_tips:
                st.write(f"- {tip}")
        with tab2:
            st.subheader("Exercises Recommendations:")
            c1, c2, c3 = st.columns([1, 1, 1], gap="small")
            with c1:
                exercise_recommendation = [
                    "Exercises for Lowering Blood Pressure:",
                    "Taking regular exercise lowers blood pressure. Adults should do at least 150 minutes (2 hours and 30 minutes) of moderate-intensity aerobic activity, such as cycling or fast walking every week."
                ]
                for tip in exercise_recommendation:
                    st.write(f"- {tip}")
            with c3:
                st_lottie(
                    lottie_coding,
                    speed=1,
                    reverse=False,
                    loop=True,
                    quality="medium",
                    height=None,
                    width=None,
                    key=None,
                )
        with tab3:
            st.subheader("Dietary Recommendations:")
            dietary_recommendations = [
                "Eat plenty of fruits and vegetables.",
                "Eat more fish, including a portion of oily fish.",
                "Cut down on saturated fat and sugar.",
                "Eat less salt (no more than 6g a day for adults)."
            ]
            for tip in dietary_recommendations:
                st.write(f"- {tip}")

    # Create a multiselect for all the plot options
    selected_plots = st.multiselect("You can see all plots here👇",
                                    ["Naive Bayes", "K-Nearest Neighbors", "Decision Tree", "Logistic Regression",
                                     "Random Forest"], default=[], key="ms_hy")
    if "ms_hy" not in st.session_state:
        st.session_state.selected_plots = []
    # Check the selected plots and call the corresponding plot functions
    if selected_plots:
        col1, col2 = st.columns(2)
        with col1:
            st.text('Why Classifier Report',
                    help="It helps assess the model's ability to correctly identify classes and its overall performance in classifying data.")
        with col2:
            st.text('How to read',
                    help="By looking at the cells where the true and predicted labels intersect, you can see the counts of correct and incorrect predictions. This helps evaluate the model's performance in distinguishing between 'No Disease' and 'Disease' categories.")

    placeholder = st.empty()

    # Check the selected plots and call the corresponding plot functions
    if "Naive Bayes" in selected_plots:
        with st.spinner("Generating Naive Bayes...."):
            cmb.plt_NB()
            time.sleep(1)

    if "K-Nearest Neighbors" in selected_plots:
        with st.spinner("Generating KNN...."):
            cmb.plt_KNN()
            time.sleep(1)

    if "Decision Tree" in selected_plots:
        with st.spinner("Generating Decision Tree...."):
            cmb.plt_DT()
            time.sleep(1)

    if "Logistic Regression" in selected_plots:
        with st.spinner("Generating Logistic Regression...."):
            cmb.plt_LR()
            time.sleep(1)

    if "Random Forest" in selected_plots:
        with st.spinner("Generating Random Forest...."):
            cmb.plt_RF()
            time.sleep(1)

    # Remove the placeholder to display the list options
    placeholder.empty()
