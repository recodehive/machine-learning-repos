import time
import streamlit as st
import pandas as pd
from Classifier_Models import Classifier_model_builder_kidney as cmb
from streamlit_toggle import st_toggle_switch
import pickle
import numpy as np
import json
from streamlit_lottie import st_lottie

def app():
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

    lottie_coding = load_lottiefile("res/Yoga_Bhujangasana.json")
    st.title("Kidney Disease Detector")
    st.info("This app predicts whether a person have any kidney disease or not")

    st.sidebar.header('User Input Features')
    # st.sidebar.markdown("""
    # [Import input CSV file](https://github.com/ChakraDeep8/Heart-Disease-Detector/tree/master/res)""")

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def patient_details():
            age = st.sidebar.slider('Age', 2, 90)
            bp = st.sidebar.slider('Blood Pressure', 50, 180)
            sg = st.sidebar.slider('Salmonella Gallinarum - SG', 1.00, 1.02, step=0.01)
            al = st.sidebar.slider('Albumin', 0, 5)
            su = st.sidebar.slider('Sugar - SU', 0, 5)
            bgr = st.sidebar.slider('Blood Glucose Regulator - BGR', 22, 490)
            bu = st.sidebar.slider('Blood Urea - BU', 2, 90)
            sc = st.sidebar.slider('Serum Creatinine - SC', 1.5, 391.0, step=0.1)
            sod = st.sidebar.slider('Sodium', 45, 163)
            pot = st.sidebar.slider('Potassium', 2.5, 47.0, step=0.1)
            hemo = st.sidebar.slider('Hemoglobin', 3.1, 17.8, step=0.1)
            pcv = st.sidebar.slider('Packed Cell Volume - PCV', 9, 54)
            wc = st.sidebar.slider('White Blood Cell Count - WC', 3800, 26400)
            rc = st.sidebar.slider('Red Blood Cell Count - RC', 4, 80)
            rbc = st.sidebar.selectbox('Red Blood Corpulence', ('normal', 'abnormal'))
            pc = st.sidebar.selectbox('Post Cibum - PC', ('normal', 'abnormal'))
            pcc = st.sidebar.selectbox('Prothrombin Complex Concentrates - PCC', ('present', 'notpresent'))
            ba = st.sidebar.selectbox('Bronchial Asthma - BA', ('present', 'notpresent'))
            htn = st.sidebar.selectbox('Hypertension - HTN', ('yes', 'no'))
            dm = st.sidebar.selectbox('Diabetes Mellitus', ('yes', 'no'))
            cad = st.sidebar.selectbox('Coronary Artery Disease - CAD', ('yes', 'no'))
            appet = st.sidebar.selectbox('Appetite', ('poor', 'good'))
            pe = st.sidebar.selectbox('Pulmonary Embolism - PE', ('yes', 'no'))
            ane = st.sidebar.selectbox('Acute Necrotizing Encephalopathy - ANE', ('yes', 'no'))

            data = {'age': age,
                    'rbc': rbc,
                    'pc': pc,
                    'pcc': pcc,
                    'bp': bp,
                    'ba': ba,
                    'htn': htn,
                    'dm': dm,
                    'cad': cad,
                    'appet': appet,
                    'pe': pe,
                    'ane': ane,
                    'sg': sg,
                    'al': al,
                    'su': su,
                    'bgr': bgr,
                    'bu': bu,
                    'sc': sc,
                    'sod': sod,
                    'pot': pot,
                    'hemo': hemo,
                    'pcv': pcv,
                    'wc': wc,
                    'rc': rc,
                    }

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = patient_details()

    url = "res/dataset/kidney.csv"
    kidney_disease_raw = pd.read_csv(url)
    kidney_disease_raw = kidney_disease_raw.loc[:, ~kidney_disease_raw.columns.str.contains('^Unnamed')]
    kidney = kidney_disease_raw.drop(columns=['target'])
    df = pd.concat([input_df, kidney], axis=0)

    # Encoding of ordinal features
    encode = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
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
    load_clf_NB = pickle.load(open('res/pickle/kidney_disease_classifier_NB.pkl', 'rb'))
    load_clf_KNN = pickle.load(open('res/pickle/kidney_disease_classifier_KNN.pkl', 'rb'))
    load_clf_DT = pickle.load(open('res/pickle/kidney_disease_classifier_DT.pkl', 'rb'))
    load_clf_LR = pickle.load(open('res/pickle/kidney_disease_classifier_LR.pkl', 'rb'))
    load_clf_RF = pickle.load(open('res/pickle/kidney_disease_classifier_RF.pkl', 'rb'))

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
            st.write("<p style='font-size:20px;color: orange'><b>You have kidney disease.</b></p>",
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
            st.write("<p style='font-size:20px;color: orange'><b>You have kidney disease.</b></p>",
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
            st.write("<p style='font-size:20px; color: orange'><b>You have kidney disease.</b></p>",
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
            st.write("<p style='font-size:20px; color: orange'><b>You have kidney disease.<b></p>",
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
            st.write("<p style='font-size:20px; color: orange'><b>You have kidney disease.</b></p>",
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
    with st.expander("Prediction Results"):
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
            st.subheader("Tips for Managing Kidney Disease:")
            management_tips = [
                "Lose weight if overweight.",
                "Stay physically active to control blood sugar levels.",
                "Quit smoking.",
                "Get regular checkups and include kidney health checks.",
                "Take medications as prescribed.",
                "Maintain blood pressure below 140/90 or follow your doctor's advice.",
                "Control blood sugar levels if you have diabetes.",
                "Manage cholesterol levels within your target range.",
                "Consume less salt and opt for homemade, low-sodium foods.",
                "Limit potassium, phosphorus, and protein intake."
            ]
            for tip in management_tips:
                st.write(f"- {tip}")
        with tab2:
            st.subheader("Kidney-Friendly Exercises:")
            c1, c2, c3 = st.columns([1, 1, 1], gap="small")
            with c1:
                kidney_exercises = [
                    "Walking",
                    "Swimming",
                    "Bicycling (indoors or out)",
                    "Skiing",
                    "Aerobic dancing"
                ]
                for exercise in kidney_exercises:
                    st.write(f"- {exercise}")
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
            st.subheader("Kidney-Friendly Diet:")
            kidney_diet = [
                "Choose low-sodium, fresh, homemade foods.",
                "Reduce phosphorus and potassium by avoiding certain foods (e.g., meats, dairy, beans for phosphorus; oranges, potatoes, tomatoes for potassium).",
                "Consume the right amount of protein to ease kidney workload.",
                "Include fruits, vegetables, lean meats, eggs, and unsalted seafood.",
                "Opt for low-phosphorus and low-potassium carb sources (e.g., white bread, pasta).",
                "Drink water, clear diet sodas, and unsweetened tea."
            ]
            for diet in kidney_diet:
                st.write(f"- {diet}")

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
