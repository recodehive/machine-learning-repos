import time
import streamlit as st
import pandas as pd
import Classifier_Models.Classifier_model_builder_breast_cancer as cmb
import pickle
import numpy as np
from streamlit_toggle import st_toggle_switch

def app():
    st.title("Breast Cancer Detector")
    st.info("This app predicts whether a person have any breast cancer or not")
    st.markdown("""
    **Note** - :red[This Prediction Model is only applicable for Females.]
    """)

    st.sidebar.header('User Input Features')
    # st.sidebar.markdown("""
    # [Import input CSV file](https://github.com/ChakraDeep8/Heart-Disease-Detector/tree/master/res)""")

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def patient_details():
            radius_mean = st.sidebar.slider('Radius of Lobes', 6.98, 28.10, step=0.01)
            texture_mean = st.sidebar.slider('Mean of Surface Texture', 9.71, 39.30, step=0.01)
            perimeter_mean = st.sidebar.slider('Outer Perimeter of Lobes', 43.8, 189.0, step=0.1)
            area_mean = st.sidebar.slider('Mean Area of Lobes', 144, 2510)
            smoothness_mean = st.sidebar.slider('Mean of Smoothness Levels', 0.05, 0.16, step=0.01)
            compactness_mean = st.sidebar.slider('Mean of Compactness', 0.02, 0.35, step=0.01)
            concavity_mean = st.sidebar.slider('Mean of Concavity', 0.00, 0.43, step=0.01)
            concave_points_mean = st.sidebar.slider('Mean of Cocave Points', 0.00, 0.20, step=0.01)
            symmetry_mean = st.sidebar.slider('Mean of Symmetry', 0.11, 0.30, step=0.01)
            fractal_dimension_mean = st.sidebar.slider('Mean of Fractal Dimension', 0.05, 0.10, step=0.01)
            radius_se = st.sidebar.slider('SE of Radius', 0.11, 2.87, step=0.01)
            texture_se = st.sidebar.slider('SE of Texture', 0.36, 4.88, step=0.01)
            perimeter_se = st.sidebar.slider('Perimeter of SE', 0.76, 22.00, step=0.01)
            area_se = st.sidebar.slider('Area of SE', 6.8, 542.0, step=0.1)
            smoothness_se = st.sidebar.slider('SE of Smoothness', 0.00, 0.03, step=0.01)
            compactness_se = st.sidebar.slider('SE of compactness', 0.00, 0.14, step=0.01)
            concavity_se = st.sidebar.slider('SE of concavity', 0.00, 0.40, step=0.01)
            concave_points_se = st.sidebar.slider('SE of concave points', 0.00, 0.05, step=0.01)
            symmetry_se = st.sidebar.slider('SE of symmetry', 0.01, 0.08, step=0.01)
            fractal_dimension_se = st.sidebar.slider('SE of Fractal Dimension', 0.00, 0.03, step=0.01)
            radius_worst = st.sidebar.slider('Worst Radius', 7.93, 36.00, step=0.01)
            texture_worst = st.sidebar.slider('Worst Texture', 12.0, 49.5, step=0.1)
            perimeter_worst = st.sidebar.slider('Worst Permimeter', 50.40, 251.20, step=0.01)
            area_worst = st.sidebar.slider('Worst Area', 185.20, 4250.00, step=0.01)
            smoothness_worst = st.sidebar.slider('Worst Smoothness', 0.07, 0.22, step=0.01)
            compactness_worst = st.sidebar.slider('Worst Compactness', 0.03, 1.06, step=0.01)
            concavity_worst = st.sidebar.slider('Worst Concavity', 0.00, 1.25, step=0.01)
            concave_points_worst= st.sidebar.slider('Worst Concave Points', 0.00, 0.29, step=0.01)
            symmetry_worst = st.sidebar.slider('Worst Symmetry', 0.16, 0.66, step=0.01)
            fractal_dimension_worst = st.sidebar.slider('Worst Fractal Dimension', 0.06, 0.21, step=0.01)

            data = {'radius_mean': radius_mean,
                    'texture_mean': texture_mean,
                    'perimeter_mean': perimeter_mean,
                    'area_mean': area_mean,
                    'smoothness_mean': smoothness_mean,
                    'compactness_mean': compactness_mean,
                    'concavity_mean': concavity_mean,
                    'concave points_mean': concave_points_mean,
                    'symmetry_mean': symmetry_mean,
                    'fractal_dimension_mean': fractal_dimension_mean,
                    'radius_se': radius_se,
                    'texture_se': texture_se,
                    'perimeter_se': perimeter_se,
                    'area_se': area_se,
                    'smoothness_se': smoothness_se,
                    'compactness_se': compactness_se,
                    'concavity_se': concavity_se,
                    'concave points_se': concave_points_se,
                    'symmetry_se': symmetry_se,
                    'fractal_dimension_se': fractal_dimension_se,
                    'radius_worst': radius_worst,
                    'texture_worst': texture_worst,
                    'perimeter_worst': perimeter_worst,
                    'area_worst': area_worst,
                    'smoothness_worst': smoothness_worst,
                    'compactness_worst': compactness_worst,
                    'concavity_worst': concavity_worst,
                    'concave points_worst': concave_points_worst,
                    'symmetry_worst': symmetry_worst,
                    'fractal_dimension_worst': fractal_dimension_worst, }

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = patient_details()
    heart = cmb.X
    df = pd.concat([input_df, heart], axis=0)
    df = df[:1]  # Selects only the first row (the user input data)
    df.loc[:, ~df.columns.duplicated()]

    if uploaded_file is not None:
        st.write(df)

    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        df = df.loc[:, ~df.columns.duplicated()]
        st.write(df)

    # Load the classification models
    load_clf_NB = pickle.load(open('res/pickle/breast-cancer_disease_classifier_NB.pkl', 'rb'))
    load_clf_KNN = pickle.load(open('res/pickle/breast-cancer_disease_classifier_KNN.pkl', 'rb'))
    load_clf_DT = pickle.load(open('res/pickle/breast-cancer_disease_classifier_DT.pkl', 'rb'))
    load_clf_LR = pickle.load(open('res/pickle/breast-cancer_disease_classifier_LR.pkl', 'rb'))
    load_clf_RF = pickle.load(open('res/pickle/breast-cancer_disease_classifier_RF.pkl', 'rb'))
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
            st.write("<p style='font-size:20px; color: orange'><b>You have Malignant Tumors.</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Malignant tumors are cancerous and have the potential to spread and invade nearby tissues or other parts of the body.`
                        """)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You have Benign Tumors.</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Benign tumors are non-cancerous and usually do not invade nearby tissues or spread to other parts of the body.`
                        """)
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
            st.write("<p style='font-size:20px; color: orange'><b>You have Malignant Tumors.</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Malignant tumors are cancerous and have the potential to spread and invade nearby tissues or other parts of the body.`
                        """)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You have Benign Tumors.</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Benign tumors are non-cancerous and usually do not invade nearby tissues or spread to other parts of the body.`
                        """)
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
            st.write("<p style='font-size:20px; color: orange'><b>You have Malignant Tumors.</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Malignant tumors are cancerous and have the potential to spread and invade nearby tissues or other parts of the body.`
                        """)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You have Benign Tumors.</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Benign tumors are non-cancerous and usually do not invade nearby tissues or spread to other parts of the body.`
                        """)
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
            st.write("<p style='font-size:20px; color: orange'><b>You have Malignant Tumors.</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Malignant tumors are cancerous and have the potential to spread and invade nearby tissues or other parts of the body.`
                        """)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You have Benign Tumors.</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Benign tumors are non-cancerous and usually do not invade nearby tissues or spread to other parts of the body.`
                        """)
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
            st.write("<p style='font-size:20px; color: orange'><b>You have Malignant Tumors.</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Malignant tumors are cancerous and have the potential to spread and invade nearby tissues or other parts of the body.`
                        """)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You have Benign Tumors.</b></p>",
                     unsafe_allow_html=True)
            st.markdown("""
                        ##### `Benign tumors are non-cancerous and usually do not invade nearby tissues or spread to other parts of the body.`
                        """)
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
            st.write("<p style='font-size:20px;color: green'><b>You have Benign Tumors.</b></p>",
                    unsafe_allow_html=True)
            st.markdown("""
                        ##### `Benign tumors are non-cancerous and usually do not invade nearby tissues or spread to other parts of the body.`
                        """)

    st.markdown("👈 Provide your input data in the sidebar")
    # Displays the user input features
    with st.expander("Prediction Results", expanded=False):
        # Display the input dataframe
        st.write("Your input values are shown below:")
        st.dataframe(input_df)
        # Call the predict_best_algorithm() function
        st.text('Here, The best algorithm is selected among all algorithm', help='It is based on classifier report')
        predict_best_algorithm()

        # Tips, Diagnosis, Treatment, and Recommendations.
        st.subheader("👨‍⚕️ Expert Insights on Disease")
        tab1, tab2, tab3 = st.tabs(["Tips", "Diagnosis", "Treatment"])
        with tab1:
            st.subheader("Breast Cancer Prevention Tips:")
            prevention_tips = [
                "Maintain a healthy weight.",
                "Exercise regularly.",
                "Limit saturated fat intake.",
                "Avoid alcohol consumption.",
                "Consider breastfeeding, as it may reduce the risk of breast cancer."
            ]
            for tip in prevention_tips:
                st.write(f"- {tip}")
        with tab2:
            st.subheader("Diagnosis Methods:")
            diagnosis_methods = [
                "Mammogram and breast ultrasound.",
                "Biopsy (needle aspiration, needle biopsy, vacuum-assisted biopsy)."
            ]
            for method in diagnosis_methods:
                st.write(f"- {method}")
        with tab3:
            st.subheader("Treatment Options:")
            treatment_options = [
                "Surgery (breast-conserving surgery, mastectomy).",
                "Radiotherapy.",
                "Chemotherapy.",
                "Hormone therapy.",
                "Targeted therapy."
            ]
            for option in treatment_options:
                st.write(f"- {option}")

            st.subheader("Note:")
            st.write("Treatment decisions are based on factors like cancer stage, grade, overall health, and menopause status.")
            st.write("You can discuss your treatment options with your healthcare team and ask questions at any time.")


    # Create a multiselect for all the plot options
    selected_plots = st.multiselect("You can see all plots here👇",
                                    ["Naive Bayes", "K-Nearest Neighbors", "Decision Tree", "Logistic Regression",
                                     "Random Forest"], default=[],key="ms_B")
    if "ms_B" not in st.session_state:
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