import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load the dataset
df = pd.read_csv('datasets/Placement_Data_Full_Class.csv')

# Encode categorical variables
label_encoders = {}
for column in ['Gender', 'SSC Board', 'HSC Board', '12th Stream', 'Degree stream', 'Work exp', 'specialisation', 'status']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target variable
X = df.drop(columns=['Sno', 'status', 'salary'])
y = df['status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Streamlit App
st.title("Placement Prediction App")

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ['Home', 'Correlation Heatmap', 'Distributions', 'Barcharts', 'Scatter Plots', 'Predict Placement'])

if options == 'Home':
    st.header("Welcome to the Placement Prediction App")
    st.write("Identifying Patterns and Trends in Campus Placement Data using Machine Learning model and python libraries")
    st.write("Use the sidebar to navigate to different sections of the app.")

elif options == 'Correlation Heatmap':
    st.header("Correlation Heatmap")

    # Plot the heatmap
    plt.figure(figsize=(12, 10))  # Increase the figure size
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 8})
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.yticks(rotation=0)  # Rotate y-axis labels
    plt.title('Correlation Matrix')
    st.pyplot(plt)

elif options == 'Distributions':
    st.header("Distributions of Features")

    feature = st.selectbox("Select a feature to plot", df.columns.drop(['Sno', 'salary']))

    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    st.pyplot(plt)

elif options == 'Barcharts':
    st.header("Bar Charts")

    bar_chart_options = [
        'Gender vs Sno', 'SSC Board vs Sno', 'SSC Board vs Sno ',
        'HSC Board vs Sno', 'HSC Board vs Sno ', '12th Stream vs Sno',
        '12th Stream vs Sno ', 'Degree stream vs Sno', 'Degree stream vs Sno',
        'Work exp vs Sno', 'Work exp vs Sno', 'Specialisation vs Sno',
        'Status vs Sno', 'Gender vs 10th %', 'SSC Board vs 12th % ',
        'HSC Board vs Degree % ', '12th Stream vs Mba % ', 'Degree stream vs Salary'
    ]

    selected_chart = st.selectbox("Select a bar chart", bar_chart_options)

    plt.figure(figsize=(12, 8))

    if selected_chart == 'Gender vs Sno':
        sns.barplot(x='Gender', y='Sno', data=df, palette='Accent')
    elif selected_chart == 'SSC Board vs Sno':
        sns.barplot(x='SSC Board', y='Sno', data=df, palette='Greys')
    elif selected_chart == 'SSC Board vs Sno':
        sns.barplot(x='SSC Board', y='Sno', data=df, hue='Gender')
    elif selected_chart == 'HSC Board vs Sno':
        sns.barplot(x='HSC Board', y='Sno', data=df, palette='terrain')
    elif selected_chart == 'HSC Board vs Sno':
        sns.barplot(x='HSC Board', y='Sno', data=df, hue='Gender')
    elif selected_chart == '12th Stream vs Sno':
        sns.barplot(x='12th Stream', y='Sno', data=df, palette='icefire')
    elif selected_chart == '12th Stream vs Sno':
        sns.barplot(x='12th Stream', y='Sno', data=df, hue='Gender')
    elif selected_chart == 'Degree stream vs Sno':
        sns.barplot(x='Degree stream', y='Sno', data=df, palette='terrain')
    elif selected_chart == 'Degree stream vs Sno ':
        sns.barplot(x='Degree stream', y='Sno', data=df, hue='Gender')
    elif selected_chart == 'Work exp vs Sno':
        sns.barplot(x='Work exp', y='Sno', data=df, palette='BuGn')
    elif selected_chart == 'Work exp vs Sno ':
        sns.barplot(x='Work exp', y='Sno', data=df, hue='Gender')
    elif selected_chart == 'Specialisation vs Sno':
        sns.barplot(x='specialisation', y='Sno', data=df, palette='gist_heat')
    elif selected_chart == 'Status vs Sno':
        sns.barplot(x='status', y='Sno', data=df, palette='Blues_d')
    elif selected_chart == 'Gender vs 10th % ':
        sns.barplot(x='Gender', y='10th %', data=df, hue='Work exp', palette='icefire')
    elif selected_chart == 'SSC Board vs 12th %':
        sns.barplot(x='SSC Board', y='12th %', data=df, hue='specialisation', palette='magma')
    elif selected_chart == 'HSC Board vs Degree % ':
        sns.barplot(x='HSC Board', y='Degree %', data=df, hue='status', palette='Accent')
    elif selected_chart == '12th Stream vs Mba %':
        sns.barplot(x='12th Stream', y='Mba %', data=df, hue='SSC Board', palette='viridis')
    elif selected_chart == 'Degree stream vs Salary':
        sns.barplot(x='Degree stream', y='salary', data=df, hue='Gender', palette='terrain')

    st.pyplot(plt)

elif options == 'Scatter Plots':
    st.header("Scatter Plots")

    x_feature = st.selectbox("Select the X-axis feature", df.columns.drop(['Sno', 'salary']))
    y_feature = st.selectbox("Select the Y-axis feature", df.columns.drop(['Sno', 'salary']))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='status')
    plt.title(f'Scatter plot of {x_feature} vs {y_feature}')
    st.pyplot(plt)

elif options == 'Predict Placement':
    st.header("Predict Placement Status")

    # Input fields for user data
    gender = st.selectbox("Gender", ["M", "F"])
    ssc_board = st.selectbox("SSC Board", ["Others", "Central"])
    hsc_board = st.selectbox("HSC Board", ["Others", "Central"])
    hsc_stream = st.selectbox("12th Stream", ["Commerce", "Science", "Arts"])
    degree_stream = st.selectbox("Degree Stream", ["Sci&Tech", "Comm&Mgmt", "Others"])
    work_exp = st.selectbox("Work Experience", ["Yes", "No"])
    specialisation = st.selectbox("Specialisation", ["Mkt&HR", "Mkt&Fin"])

    tenth_percent = st.slider("10th %", 0.0, 100.0, 50.0)
    twelfth_percent = st.slider("12th %", 0.0, 100.0, 50.0)
    degree_percent = st.slider("Degree %", 0.0, 100.0, 50.0)
    mba_percent = st.slider("MBA %", 0.0, 100.0, 50.0)

    # Prepare the input data
    input_data = pd.DataFrame({
        'Gender': [gender],
        '10th %': [tenth_percent],
        'SSC Board': [ssc_board],
        '12th %': [twelfth_percent],
        'HSC Board': [hsc_board],
        '12th Stream': [hsc_stream],
        'Degree %': [degree_percent],
        'Degree stream': [degree_stream],
        'Work exp': [work_exp],
        'specialisation': [specialisation],
        'Mba %': [mba_percent]
    })

    # Encode the input data
    for column in ['Gender', 'SSC Board', 'HSC Board', '12th Stream', 'Degree stream', 'Work exp', 'specialisation']:
        input_data[column] = label_encoders[column].transform(input_data[column])

    # Predict the placement
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Display the prediction
    status = "Placed" if prediction == 1 else "Not Placed"
    st.write(f"The candidate is predicted to be: **{status}**")
    st.write(f"Prediction confidence: {prediction_proba.max() * 100:.2f}%")
