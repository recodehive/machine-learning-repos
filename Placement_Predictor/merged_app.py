import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
data1 = pd.read_csv('Placement_Data_Full_Class.csv')
data2 = pd.read_csv('placement-dataset.csv')

# Preprocess data1
data1 = data1.drop(['Sno'], axis=1)
data1 = pd.get_dummies(data1, drop_first=True)

# Preprocess data2
data2 = data2.rename(columns={'Unnamed: 0': 'Index'})
data2 = data2.drop(['Index'], axis=1)

# Concatenate datasets horizontally
combined_data = pd.concat([data1.reset_index(drop=True), data2.reset_index(drop=True)], axis=1)

# Define features and target
X = combined_data.drop(['status_Placed', 'salary'], axis=1)
y = combined_data['status_Placed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'placement_model.pkl')

# Streamlit app
st.title('Placement Prediction App')

# Sidebar for bar chart selection
st.sidebar.title("Bar Chart Options")
chart_type = st.sidebar.selectbox(
    'Select a bar chart',
    [
        'Gender vs Sno', 'SSC Board vs Sno', 'SSC Board vs Sno (hue=Gender)',
        'HSC Board vs Sno', 'HSC Board vs Sno (hue=Gender)', '12th Stream vs Sno',
        '12th Stream vs Sno (hue=Gender)', 'Degree stream vs Sno', 'Degree stream vs Sno (hue=Gender)',
        'Work exp vs Sno', 'Work exp vs Sno (hue=Gender)', 'Specialisation vs Sno',
        'Status vs Sno', 'Gender vs 10th % (hue=Work exp)', 'SSC Board vs 12th % (hue=specialisation)',
        'HSC Board vs Degree % (hue=status)', '12th Stream vs Mba % (hue=SSC Board)', 'Degree stream vs Salary (hue=Gender)'
    ]
)

# Load the dataset for bar charts
df = pd.read_csv('Placement_Data_Full_Class.csv')

# Function to create and display the selected bar chart
def show_bar_chart(chart_type):
    if chart_type == 'Gender vs Sno':
        sns.barplot(x='Gender', y='Sno', data=df, palette='Accent')
    elif chart_type == 'SSC Board vs Sno':
        sns.barplot(x='SSC Board', y='Sno', data=df, palette='Greys')
    elif chart_type == 'SSC Board vs Sno (hue=Gender)':
        sns.barplot(x='SSC Board', y='Sno', data=df, hue='Gender')
    elif chart_type == 'HSC Board vs Sno':
        sns.barplot(x='HSC Board', y='Sno', data=df, palette='terrain')
    elif chart_type == 'HSC Board vs Sno (hue=Gender)':
        sns.barplot(x='HSC Board', y='Sno', data=df, hue='Gender')
    elif chart_type == '12th Stream vs Sno':
        sns.barplot(x='12th Stream', y='Sno', data=df, palette='icefire')
    elif chart_type == '12th Stream vs Sno (hue=Gender)':
        sns.barplot(x='12th Stream', y='Sno', data=df, hue='Gender')
    elif chart_type == 'Degree stream vs Sno':
        sns.barplot(x='Degree stream', y='Sno', data=df, palette='terrain')
    elif chart_type == 'Degree stream vs Sno (hue=Gender)':
        sns.barplot(x='Degree stream', y='Sno', data=df, hue='Gender')
    elif chart_type == 'Work exp vs Sno':
        sns.barplot(x='Work exp', y='Sno', data=df, palette='BuGn')
    elif chart_type == 'Work exp vs Sno (hue=Gender)':
        sns.barplot(x='Work exp', y='Sno', data=df, hue='Gender')
    elif chart_type == 'Specialisation vs Sno':
        sns.barplot(x='specialisation', y='Sno', data=df, palette='gist_heat')
    elif chart_type == 'Status vs Sno':
        sns.barplot(x='status', y='Sno', data=df, palette='Blues_d')
    elif chart_type == 'Gender vs 10th % (hue=Work exp)':
        sns.barplot(x='Gender', y='10th %', data=df, hue='Work exp', palette='icefire')
    elif chart_type == 'SSC Board vs 12th % (hue=specialisation)':
        sns.barplot(x='SSC Board', y='12th %', data=df, hue='specialisation', palette='magma')
    elif chart_type == 'HSC Board vs Degree % (hue=status)':
        sns.barplot(x='HSC Board', y='Degree %', data=df, hue='status', palette='Accent')
    elif chart_type == '12th Stream vs Mba % (hue=SSC Board)':
        sns.barplot(x='12th Stream', y='Mba %', data=df, hue='SSC Board', palette='viridis')
    elif chart_type == 'Degree stream vs Salary (hue=Gender)':
        sns.barplot(x='Degree stream', y='salary', data=df, hue='Gender', palette='terrain')

    st.pyplot()

# Display the selected bar chart
show_bar_chart(chart_type)

# User inputs for placement prediction
st.header('Predict Placement')
gender = st.selectbox('Gender', ['M', 'F'])
ssc_p = st.number_input('10th Percentage')
hsc_p = st.number_input('12th Percentage')
degree_p = st.number_input('Degree Percentage')
work_exp = st.selectbox('Work Experience', ['Yes', 'No'])
specialisation = st.selectbox('Specialisation', ['Mkt&HR', 'Mkt&Fin'])
mba_p = st.number_input('MBA Percentage')
cgpa = st.number_input('CGPA')
iq = st.number_input('IQ')

# Convert inputs to dataframe
input_data = pd.DataFrame({
    'Gender': [gender],
    '10th %': [ssc_p],
    '12th %': [hsc_p],
    'Degree %': [degree_p],
    'Work exp': [work_exp],
    'specialisation': [specialisation],
    'Mba %': [mba_p],
    'cgpa': [cgpa],
    'iq': [iq]
})

input_data = pd.get_dummies(input_data, drop_first=True)

# Load model and make prediction
model = joblib.load('placement_model.pkl')
prediction = model.predict(input_data)[0]

# Display result
if prediction == 1:
    st.success('The student is likely to be placed.')
else:
    st.error('The student is not likely to be placed.')
