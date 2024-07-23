import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def load_dataset(selected_dataset):
    if selected_dataset == "Breast Cancer":
        data = load_breast_cancer()
    elif selected_dataset == "Iris":
        data = load_iris()
    elif selected_dataset == "Synthetic":
        data = make_classification(n_samples=10000, n_features=10, n_informative=8, n_redundant=2, random_state=42)
    else:
        raise ValueError("Invalid dataset selection.")
    return data

# Sidebar
st.sidebar.title("KNN Classifier Hyperparameter Tuning")
selected_dataset = st.sidebar.selectbox("Select Dataset", ["Breast Cancer", "Iris", "Synthetic"])
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, step=0.05)
n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 20, 5)
weights = st.sidebar.selectbox("Weight function", ["uniform", "distance"])
algorithm = st.sidebar.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
p_value = st.sidebar.slider("Minkowski Power (p)", 1, 10, 2)
leaf_size = st.sidebar.slider("Leaf Size", 1, 50, 30)

# Load selected dataset
data = load_dataset(selected_dataset)

if selected_dataset == "Synthetic":
    X, y = data
else:
    X = data.data
    y = data.target

# Display dataset information in the main window
st.title(f"Dataset Information for {selected_dataset}")
st.write(f"No. of Data Points: {X.shape[0]}")
st.write(f"No. of Features: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# KNN model
knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p_value, leaf_size=leaf_size)
knn_model.fit(X_train, y_train)

# Make predictions
y_pred = knn_model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display metrics
st.title("Model Evaluation Metrics")
st.write(f"Accuracy: {accuracy:.2f}")
st.write("Confusion Matrix:")
st.write(conf_matrix)
