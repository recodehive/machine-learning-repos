import os
import pickle
import numpy as np
import streamlit as st

# Load model and pre-fitted scaler safely using relative path
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "deploy.pkl")

with open(model_path, "rb") as f:
    saved_artifacts = pickle.load(f)
    model1 = saved_artifacts['model']
    scaler = saved_artifacts['scaler']

def fun():
    features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    st.title("Cancer Prediction System")
    st.write("Enter patient cell nucleus attributes to predict tumor type:")
    
    user_input = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            user_input[feature] = st.number_input(f"{feature}:", value=0.0, format="%.4f")
            
    if st.button("Predict"):
        prediction_data = [user_input[feature] for feature in features]
        # Reshape to 2D array and transform using pre-fitted scaler
        user_data_2d = np.array(prediction_data).reshape(1, -1)
        user_data_scaled = scaler.transform(user_data_2d)
        
        prediction = model1.predict(user_data_scaled)[0]
        
        if prediction == 1:
            st.error("⚠️ Prediction Result: Malignant")
        else:
            st.success("✅ Prediction Result: Benign")

if __name__ == '__main__':
    fun()
