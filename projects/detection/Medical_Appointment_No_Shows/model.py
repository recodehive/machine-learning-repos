"""Medical Appointment No-Shows Prediction Model
This module implements a machine learning model to predict whether a patient
will miss their medical appointment using preprocessing pipeline and model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class AppointmentPreprocessor:
    """Preprocessing pipeline for medical appointment data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def preprocess(self, data):
        """Preprocess medical appointment data.
        
        Args:
            data (pd.DataFrame): Raw appointment data
            
        Returns:
            pd.DataFrame: Preprocessed data ready for model training
        """
        # Handle missing values
        data = data.fillna(data.mean(numeric_only=True))
        
        # Encode categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
        
        # Scale numerical features
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        
        return data


class AppointmentNoShowModel:
    """Machine Learning model for predicting appointment no-shows."""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.preprocessor = AppointmentPreprocessor()
    
    def train(self, X_train, y_train):
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            predictions: Predicted labels
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Performance metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1': f1_score(y_test, predictions, average='weighted')
        }
        
        return metrics


if __name__ == '__main__':
    # Example usage
    print('Medical Appointment No-Shows Prediction Model initialized.')
