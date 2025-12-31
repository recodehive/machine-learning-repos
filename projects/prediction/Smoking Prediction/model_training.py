
#? STAGE 4: MODEL TRAINING

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Paths to datasets
dataset_paths = {
    'ml_olympiad': {
        'train': 'Y:/SmokingML V2/data/processed/ml_olympiad_train.csv',
        'test': 'Y:/SmokingML V2/data/processed/ml_olympiad_test.csv'
    },
    'archive': {
        'train': 'Y:/SmokingML V2/data/processed/archive_train.csv',
        'test': 'Y:/SmokingML V2/data/processed/archive_test.csv'
    }
}

# Directory to save models
os.makedirs("models", exist_ok=True)

# Best performing models for each dataset
models = {
    'ml_olympiad': {
        'name': 'XGBoost',
        'model': XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            enable_categorical=False,  # Modern replacement for use_label_encoder
            verbosity=1
        )
    },
    'archive': {
        'name': 'XGBoost',
        'model': XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            enable_categorical=False,  # Modern replacement for use_label_encoder
            verbosity=1
        )
    }
}

# Dictionary to store trained models and their metrics
model_info = {}

for dataset_name, paths in dataset_paths.items():
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Load datasets
    train_df = pd.read_csv(paths['train'])
    test_df = pd.read_csv(paths['test'])
    
    # Get all features except 'smoking' (target)
    features = [col for col in train_df.columns if col != 'smoking']
    
    # Split into features and target
    x_train = train_df[features]
    y_train = train_df['smoking']
    x_val = test_df[features]
    y_val = test_df['smoking']

    print(f"\n======= Dataset Info: {dataset_name.replace('_', ' ').title()} =======")
    print(f"Training data shape: {x_train.shape}")
    print(f"Number of features: {len(features)}")
    print("Features:", features)
    
    # Store model info
    model_info[dataset_name] = {
        'name': models[dataset_name]['name'],
        'features': features  # Store all features for this dataset
    }
    
    # Get and train the appropriate model
    model = models[dataset_name]['model']
    print(f"\nTraining {models[dataset_name]['name']} on {dataset_name} dataset...")
    model.fit(x_train, y_train)

    # Save model
    model_filename = f"models/{dataset_name}_{models[dataset_name]['name']}.pkl"
    joblib.dump(model, model_filename)
    print(f"Saved {models[dataset_name]['name']} model at {model_filename}")

    # Evaluate model
    y_pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\n{models[dataset_name]['name']} Accuracy on {dataset_name}: {accuracy:.4f}")

    # Store metrics in model_info
    model_info[dataset_name].update({
        'accuracy': accuracy,
        'model_path': model_filename
    })

    # Extra Evaluation Metrics
    print(f"\nConfusion Matrix ({models[dataset_name]['name']} - {dataset_name}):")
    print(confusion_matrix(y_val, y_pred))

    print(f"\nClassification Report ({models[dataset_name]['name']} - {dataset_name}):")
    print(classification_report(y_val, y_pred))

# Save model information for use in evaluation and API
model_info_path = "artifacts/models/model_info.json"
os.makedirs(os.path.dirname(model_info_path), exist_ok=True)
with open(model_info_path, 'w') as f:
    import json
    json.dump(model_info, f, indent=4)
