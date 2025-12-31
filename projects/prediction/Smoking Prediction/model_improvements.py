
#? STAGE 7: MODEL IMPROVEMENTS

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before other matplotlib imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, precision_recall_curve, 
                           roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def custom_scorer(y_true, y_pred):
    """Custom scorer that emphasizes precision while maintaining other metrics"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # Weight precision more heavily
    return (2 * precision + recall + f1) / 4

def create_advanced_features(df):
    """Create advanced feature set with enhanced interactions"""
    # Original features except 'smoking'
    original_features = [col for col in df.columns if col != 'smoking']
    
    # Enhanced health indicators
    df['bmi'] = df['weight(kg)'] / ((df['height(cm)']/100) ** 2)
    df['liver_function'] = (df['AST'] + df['ALT'] + df['Gtp']) / 3
    df['cardiovascular_risk'] = (df['systolic'] * df['triglyceride']) / (df['HDL'] + 1)
    df['metabolic_index'] = df['fasting blood sugar'] * df['bmi'] / (df['HDL'] + 1)
    df['age_health_index'] = df['age'] * df['hemoglobin'] / df['liver_function']
    
    # Polynomial features for key health indicators
    poly = PolynomialFeatures(degree=2, include_bias=False)
    key_features = ['bmi', 'liver_function', 'cardiovascular_risk', 'metabolic_index']
    poly_features = poly.fit_transform(df[key_features])
    poly_names = [f'health_poly_{i}' for i in range(poly_features.shape[1])]
    df[poly_names] = poly_features
    
    # Feature ratios
    df['hdl_ldl_ratio'] = df['HDL'] / (df['LDL'] + 1)
    df['ast_alt_ratio'] = df['AST'] / (df['ALT'] + 1)
    df['bp_ratio'] = df['systolic'] / (df['relaxation'] + 1)
    
    # All features
    all_features = (
        original_features + 
        poly_names + 
        ['bmi', 'liver_function', 'cardiovascular_risk', 'metabolic_index', 
         'age_health_index', 'hdl_ldl_ratio', 'ast_alt_ratio', 'bp_ratio']
    )
    
    # Normalize features
    scaler = StandardScaler()
    df[all_features] = scaler.fit_transform(df[all_features])
    
    return df[all_features]

def create_efficient_ensemble(dataset_name):
    """Create an enhanced voting ensemble with XGBoost and Random Forest"""
    if dataset_name == 'archive':
        rf = RandomForestClassifier(
            n_estimators=1200,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=6,
            max_features=0.7,
            min_impurity_decrease=0.004,
            class_weight={0: 1.2, 1: 1},  # Reduced class weight difference
            criterion='entropy',
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
            max_samples=0.85
        )
        
        xgb = XGBClassifier(
            max_depth=7,
            learning_rate=0.03,
            n_estimators=400,
            min_child_weight=3,
            gamma=0.15,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=1.2,  # Reduced scale weight
            random_state=42,
            n_jobs=-1
        )
        
        return VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb)
            ],
            voting='soft',
            weights=[0.5, 0.5]  # Equal weights for better balance
        )
    else:
        # Keep existing XGBoost for ml_olympiad
        return XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            min_child_weight=3,
            gamma=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='logloss',
            enable_categorical=False
        )

def select_best_features(X, y, threshold=0.55):  # Adjusted threshold
    """Select best features using enhanced selection"""
    selector = SelectFromModel(
        estimator=XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4,
            random_state=42,
            n_jobs=-1
        ),
        threshold=threshold
    )
    selector.fit(X, y)
    return selector

def create_visualizations(y_true, y_pred, y_pred_proba, dataset_name, model_name):
    """Create and save visualization plots for model evaluation"""
    os.makedirs('artifacts/visualizations', exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name} ({model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'artifacts/visualizations/confusion_matrix_{dataset_name}_{model_name}.png')
    plt.close()

    # Percentage Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues')
    plt.title(f'Confusion Matrix (%) - {dataset_name} ({model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'artifacts/visualizations/confusion_matrix_percent_{dataset_name}_{model_name}.png')
    plt.close()

    # ROC Curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} ({model_name})')
    plt.legend(loc="lower right")
    plt.savefig(f'artifacts/visualizations/roc_curve_{dataset_name}_{model_name}.png')
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name} ({model_name})')
    plt.legend(loc="lower right")
    plt.savefig(f'artifacts/visualizations/pr_curve_{dataset_name}_{model_name}.png')
    plt.close()

def main():
    """Main function to train and evaluate improved models"""
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

    print("Using dataset paths:")
    for dataset, paths in dataset_paths.items():
        print(f"{dataset}:")
        print(f"  Train: {paths['train']}")
        print(f"  Test:  {paths['test']}")

    results = {}
    
    for dataset_name, paths in dataset_paths.items():
        print(f"\nImproving model for {dataset_name} dataset...")
        
        print("Loading data...")
        # Load data
        train_df = pd.read_csv(paths['train'])
        test_df = pd.read_csv(paths['test'])
        print(f"Loaded training data shape: {train_df.shape}")
        print(f"Loaded test data shape: {test_df.shape}")
        
        print("Creating advanced features...")
        # Create advanced features
        X_train = create_advanced_features(train_df)
        y_train = train_df['smoking']
        X_test = create_advanced_features(test_df)
        y_test = test_df['smoking']
        print(f"Features created. Training features shape: {X_train.shape}")
        
        print("Selecting best features...")
        # Feature selection
        selector = select_best_features(X_train, y_train, threshold='median')
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        print(f"Selected {X_train_selected.shape[1]} features")
        
        # Apply modified SMOTE for archive dataset
        if dataset_name == 'archive':
            print("Applying SMOTE resampling...")
            smote = SMOTE(
                random_state=42,
                k_neighbors=5,
                sampling_strategy=0.85
            )
            X_train_selected, y_train = smote.fit_resample(X_train_selected, y_train)
            print(f"After SMOTE - Training data shape: {X_train_selected.shape}")
        
        print("Training model...")
        # Create and train model
        model = create_efficient_ensemble(dataset_name)
        model.fit(X_train_selected, y_train)
        
        print("Making predictions...")
        # Get predictions
        y_pred = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
        
        print("Calculating metrics...")
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        print("Creating visualizations...")
        # Create visualizations
        model_name = 'Ensemble' if dataset_name == 'archive' else 'XGBoost'
        create_visualizations(y_test, y_pred, y_pred_proba, dataset_name, model_name)
        
        # Save results
        results[dataset_name] = {
            'metrics': metrics,
            'n_features_selected': int(X_train_selected.shape[1]),
            'features': list(X_train.columns[selector.get_support()])
        }
        
        # Save model and feature selector
        model_artifacts = {
            'model': model,
            'selector': selector,
            'feature_names': list(X_train.columns)
        }
        model_path = f"models/{dataset_name}_improved_final.pkl"
        joblib.dump(model_artifacts, model_path)
        print(f"Saved improved model to {model_path}")
        
        print(f"\nFinal Results for {dataset_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Save results
    os.makedirs('artifacts/improvements', exist_ok=True)
    with open('artifacts/improvements/final_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nFinal improvements completed! Results saved to artifacts/improvements/final_results.json")

if __name__ == "__main__":
    main()