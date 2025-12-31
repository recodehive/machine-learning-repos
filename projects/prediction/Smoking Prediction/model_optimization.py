
#  ? STAGE 5: MODEL OPTIMIZATION


import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

def create_balanced_scorer(dataset_name):
    """Create a custom scorer that ensures all metrics improve for the specific dataset"""
    def balanced_scorer(y_true, y_pred):
        # Calculate all metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Select thresholds based on dataset
        if dataset_name == 'ml_olympiad':
            thresholds = {
                'accuracy': 0.7777,
                'precision': 0.7203,
                'recall': 0.7981,
                'f1': 0.7572
            }
        else:  # archive dataset
            thresholds = {
                'accuracy': 0.7724,
                'precision': 0.6957,
                'recall': 0.6768,
                'f1': 0.6861
            }
        
        # Calculate improvements relative to thresholds
        acc_imp = (acc - thresholds['accuracy'])
        prec_imp = (prec - thresholds['precision'])
        rec_imp = (rec - thresholds['recall'])
        f1_imp = (f1 - thresholds['f1'])
        
        # If any metric is below threshold, heavily penalize
        if acc < thresholds['accuracy'] or prec < thresholds['precision'] or \
           rec < thresholds['recall'] or f1 < thresholds['f1']:
            return -100.0  # Strong penalty for any decrease
        
        # Otherwise, reward based on minimum improvement
        min_improvement = min(acc_imp, prec_imp, rec_imp, f1_imp)
        avg_improvement = (acc_imp + prec_imp + rec_imp + f1_imp) / 4
        
        # Combine minimum and average improvements
        # This ensures we prioritize solutions where all metrics improve
        return min_improvement + avg_improvement
    
    return make_scorer(balanced_scorer, greater_is_better=True)

def optimize_model(X, y, model_type, param_grid, dataset_name):
    """Optimize model hyperparameters using GridSearchCV with strict improvement requirements"""
    # Create dataset-specific balanced scorer
    balanced_scorer = create_balanced_scorer(dataset_name)
    
    # Perform grid search with reduced CV to speed up search
    search = RandomizedSearchCV(
        estimator=model_type,
        param_distributions=param_grid,
        n_iter=30,
        scoring=balanced_scorer,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )
    
    search.fit(X, y)
    
    # Get predictions using best model
    y_pred = search.best_estimator_.predict(X)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred)
    }
    
    return search.best_estimator_,{
        'best_params': search.best_params_,
        'best_score': float(search.best_score_),
        'cv_results': metrics
    }
    

def main():
    """Main function to optimize models"""
    # Load data
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

    # Load model information
    with open('artifacts/models/model_info.json', 'r') as f:
        model_info = json.load(f)

    # Updated parameter grids with more focused ranges
    param_grids = {
        'XGBoost': {
            'max_depth': [5, 6, 7],
            'learning_rate': stats.uniform(0.01, 0.1),
            'n_estimators': [300, 400, 500],
            'min_child_weight': [2, 3, 4],
            'gamma': stats.uniform(0, 0.3),
            'subsample': stats.uniform(0.7, 0.3),
            'colsample_bytree': stats.uniform(0.7, 0.3),
            'scale_pos_weight': [1.0, 1.1],
            'reg_alpha': stats.uniform(0.0, 0.3),
            'reg_lambda': stats.uniform(1.0, 2.0),
        },
        'Random_Forest': {
            'n_estimators': [200, 300],  # Reduced from 3 values to 2
            'max_depth': [15, 20],       # Still allows deep trees but not overly large
            'min_samples_split': [4],    # Fixed to one optimal value
            'min_samples_leaf': [2],     # Fixed to one optimal value
            'max_features': ['sqrt'],    # Typically best for classification
            'class_weight': ['balanced'] # Good for imbalance handling
        }
    }

    optimization_results = {}

    for dataset_name, paths in dataset_paths.items():
        print(f"\nOptimizing model for {dataset_name} dataset...")
        
        # Load training data
        train_df = pd.read_csv(paths['train'])
        test_df = pd.read_csv(paths['test'])
        
        # Get features from model info
        features = model_info[dataset_name]['features']
        
        # Prepare data
        X_train = train_df[features]
        y_train = train_df['smoking']
        X_test = test_df[features]
        y_test = test_df['smoking']
        
        # Select model type and parameter grid
        model_name = model_info[dataset_name]['name']
        # Using XGBoost for both datasets
        model_type = XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            enable_categorical=False
        )
        param_grid = param_grids['XGBoost']  # Use same parameter grid for both datasets

        
        # Optimize model
        print(f"Performing grid search with cross-validation for {model_name}...")
        best_model, cv_results = optimize_model(X_train, y_train, model_type, param_grid, dataset_name)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # Save optimized model
        model_path = f"models/{dataset_name}_{model_name}_optimized.pkl"
        joblib.dump(best_model, model_path)
        
        # Store results
        optimization_results[dataset_name] = {
            'model_name': model_name,
            'best_params': cv_results['best_params'],
            'cv_scores': cv_results['cv_results'],
            'test_scores': test_metrics,
            'model_path': model_path
        }
        
        print(f"\nOptimization results for {dataset_name}:")
        print(f"Best parameters: {cv_results['best_params']}")
        print("\nTest set scores:")
        for metric, score in test_metrics.items():
            print(f"{metric}: {score:.4f}")

    # Save optimization results
    os.makedirs('artifacts/optimization', exist_ok=True)
    with open('artifacts/optimization/optimization_results.json', 'w') as f:
        json.dump(optimization_results, f, indent=4)
    
    print("\nOptimization completed! Results saved to artifacts/optimization/optimization_results.json")

if __name__ == "__main__":
    main()