
#? STAGE 6: MODEL EVALUATION


import os
import json
import joblib
import numpy as np
import pandas as pd
# Set the backend to 'Agg' before importing matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

def convert_to_python_types(d):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(d, dict):
        return {k: convert_to_python_types(v) for k, v in d.items()}
    elif isinstance(d, (np.integer)):  # Updated for NumPy 2.0+
        return int(d)
    elif isinstance(d, (np.floating)):  # Updated for NumPy 2.0+
        return float(d)
    elif isinstance(d, (np.ndarray, pd.Series)):
        return convert_to_python_types(d.tolist())
    elif isinstance(d, list):
        return [convert_to_python_types(i) for i in d]
    else:
        return d

def evaluate_model(model, X_test, y_test, model_name, features):
    """
    Evaluate model performance and generate visualizations
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }

    # Create visualizations directory
    os.makedirs('artifacts/visualizations', exist_ok=True)

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'artifacts/visualizations/roc_curve_{model_name}.png')
    plt.close()

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'artifacts/visualizations/confusion_matrix_{model_name}.png')
    plt.close()

    # Enhanced error analysis
    errors_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Probability': y_pred_proba
    })
    errors_df['Error_Type'] = 'Correct'
    errors_df.loc[(errors_df['Actual'] == 1) & (errors_df['Predicted'] == 0), 'Error_Type'] = 'False Negative'
    errors_df.loc[(errors_df['Actual'] == 0) & (errors_df['Predicted'] == 1), 'Error_Type'] = 'False Positive'
    
    # Add feature values for error analysis
    errors_df = pd.concat([errors_df, X_test.reset_index(drop=True)], axis=1)
    
    # Save error analysis with converted types
    error_analysis = {
        'false_positives': {
            'count': int(len(errors_df[errors_df['Error_Type'] == 'False Positive'])),
            'avg_probability': float(errors_df[errors_df['Error_Type'] == 'False Positive']['Probability'].mean()),
            'feature_means': convert_to_python_types(
                errors_df[errors_df['Error_Type'] == 'False Positive'][features].mean().to_dict()
            )
        },
        'false_negatives': {
            'count': int(len(errors_df[errors_df['Error_Type'] == 'False Negative'])),
            'avg_probability': float(errors_df[errors_df['Error_Type'] == 'False Negative']['Probability'].mean()),
            'feature_means': convert_to_python_types(
                errors_df[errors_df['Error_Type'] == 'False Negative'][features].mean().to_dict()
            )
        }
    }
    
    # Save detailed error analysis
    with open(f'artifacts/visualizations/error_analysis_{model_name}.json', 'w') as f:
        json.dump(error_analysis, f, indent=4)
    
    # Plot confusion matrix with percentages
    plt.figure(figsize=(10, 8))
    cm_percent = confusion_matrix(y_test, y_pred, normalize='true') * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Non-Smoker', 'Smoker'],
                yticklabels=['Non-Smoker', 'Smoker'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (%) - {model_name}')
    plt.savefig(f'artifacts/visualizations/confusion_matrix_percent_{model_name}.png')
    plt.close()

    # Feature Importance Plot (if available)
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': features,
            'importance': [float(i) for i in model.feature_importances_]  # Convert to Python float
        }).sort_values('importance', ascending=False)

        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_20_features = importances.head(20)
        sns.barplot(data=top_20_features, x='importance', y='feature')
        plt.title(f'Top 20 Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'artifacts/visualizations/feature_importance_{model_name}.png')
        plt.close()

        # Save complete feature importance to JSON
        importance_dict = {k: float(v) for k, v in importances.set_index('feature')['importance'].to_dict().items()}
        with open(f'artifacts/visualizations/feature_importance_{model_name}.json', 'w') as f:
            json.dump(importance_dict, f, indent=4)

    return metrics

def main():
    # Load model information
    with open('artifacts/models/model_info.json', 'r') as f:
        model_info = json.load(f)

    # Paths to test datasets
    dataset_paths = {
        'ml_olympiad': 'Y:/SmokingML V2/data/processed/ml_olympiad_test.csv',
        'archive': 'Y:/SmokingML V2/data/processed/archive_test.csv'
    }

    evaluation_results = {}

    for dataset_name, test_path in dataset_paths.items():
        print(f"\nEvaluating model for {dataset_name} dataset...")
        
        # Load the model
        model_path = model_info[dataset_name]['model_path']
        model = joblib.load(model_path)
        
        # Load test data
        test_df = pd.read_csv(test_path)
        
        # Get dataset-specific features from model info
        features = model_info[dataset_name]['features']
        
        # Get features and target
        X_test = test_df[features]
        y_test = test_df['smoking']
        
        print(f"Number of features being used for {dataset_name}: {len(features)}")
        
        # Evaluate model
        metrics = evaluate_model(
            model, 
            X_test, 
            y_test, 
            f"{dataset_name}_{model_info[dataset_name]['name']}",
            features
        )
        
        # Store results
        evaluation_results[dataset_name] = {
            'model_name': model_info[dataset_name]['name'],
            'metrics': metrics,
            'num_features': len(features),
            'features': features
        }
        
        print(f"\nResults for {dataset_name}:")
        print(f"Model: {model_info[dataset_name]['name']}")
        print(f"Number of features: {len(features)}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Save evaluation results
    with open('artifacts/models/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
        
    print("\nEvaluation completed! Results saved to artifacts/models/evaluation_results.json")
    print("Visualizations saved to artifacts/visualizations/")

if __name__ == "__main__":
    main()
