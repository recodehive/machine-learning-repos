"""Evaluation metrics module for Medical Appointment No-Shows model.

This module provides comprehensive evaluation metrics including accuracy,
precision, recall, F1-score, confusion matrix, and ROC-AUC score.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
    roc_curve, auc
)
import seaborn as sns


class MetricsEvaluator:
    """Evaluate machine learning model performance with comprehensive metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.predictions = None
        self.actual = None
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        self.actual = y_true
        self.predictions = y_pred
        
        # Calculate basic metrics
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        self.metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        self.metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Calculate ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                self.metrics['roc_auc'] = None
        
        # Confusion matrix
        self.metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return self.metrics
    
    def print_report(self):
        """Print detailed classification report."""
        if self.actual is None or self.predictions is None:
            print("No metrics calculated yet. Call calculate_metrics first.")
            return
        
        print("\n=== Classification Report ===")
        print(classification_report(self.actual, self.predictions))
    
    def display_metrics(self):
        """Display all calculated metrics."""
        print("\n=== Evaluation Metrics ===")
        print(f"Accuracy:  {self.metrics.get('accuracy', 'N/A'):.4f}")
        print(f"Precision: {self.metrics.get('precision', 'N/A'):.4f}")
        print(f"Recall:    {self.metrics.get('recall', 'N/A'):.4f}")
        print(f"F1-Score:  {self.metrics.get('f1_score', 'N/A'):.4f}")
        if self.metrics.get('roc_auc'):
            print(f"ROC-AUC:   {self.metrics.get('roc_auc', 'N/A'):.4f}")
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix heatmap.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        cm = self.metrics.get('confusion_matrix')
        if cm is None:
            print("No confusion matrix available.")
            return
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


if __name__ == '__main__':
    print('Metrics Evaluator module for Medical Appointment No-Shows prediction model')
