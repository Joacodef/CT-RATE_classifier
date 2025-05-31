import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, 
    recall_score, precision_recall_curve, classification_report
)
from typing import List, Dict

import logging
logger = logging.getLogger(__name__)



def compute_metrics(predictions: np.ndarray, labels: np.ndarray, 
                   pathology_names: List[str]) -> Dict[str, float]:
    """Compute comprehensive metrics for medical classification"""
    
    # Convert predictions to probabilities
    probabilities = 1 / (1 + np.exp(-predictions))  # Sigmoid
    binary_predictions = (probabilities > 0.5).astype(int)
    
    metrics = {}
    
    # Overall metrics
    try:
        # Only compute AUC if both classes are present
        if labels.min() == 0 and labels.max() == 1:
            metrics["roc_auc_macro"] = roc_auc_score(labels, probabilities, average='macro')
            metrics["roc_auc_micro"] = roc_auc_score(labels, probabilities, average='micro')
            if np.isnan(metrics["roc_auc_macro"]) or np.isnan(metrics["roc_auc_micro"]):
                raise ValueError("AUC score is NaN, likely due to insufficient class samples.")
        else:
            metrics["roc_auc_macro"] = 0.0
            metrics["roc_auc_micro"] = 0.0
    except ValueError as e:
        logger.warning(f"AUC computation error: {e}")
        metrics["roc_auc_macro"] = 0.0
        metrics["roc_auc_micro"] = 0.0
    
    metrics["f1_macro"] = f1_score(labels, binary_predictions, average='macro', zero_division=0)
    metrics["f1_micro"] = f1_score(labels, binary_predictions, average='micro', zero_division=0)
    metrics["accuracy"] = accuracy_score(labels, binary_predictions)
    metrics["precision_macro"] = precision_score(labels, binary_predictions, average='macro', zero_division=0)
    metrics["recall_macro"] = recall_score(labels, binary_predictions, average='macro', zero_division=0)
    
    # Per-pathology metrics
    for i, pathology in enumerate(pathology_names):
        y_true = labels[:, i]
        y_pred = binary_predictions[:, i]
        y_prob = probabilities[:, i]
        
        if len(np.unique(y_true)) > 1:  # Both classes present
            try:
                auc = roc_auc_score(y_true, y_prob)
                metrics[f"{pathology}_auc"] = auc
            except ValueError:
                metrics[f"{pathology}_auc"] = 0.0
                
            metrics[f"{pathology}_f1"] = f1_score(y_true, y_pred, zero_division=0)
            metrics[f"{pathology}_precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics[f"{pathology}_recall"] = recall_score(y_true, y_pred, zero_division=0)
            
            # Sensitivity and Specificity
            tn = np.sum((y_true == 0) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics[f"{pathology}_sensitivity"] = sensitivity
            metrics[f"{pathology}_specificity"] = specificity
        else:
            # If only one class present, set metrics to 0
            metrics[f"{pathology}_auc"] = 0.0
            metrics[f"{pathology}_f1"] = 0.0
            metrics[f"{pathology}_precision"] = 0.0
            metrics[f"{pathology}_recall"] = 0.0
            metrics[f"{pathology}_sensitivity"] = 0.0
            metrics[f"{pathology}_specificity"] = 0.0
    
    return metrics
