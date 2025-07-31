# src/training/metrics.py

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
)
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def compute_metrics(predictions: np.ndarray, labels: np.ndarray, 
                   pathology_names: List[str]) -> Dict[str, float]:
    """
    Compute comprehensive metrics for multi-label classification.
    
    This function calculates a wide range of metrics, including overall
    (micro and macro) and per-pathology scores.
    
    Args:
        predictions: Logits output by the model, shape (n_samples, n_classes).
        labels: Ground truth labels, shape (n_samples, n_classes).
        pathology_names: List of strings with the name for each class.
        
    Returns:
        A dictionary mapping metric names to their float values.
    """
    
    # Convert model logits to probabilities and binary predictions
    probabilities = 1 / (1 + np.exp(-predictions))  # Sigmoid activation
    binary_predictions = (probabilities > 0.5).astype(int)
    
    metrics = {}
    
    # --- Overall Metrics (Micro and Macro Averages) ---
    
    # Calculate AUC scores, with a check for cases with only one class in labels
    try:
        if labels.min() == 0 and labels.max() == 1:
            metrics["roc_auc_macro"] = roc_auc_score(labels, probabilities, average='macro')
            metrics["roc_auc_micro"] = roc_auc_score(labels, probabilities, average='micro')
        else:
            metrics["roc_auc_macro"] = 0.0
            metrics["roc_auc_micro"] = 0.0
    except ValueError as e:
        logger.warning(f"Could not compute AUC score, likely due to single-class samples in batch: {e}")
        metrics["roc_auc_macro"] = 0.0
        metrics["roc_auc_micro"] = 0.0
        
    # Standard micro and macro metrics from sklearn
    metrics["f1_macro"] = f1_score(labels, binary_predictions, average='macro', zero_division=0)
    metrics["f1_micro"] = f1_score(labels, binary_predictions, average='micro', zero_division=0)
    metrics["precision_macro"] = precision_score(labels, binary_predictions, average='macro', zero_division=0)
    metrics["precision_micro"] = precision_score(labels, binary_predictions, average='micro', zero_division=0)
    metrics["recall_macro"] = recall_score(labels, binary_predictions, average='macro', zero_division=0)
    metrics["recall_micro"] = recall_score(labels, binary_predictions, average='micro', zero_division=0)
    
    # Overall accuracy
    metrics["accuracy"] = accuracy_score(labels, binary_predictions)
    
    # --- Per-Pathology and Manual Micro/Macro Calculations ---
    
    all_specificities = []
    total_tn = 0
    total_fp = 0
    
    # Iterate over each pathology to calculate individual metrics
    for i, pathology in enumerate(pathology_names):
        y_true = labels[:, i]
        y_pred = binary_predictions[:, i]
        y_prob = probabilities[:, i]
        
        # Confusion matrix components for the current class
        tn = np.sum((y_true == 0) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        # Accumulate for micro-average specificity
        total_tn += tn
        total_fp += fp
        
        # Per-pathology metrics are only valid if both classes are present
        if len(np.unique(y_true)) > 1:
            try:
                metrics[f"{pathology}_auc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics[f"{pathology}_auc"] = 0.0
                
            metrics[f"{pathology}_f1"] = f1_score(y_true, y_pred, zero_division=0)
            metrics[f"{pathology}_precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics[f"{pathology}_recall"] = recall_score(y_true, y_pred, zero_division=0)
            
            # Sensitivity (same as recall) and Specificity
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            metrics[f"{pathology}_sensitivity"] = sensitivity
            metrics[f"{pathology}_specificity"] = specificity
            all_specificities.append(specificity)
            
        else:
            # If only one class is present, metrics are set to 0
            metrics[f"{pathology}_auc"] = 0.0
            metrics[f"{pathology}_f1"] = 0.0
            metrics[f"{pathology}_precision"] = 0.0
            metrics[f"{pathology}_recall"] = 0.0
            metrics[f"{pathology}_sensitivity"] = 0.0
            metrics[f"{pathology}_specificity"] = 0.0

    # --- Final Macro/Micro Specificity and Balanced Accuracy ---
    
    # Macro-average specificity (average of per-class specificities)
    if all_specificities:
        metrics["specificity_macro"] = np.mean(all_specificities)
    else:
        metrics["specificity_macro"] = 0.0
        
    # Micro-average specificity (calculated from total TN and FP)
    if (total_tn + total_fp) > 0:
        metrics["specificity_micro"] = total_tn / (total_tn + total_fp)
    else:
        metrics["specificity_micro"] = 0.0

    # Balanced accuracy (average of recall/sensitivity and specificity)
    metrics["balanced_accuracy_macro"] = (metrics["recall_macro"] + metrics["specificity_macro"]) / 2.0
            
    return metrics