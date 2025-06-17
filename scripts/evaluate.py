#!/usr/bin/env python3
"""
Evaluation script for CT 3D Classifier

This script evaluates trained models on validation or test datasets,
generates detailed reports, and provides various analysis options.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, 
    recall_score, precision_recall_curve, roc_curve, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
import logging

# Add project root to path
# Add project root and src directory to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import load_config
from src.training.trainer import create_model
from src.data.dataset import CTDataset3D
from src.training.metrics import compute_metrics
from src.utils.logging_config import setup_logging

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, config, model_path, device: str = 'auto'):
        self.config = config # Store config instance
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.logger = logging.getLogger(__name__) # Use standard logger

        # Load model
        self.model = self._load_model()
        
    def _setup_device(self, device_option: str) -> torch.device: # Renamed for clarity
        """Setup computation device"""
        if device_option == 'auto':
            selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            selected_device = device_option
        self.logger.info(f"Computation device set to: {selected_device}") # Log device selection
        return torch.device(selected_device)
    
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint"""
        if not self.model_path.exists():
            self.logger.error(f"Model file not found: {self.model_path}") # Log error
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
         # Create model architecture using the centralized create_model function
        self.logger.info(f"Creating model architecture: {self.config.model.type} (variant: {self.config.model.variant}) using create_model.")
        # Ensure gradient checkpointing is disabled for evaluation/inference
        original_gradient_checkpointing_setting = self.config.optimization.gradient_checkpointing
        self.config.optimization.gradient_checkpointing = False
        model = create_model(self.config)
        self.config.optimization.gradient_checkpointing = original_gradient_checkpointing_setting # Restore original setting
        self.logger.info(f"Model created for {self.config.pathologies.num_pathologies} classes.")
        
        # Load weights
        self.logger.info(f"Loading model weights from: {self.model_path}") # Log path
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            # Log epoch if available in checkpoint
            epoch_info = checkpoint.get('epoch', 'unknown')
            self.logger.info(f"Loaded model state_dict from epoch {epoch_info}.")
        else:
            # Assumes the checkpoint is the state_dict itself
            model.load_state_dict(checkpoint)
            self.logger.info("Loaded model state_dict directly (no epoch info in checkpoint).")
        
        model = model.to(self.device) # Move model to the specified device
        model.eval() # Set model to evaluation mode
        
        self.logger.info(f"Model loaded successfully on {self.device} and set to evaluation mode.")
        return model
    
    def _prepare_dataset(self, dataset_type: str = 'validation') -> Tuple[pd.DataFrame, Path]:
        """Prepare dataset for evaluation"""
        self.logger.info(f"Preparing dataset for '{dataset_type}' evaluation.") # Log dataset type
        if dataset_type == 'validation':
            volumes_csv = self.config.paths.data_subsets.selected_valid_volumes
            labels_csv = self.config.paths.labels.valid
            img_dir = self.config.paths.valid_img_dir
        elif dataset_type == 'train': # Added train dataset option
            volumes_csv = self.config.paths.data_subsets.selected_train_volumes
            labels_csv = self.config.paths.labels.train
            img_dir = self.config.paths.train_img_dir
        else:
            self.logger.error(f"Unknown dataset type for evaluation: {dataset_type}") # Log error
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Load and merge data
        self.logger.info(f"Loading volumes from: {volumes_csv}") # Log path
        self.logger.info(f"Loading labels from: {labels_csv}") # Log path
        volumes_df = pd.read_csv(volumes_csv)[['VolumeName']]
        labels_df = pd.read_csv(labels_csv)
        
        eval_df = pd.merge(volumes_df, labels_df, on='VolumeName', how='inner')
        # Fill NaN and convert pathology columns to int
        eval_df[self.config.pathologies.columns] = eval_df[self.config.pathologies.columns].fillna(0).astype(int)
        
        self.logger.info(f"Loaded {len(eval_df)} samples for {dataset_type} evaluation.")
        return eval_df, img_dir
    
    @torch.no_grad() # Disable gradient calculations for evaluation
    def evaluate_dataset(self, dataset_type: str = 'validation', 
                        save_predictions: bool = True) -> Dict:
        """Evaluate model on specified dataset"""
        
        self.logger.info(f"Starting evaluation on {dataset_type} dataset...")
        
        # Prepare dataset
        eval_df, img_dir = self._prepare_dataset(dataset_type)
        
        # Create dataset and dataloader
        self.logger.info("Creating evaluation CTDataset3D instance.") # Log dataset creation
        eval_dataset = CTDataset3D(
            dataframe=eval_df, # Pass dataframe
            img_dir=img_dir, 
            pathology_columns=self.config.pathologies.columns,
            target_spacing_xyz=self.config.image_processing.target_spacing, 
            target_shape_dhw=self.config.image_processing.target_shape_dhw,
            clip_hu_min=self.config.image_processing.clip_hu_min, 
            clip_hu_max=self.config.image_processing.clip_hu_max,
            use_cache=self.config.cache.use_cache, 
            cache_dir=self.config.paths.cache_dir,
            augment=False,  # No augmentation for evaluation
            orientation_axcodes=self.config.image_processing.orientation_axcodes
        )
        
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=self.config.training.batch_size, 
            shuffle=False, # No shuffle for evaluation
            num_workers=self.config.training.num_workers, 
            pin_memory=self.config.training.pin_memory,
            persistent_workers=self.config.training.num_workers > 0
        )
        
        # Collect predictions and labels
        all_predictions = [] # Stores raw model outputs (logits)
        all_labels = [] # Stores true labels
        all_volume_names = [] # Stores volume names for traceability
        
        self.model.eval() # Ensure model is in evaluation mode
        self.logger.info(f"Iterating through DataLoader for {dataset_type} set...") # Log loader iteration
        for batch_idx, batch in enumerate(eval_loader):
            # Move data to the configured device
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            volume_names = batch["volume_name"] # List of volume names in the batch
            
            # Forward pass through the model
            outputs = self.model(pixel_values)
            
            # Collect results, moving to CPU and converting to NumPy arrays
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_volume_names.extend(volume_names) # Extend list of volume names
            
            # Log progress periodically
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(eval_loader): # Log every 10 batches or last batch
                self.logger.info(f"Processed {batch_idx + 1}/{len(eval_loader)} batches for {dataset_type} evaluation.")
        
        # Concatenate all results from batches
        predictions_np = np.concatenate(all_predictions, axis=0) # Corrected variable name
        labels_np = np.concatenate(all_labels, axis=0) # Corrected variable name
        
        # Convert predictions (logits) to probabilities using sigmoid
        probabilities_np = 1 / (1 + np.exp(-predictions_np)) # Sigmoid function
        
        # Compute metrics using the dedicated metrics computation function
        self.logger.info("Computing metrics from predictions and labels.") # Log metrics computation
        metrics = compute_metrics(predictions_np, labels_np, self.config.pathologies.columns) # Pass logits
        
        # Prepare results dictionary
        results = {
            'dataset_type': dataset_type,
            'num_samples': len(labels_np), # Use length of concatenated labels
            'metrics': metrics,
            'predictions_logits': predictions_np, # Store raw logits
            'probabilities': probabilities_np, # Store probabilities
            'labels': labels_np, # Store true labels
            'volume_names': all_volume_names, # Store volume names
            'pathology_columns': self.config.pathologies.columns # Store pathology names
        }
        
        # Save predictions to a CSV file if requested
        if save_predictions:
            self._save_predictions(results, dataset_type) # Call helper to save
        
        self.logger.info(f"Evaluation completed on {len(labels_np)} samples from {dataset_type} dataset.")
        self.logger.info(f"Overall AUC (macro): {metrics.get('roc_auc_macro', float('nan')):.4f}") # Use .get for safety
        self.logger.info(f"Overall F1 (macro): {metrics.get('f1_macro', float('nan')):.4f}") # Use .get for safety
        
        return results
    
    def _save_predictions(self, results: Dict, dataset_type: str):
        """Save predictions and probabilities to CSV file"""
        output_dir = self.config.paths.output_dir # Get output directory from config
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        # Prepare data for DataFrame
        predictions_data = []
        num_samples = results['num_samples']
        pathologies = results['pathology_columns']
        
        for i in range(num_samples):
            row = {'VolumeName': results['volume_names'][i]}
            for j, pathology in enumerate(pathologies):
                row[f'{pathology}_true'] = int(results['labels'][i, j])
                row[f'{pathology}_prob'] = float(results['probabilities'][i, j])
                # Binary prediction based on 0.5 threshold
                row[f'{pathology}_pred'] = int(results['probabilities'][i, j] > 0.5) 
            predictions_data.append(row)
        
        predictions_df = pd.DataFrame(predictions_data)
        
        # Define file paths for predictions and metrics
        predictions_file = output_dir / f'predictions_{dataset_type}_{self.model_path.stem}.csv' # Add model name to filename
        metrics_file = output_dir / f'metrics_{dataset_type}_{self.model_path.stem}.json' # Add model name to filename
        
        # Save predictions DataFrame to CSV
        try:
            predictions_df.to_csv(predictions_file, index=False)
            self.logger.info(f"Predictions for {dataset_type} saved to: {predictions_file}")
        except Exception as e:
            self.logger.error(f"Failed to save predictions CSV to {predictions_file}: {e}") # Log error
            
        # Save metrics dictionary to JSON
        try:
            with open(metrics_file, 'w') as f:
                # Use a helper to ensure all items in metrics are JSON serializable
                serializable_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                       for k, v in results['metrics'].items()}
                json.dump(serializable_metrics, f, indent=2)
            self.logger.info(f"Metrics for {dataset_type} saved to: {metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics JSON to {metrics_file}: {e}") # Log error
    
    def generate_evaluation_report(self, results: Dict, save_path: Optional[str] = None):
        """Generate comprehensive evaluation report with visualizations"""
        
        # Define save path for the report image
        if save_path is None:
            report_filename = f"evaluation_report_{results['dataset_type']}_{self.model_path.stem}.png" # Add model name
            save_path = self.config.paths.output_dir / report_filename
        else:
            save_path = Path(save_path) # Ensure it's a Path object
        save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

        self.logger.info(f"Generating evaluation report. Saving to: {save_path}") # Log save path
        
        # Create figure for the report
        fig, axes = plt.subplots(3, 3, figsize=(22, 18)) # Adjusted figsize
        fig.suptitle(f'Model Evaluation Report - {results["dataset_type"].title()} Set\nModel: {self.model_path.name}', 
                    fontsize=18, fontweight='bold') # Increased title font size
        
        # 1. Overall Metrics Bar Chart
        ax = axes[0, 0]
        # Define which overall metrics to plot
        overall_metrics_keys = ['roc_auc_macro', 'roc_auc_micro', 'f1_macro', 'f1_micro', 
                                'accuracy', 'precision_macro', 'recall_macro']
        metric_values = [results['metrics'].get(m, 0.0) for m in overall_metrics_keys] # Use .get for safety
        metric_labels = [m.replace('_', ' ').title() for m in overall_metrics_keys]
        
        bars = ax.bar(metric_labels, metric_values, color='skyblue', width=0.6) # Adjusted bar width
        ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=10) # Adjusted font size
        ax.set_ylabel('Score', fontsize=12) # Adjusted font size
        ax.set_title('Overall Performance Metrics', fontsize=14) # Adjusted font size
        ax.set_ylim(0, 1.05) # Slightly higher limit for text
        ax.grid(True, linestyle='--', alpha=0.7) # Added grid
        
        # Add value labels on bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', 
                    ha='center', va='bottom', fontsize=9) # Adjusted font size
        
        # 2. Per-Pathology AUC Scores (Horizontal Bar Chart)
        ax = axes[0, 1]
        pathology_aucs = []
        pathology_names_short = [] # For shortened names
        
        for pathology in results['pathology_columns']:
            auc_key = f"{pathology}_auc"
            if auc_key in results['metrics']: # Check if AUC score exists for pathology
                pathology_aucs.append(results['metrics'][auc_key])
                # Shorten name if too long for display
                name_to_display = pathology[:18] + '...' if len(pathology) > 18 else pathology 
                pathology_names_short.append(name_to_display)
        
        if pathology_aucs: # Proceed only if there are AUCs to plot
            y_pos = np.arange(len(pathology_names_short))
            bars = ax.barh(y_pos, pathology_aucs, color='lightgreen', height=0.7) # Adjusted bar height
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pathology_names_short, fontsize=9) # Adjusted font size
            ax.set_xlabel('AUC Score', fontsize=12) # Adjusted font size
            ax.set_title('Per-Pathology AUC Scores', fontsize=14) # Adjusted font size
            ax.set_xlim(0, 1.05) # Slightly higher limit for text
            ax.invert_yaxis() # Display top-performing pathologies at the top
            ax.grid(True, linestyle='--', alpha=0.7, axis='x') # Grid on x-axis

            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.0, f'{width:.3f}',
                        ha='left', va='center', fontsize=8) # Adjusted font size
        else:
            ax.text(0.5, 0.5, "No per-pathology AUCs available.", ha='center', va='center') # Placeholder text
        
        # 3. ROC Curves for top pathologies
        ax = axes[0, 2]
        # Sort pathologies by AUC to select top ones
        auc_sorted_pathologies = sorted(
            [(p, results['metrics'].get(f"{p}_auc", 0.0)) for p in results['pathology_columns']], 
            key=lambda x: x[1], reverse=True
        )
        top_n_roc = min(5, len(auc_sorted_pathologies)) # Plot up to 5 top pathologies

        for i in range(top_n_roc):
            pathology, auc_score = auc_sorted_pathologies[i]
            pathology_idx = results['pathology_columns'].index(pathology)
            y_true = results['labels'][:, pathology_idx]
            y_prob = results['probabilities'][:, pathology_idx]
            
            # Ensure both classes are present for ROC curve calculation
            if len(np.unique(y_true)) > 1:  
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                # Shorten name for legend
                legend_name = pathology[:15] + '...' if len(pathology) > 15 else pathology
                ax.plot(fpr, tpr, label=f'{legend_name} (AUC={auc_score:.3f})', linewidth=1.5) # Adjusted linewidth
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6) # Diagonal reference line
        ax.set_xlabel('False Positive Rate', fontsize=12) # Adjusted font size
        ax.set_ylabel('True Positive Rate', fontsize=12) # Adjusted font size
        ax.set_title(f'ROC Curves (Top {top_n_roc} Pathologies)', fontsize=14) # Adjusted font size
        ax.legend(fontsize=8, loc='lower right') # Adjusted font size and location
        ax.grid(True, linestyle='--', alpha=0.7) # Added grid
        ax.set_aspect('equal', adjustable='box') # Make ROC plot square
        
        # 4. Precision-Recall Curves for top pathologies (same top N as ROC)
        ax = axes[1, 0]
        for i in range(top_n_roc):
            pathology, _ = auc_sorted_pathologies[i] # AUC score not needed for PR curve directly here
            pathology_idx = results['pathology_columns'].index(pathology)
            y_true = results['labels'][:, pathology_idx]
            y_prob = results['probabilities'][:, pathology_idx]
            
            if len(np.unique(y_true)) > 1: # Ensure both classes are present
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                legend_name = pathology[:15] + '...' if len(pathology) > 15 else pathology
                ax.plot(recall, precision, label=f'{legend_name}', linewidth=1.5) # Adjusted linewidth
        
        ax.set_xlabel('Recall', fontsize=12) # Adjusted font size
        ax.set_ylabel('Precision', fontsize=12) # Adjusted font size
        ax.set_title('Precision-Recall Curves', fontsize=14) # Adjusted font size
        ax.legend(fontsize=8, loc='lower left') # Adjusted font size and location
        ax.grid(True, linestyle='--', alpha=0.7) # Added grid
        ax.set_aspect('equal', adjustable='box') # Make PR plot square
        
        # 5. Prediction Probability Distribution
        ax = axes[1, 1]
        all_probs_flat = results['probabilities'].flatten() # Flatten all probabilities
        sns.histplot(all_probs_flat, bins=50, kde=True, ax=ax, color='teal', stat="density") # Added KDE
        ax.set_xlabel('Predicted Probability', fontsize=12) # Adjusted font size
        ax.set_ylabel('Density', fontsize=12) # Adjusted font size
        ax.set_title('Distribution of Predicted Probabilities', fontsize=14) # Adjusted font size
        ax.grid(True, linestyle='--', alpha=0.7) # Added grid
        
        # 6. Calibration Plot (Overall)
        ax = axes[1, 2]
        all_labels_flat = results['labels'].flatten() # Flatten all labels
        # Ensure both classes are present for calibration curve
        if len(np.unique(all_labels_flat)) > 1 and len(all_labels_flat) == len(all_probs_flat):
            fraction_of_positives, mean_predicted_value = calibration_curve(
                all_labels_flat, all_probs_flat, n_bins=10, strategy='uniform' # Uniform strategy
            )
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model", markersize=5) # Added markersize
            ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated", alpha=0.7) # Adjusted alpha
            ax.set_xlabel('Mean Predicted Probability (Bin)', fontsize=12) # Adjusted font size
            ax.set_ylabel('Fraction of Positives (Bin)', fontsize=12) # Adjusted font size
            ax.set_title('Calibration Plot (Overall)', fontsize=14) # Adjusted font size
            ax.legend(fontsize=9) # Adjusted font size
            ax.grid(True, linestyle='--', alpha=0.7) # Added grid
            ax.set_aspect('equal', adjustable='box') # Make plot square
        else:
            ax.text(0.5, 0.5, "Calibration plot not available\n(requires multiple classes).", ha='center', va='center')

        # 7. Per-Pathology F1 Scores (Companion to AUC plot)
        ax = axes[2, 0]
        pathology_f1s = []
        # Using pathology_names_short from AUC plot for consistency
        for pathology in results['pathology_columns']: # Iterate in original order to match names if needed
            f1_key = f"{pathology}_f1"
            if f1_key in results['metrics']:
                pathology_f1s.append(results['metrics'][f1_key])
            else:
                # If some pathologies were not in AUC plot due to missing AUC,
                # ensure f1 list matches pathology_names_short length or handle carefully.
                # For simplicity, assuming pathology_names_short is based on pathologies that have AUC.
                # This plot should ideally use names for which F1 is available.
                pass # Or append 0.0 if name is in pathology_names_short but F1 is missing
        
        # Re-filter names to match available F1 scores
        pathology_names_for_f1 = []
        pathology_f1_values_for_plot = []
        for pathology in results['pathology_columns']:
            f1_key = f"{pathology}_f1"
            if f1_key in results['metrics']:
                 name_to_display = pathology[:18] + '...' if len(pathology) > 18 else pathology
                 pathology_names_for_f1.append(name_to_display)
                 pathology_f1_values_for_plot.append(results['metrics'][f1_key])

        if pathology_f1_values_for_plot:
            y_pos_f1 = np.arange(len(pathology_names_for_f1))
            bars_f1 = ax.barh(y_pos_f1, pathology_f1_values_for_plot, color='lightcoral', height=0.7)
            ax.set_yticks(y_pos_f1)
            ax.set_yticklabels(pathology_names_for_f1, fontsize=9)
            ax.set_xlabel('F1 Score', fontsize=12)
            ax.set_title('Per-Pathology F1 Scores', fontsize=14)
            ax.set_xlim(0, 1.05)
            ax.invert_yaxis() 
            ax.grid(True, linestyle='--', alpha=0.7, axis='x')
            for bar in bars_f1: # Add labels
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.0, f'{width:.3f}',
                        ha='left', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, "No per-pathology F1s available.", ha='center', va='center')
        
        # 8. Class Distribution in Evaluated Set
        ax = axes[2, 1]
        if results['labels'].size > 0: # Check if labels exist
            positive_counts = results['labels'].sum(axis=0)
            # Ensure no negative counts if sum is 0 (can happen with uint types if not careful)
            negative_counts = len(results['labels']) - positive_counts 
            
            x_indices = np.arange(len(results['pathology_columns']))
            width = 0.35 # Bar width
            
            rects1 = ax.bar(x_indices - width/2, positive_counts, width, label='Positive', color='forestgreen', alpha=0.8)
            rects2 = ax.bar(x_indices + width/2, negative_counts, width, label='Negative', color='indianred', alpha=0.8)
            
            ax.set_xlabel('Pathology', fontsize=12)
            ax.set_ylabel('Number of Samples', fontsize=12)
            ax.set_title('Class Distribution in Evaluated Set', fontsize=14)
            ax.set_xticks(x_indices)
            # Shorten pathology names for x-axis labels
            xtick_labels = [p[:10] + '...' if len(p) > 10 else p for p in results['pathology_columns']]
            ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        else:
            ax.text(0.5,0.5, "Label data not available.", ha='center', va='center')

        # 9. Summary Text Box
        ax = axes[2, 2]
        ax.axis('off') # Hide axes for text box
        
        # Construct summary text
        summary_text_lines = [
            f"Evaluation Summary - {self.model_path.name}",
            "=" * 35,
            f"Dataset: {results['dataset_type'].title()}",
            f"Samples: {results['num_samples']}",
            f"Model Type: {self.config.model.type}",
            "",
            "Overall Metrics:",
            f"  ROC AUC (Macro): {results['metrics'].get('roc_auc_macro', 'N/A'):.4f}",
            f"  F1 Score (Macro): {results['metrics'].get('f1_macro', 'N/A'):.4f}",
            f"  Accuracy: {results['metrics'].get('accuracy', 'N/A'):.4f}",
            f"  Precision (Macro): {results['metrics'].get('precision_macro', 'N/A'):.4f}",
            f"  Recall (Macro): {results['metrics'].get('recall_macro', 'N/A'):.4f}",
        ]
        summary_text = "\n".join(summary_text_lines)
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.9)) # Adjusted bbox
        
        # Adjust layout to prevent overlap and save
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Evaluation report saved successfully to: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save evaluation report to {save_path}: {e}") # Log error
        plt.close(fig) # Close the figure to free memory
    
    def compare_thresholds(self, results: Dict, pathology: str, save_path: Optional[str] = None):
        """Analyze performance at different thresholds for a specific pathology"""
        
        if pathology not in self.config.pathologies.columns:
            self.logger.error(f"Pathology '{pathology}' not found in configuration.") # Log error
            raise ValueError(f"Pathology '{pathology}' not found in configuration")
        
        pathology_idx = self.config.pathologies.columns.index(pathology)
        y_true = results['labels'][:, pathology_idx]
        y_prob = results['probabilities'][:, pathology_idx]
        
        if len(np.unique(y_true)) <= 1: # Check if only one class is present
            self.logger.warning(f"Only one class present for {pathology}, skipping threshold analysis.")
            return None, None # Return None if analysis cannot be performed
        
        # Define thresholds to test
        thresholds = np.linspace(0.05, 0.95, 19) # More granular thresholds
        metrics_at_thresholds = {
            'threshold': [], 'f1': [], 'precision': [], 'recall': [],
            'specificity': [], 'accuracy': []
        }
        
        # Calculate metrics for each threshold
        for threshold_val in thresholds: # Renamed variable
            y_pred = (y_prob >= threshold_val).astype(int) # Predictions based on current threshold
            
            tn = np.sum((y_true == 0) & (y_pred == 0)) # True Negatives
            tp = np.sum((y_true == 1) & (y_pred == 1)) # True Positives
            fn = np.sum((y_true == 1) & (y_pred == 0)) # False Negatives
            fp = np.sum((y_true == 0) & (y_pred == 1)) # False Positives
            
            # Calculate metrics, handling division by zero by setting to 0.0
            metrics_at_thresholds['threshold'].append(threshold_val)
            metrics_at_thresholds['f1'].append(f1_score(y_true, y_pred, zero_division=0))
            metrics_at_thresholds['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics_at_thresholds['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            metrics_at_thresholds['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            metrics_at_thresholds['accuracy'].append(accuracy_score(y_true, y_pred))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7)) # Adjusted figsize
        
        # Plot each metric vs threshold
        ax.plot(metrics_at_thresholds['threshold'], metrics_at_thresholds['f1'], 'o-', label='F1 Score', linewidth=2, markersize=4)
        ax.plot(metrics_at_thresholds['threshold'], metrics_at_thresholds['precision'], 's-', label='Precision', linewidth=2, markersize=4)
        ax.plot(metrics_at_thresholds['threshold'], metrics_at_thresholds['recall'], '^-', label='Recall (Sensitivity)', linewidth=2, markersize=4)
        ax.plot(metrics_at_thresholds['threshold'], metrics_at_thresholds['specificity'], 'd-', label='Specificity', linewidth=2, markersize=4)
        # ax.plot(metrics_at_thresholds['threshold'], metrics_at_thresholds['accuracy'], 'v-', label='Accuracy', linewidth=2, markersize=4) # Accuracy often less informative here
        
        # Find optimal threshold (e.g., maximizing F1-score)
        best_f1_idx = np.argmax(metrics_at_thresholds['f1'])
        best_threshold_val = metrics_at_thresholds['threshold'][best_f1_idx] # Renamed variable
        best_f1_score = metrics_at_thresholds['f1'][best_f1_idx] # Renamed variable
        
        ax.axvline(x=best_threshold_val, color='crimson', linestyle='--', alpha=0.8, 
                  label=f'Optimal Thr (Max F1): {best_threshold_val:.2f} (F1={best_f1_score:.3f})')
        
        ax.set_xlabel('Threshold', fontsize=12) # Adjusted font size
        ax.set_ylabel('Score', fontsize=12) # Adjusted font size
        ax.set_title(f'Performance vs. Threshold - {pathology}', fontsize=14) # Adjusted font size
        ax.legend(fontsize=10) # Adjusted font size
        ax.grid(True, linestyle='--', alpha=0.7) # Added grid
        ax.set_ylim(-0.05, 1.05) # Set y-axis limits
        ax.set_xticks(np.arange(0, 1.01, 0.1)) # Set x-axis ticks for clarity

        # Define save path for the threshold analysis plot
        if save_path is None:
            plot_filename = f"threshold_analysis_{pathology.replace(' ', '_').lower()}_{self.model_path.stem}.png" # Add model name
            save_path = self.config.paths.output_dir / plot_filename
        save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        try:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Threshold analysis for '{pathology}' saved to: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save threshold analysis plot to {save_path}: {e}") # Log error
        plt.close(fig) # Close figure
        
        return pd.DataFrame(metrics_at_thresholds), best_threshold_val # Return DataFrame and best threshold


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate CT 3D Classifier')
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to the YAML configuration file.'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to trained model checkpoint (.pth file).'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='validation',
        choices=['validation', 'train'],
        help='Dataset to evaluate on (default: validation).'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None,
        help='Override the output directory specified in the config file.'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for evaluation (default: auto).'
    )
    parser.add_argument(
        '--no-save-predictions', 
        action='store_true',
        help='If set, do not save detailed predictions to a CSV file.'
    )
    parser.add_argument(
        '--generate-report', 
        action='store_true',
        help='If set, generate a comprehensive visual evaluation report.'
    )
    parser.add_argument(
        '--threshold-analysis', 
        type=str, 
        default=None,
        help='Name of a single pathology for which to generate a threshold analysis plot.'
    )
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Override the output directory if provided via command line
    if args.output_dir:
        config.paths.output_dir = Path(args.output_dir)
        
    # Setup logging to use the (potentially overridden) output directory
    setup_logging(log_file_path=config.paths.output_dir / "evaluation.log")
    logger = logging.getLogger(__name__)
    
    if args.output_dir:
        logger.info(f"Output directory overridden by CLI: {config.paths.output_dir}")

    # Create ModelEvaluator instance
    try:
        evaluator = ModelEvaluator(config, args.model, device=args.device)
    except Exception as e:
        logger.critical(f"Failed to initialize ModelEvaluator: {e}", exc_info=True)
        sys.exit(1)
    
    # Create ModelEvaluator instance
    try:
        evaluator = ModelEvaluator(config, args.model, args.device)
    except Exception as e:
        logger.critical(f"Failed to initialize ModelEvaluator: {e}", exc_info=True) # Log critical error
        sys.exit(1) # Exit if evaluator cannot be created
    
    # Run evaluation on the specified dataset
    try:
        results = evaluator.evaluate_dataset(
            dataset_type=args.dataset,
            save_predictions=not args.no_save_predictions # Save if flag is not set
        )
    except Exception as e:
        logger.critical(f"Error during dataset evaluation: {e}", exc_info=True) # Log critical error
        sys.exit(1) # Exit on evaluation failure
    
    # Generate visual report if requested
    if args.generate_report:
        try:
            evaluator.generate_evaluation_report(results)
        except Exception as e:
            logger.error(f"Error generating visual report: {e}", exc_info=True) # Log error
    
    # Perform threshold analysis for a specific pathology if requested
    if args.threshold_analysis:
        try:
            # Ensure the pathology name is valid before proceeding
            if args.threshold_analysis not in config.PATHOLOGY_COLUMNS:
                logger.error(f"Invalid pathology name for threshold analysis: '{args.threshold_analysis}'. "
                             f"Available: {config.PATHOLOGY_COLUMNS}")
            else:
                evaluator.compare_thresholds(results, args.threshold_analysis)
        except Exception as e:
            logger.error(f"Error during threshold analysis for '{args.threshold_analysis}': {e}", exc_info=True)
    
    # Print summary of main metrics to console
    metrics_summary = results.get('metrics', {}) # Get metrics dict, empty if not found
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY (from evaluate.py)")
    logger.info("="*60)
    logger.info(f"Model Evaluated: {args.model}")
    logger.info(f"Evaluated Dataset: {args.dataset.title()}")
    logger.info(f"Number of Samples: {results.get('num_samples', 'N/A')}")
    logger.info("-" * 60)
    logger.info("Overall Metrics:")
    for metric_key in ['roc_auc_macro', 'roc_auc_micro', 'f1_macro', 'f1_micro', 'accuracy', 'precision_macro', 'recall_macro']:
        value = metrics_summary.get(metric_key, float('nan')) # Default to NaN if metric missing
        logger.info(f"  {metric_key.replace('_', ' ').title():<20}: {value:.4f}")
    
    # Log top/bottom performing pathologies by AUC
    if metrics_summary: # Check if metrics were computed
        pathology_aucs_list = []
        for p_col in config.PATHOLOGY_COLUMNS:
            auc_val = metrics_summary.get(f"{p_col}_auc", None)
            if auc_val is not None:
                pathology_aucs_list.append((p_col, auc_val))
        
        if pathology_aucs_list: # If any per-pathology AUCs exist
            pathology_aucs_list.sort(key=lambda x: x[1], reverse=True) # Sort by AUC desc
            
            logger.info("-" * 60)
            logger.info("Top 5 Pathologies by AUC:")
            for i, (pathology, auc) in enumerate(pathology_aucs_list[:5]):
                logger.info(f"  {i+1}. {pathology:<30}: {auc:.4f}")
            
            if len(pathology_aucs_list) > 5: # Only show bottom if more than 5
                logger.info("\nBottom 5 Pathologies by AUC:")
                # Iterate from end of list for bottom performers
                for i, (pathology, auc) in enumerate(pathology_aucs_list[-5:]):
                    logger.info(f"  {len(pathology_aucs_list)-4+i}. {pathology:<30}: {auc:.4f}")
        else:
            logger.info("Per-pathology AUC scores not available in metrics.")
    else:
        logger.warning("Metrics dictionary is empty, cannot display detailed performance.")

    logger.info("="*60)
    logger.info("Evaluation script completed successfully!")


if __name__ == "__main__":
    main()