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
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.resnet3d import resnet18_3d, resnet34_3d
from data.dataset import CTDataset3D
from data.utils import get_dynamic_image_path
from training.metrics import compute_metrics
from utils.logging_config import setup_logging


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, config: Config, model_path: str, device: str = 'auto'):
        self.config = config
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = self._load_model()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create model architecture
        if self.config.MODEL_TYPE == "resnet3d":
            model = resnet18_3d(
                num_classes=self.config.NUM_PATHOLOGIES,
                use_checkpointing=False  # Disable checkpointing for inference
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.MODEL_TYPE}")
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            self.logger.info("Loaded model weights")
        
        model = model.to(self.device)
        model.eval()
        
        self.logger.info(f"Model loaded successfully on {self.device}")
        return model
    
    def _prepare_dataset(self, dataset_type: str = 'validation') -> Tuple[pd.DataFrame, Path]:
        """Prepare dataset for evaluation"""
        if dataset_type == 'validation':
            volumes_csv = self.config.SELECTED_VALID_VOLUMES_CSV
            labels_csv = self.config.VALID_LABELS_CSV
            img_dir = self.config.VALID_IMG_DIR
        elif dataset_type == 'train':
            volumes_csv = self.config.SELECTED_TRAIN_VOLUMES_CSV
            labels_csv = self.config.TRAIN_LABELS_CSV
            img_dir = self.config.TRAIN_IMG_DIR
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Load and merge data
        volumes_df = pd.read_csv(volumes_csv)[['VolumeName']]
        labels_df = pd.read_csv(labels_csv)
        
        eval_df = pd.merge(volumes_df, labels_df, on='VolumeName', how='inner')
        eval_df[self.config.PATHOLOGY_COLUMNS] = eval_df[self.config.PATHOLOGY_COLUMNS].fillna(0).astype(int)
        
        self.logger.info(f"Loaded {len(eval_df)} samples for {dataset_type} evaluation")
        return eval_df, img_dir
    
    @torch.no_grad()
    def evaluate_dataset(self, dataset_type: str = 'validation', 
                        save_predictions: bool = True) -> Dict:
        """Evaluate model on specified dataset"""
        
        self.logger.info(f"Starting evaluation on {dataset_type} dataset...")
        
        # Prepare dataset
        eval_df, img_dir = self._prepare_dataset(dataset_type)
        
        # Create dataset and dataloader
        eval_dataset = CTDataset3D(
            eval_df, img_dir, self.config.PATHOLOGY_COLUMNS,
            self.config.TARGET_SPACING, self.config.TARGET_SHAPE_DHW,
            self.config.CLIP_HU_MIN, self.config.CLIP_HU_MAX,
            use_cache=self.config.USE_CACHE, cache_dir=self.config.CACHE_DIR,
            augment=False  # No augmentation for evaluation
        )
        
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.config.NUM_WORKERS, 
            pin_memory=self.config.PIN_MEMORY
        )
        
        # Collect predictions and labels
        all_predictions = []
        all_labels = []
        all_volume_names = []
        
        self.model.eval()
        for batch_idx, batch in enumerate(eval_loader):
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            volume_names = batch["volume_name"]
            
            # Forward pass
            outputs = self.model(pixel_values)
            
            # Collect results
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_volume_names.extend(volume_names)
            
            if (batch_idx + 1) % 10 == 0:
                self.logger.info(f"Processed {batch_idx + 1}/{len(eval_loader)} batches")
        
        # Concatenate all results
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        # Convert predictions to probabilities
        probabilities = 1 / (1 + np.exp(-predictions))  # Sigmoid
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels, self.config.PATHOLOGY_COLUMNS)
        
        # Prepare results dictionary
        results = {
            'dataset_type': dataset_type,
            'num_samples': len(labels),
            'metrics': metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'labels': labels,
            'volume_names': all_volume_names,
            'pathology_columns': self.config.PATHOLOGY_COLUMNS
        }
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(results, dataset_type)
        
        self.logger.info(f"Evaluation completed on {len(labels)} samples")
        self.logger.info(f"Overall AUC (macro): {metrics['roc_auc_macro']:.4f}")
        self.logger.info(f"Overall F1 (macro): {metrics['f1_macro']:.4f}")
        
        return results
    
    def _save_predictions(self, results: Dict, dataset_type: str):
        """Save predictions to CSV file"""
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create predictions DataFrame
        predictions_data = []
        
        for i, volume_name in enumerate(results['volume_names']):
            row = {'VolumeName': volume_name}
            
            # Add true labels
            for j, pathology in enumerate(self.config.PATHOLOGY_COLUMNS):
                row[f'{pathology}_true'] = int(results['labels'][i, j])
                row[f'{pathology}_prob'] = float(results['probabilities'][i, j])
                row[f'{pathology}_pred'] = int(results['probabilities'][i, j] > 0.5)
            
            predictions_data.append(row)
        
        predictions_df = pd.DataFrame(predictions_data)
        
        # Save to CSV
        predictions_file = output_dir / f'predictions_{dataset_type}.csv'
        predictions_df.to_csv(predictions_file, index=False)
        self.logger.info(f"Predictions saved to: {predictions_file}")
        
        # Save metrics to JSON
        metrics_file = output_dir / f'metrics_{dataset_type}.json'
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        self.logger.info(f"Metrics saved to: {metrics_file}")
    
    def generate_evaluation_report(self, results: Dict, save_path: Optional[str] = None):
        """Generate comprehensive evaluation report with visualizations"""
        
        if save_path is None:
            save_path = Path(self.config.OUTPUT_DIR) / f"evaluation_report_{results['dataset_type']}.png"
        else:
            save_path = Path(save_path)
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'Model Evaluation Report - {results["dataset_type"].title()} Set', 
                    fontsize=16, fontweight='bold')
        
        # 1. Overall Metrics Bar Chart
        ax = axes[0, 0]
        overall_metrics = ['roc_auc_macro', 'roc_auc_micro', 'f1_macro', 'f1_micro', 
                          'accuracy', 'precision_macro', 'recall_macro']
        metric_values = [results['metrics'][m] for m in overall_metrics]
        metric_labels = [m.replace('_', ' ').title() for m in overall_metrics]
        
        bars = ax.bar(range(len(metric_values)), metric_values, color='skyblue')
        ax.set_xticks(range(len(metric_values)))
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Overall Performance Metrics')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Per-Pathology AUC Scores
        ax = axes[0, 1]
        pathology_aucs = []
        pathology_names = []
        
        for pathology in self.config.PATHOLOGY_COLUMNS:
            auc_key = f"{pathology}_auc"
            if auc_key in results['metrics']:
                pathology_aucs.append(results['metrics'][auc_key])
                pathology_names.append(pathology[:15] + '...' if len(pathology) > 15 else pathology)
        
        if pathology_aucs:
            y_pos = np.arange(len(pathology_names))
            bars = ax.barh(y_pos, pathology_aucs, color='lightgreen')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pathology_names)
            ax.set_xlabel('AUC Score')
            ax.set_title('Per-Pathology AUC Scores')
            ax.set_xlim(0, 1)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, pathology_aucs)):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='left', va='center')
        
        # 3. ROC Curves for top pathologies
        ax = axes[0, 2]
        top_pathologies = sorted(
            [(p, results['metrics'][f"{p}_auc"]) for p in self.config.PATHOLOGY_COLUMNS 
             if f"{p}_auc" in results['metrics']], 
            key=lambda x: x[1], reverse=True
        )[:5]  # Top 5 pathologies by AUC
        
        for pathology, auc_score in top_pathologies:
            pathology_idx = self.config.PATHOLOGY_COLUMNS.index(pathology)
            y_true = results['labels'][:, pathology_idx]
            y_prob = results['probabilities'][:, pathology_idx]
            
            if len(np.unique(y_true)) > 1:  # Only if both classes present
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                ax.plot(fpr, tpr, label=f'{pathology[:15]}... (AUC={auc_score:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (Top 5 Pathologies)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 4. Precision-Recall Curves
        ax = axes[1, 0]
        for pathology, auc_score in top_pathologies:
            pathology_idx = self.config.PATHOLOGY_COLUMNS.index(pathology)
            y_true = results['labels'][:, pathology_idx]
            y_prob = results['probabilities'][:, pathology_idx]
            
            if len(np.unique(y_true)) > 1:
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                ax.plot(recall, precision, label=f'{pathology[:15]}...')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Prediction Distribution
        ax = axes[1, 1]
        all_probs = results['probabilities'].flatten()
        ax.hist(all_probs, bins=50, alpha=0.7, density=True, color='orange')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Predicted Probabilities')
        ax.grid(True, alpha=0.3)
        
        # 6. Calibration Plot
        ax = axes[1, 2]
        all_labels = results['labels'].flatten()
        all_probs = results['probabilities'].flatten()
        
        if len(np.unique(all_labels)) > 1:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                all_labels, all_probs, n_bins=10
            )
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 7. F1 Scores per Pathology
        ax = axes[2, 0]
        pathology_f1s = []
        for pathology in self.config.PATHOLOGY_COLUMNS:
            f1_key = f"{pathology}_f1"
            if f1_key in results['metrics']:
                pathology_f1s.append(results['metrics'][f1_key])
        
        if pathology_f1s:
            y_pos = np.arange(len(pathology_names))
            bars = ax.barh(y_pos, pathology_f1s, color='lightcoral')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pathology_names)
            ax.set_xlabel('F1 Score')
            ax.set_title('Per-Pathology F1 Scores')
            ax.set_xlim(0, 1)
        
        # 8. Class Distribution
        ax = axes[2, 1]
        positive_counts = results['labels'].sum(axis=0)
        negative_counts = len(results['labels']) - positive_counts
        
        x = np.arange(len(self.config.PATHOLOGY_COLUMNS))
        width = 0.35
        
        ax.bar(x - width/2, positive_counts, width, label='Positive', color='green', alpha=0.7)
        ax.bar(x + width/2, negative_counts, width, label='Negative', color='red', alpha=0.7)
        
        ax.set_xlabel('Pathology')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels([p[:10] + '...' if len(p) > 10 else p for p in self.config.PATHOLOGY_COLUMNS], 
                          rotation=45, ha='right')
        ax.legend()
        
        # 9. Summary Statistics
        ax = axes[2, 2]
        ax.axis('off')
        
        summary_text = f"Evaluation Summary\n"
        summary_text += "=" * 30 + "\n\n"
        summary_text += f"Dataset: {results['dataset_type'].title()}\n"
        summary_text += f"Samples: {results['num_samples']}\n"
        summary_text += f"Model: {self.config.MODEL_TYPE}\n\n"
        summary_text += f"Overall Metrics:\n"
        summary_text += f"  ROC AUC (Macro): {results['metrics']['roc_auc_macro']:.4f}\n"
        summary_text += f"  F1 Score (Macro): {results['metrics']['f1_macro']:.4f}\n"
        summary_text += f"  Accuracy: {results['metrics']['accuracy']:.4f}\n"
        summary_text += f"  Precision (Macro): {results['metrics']['precision_macro']:.4f}\n"
        summary_text += f"  Recall (Macro): {results['metrics']['recall_macro']:.4f}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Evaluation report saved to: {save_path}")
        plt.close()
    
    def compare_thresholds(self, results: Dict, pathology: str, save_path: Optional[str] = None):
        """Analyze performance at different thresholds for a specific pathology"""
        
        if pathology not in self.config.PATHOLOGY_COLUMNS:
            raise ValueError(f"Pathology '{pathology}' not found in configuration")
        
        pathology_idx = self.config.PATHOLOGY_COLUMNS.index(pathology)
        y_true = results['labels'][:, pathology_idx]
        y_prob = results['probabilities'][:, pathology_idx]
        
        if len(np.unique(y_true)) <= 1:
            self.logger.warning(f"Only one class present for {pathology}, skipping threshold analysis")
            return
        
        # Test different thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        metrics_at_thresholds = {
            'threshold': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            'accuracy': []
        }
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate metrics
            tn = np.sum((y_true == 0) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            
            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy = accuracy_score(y_true, y_pred)
            
            metrics_at_thresholds['threshold'].append(threshold)
            metrics_at_thresholds['f1'].append(f1)
            metrics_at_thresholds['precision'].append(precision)
            metrics_at_thresholds['recall'].append(recall)
            metrics_at_thresholds['specificity'].append(specificity)
            metrics_at_thresholds['accuracy'].append(accuracy)
        
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(thresholds, metrics_at_thresholds['f1'], 'o-', label='F1 Score', linewidth=2)
        ax.plot(thresholds, metrics_at_thresholds['precision'], 's-', label='Precision', linewidth=2)
        ax.plot(thresholds, metrics_at_thresholds['recall'], '^-', label='Recall', linewidth=2)
        ax.plot(thresholds, metrics_at_thresholds['specificity'], 'd-', label='Specificity', linewidth=2)
        ax.plot(thresholds, metrics_at_thresholds['accuracy'], 'v-', label='Accuracy', linewidth=2)
        
        # Find optimal threshold (max F1)
        best_f1_idx = np.argmax(metrics_at_thresholds['f1'])
        best_threshold = thresholds[best_f1_idx]
        best_f1 = metrics_at_thresholds['f1'][best_f1_idx]
        
        ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Optimal Threshold: {best_threshold:.2f} (F1={best_f1:.3f})')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'Performance vs Threshold - {pathology}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        if save_path is None:
            save_path = Path(self.config.OUTPUT_DIR) / f"threshold_analysis_{pathology.replace(' ', '_')}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Threshold analysis saved to: {save_path}")
        plt.close()
        
        return metrics_at_thresholds, best_threshold


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate CT 3D Classifier')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='validation',
                       choices=['validation', 'train'],
                       help='Dataset to evaluate on')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for evaluation')
    parser.add_argument('--no-save-predictions', action='store_true',
                       help='Do not save predictions to CSV')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive evaluation report')
    parser.add_argument('--threshold-analysis', type=str, default=None,
                       help='Pathology name for threshold analysis')
    parser.add_argument('--config-override', type=str, default=None,
                       help='JSON file to override config parameters')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('evaluation.log')
    
    # Load configuration
    config = Config()
    
    # Override output directory if specified
    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)
    
    # Override config parameters if specified
    if args.config_override and Path(args.config_override).exists():
        with open(args.config_override, 'r') as f:
            override_params = json.load(f)
        for key, value in override_params.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Overrode config.{key} = {value}")
    
    # Create evaluator
    evaluator = ModelEvaluator(config, args.model, args.device)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        dataset_type=args.dataset,
        save_predictions=not args.no_save_predictions
    )
    
    # Generate report if requested
    if args.generate_report:
        evaluator.generate_evaluation_report(results)
    
    # Run threshold analysis if requested
    if args.threshold_analysis:
        evaluator.compare_thresholds(results, args.threshold_analysis)
    
    # Print summary
    metrics = results['metrics']
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {results['num_samples']}")
    logger.info(f"Model: {args.model}")
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    logger.info(f"  ROC AUC (Micro): {metrics['roc_auc_micro']:.4f}")
    logger.info(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
    
    # Show top/bottom performing pathologies
    pathology_aucs = [(p, metrics[f"{p}_auc"]) for p in config.PATHOLOGY_COLUMNS 
                     if f"{p}_auc" in metrics]
    pathology_aucs.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"\nTop 5 Pathologies by AUC:")
    for i, (pathology, auc) in enumerate(pathology_aucs[:5]):
        logger.info(f"  {i+1}. {pathology}: {auc:.4f}")
    
    logger.info(f"\nBottom 5 Pathologies by AUC:")
    for i, (pathology, auc) in enumerate(pathology_aucs[-5:]):
        logger.info(f"  {len(pathology_aucs)-4+i}. {pathology}: {auc:.4f}")
    
    logger.info("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()