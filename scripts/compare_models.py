#!/usr/bin/env python3
"""
Compare multiple trained models on the same dataset
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path to allow imports from `src`
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import load_config
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logging_config import setup_logging


from scripts.evaluate import ModelEvaluator

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import Config
from utils.logging_config import setup_logging

def compare_models(model_paths: list, model_names: list, config, dataset_type: str = 'validation'):
    """Compare multiple models on the same dataset"""
    
    logger = setup_logging(log_file_path=config.paths.output_dir / "model_comparison.log")
    results = {}
    
    # Evaluate each model
    for model_path, model_name in zip(model_paths, model_names):
        logger.info(f"Evaluating {model_name}...")
        evaluator = ModelEvaluator(config, model_path)
        model_results = evaluator.evaluate_dataset(dataset_type, save_predictions=False)
        results[model_name] = model_results['metrics']
    
    # Create comparison DataFrame
    comparison_data = []
    overall_metrics = ['roc_auc_macro', 'roc_auc_micro', 'f1_macro', 'f1_micro', 
                      'accuracy', 'precision_macro', 'recall_macro']
    
    for model_name in model_names:
        row = {'Model': model_name}
        for metric in overall_metrics:
            row[metric] = results[model_name][metric]
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison results
    output_path = config.paths.output_dir / 'model_comparison.csv'
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"Model comparison saved to: {output_path}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison Report', fontsize=16, fontweight='bold')
    
    # Overall metrics comparison
    ax = axes[0, 0]
    x = np.arange(len(model_names))
    width = 0.1
    
    for i, metric in enumerate(overall_metrics):
        values = [results[name][metric] for name in model_names]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics Comparison')
    ax.set_xticks(x + width * len(overall_metrics) / 2)
    ax.set_xticklabels(model_names)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    
    # Per-pathology AUC heatmap
    ax = axes[0, 1]
    pathology_data = []
    for model_name in model_names:
        row = []
        for pathology in config.pathologies.columns:
            auc_key = f"{pathology}_auc"
            row.append(results[model_name].get(auc_key, 0.0))
        pathology_data.append(row)
    
    pathology_data = np.array(pathology_data)
    im = ax.imshow(pathology_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(config.pathologies.columns)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels([p[:15] + '...' if len(p) > 15 else p for p in config.pathologies.columns], 
                      rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    ax.set_title('Per-Pathology AUC Scores')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Performance differences
    ax = axes[1, 0]
    if len(model_names) == 2:
        # Show difference between two models
        model1, model2 = model_names
        differences = []
        pathology_names = []
        
        for pathology in config.pathologies.columns:
            auc_key = f"{pathology}_auc"
            if auc_key in results[model1] and auc_key in results[model2]:
                diff = results[model2][auc_key] - results[model1][auc_key]
                differences.append(diff)
                pathology_names.append(pathology[:15] + '...' if len(pathology) > 15 else pathology)
        
        colors = ['green' if d > 0 else 'red' for d in differences]
        y_pos = np.arange(len(pathology_names))
        bars = ax.barh(y_pos, differences, color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pathology_names)
        ax.set_xlabel(f'AUC Difference ({model2} - {model1})')
        ax.set_title('Performance Differences')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
    
    # Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Find best model for each metric
    summary_text = "Best Model per Metric:\n"
    summary_text += "=" * 25 + "\n\n"
    
    for metric in overall_metrics:
        values = [(name, results[name][metric]) for name in model_names]
        best_model, best_value = max(values, key=lambda x: x[1])
        summary_text += f"{metric.replace('_', ' ').title()}:\n"
        summary_text += f"  {best_model}: {best_value:.4f}\n\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
           fontsize=12, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # Save comparison plot
    plot_path = config.paths.output_dir / 'model_comparison_report.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to: {plot_path}")
    plt.close()
    
    return comparison_df

def main():
    parser = argparse.ArgumentParser(description='Compare multiple trained models')
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument('--models', nargs='+', required=True,
                       help='Paths to model checkpoints to compare.')
    parser.add_argument('--names', nargs='+', required=True,
                       help='Names for the models, one for each path.')
    parser.add_argument('--dataset', type=str, default='validation',
                       choices=['validation', 'train'],
                       help="Dataset split to use for comparison.")
    parser.add_argument('--output-dir', type=str, default=None,
                       help="Override the output directory from the config file.")
    
    args = parser.parse_args()
    
    if len(args.models) != len(args.names):
        raise ValueError("The number of model paths and model names must be equal.")
    
    # Load the base configuration from the YAML file
    config = load_config(args.config)
    
    # Override the output directory if specified on the command line
    if args.output_dir:
        config.paths.output_dir = Path(args.output_dir)
    
    # Run the comparison
    comparison_df = compare_models(args.models, args.names, config, args.dataset)
    
    # Print the results to the console
    print("\n--- Model Comparison Results ---")
    print(comparison_df.to_string())

if __name__ == "__main__":
    main()