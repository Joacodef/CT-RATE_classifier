# Evaluation Module (`/src/evaluation`)

This directory is responsible for generating comprehensive reports that summarize the results of a training run. It provides both detailed data-driven files and rich visualizations to allow for a thorough analysis of the model's performance over time.

## Core Components

The primary component of this module is the `reporting.py` script.

---

### `reporting.py`

This file contains the functions needed to create final training reports after the training loop is complete. It is called by the `trainer.py` script and uses the training history (losses and metrics for each epoch) to generate its outputs.

**Key Functions:**

* **`generate_final_report`**: This is the main function for creating a multi-faceted visual report. Using `matplotlib`, it generates a single figure containing several plots to provide a complete overview of the training process. The generated report includes:
    * **Training vs. Validation Loss**: A line plot showing the loss curves for both training and validation sets over all epochs.
    * **Performance Metrics Over Time**: Line plots for key metrics like ROC AUC (macro and micro), F1-Score, Precision, and Recall.
    * **Per-Pathology AUC Scores**: A horizontal bar chart displaying the final AUC score for each individual pathology.
    * **Learning Rate Schedule**: A plot showing how the learning rate changed across epochs.
    * **Best Epoch Summary**: A text box highlighting the key performance metrics from the best-performing epoch.
    * **Performance Heatmap**: A detailed heatmap showing multiple per-pathology metrics (AUC, F1, precision, recall, sensitivity, specificity) for the best epoch.
    
    This visual report is saved in both **`.png`** and high-quality **`.pdf`** formats.

* **`generate_csv_report`**: This function generates data-based reports for detailed analysis and record-keeping. It produces two files:
    * **`training_metrics_detailed.csv`**: A CSV file containing all metrics for every epoch of the training run. This is useful for custom plotting or in-depth analysis.
    * **`training_summary.json`**: A JSON file that provides a high-level summary of the training run, including the best epoch, final metrics, and key configuration parameters like model type, batch size, and learning rate.