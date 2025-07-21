# Training Module (`/src/training`)

This directory contains the core logic for training the 3D CT scan classifier. It orchestrates the entire training process, from data loading and model instantiation to epoch-based training, validation, metric computation, and checkpointing.

## Core Components

The training module is composed of three main files:

1.  [**`trainer.py`**](#trainerpy): The central orchestrator of the training pipeline.
2.  [**`metrics.py`**](#metricspy): A dedicated module for computing a wide range of performance metrics.
3.  [**`utils.py`**](#utilspy): Contains essential helper classes and functions for the training process, such as early stopping and checkpoint management.

---

### `trainer.py`

This is the main entry point for any training run. It houses the `train_model` function, which manages the entire lifecycle of a training session.

**Key Responsibilities:**

* **Orchestration**: The `train_model` function initializes all necessary components, including the model, optimizer, loss function, data loaders, and experiment tracking (Weights & Biases).
* **Data Pipeline**: It sets up the complete data pipeline using MONAI, integrating `PersistentDataset` and `CacheDataset` for efficient, multi-stage caching (disk and RAM) to accelerate data loading. It also handles the application of on-the-fly data augmentation.
* **Model Creation**: Includes a `create_model` factory function to dynamically instantiate different 3D architectures (`ResNet3D`, `DenseNet3D`, `ViT3D`) based on the project configuration.
* **Training and Validation Loops**: Contains the `train_epoch` and `validate_epoch` functions, which handle the forward/backward passes, loss calculation, gradient accumulation, and metric collection for each epoch.
* **Checkpointing**: Manages saving the best model, the final model, and the last checkpoint for resumability.
* **Reporting**: At the end of training, it calls the reporting module to generate a comprehensive visual and data-based summary of the run.

### `metrics.py`

This module is responsible for evaluating the model's performance on the validation set at the end of each epoch.

**Key Functions:**

* **`compute_metrics`**: Takes the raw logits from the model and the ground truth labels to calculate a comprehensive dictionary of metrics.
* **Multi-Label Metrics**: It calculates standard multi-label classification metrics, including:
    * **Overall Scores**: `roc_auc_macro`, `roc_auc_micro`, `f1_macro`, `precision_macro`, `recall_macro`, and overall `accuracy`.
    * **Per-Pathology Scores**: It computes AUC, F1-score, precision, recall, sensitivity, and specificity for each individual pathology defined in the configuration.
    * **Specificity**: Manually calculates macro and micro-averaged specificity, which are not standard in scikit-learn.

### `utils.py`

This script provides crucial utilities that support the training loop in `trainer.py`.

**Key Components:**

* **`EarlyStopping`**: A flexible class that monitors a specified metric (e.g., `roc_auc_macro`) and stops the training process if the metric does not improve for a given number of epochs (`patience`). This helps prevent overfitting and saves computational resources.
* **Checkpoint Management**:
    * `save_checkpoint`: Saves the model's state dictionary, optimizer state, and other training metadata to a `.pth` file.
    * `load_checkpoint`: Loads a checkpoint to resume training, restoring the model weights, optimizer state, and scaler state.
    * `find_latest_checkpoint`: A helper function that automatically finds the most recent checkpoint file in an output directory, simplifying the process of resuming an interrupted run.