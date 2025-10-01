# Training Module (`/src/training`)

This directory contains the logic for training the 3D CT scan classifier. It orchestrates the training process, from data loading and model instantiation to epoch-based training, validation, metric computation, and checkpointing.

## Core Components

The training module is composed of two main files:

1.  `trainer.py`: The orchestrator of the training pipeline.
2.  `utils.py`: Contains helper classes and functions for the training process, such as early stopping and checkpoint management.

-----

### `trainer.py`

This file houses the `train_model` function, which manages the lifecycle of a training session.

**Key Responsibilities:**

  * **Orchestration**: The `train_model` function initializes components, including the model, optimizer, loss function, data loaders, and experiment tracking with Weights & Biases.
  * **Data Pipeline**: It sets up the data pipeline using MONAI, integrating `PersistentDataset` and `CacheDataset` for multi-stage caching (disk and RAM).
  * **On-the-Fly GPU Augmentation**: Augmentations are defined as a separate MONAI `Compose` pipeline that is passed directly to the `train_epoch` function. They are applied to each batch on the GPU, ensuring high performance and no interference with the caching mechanism.
  * **Model Creation**: Includes a `create_model` factory function to instantiate different 3D architectures (`ResNet3D`, `DenseNet3D`, `ViT3D`) based on the project configuration.
  * **Training and Validation Loops**: Contains the `train_epoch` and `validate_epoch` functions.
      * `train_epoch` handles the forward/backward passes, loss calculation, and gradient accumulation.
      * `validate_epoch` collects model predictions and uses MONAI's metric classes (`ROCAUCMetric`, `FBetaScore`) to compute performance on the validation set.
  * **Checkpointing**: Manages saving the best model, the final model, and the last checkpoint.
  * **Reporting**: At the end of training, it calls the reporting module to generate a summary of the run.

### `utils.py`

This script provides utilities that support the training loop in `trainer.py`.

**Key Components:**

  * **`EarlyStopping`**: A class that monitors a specified metric and stops the training process if the metric does not improve for a given number of epochs.
  * **Checkpoint Management**:
      * `save_checkpoint`: Saves the model's state dictionary, optimizer state, and other training metadata to a `.pth` file.
      * `load_checkpoint`: Loads a checkpoint to resume training, restoring the model weights, optimizer state, and scaler state.
      * `find_latest_checkpoint`: A helper function that finds the most recent checkpoint file in an output directory.