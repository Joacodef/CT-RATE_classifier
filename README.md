# 3D CT Scan Pathology Classifier

This project provides a comprehensive system for classifying pathologies in 3D CT (Computed Tomography) scans using deep learning. It is built with PyTorch and leverages architectures like 3D ResNet, 3D DenseNet, and 3D Vision Transformers (ViT). The framework is designed for production-readiness, with a focus on modularity, configurability, and extensive testing.

## Features

* **Multiple Model Architectures**: Choose between 3D versions of ResNet, DenseNet, and Vision Transformers.
* **End-to-End Workflow**: Scripts for preprocessing, training, inference, and hyperparameter optimization are provided.
* **Advanced Preprocessing**: Utilizes the MONAI library for robust medical image processing, including orientation correction, voxel spacing normalization, intensity clipping, and resizing.
* **Configuration-Driven**: All aspects of the project, from file paths to model hyperparameters, are controlled via a central YAML configuration file.
* **Experiment Tracking**: Integrated with Weights & Biases (W&B) for logging metrics, visualizing results, and managing experiments.
* **Hyperparameter Optimization**: Includes a script to perform automated hyperparameter searches using Optuna.
* **Resumable Training**: Training can be paused and resumed from the last saved checkpoint.
* **Comprehensive Reporting**: Generates detailed reports at the end of training, including learning curves, performance metrics, and per-pathology analysis.
* **Testing**: A suite of unit and integration tests to ensure code quality and reliability.

## Repository Structure

```
.
├── .github/workflows         # GitHub Actions for CI/CD
├── configs/                  # Configuration files
├── data/                     # (Git-ignored) Raw and processed data
├── notebooks/                # (Git-ignored) Jupyter notebooks for exploration
├── output/                   # (Git-ignored) Saved models, logs, and reports
├── requirements.txt          # Project dependencies
├── scripts/                  # High-level scripts for core tasks
│   ├── preprocess_and_save.py # Preprocesses the entire dataset
│   ├── train.py               # Main training script
│   ├── inference.py           # Runs inference on new data
│   └── optimize_hyperparams.py# Performs hyperparameter search
├── src/                      # Source code for the project
│   ├── config/               # Configuration loading and parsing
│   ├── data/                 # Dataset, dataloader, and preprocessing logic
│   ├── evaluation/           # Evaluation metrics and reporting
│   ├── models/               # Model architectures (ResNet, DenseNet, ViT)
│   ├── training/             # Core training and validation loops
│   └── utils/                # Utility functions
└── tests/                    # Unit and integration tests
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ct_classifier
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project requires specific versions of PyTorch that are compatible with CUDA. It is recommended to install these first.
    ```bash
    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
    ```
    Then, install the remaining packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The entire project is controlled by two main configuration files:

1.  **.env**: This file should be created at the project root to store environment-specific paths. It is ignored by Git.

    **Example `.env` file:**
    ```
    BASE_PROJECT_DIR=/path/to/your/project/ct_classifier
    TRAIN_IMG_DIR=/path/to/your/training_images
    VALID_IMG_DIR=/path/to/your/validation_images
    CACHE_DIR=/path/to/your/cache
    DATA_DIR=${BASE_PROJECT_DIR}/data
    ```

2.  **config.yaml**: A copy of `configs/config_example.yaml` should be made and customized for your specific experiment. This file defines model type, hyperparameters, image processing parameters, and more.

    ```bash
    cp configs/config_example.yaml configs/my_experiment.yaml
    ```
    Now, you can modify `configs/my_experiment.yaml` as needed.

## Quickstart

### 1. Preprocessing Data

If you need to preprocess the entire dataset before training (e.g., to save preprocessed volumes to disk), use the `preprocess_and_save.py` script.

```bash
python scripts/preprocess_and_save.py \
    --config configs/my_experiment.yaml \
    --dataframe_path /path/to/your/dataset.csv \
    --img_dir /path/to/raw/images \
    --output_dir /path/to/processed/images
```

### 2. Training a Model

To start a new training run, use the `train.py` script with your configuration file.

```bash
python scripts/train.py --config configs/my_experiment.yaml
```

You can override specific model parameters from the command line:

```bash
python scripts/train.py --config configs/my_experiment.yaml --model-type vit3d --model-variant base
```

To resume training from the latest checkpoint in the output directory specified in your config:

```bash
python scripts/train.py --config configs/my_experiment.yaml --resume
```

### 3. Running Inference

To run inference on a single volume or a directory of volumes, use the `inference.py` script.

**Single Volume:**
```bash
python scripts/inference.py \
    --config /path/to/your/output/from_training/config.yaml \
    --model /path/to/your/output/from_training/best_model.pth \
    --input /path/to/single/volume.nii.gz \
    --output /path/to/your/results/single_result
```
This will print the predictions to the console and save a detailed JSON file.

**Batch of Volumes:**
```bash
python scripts/inference.py \
    --config /path/to/your/output/from_training/config.yaml \
    --model /path/to/your/output/from_training/best_model.pth \
    --input /path/to/directory_of_volumes/ \
    --output /path/to/your/results/batch_results
```
This will generate a CSV file with predictions for all volumes in the input directory.

### 4. Hyperparameter Optimization

To run a hyperparameter optimization study using Optuna, you first need to create stratified subsets of your training data.

**Create Subsets:**
```bash
python scripts/create_training_subsets.py \
    --config configs/my_experiment.yaml \
    --input-file /path/to/your/full_train_split.csv \
    --fractions 0.5 0.2 0.05
```

**Run Optimization:**
```bash
python scripts/optimize_hyperparams.py \
    --config configs/my_experiment.yaml \
    --n-trials 100 \
    --study-name "vit3d-optimization-study"
```

## Testing

To run the full suite of unit and integration tests, use `pytest`:

```bash
pytest
```
