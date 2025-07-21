````markdown
# Scripts Module (`/scripts`)

This directory contains the high-level, executable scripts that drive the entire machine learning pipeline. These scripts are the primary user-facing tools for data preparation, model training, hyperparameter optimization, and inference.

## Recommended Workflow

The scripts are designed to be run in a logical sequence to manage the lifecycle of a project. A typical workflow would be:

1.  **Initial Data Filtering**:
    * `create_filtered_dataset.py`: (Run once) Process raw metadata and exclusion lists to generate a master list of volumes for the project.
2.  **Dataset Setup & Verification**:
    * `verify_dataset.py`: Check for missing CT scan files and download them from the Hugging Face repository if needed.
3.  **Data Splitting**:
    * `create_kfold_splits.py`: Create stratified, grouped cross-validation splits to prevent data leakage and ensure balanced folds.
4.  **Preprocessing & Caching**:
    * `verify_dataset.py --generate-cache`: (Long-running process) Pre-process and cache the entire dataset to disk according to the current configuration. This dramatically speeds up subsequent training runs.
5.  **Model Training**:
    * `train.py`: Run a training session on a specific fold, with options to resume from checkpoints and override model parameters.
6.  **Inference**:
    * `inference.py`: Use a trained model checkpoint to run predictions on new, unseen CT volumes.

---

## Detailed Script Descriptions

### Data Preparation

#### `create_filtered_dataset.py`
This script performs the initial data cleaning by taking all volume metadata and applying a set of exclusion rules (e.g., removing brain scans, scans with missing data). It produces a single `filtered_master_list.csv` that serves as the basis for all subsequent data splits.

**Usage:**
```bash
# This script is typically run once after initial data setup.
# Paths are hardcoded and should be reviewed before running.
python scripts/create_filtered_dataset.py
````

-----

#### `verify_dataset.py`

This script serves two critical purposes: downloading the dataset and pre-caching it.

1.  **Download Mode**: It checks which files from your dataset CSV are missing from your local image directory and downloads them from the specified Hugging Face repository.
2.  **Cache Generation Mode**: It applies the entire preprocessing pipeline (resizing, resampling, normalization, etc.) to every volume in the dataset and saves the resulting tensors to a cache directory. This is computationally expensive but only needs to be done once per preprocessing configuration.

**Usage:**

```bash
# To check for and download missing files
python scripts/verify_dataset.py --config configs/config.yaml

# To pre-process and cache the entire dataset (can be sharded for parallel execution)
python scripts/verify_dataset.py --config configs/config.yaml --generate-cache --num-workers 8
```

-----

#### `create_kfold_splits.py`

This script creates robust, stratified, and grouped k-fold cross-validation splits. It ensures that all scans from a single patient remain in the same fold (train or valid) to prevent data leakage, and that the distribution of pathologies is balanced across all folds.

**Usage:**

```bash
python scripts/create_kfold_splits.py \
    --config configs/config.yaml \
    --n-splits 5 \
    --output-dir "splits/kfold_5"
```

-----

### Training & Inference

#### `train.py`

This is the main script for training a model. It can train on a specific cross-validation fold, override model parameters from the config, and resume training from checkpoints.

**Usage:**

```bash
# Train fold 0 using the specified config
python scripts/train.py --config configs/config.yaml --fold 0

# Train with a different model, overriding the config
python scripts/train.py --config configs/config.yaml --fold 0 --model-type vit3d --model-variant small

# Resume the last run for fold 0 automatically
python scripts/train.py --config configs/config.yaml --fold 0 --resume
```

-----

#### `inference.py`

Use this script to make predictions on new data using a trained model. It can process a single `.nii.gz` file or a directory of them. It ensures the exact same preprocessing transforms from training are applied.

**Usage:**

```bash
# Run inference on a single volume
python scripts/inference.py \
    --config output/fold_0/config.yaml \
    --model output/fold_0/best_model.pth \
    --input /path/to/new_volume.nii.gz \
    --output /path/to/results/single_pred

# Run inference on a directory of volumes
python scripts/inference.py \
    --config output/fold_0/config.yaml \
    --model output/fold_0/best_model.pth \
    --input /path/to/new_volumes/ \
    --output /path/to/results/batch_preds
```

-----

### Hyperparameter Optimization

#### `create_training_subsets.py`

This script prepares the data for efficient hyperparameter optimization. It creates smaller, stratified, and nested subsets of the training data (e.g., 5%, 20%, 50%). The optimization script uses these subsets to quickly evaluate many trials on smaller data before promoting the best ones to larger datasets.

**Usage:**

```bash
python scripts/create_training_subsets.py \
    --config configs/config.yaml \
    --input-file data/splits/kfold_5/train_fold_0.csv \
    --fractions 0.5 0.2 0.05
```

-----

#### `optimize_hyperparams.py`

This script uses the Optuna framework to perform an automated search for the best hyperparameters. It uses the subsets created by `create_training_subsets.py` for a staged optimization approach, saving time and resources.

**Usage:**

```bash
python scripts/optimize_hyperparams.py \
    --config configs/config.yaml \
    --n-trials 100 \
    --study-name "vit3d-lr-optimization"
```

-----

### Maintenance

#### `verify_cache_integrity.py`

A utility script to scan a MONAI cache directory for corrupted files. This can be useful if a caching process was interrupted. It can identify unreadable files and optionally delete them.

**Usage:**

```bash
# Do a dry-run to find corrupted files
python scripts/verify_cache_integrity.py --config configs/config.yaml

# Find and delete corrupted files
python scripts/verify_cache_integrity.py --config configs/config.yaml --fix
```

```
```