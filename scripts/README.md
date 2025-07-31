# Scripts Module (`/scripts`)

This directory contains the high-level, executable scripts that drive the machine learning pipeline, including training, inference, and hyperparameter optimization.

The recommended workflow is:

1.  **Prepare Data**: Use the scripts in `data_preparation/` to download, filter, and create k-fold splits of your dataset.
2.  **Generate Cache**: Run `cache_management/generate_cache.py` to pre-process and cache the dataset.
3.  **Optimize Hyperparameters**: Optionally, run `optimize_hyperparams.py` on data subsets to find optimal settings.
4.  **Train Model**: Use `train.py` to train a model on a specific data fold.
5.  **Run Inference**: Use `inference.py` with a trained model to make predictions on new data.

-----

## Directory Overviews

### Data Preparation (`/scripts/data_preparation`)

This directory contains scripts for dataset acquisition and organization, including downloading, filtering, and creating stratified data splits.

### Cache Management (`/scripts/cache_management`)

This directory holds utilities for creating and managing the MONAI `PersistentDataset` cache.

-----

## Core Scripts

### `train.py`

This script runs the training pipeline for a specific cross-validation fold. It loads a base configuration file and can be customized with command-line arguments to specify the fold, override model architecture, or resume from a checkpoint. The script dynamically adjusts the data paths and output directory for the selected fold.

**Usage:**

```bash
# Train on fold 0 using the settings in config.yaml
python scripts/train.py \
    --config configs/config.yaml \
    --fold 0

# Resume training automatically from the last checkpoint for fold 1
python scripts/train.py \
    --config configs/config.yaml \
    --fold 1 \
    --resume
```

### `inference.py`

This script runs inference on CT volumes using a trained model and a consistent MONAI-based preprocessing pipeline. It can process a single NIfTI file or an entire directory of them, generating predictions and probabilities for each pathology.

**Usage:**

```bash
# Run inference on a single volume
python scripts/inference.py \
    --config output/fold_0/config.yaml \
    --model output/fold_0/best_model.pth \
    --input /path/to/volume.nii.gz \
    --output /path/to/results

# Run inference on a directory of volumes
python scripts/inference.py \
    --config output/fold_0/config.yaml \
    --model output/fold_0/best_model.pth \
    --input /path/to/volume_directory/ \
    --output /path/to/batch_results
```

### `optimize_hyperparams.py`

This script uses the Optuna framework to perform hyperparameter optimization. It defines a search space for parameters like model architecture, loss function, and learning rate. The script supports staged optimization, where trials are first run on smaller data subsets to quickly discard unpromising configurations. It also integrates with pruners to stop unpromising trials early.

**Usage:**

```bash
python scripts/optimize_hyperparams.py \
    --config configs/config.yaml \
    --n-trials 100 \
    --study-name "resnet3d-optimization" \
    --storage-db "study.db"
```