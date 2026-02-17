# Scripts Module (`/scripts`)

This directory contains the high-level, executable scripts that drive the machine learning pipeline, including training, inference, and hyperparameter optimization.

The recommended workflow is:

1.  **Prepare Data**: Use the scripts in `data_preparation/` to download, filter, and prepare fixed train/valid splits.
2.  **Generate Cache**: Run `cache_management/generate_cache.py` to pre-process and cache the dataset.
3.  **Optimize Hyperparameters**: Optionally, run `optimize_hyperparams.py` on data subsets to find optimal settings.
4.  **Train Model**: Use `train.py` to train a model using the configured data split.
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

This script runs the training pipeline. It loads a base configuration file and can be customized with command-line arguments to override model architecture, workflow mode, or resume from a checkpoint.

**Output behavior:**

- New runs are created directly under `paths.output_dir` as timestamped folders (for example: `run_YYYYMMDD-HHMMSS_resnet3d_18`).
- `--resume` (without a path) searches the output root for the latest run and resumes from the latest checkpoint found.

**W&B naming behavior:**

- If Weights & Biases is enabled, run names are auto-generated from key config fields (model, split, workflow, core hyperparameters, shape/classes, and optimization flags) plus a short hash suffix.

**Usage:**

```bash
# Train using the split paths defined in config.yaml
python scripts/train.py \
    --config configs/config.yaml

# Resume training automatically from the last checkpoint
python scripts/train.py \
    --config configs/config.yaml \
    --resume
```

### `inference.py`

This script runs inference on CT volumes using a trained model and a consistent MONAI-based preprocessing pipeline. It can process a single NIfTI file or an entire directory of them, generating predictions and probabilities for each pathology.

**Usage:**

```bash
# Run inference on a single volume
python scripts/inference.py \
    --config /path/to/configs/config.yaml \
    --model /path/to/output/run_YYYYMMDD-HHMMSS_resnet3d_18/best_model.pth \
    --input /path/to/volume.nii.gz \
    --output /path/to/results

# Run inference on a directory of volumes
python scripts/inference.py \
    --config /path/to/configs/config.yaml \
    --model /path/to/output/run_YYYYMMDD-HHMMSS_resnet3d_18/best_model.pth \
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