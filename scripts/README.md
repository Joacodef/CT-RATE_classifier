# Scripts Module (`/scripts`)

This directory contains the high-level, executable scripts that drive the entire machine learning pipeline. They are organized into subdirectories based on their function, with the core ML scripts residing in the root.

## Recommended Workflow

A typical workflow involves running these scripts in the following sequence:

1.  **Prepare Data**: Use the scripts in `data_preparation/` to download, filter, and create k-fold splits of your dataset.
2.  **Generate Cache**: Run `cache_management/generate_cache.py` to pre-process and cache the dataset, which dramatically speeds up training.
3.  **Optimize Hyperparameters**: Optionally, run `optimize_hyperparams.py` on a small subset of the data to find the best settings for a full training run.
4.  **Train Model**: Use `train.py` to train your model on a specific data fold.
5.  **Run Inference**: Use `inference.py` with a trained model checkpoint to make predictions on new data.

---

## Directory Overviews

### Data Preparation (`/scripts/data_preparation`)
This directory contains all scripts related to getting and organizing the raw dataset before processing. This includes downloading files from Hugging Face, filtering out unwanted volumes based on exclusion criteria, and creating stratified data splits for training and validation.

**Key scripts:**
- `verify_and_download.py`
- `create_filtered_dataset.py`
- `create_kfold_splits.py`
- `create_training_subsets_hpo.py`

### Cache Management (`/scripts/cache_management`)
This directory holds utilities for creating and maintaining the MONAI `PersistentDataset` cache. Caching involves pre-processing all images with the defined transforms and saving the resulting tensors to disk for rapid access during training. These scripts can generate the cache, verify its integrity, and help debug it by mapping hashed filenames back to their original volume names.

**Key scripts:**
- `generate_cache.py`
- `verify_cache_integrity.py`
- `map_hashes_to_volumes.py`

---

## Core Script Explanations

### `inference.py`

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
````

### `optimize_hyperparams.py`

This script uses the Optuna framework to perform an automated search for the best hyperparameters. It uses subsets of the data (created via `create_training_subsets_hpo.py`) for a staged optimization approach, saving time and resources.

**Usage:**

```bash
python scripts/optimize_hyperparams.py \
    --config configs/config.yaml \
    --n-trials 100 \
    --study-name "vit3d-lr-optimization"
```
