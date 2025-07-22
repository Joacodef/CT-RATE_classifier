# Data Preparation Scripts (`/scripts/data_preparation`)

This directory contains all the scripts needed to take the raw dataset and transform it into a well-structured, clean, and complete set of files ready for the caching and training stages.

These scripts handle everything from filtering out unwanted scans to creating robust cross-validation splits.

## Recommended Order of Operations

For setting up the project from scratch, these scripts should be run in the following order:

1.  **`create_filtered_dataset.py`**: Run this once to generate the master list of all volumes to be used in the project.
2.  **`verify_and_download.py`**: Run this to ensure you have all the necessary `.nii.gz` files downloaded locally.
3.  **`create_kfold_splits.py`**: Use the master list to generate your cross-validation folds (e.g., for 5-fold CV).
4.  **`create_training_subsets_hpo.py`**: Run this on a specific training fold (e.g., `train_fold_0.csv`) to create smaller subsets for hyperparameter optimization.

---

## Script Descriptions

### `create_filtered_dataset.py`

This is typically the first script you will run. It takes the raw metadata and applies a set of exclusion rules defined within the script (e.g., removing brain scans, scans with missing data). It produces a single `filtered_master_list.csv` that serves as the definitive source for all subsequent data splitting and processing.

**Usage:**
```bash
# NOTE: This script contains hardcoded paths.
# You must review and edit the paths inside the script before running.
python scripts/data_preparation/create_filtered_dataset.py
````

-----

### `verify_and_download.py`

This script ensures your local dataset is complete. It checks which files from your dataset CSV are missing from your local image directory and, if any are found, prompts you to download them from the specified Hugging Face repository.

**Usage:**

```bash
# Check for and download missing files based on the config
python scripts/data_preparation/verify_and_download.py --config configs/config.yaml
```

-----

### `create_kfold_splits.py`

This script creates robust, stratified, and grouped k-fold cross-validation splits from the master dataset list. This is crucial for reliable model evaluation, as it:

1.  **Groups** scans from the same patient to prevent data leakage between training and validation sets.
2.  **Stratifies** the splits to ensure the distribution of pathologies is balanced across all folds.

**Usage:**

```bash
# Example: Create 5 cross-validation folds
python scripts/data_preparation/create_kfold_splits.py \
    --config configs/config.yaml \
    --n-splits 5 \
    --output-dir "data/splits/kfold_5"
```

-----

### `create_training_subsets_hpo.py`

This script is used to prepare for hyperparameter optimization. It takes a training data file (e.g., a single fold from `create_kfold_splits.py`) and creates smaller, stratified, and nested subsets (e.g., 5%, 20%, 50% of the data). This allows the `optimize_hyperparams.py` script to test many parameter combinations quickly on smaller data before promoting the best ones to larger datasets.

**Usage:**

```bash
# Create 50%, 20%, and 5% subsets from the first training fold
python scripts/data_preparation/create_training_subsets_hpo.py \
    --config configs/config.yaml \
    --input-file data/splits/kfold_5/train_fold_0.csv \
    --fractions 0.5 0.2 0.05
```

