# Data Preparation Scripts (`/scripts/data_preparation`)

This directory contains the scripts required to process the raw dataset into a structured and filtered set of files suitable for the caching and training stages. The scripts handle operations from initial filtering to the creation of cross-validation splits.

## Data Handling Philosophy

**It is critical to note that this project does not use the original train/validation splits provided with the CT-RATE dataset.** Instead, a custom data filtering and preparation workflow is enforced by the scripts in this directory.

The core motivation is to create a robust and reliable dataset by:

1.  **Excluding Problematic Scans**: The `create_filtered_dataset.py` script removes scans that are known to be problematic, such as brain scans or those with missing metadata, which are not suitable for a chest CT classifier.
2.  **Separating a Gold Standard Test Set**: A set of scans with high-quality, manual labels are treated as a "gold standard" evaluation set. These are also removed from the main training corpus to ensure that the model is evaluated on data that has not been seen in any form during training.

This process results in a `filtered_master_list.csv`, which serves as the definitive source for all subsequent steps, such as creating custom cross-validation folds. While this methodology enhances the reliability of the experimental results, it means that performance metrics may not be directly comparable to other works that use the original, unfiltered dataset.

## Recommended Order of Operations

The following sequence is recommended for initial project setup:

1.  **`create_filtered_dataset.py`**: Generates the master list of all volumes to be used.
2.  **`verify_and_download.py`**: Ensures all required `.nii.gz` files are downloaded locally.
3.  **`create_kfold_splits.py`**: Uses the master list to generate cross-validation folds.
4.  **`create_training_subsets_hpo.py`**: Generates smaller, stratified subsets from a training fold for hyperparameter optimization.

-----

## Script Descriptions

### `create_filtered_dataset.py`

This script generates a definitive master list of volumes for the project. It loads the complete list of available volumes and applies patient-level filtering by aggregating patient IDs from multiple exclusion files (e.g., brain scans, scans with missing metadata). If a patient appears in any exclusion list, all of that patient's scans are removed from the final dataset to prevent data leakage. The final list is saved as a CSV file.

**Usage:**

```bash
python scripts/data_preparation/create_filtered_dataset.py --config configs/config.yaml
```

-----

### `verify_and_download.py`

This script ensures the local dataset is complete by cross-referencing a dataset CSV file with the local image directory. It identifies any required NIfTI files that are missing locally and downloads them from the specified Hugging Face repository using a multi-threaded downloader.

**Usage:**

```bash
python scripts/data_preparation/verify_and_download.py --config configs/config.yaml
```

-----

### `create_kfold_splits.py`

This script generates k-fold cross-validation splits from the master dataset list. The process is designed to be robust for reliable model evaluation by implementing two key strategies:

1.  **Patient Grouping**: It ensures all scans from the same patient are kept together within a single split (either training or validation) to prevent data leakage.
2.  **Iterative Stratification**: It stratifies the patient-level splits to maintain a balanced distribution of all pathology labels across every fold.

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

This script prepares data for hyperparameter optimization by creating smaller, nested subsets from a larger training file (e.g., `train_fold_0.csv`). It uses multi-label iterative stratification to ensure that the pathology distribution in the smaller subsets (e.g., 5%, 20%, 50%) is representative of the original set. This allows for faster evaluation of multiple hyperparameter combinations.

**Usage:**

```bash
# Create 50%, 20%, and 5% subsets from the first training fold
python scripts/data_preparation/create_training_subsets_hpo.py \
    --config configs/config.yaml \
    --input-file data/splits/kfold_5/train_fold_0.csv \
    --fractions 0.5 0.2 0.05
```