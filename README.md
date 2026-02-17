# 3D CT Scan Pathology Classifier

This project provides a system for classifying pathologies in 3D CT (Computed Tomography) scans using self-contained PyTorch implementations of 3D ResNet, 3D DenseNet, and 3D Vision Transformers (ViT), with preprocessing support from MONAI.

## Core Workflow

The recommended workflow for using this repository is as follows:

1.  **Setup**: Install dependencies and configure the environment.
2.  **Data Preparation**: Use scripts to filter the dataset, download files, and prepare fixed training/validation splits.
3.  **Cache Generation**: Pre-process and cache the dataset to accelerate training.
4.  **Training**: Train a model using the configured training/validation split.
5.  **Inference**: Use a trained model to make predictions on new scans.
6.  **Hyperparameter Optimization (Optional)**: Run a search to find optimal hyperparameters.

-----

## Repository Structure

```
.
├── configs/                    # Configuration files
├── data/                       # (Git-ignored) Raw and processed data
├── output/                     # (Git-ignored) Training runs (e.g., run_YYYYMMDD-HHMMSS_model_variant)
├── requirements.txt            # Project dependencies
├── scripts/                    # High-level scripts for core tasks
│   ├── data_preparation/       # Scripts for downloading, filtering, and splitting data
│   │   ├── create_filtered_dataset.py
│   │   ├── verify_and_download.py
│   │   └── create_training_subsets_hpo.py
│   ├── cache_management/       # Scripts for generating and verifying the cache
│   │   ├── generate_cache.py
│   │   └── verify_cache_integrity.py
│   ├── train.py                # Main training script
│   ├── inference.py            # Runs inference on new data
│   └── optimize_hyperparams.py # Performs hyperparameter search
└── src/                        # Source code for the project
    ├── config/                 # Configuration loading and parsing
    ├── data/                   # Dataset, dataloader, and preprocessing logic
    ├── models/                 # Model architectures (ResNet, DenseNet, ViT)
    ├── training/               # Core training and validation loops
    └── utils/                  # Utility functions
```

-----

## 1\. Setup and Configuration

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Joacodef/CT_classifier.git
    cd CT_classifier
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:** First, install the correct PyTorch version for your CUDA setup.
    ```bash
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
    ```
    Then, install the remaining packages:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The project is controlled by two types of files:

1.  **.env**: This file, located at the project root, stores machine-specific absolute paths and is ignored by Git. Create it from the example `cp .envexample .env` and edit it to match your system's paths.
2.  **config.yaml**: These files, located in `configs/`, define experiment parameters like model type, hyperparameters, and image processing settings.

-----

## 2\. Data Preparation

Scripts in `scripts/data_preparation/` are used to prepare the dataset.

> **Important Note on Data Handling:** This repository does not use the original train/validation splits provided with the CT-RATE dataset. Instead, it generates a custom, filtered master list of volumes. This is done to:
> 1.  **Separate Manually Labeled Scans**: A subset of scans with manual "gold standard" labels are excluded from the main training corpus to be used for high-quality, independent evaluation.
> 2.  **Clean the Dataset**: Known problematic scans (e.g., brain scans, scans with missing z-space) are removed to improve model robustness.
>
> This approach was chosen for the purposes of the experiments of this repo, though it means results may not be directly comparable to models trained on the original, unfiltered data splits.

* **`create_filtered_dataset.py`**: Generates a master list of volumes by applying patient-level exclusion rules from multiple source files to prevent data leakage.
* **`verify_and_download.py`**: Checks for missing NIfTI files locally and downloads them from the specified Hugging Face repository.
**Example Workflow:**

```bash
# 1. Create the filtered master list of volumes
python scripts/data_preparation/create_filtered_dataset.py --config configs/my_experiment.yaml

# 2. Download any missing files from Hugging Face
python scripts/data_preparation/verify_and_download.py --config configs/my_experiment.yaml

# 3. Prepare fixed split CSVs (train.csv / valid.csv) under data/splits/
# This repository expects split files referenced by paths.data_subsets in your config.
```

-----

## 3\. Cache Generation

Caching is the process of pre-computing the results of the data preprocessing pipeline (e.g., resampling, resizing, normalizing) and saving the final tensors to disk. This accelerates training by allowing the `DataLoader` to fetch ready-to-use tensors instead of repeating these steps every epoch. The cache directory is determined by a hash of the preprocessing transforms, so a new cache is automatically created if settings change.

  * **`generate_cache.py`**: Builds the cache by identifying missing volumes, downloading them in batches, applying transforms, and saving the processed tensors. It cleans up the raw downloaded files after they are cached.
  * **`verify_cache_integrity.py`**: A utility to scan the cache directory for corrupted files and optionally delete them.

**Example Usage:**

```bash
# Generate the cache using 8 worker processes
python scripts/cache_management/generate_cache.py \
    --config configs/my_experiment.yaml \
    --num-workers 8
```

-----

## 4\. Model Training

The `train.py` script orchestrates the training process.

### Data Pipeline during Training

The training data pipeline is designed for efficiency, separating preprocessing from on-the-fly augmentations.

1.  **`CTMetadataDataset`**: Reads a data split CSV and retrieves the file path for the corresponding CT volume.
2.  **`monai.PersistentDataset` (Disk Cache)**: Receives the file path from the previous step. On the first epoch, it applies the preprocessing transforms (resampling, resizing, normalizing) and saves the resulting tensor to the disk cache. On subsequent epochs, it loads the tensor directly from the cache, skipping the expensive preprocessing steps.
3.  **`LabelAttacherDataset`**: Attaches the correct pathology labels to the processed image tensor retrieved from the cache.
4.  **On-the-Fly GPU Augmentations**: If enabled, data augmentations (e.g., random flips, rotations) are applied to the batch on the GPU within the training loop. This ensures maximum efficiency by processing augmentations in parallel on the accelerator.

### Running Training

To train a model, configure `paths.data_subsets.train` and `paths.data_subsets.valid` in your config.

```bash
# Train using configured split files
python scripts/train.py \
    --config configs/my_experiment.yaml

# Resume training from the latest checkpoint
python scripts/train.py \
    --config configs/my_experiment.yaml \
    --resume
```

### Training Output Structure

Each new training run is saved under the configured `paths.output_dir` using a timestamped run folder:

```text
output/
    run_YYYYMMDD-HHMMSS_<model>_<variant>/
        training.log
        best_model.pth
        last_checkpoint.pth
        final_model.pth
        training_history.json
        training_metrics_detailed.csv
        training_summary.json
```

When using `--resume` without an explicit path, the script automatically scans the `output/` root for the most recent run and resumes from the latest checkpoint found there.

### Weights & Biases Run Names

If W&B logging is enabled, run names are auto-generated from key configuration fields to improve experiment traceability, including:

- model type and variant
- workflow mode
- train split identifier
- epochs, batch size, learning rate, and weight decay
- input shape and number of classes
- optimization/cache flags (augment, cache, mixed precision, bf16)

A short hash suffix is appended for uniqueness across similar runs.

-----

## 5\. Inference

The `inference.py` script uses a trained model to make predictions on new data. It can process a single file or a directory of volumes.

**Example Usage:**

```bash
# Run inference on a directory of volumes
python scripts/inference.py \
    --config /path/to/configs/my_experiment.yaml \
    --model /path/to/output/run_YYYYMMDD-HHMMSS_resnet3d_18/best_model.pth \
    --input /path/to/directory_of_volumes/ \
    --output /path/to/results/batch_results.csv
```

-----

## 6\. Hyperparameter Optimization (Optional)

The project includes a hyperparameter optimization script using Optuna.

1.  **`create_training_subsets_hpo.py`**: First, create smaller, stratified subsets of your training data. The optimization script uses these to quickly evaluate hyperparameter combinations.
2.  **`optimize_hyperparams.py`**: This launches an Optuna study that saves its results to a `.db` file. It uses staged optimization, where unpromising trials are run on small data fractions and pruned early, focusing resources on more promising trials which are promoted to larger data fractions.

**Example Workflow:**

```bash
# 1. Create subsets from the configured training split
python scripts/data_preparation/create_training_subsets_hpo.py \
    --config configs/my_experiment.yaml \
    --input-file data/splits/train.csv \
    --fractions 0.5 0.2 0.05

# 2. Run the optimization study
python scripts/optimize_hyperparams.py \
    --config configs/my_experiment.yaml \
    --n-trials 100 \
    --study-name "vit3d-optimization-study" \
    --storage-db "vit3d_study.db"
```