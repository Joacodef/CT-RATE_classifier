# 3D CT Scan Pathology Classifier

This project provides a comprehensive, production-ready system for classifying pathologies in 3D CT (Computed Tomography) scans using deep learning. It is built with PyTorch and MONAI, and it supports architectures like 3D ResNet, 3D DenseNet, and 3D Vision Transformers (ViT). The framework is designed with modularity, configurability, and extensive testing in mind, making it suitable for both research and production environments.

## Core Workflow

The recommended workflow for using this repository is as follows:

1.  **Setup**: Install the necessary dependencies and configure your environment.
2.  **Data Preparation**: Prepare the dataset by filtering entries and creating stratified, patient-grouped k-fold splits for cross-validation.
3.  **Training**: Train a model on a single fold or run a full k-fold cross-validation experiment. The intelligent caching system will preprocess data on-the-fly.
4.  **Inference**: Use a trained model to make predictions on new, unseen CT scans.
5.  **Hyperparameter Optimization (Optional)**: Run an efficient, staged search to find the best hyperparameters for your model and data.

---

## Features

* **Multiple Model Architectures**: Choose between 3D versions of ResNet, DenseNet, and Vision Transformers.
* **End-to-End Workflow**: Scripts for data preparation, training, inference, and hyperparameter optimization are provided.
* **Intelligent Preprocessing & Caching**: The system uses MONAI's `PersistentDataset` to automatically cache preprocessed images to disk. The cache is smartly managed using a hash of the preprocessing pipeline; any change to parameters like voxel spacing or image size automatically creates a new cache, ensuring perfect reproducibility.
* **Decoupled Data Pipeline**: A sophisticated, multi-stage data pipeline separates concerns: it first loads image paths, then loads and caches processed images, then attaches labels, and finally applies augmentations on-the-fly.
* **Configuration-Driven**: All aspects of the project, from file paths to model hyperparameters, are controlled via a central YAML configuration file.
* **Stratified Grouped K-Fold Cross-Validation**: Scripts and training logic are designed to support robust evaluation, ensuring that scans from the same patient do not leak between training and validation sets.
* **Staged Hyperparameter Optimization**: An efficient hyperparameter search script using Optuna that runs initial trials on small data subsets and allocates more data to more promising trials, dramatically speeding up the optimization process.
* **Experiment Tracking**: Integrated with Weights & Biases (W&B) for logging metrics, visualizing results, and managing experiments.
* **Resumable Training & Optimization**: Training runs can be resumed from checkpoints, and optimization studies can be resumed from a persistent database.
* **Comprehensive Reporting**: At the end of training, the system generates a detailed PDF/PNG report with learning curves, ROC AUC, F1 scores, a per-pathology performance heatmap, and a summary of the best model's metrics.
* **Testing**: A suite of unit and integration tests to ensure code quality and reliability.

---

## Repository Structure

.├── .github/workflows           # GitHub Actions for CI/CD├── configs/                    # Configuration files├── data/                       # (Git-ignored) Raw and processed data├── notebooks/                  # (Git-ignored) Jupyter notebooks for exploration├── output/                     # (Git-ignored) Saved models, logs, and reports├── requirements.txt            # Project dependencies├── scripts/                    # High-level scripts for core tasks│   ├── create_filtered_dataset.py # Filters the main dataset CSV│   ├── create_kfold_splits.py  # Creates k-fold data splits│   ├── train.py                # Main training script│   ├── inference.py            # Runs inference on new data│   └── optimize_hyperparams.py # Performs hyperparameter search├── src/                        # Source code for the project│   ├── config/                 # Configuration loading and parsing│   ├── data/                   # Dataset, dataloader, and preprocessing logic│   ├── evaluation/             # Evaluation metrics and reporting│   ├── models/                 # Model architectures (ResNet, DenseNet, ViT)│   ├── training/               # Core training and validation loops│   └── utils/                  # Utility functions└── tests/                      # Unit and integration tests
---

## 1. Setup and Configuration

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Joacodef/CT_classifier.git](https://github.com/Joacodef/CT_classifier.git)
    cd CT_classifier
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project requires specific versions of PyTorch that are compatible with your CUDA version. It is recommended to install these first. For CUDA 11.8, for example:
    ```bash
    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```
    Then, install the remaining packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The project is controlled by two main configuration files:

1.  **.env**: This file should be created at the project root to store machine-specific absolute paths and is ignored by Git. It prevents hardcoding local paths into the main configuration.

    **Create a `.env` file from the example:**
    ```bash
    cp .envexample .env
    ```
    **Then, edit `.env` to match your system's paths.** The `CACHE_DIR` is particularly important, as it stores preprocessed images to accelerate training. Ensure it points to a location with sufficient storage.

2.  **config.yaml**: This file defines all experiment parameters, including model type, hyperparameters, image processing settings, and file paths.

    **Create a new configuration file for your experiment from the example:**
    ```bash
    cp configs/config_example.yaml configs/my_experiment.yaml
    ```
    You can now modify `configs/my_experiment.yaml` for your specific needs.

---

## 2. The Data Pipeline & Caching

This project uses a sophisticated, multi-stage data pipeline that maximizes efficiency and reproducibility. Instead of a manual preprocessing step, caching is handled automatically and intelligently on-the-fly.

The data loading process for each item follows these steps:

1.  **`CTMetadataDataset`**: Reads a row from a data split CSV (e.g., `train_fold_0.csv`) and retrieves the file path for the corresponding CT volume. It is completely label-agnostic.

2.  **`monai.PersistentDataset` (The Disk Cache)**:
    * Receives the file path from the previous step.
    * Applies the computationally expensive preprocessing transforms (resampling, intensity clipping, etc.).
    * Saves the processed tensor to a cache directory on disk. The cache location is determined by a unique hash of the preprocessing transforms, ensuring that any change to the pipeline automatically creates a new, separate cache.
    * A custom `KeyCleanerD` transform ensures only the essential data (`image` and `image_meta_dict`) is saved, keeping the cache minimal and efficient.
    * On subsequent epochs, it reads the processed tensor directly from the disk, saving significant time.

3.  **`LabelAttacherDataset`**: This custom wrapper takes the processed image tensor from the cache and, using the index, finds the corresponding row in the master labels CSV (`paths.labels.all`) to attach the correct pathology labels.

4.  **`ApplyTransforms` (Augmentations)**: If augmentations are enabled (`training.augment: true`), this final wrapper applies on-the-fly augmentations (like random flips and rotations) to the processed tensor.

This decoupled design ensures that only the core, computationally heavy preprocessing is cached, while labels and augmentations remain dynamic.

---

## 3. Data Preparation

Before training, you need to prepare your dataset metadata.

### Create a Filtered Dataset (Optional)

The official dataset may contain scans with issues (e.g., missing files, corrupted headers). This script filters the main CSV to include only valid and usable scans.

```bash
python scripts/create_filtered_dataset.py \
    --input-file /path/to/your/dataset.csv \
    --img-dir /path/to/raw/images \
    --output-file ./data/filtered_dataset.csv
Create K-Fold SplitsFor robust evaluation, this script creates data splits using Stratified Grouped K-Fold. This method is ideal for medical datasets because it:Stratifies by the combination of pathology labels to ensure each fold has a similar distribution of cases.Groups by patient_id to ensure that all scans from a single patient remain in the same fold (either training or validation), preventing data leakage.python scripts/create_kfold_splits.py \
    --config configs/my_experiment.yaml \
    --input-file ./data/filtered_dataset.csv \
    --output-dir ./data/splits \
    --n-splits 5
This will create files like fold_0_train.csv, fold_0_val.csv, etc., which are used automatically by the training script.4. TrainingTraining a Single ModelTo train a single model, use the train.py script and specify the fold number. The script will automatically locate the correct split files and use the master labels file for training.python scripts/train.py \
    --config configs/my_experiment.yaml \
    --fold 0
Overriding Parameters: You can override specific parameters from the config file via the command line:python scripts/train.py --config configs/my_experiment.yaml --fold 0 --model-type vit3d --learning-rate 1e-4
Resuming Training: To resume a run from the latest checkpoint in the output directory:python scripts/train.py --config configs/my_experiment.yaml --fold 0 --resume
Running Full K-Fold Cross-ValidationTo run a full cross-validation experiment, you can loop through the folds using a simple bash script.for i in {0..4}; do
    echo "--------------------------------"
    echo "--- Starting Training, Fold $i ---"
    echo "--------------------------------"
    python scripts/train.py \
        --config configs/my_experiment.yaml \
        --fold $i
done
5. InferenceUse a trained model to make predictions on new data. The inference script requires the configuration file and the model checkpoint (.pth file) from a training output directory.Single Volume InferenceThis prints predictions to the console and saves a detailed JSON file.python scripts/inference.py \
    --config /path/to/output/from_training/config.yaml \
    --model /path/to/output/from_training/best_model.pth \
    --input /path/to/single/volume.nii.gz \
    --output /path/to/results/single_result
Batch InferenceThis processes all volumes in a directory and generates a single CSV file with predictions for each one.python scripts/inference.py \
    --config /path/to/output/from_training/config.yaml \
    --model /path/to/output/from_training/best_model.pth \
    --input /path/to/directory_of_volumes/ \
    --output /path/to/results/batch_results.csv
6. Hyperparameter Optimization (Optional)The repository includes an efficient hyperparameter optimization workflow using Optuna that leverages staged optimization to save time and computational resources.1. Create Data Subsets for Staged OptimizationFirst, create smaller, stratified subsets of your training data. The optimization script will use these to quickly evaluate a large number of hyperparameter combinations on a small amount of data before promoting the best-performing trials to larger data subsets.python scripts/create_training_subsets.py \
    --config configs/my_experiment.yaml \
    --input-file /path/to/your/full_train_split.csv \
    --fractions 0.5 0.2 0.05
2. Run the Staged Optimization StudyThis will launch an Optuna study. The study is persistent and resumable, as its results are saved to a .db file.Early, unpromising trials are run on small data fractions (e.g., 5%).More promising trials are "promoted" to run on larger fractions (e.g., 20%, 50%, and finally 100%).This prunes bad hyperparameter sets quickly, focusing resources on the ones that matter.python scripts/optimize_hyperparams.py \
    --config configs/my_experiment.yaml \
    --n-trials 100 \
    --study-name "vit3d-optimization-study" \
    --storage-db "vit3d_study.db"
7. TestingTo run the full suite of unit and integration tests, use pytest:pytest
