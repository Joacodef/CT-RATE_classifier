# scripts/create_kfold_splits.py
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import load_config

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def create_kfold_splits(config, n_splits: int, output_dir: Path):
    """
    Creates stratified, grouped k-fold splits from a master data file.

    This function ensures that:
    1.  Folds are stratified to maintain the same distribution of multi-label
        disease classes across each fold.
    2.  Scans from the same patient (group) are kept within the same fold to
        prevent data leakage between training and validation sets.

    Args:
        config: The loaded project configuration object.
        n_splits (int): The number of folds (k) to create.
        output_dir (Path): The directory where the split CSV files will be saved.
    """
    # --- 1. Load and Prepare Master Data ---
    data_dir = Path(config.paths.data_dir)
    master_list_path = data_dir / config.paths.full_dataset_csv
    labels_path = data_dir / config.paths.labels.all

    if not master_list_path.exists() or not labels_path.exists():
        logger.error(
            f"Master file not found at '{master_list_path}' or "
            f"labels file not found at '{labels_path}'. Please check config paths."
        )
        return

    logger.info(f"Loading master volume list from: {master_list_path}")
    df_master = pd.read_csv(master_list_path)

    logger.info(f"Loading all labels from: {labels_path}")
    df_labels = pd.read_csv(labels_path)

    # Merge master list with labels to create the full dataset for splitting
    df_full = pd.merge(df_master, df_labels, on="VolumeName", how="inner")
    logger.info(f"Merged dataset created with {len(df_full)} volumes.")

    # --- 2. Create Grouping and Stratification Keys ---
    
    # Create a patient/subject ID for grouping
    # Assumes VolumeName format like 'train_12345_a_1' -> 'train_12345'
    logger.info("Generating patient IDs for grouping...")
    df_full['patient_id'] = df_full['VolumeName'].str.split('_').str[:2].str.join('_')
    
    # Create a single stratification key from the combination of disease labels
    disease_cols = config.pathologies.columns
    logger.info(f"Generating stratification key based on columns: {disease_cols}")
    df_full["stratify_key"] = (
        df_full[disease_cols].astype(str).agg("-".join, axis=1)
    )

    # --- 3. Perform Stratified Group K-Fold Split ---
    logger.info(f"Performing Stratified Group K-Fold split with k={n_splits}...")
    
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=config.training.seed
    )

    X = df_full[["VolumeName"]]
    y = df_full["stratify_key"]
    groups = df_full["patient_id"]

    # --- 4. Save the Split Files ---
    logger.info(f"Saving split files to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_indices, valid_indices) in enumerate(sgkf.split(X, y, groups)):
        df_train = X.iloc[train_indices]
        df_valid = X.iloc[valid_indices]

        train_path = output_dir / f"train_fold_{fold_idx}.csv"
        valid_path = output_dir / f"valid_fold_{fold_idx}.csv"

        df_train.to_csv(train_path, index=False)
        df_valid.to_csv(valid_path, index=False)

        logger.info(
            f"  - Fold {fold_idx}: "
            f"Train={len(df_train)}, Valid={len(df_valid)}. "
            f"Saved to {train_path.name}, {valid_path.name}"
        )
    
    logger.info("K-fold split creation complete.")


def main():
    """Main function to parse arguments and initiate k-fold split creation."""
    parser = argparse.ArgumentParser(
        description="Create stratified, grouped k-fold splits for cross-validation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        required=True,
        help="The number of folds (k) for cross-validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the fold split CSV files (e.g., 'data/splits/kfold_5').",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    
    # The output path needs to be relative to the project directory
    output_path = config.paths.data_dir / args.output_dir
    
    create_kfold_splits(config, args.n_splits, output_path)


if __name__ == "__main__":
    main()