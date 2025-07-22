# scripts/create_kfold_splits.py
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from skmultilearn.model_selection import IterativeStratification

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
    Creates multi-label stratified k-fold splits from a master data file.

    This function ensures that:
    1.  Folds are stratified to maintain the same distribution of each individual
        disease label across each fold using iterative stratification.

    Warning:
        This method does NOT perform grouping by patient. Scans from the same
        patient may be split between the training and validation sets, which
        can lead to data leakage and overly optimistic validation scores.

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

    df_full = pd.merge(df_master, df_labels, on="VolumeName", how="inner")
    logger.info(f"Merged dataset created with {len(df_full)} volumes.")

    # --- 2. Define Labels for Splitting ---
    disease_cols = config.pathologies.columns
    logger.info(f"Using multi-label stratification based on columns: {disease_cols}")

    X = df_full[["VolumeName"]]
    y = df_full[disease_cols]

    # --- 3. Perform Multi-Label Stratified K-Fold Split ---
    logger.warning(
        "Performing multi-label stratification WITHOUT patient grouping. "
        "Data leakage may occur if a patient has multiple scans."
    )
    logger.info(f"Performing Iterative Stratification split with k={n_splits}...")

    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # NOTE: The 'shuffle' and 'random_state' arguments are not supported by this
    # implementation of IterativeStratification and have been removed.
    kfold = IterativeStratification(
        n_splits=n_splits,
        order=1
    )

    # --- 4. Save the Split Files ---
    logger.info(f"Saving split files to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_indices, valid_indices) in enumerate(kfold.split(X_np, y_np)):
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
        description="Create multi-label stratified k-fold splits for cross-validation.",
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
        help="Directory to save the fold split CSV files (e.g., 'splits/kfold_5').",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    data_dir = Path(config.paths.data_dir)
    output_path = data_dir / args.output_dir

    create_kfold_splits(config, args.n_splits, output_path)


if __name__ == "__main__":
    main()