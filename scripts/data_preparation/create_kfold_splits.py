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
    Creates multi-label stratified k-fold splits from a master data file,
    ensuring that scans from the same patient remain in the same split.

    This function ensures that:
    1.  Folds are stratified to maintain the same distribution of each individual
        disease label across each fold using iterative stratification.
    2.  Scans from the same patient are kept within the same split (either
        training or validation) to prevent data leakage.

    The process involves:
    - Extracting a PatientID from each VolumeName.
    - Aggregating labels at the patient level.
    - Performing iterative stratification on the patients.
    - Mapping the patient-level splits back to the individual volumes.

    Args:
        config: The loaded project configuration object.
        n_splits (int): The number of folds (k) to create.
        output_dir (Path): The directory where the split CSV files will be saved.
    """
    # 1. Load and Prepare Master Data
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

    # 2. Extract PatientID from VolumeName
    logger.info("Extracting PatientID from VolumeName...")

    try:
        df_full["PatientID"] = df_full["VolumeName"].apply(
            lambda x: x.split("_")[1]
        )
        logger.info(f"Found {df_full['PatientID'].nunique()} unique patients.")
    except IndexError:
        logger.error(
            "Could not extract PatientID from VolumeName. "
            "Please ensure the filenames follow the 'split_patientID_scanID_reconstructionID' format."
        )
        return
        
    # 3. Aggregate Labels by Patient for Stratification
    logger.info("Aggregating labels at the patient level for stratification.")
    disease_cols = config.pathologies.columns
    # Group by patient and aggregate labels. Using max() serves as a logical OR
    # for binary (0/1) encoded labels.
    patient_labels = df_full.groupby("PatientID")[disease_cols].max()
    patient_labels = patient_labels.reset_index()
    logger.info(
        f"Aggregated labels for {len(patient_labels)} patients based on columns: {disease_cols}"
    )

    # 4. Perform Patient-Level Multi-Label Stratified K-Fold Split
    logger.info("Performing multi-label stratification WITH patient grouping.")
    logger.info(
        f"Performing Iterative Stratification split with k={n_splits} "
        f"on {len(patient_labels)} patients..."
    )

    X_patient = patient_labels[["PatientID"]]
    y_patient = patient_labels[disease_cols].to_numpy()

    kfold = IterativeStratification(n_splits=n_splits, order=1)

    # 5. Map Patient Splits to Volume Splits and Save
    logger.info(f"Saving split files to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_patient_indices, valid_patient_indices) in enumerate(
        kfold.split(X_patient, y_patient)
    ):
        # Get the patient IDs for the current train and validation folds
        train_patient_ids = X_patient.iloc[train_patient_indices]["PatientID"]
        valid_patient_ids = X_patient.iloc[valid_patient_indices]["PatientID"]

        # Select all volumes belonging to the patients in each split
        df_train = df_full[df_full["PatientID"].isin(train_patient_ids)]
        df_valid = df_full[df_full["PatientID"].isin(valid_patient_ids)]

        # We only need the VolumeName for the output files
        df_train_vols = df_train[["VolumeName"]]
        df_valid_vols = df_valid[["VolumeName"]]

        train_path = output_dir / f"train_fold_{fold_idx}.csv"
        valid_path = output_dir / f"valid_fold_{fold_idx}.csv"

        df_train_vols.to_csv(train_path, index=False)
        df_valid_vols.to_csv(valid_path, index=False)

        logger.info(
            f"  - Fold {fold_idx}: "
            f"Train={len(df_train_vols)} (from {len(train_patient_ids)} patients), "
            f"Valid={len(df_valid_vols)} (from {len(valid_patient_ids)} patients). "
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