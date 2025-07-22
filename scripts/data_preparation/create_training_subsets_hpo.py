import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to the Python path to allow importing from 'config'
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import load_config

# Configure basic logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def create_training_subsets(
    config,
    input_file: str,
    fractions: list[float],
    subsample_strategy: str,
):
    """
    Creates nested subsets from a given training data file using a chosen strategy.

    This function first loads a list of volume identifiers from the specified
    input file and merges them with the full set of pathology labels from the
    path defined in the configuration (`paths.labels.all`).

    It then generates nested subsets based on the selected `subsample_strategy`:
    - 'stratified': Preserves the original distribution of diseases in subsets.
      Useful for creating representative, smaller versions of the dataset.
    - 'uniform': Actively undersamples majority classes to create subsets with a
      more uniform label distribution. This is achieved through inverse
      frequency weighting and can help mitigate class imbalance.

    Args:
        config: The loaded configuration object.
        input_file (str): Path to the input CSV file containing 'VolumeName'.
        fractions (list[float]): A list of fractions for the subsets
                                 (e.g., [0.5, 0.2, 0.05]).
        subsample_strategy (str): The method for subsampling ('stratified' or
                                  'uniform').
    """
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found at: {input_path}")
        return

    # --- Load Data and Labels ---
    logger.info(f"Loading volume list from: {input_path}")
    df_volumes = pd.read_csv(input_path)

    labels_path = Path(config.paths.data_dir) / config.paths.labels.all
    if not labels_path.exists():
        logger.error(f"Labels file not found at: {labels_path}")
        return
    logger.info(f"Loading labels from: {labels_path}")
    df_labels = pd.read_csv(labels_path)

    logger.info("Merging volume list with labels on 'VolumeName'")
    df_full = pd.merge(df_volumes, df_labels, on="VolumeName", how="left")

    # --- Multi-Label Stratification/Sampling Setup ---
    disease_cols = config.pathologies.columns
    seed = config.training.seed
    missing_cols = [col for col in disease_cols if col not in df_full.columns]
    if missing_cols:
        logger.error(
            "Label columns not found in dataframe: %s", missing_cols
        )
        return

    if df_full[disease_cols].isnull().any().any():
        logger.warning(
            "Some volumes in the input file did not have corresponding labels "
            "and will be dropped."
        )
        df_full.dropna(subset=disease_cols, inplace=True)

    logger.info(
        f"Using '{subsample_strategy}' strategy based on columns: {disease_cols}"
    )
    df_full["stratify_key"] = (
        df_full[disease_cols].astype(str).agg("-".join, axis=1)
    )

    # --- Create Nested Subsets ---
    fractions.sort(reverse=True)
    subsets = {}
    df_current = df_full.copy()
    current_fraction = 1.0

    for frac in fractions:
        if frac >= current_fraction:
            logger.warning(
                "Skipping fraction %f as it is not smaller than the current "
                "fraction %f.",
                frac,
                current_fraction,
            )
            continue

        logger.info(
            "Creating %d%% subset from %d%% subset...",
            int(frac * 100),
            int(current_fraction * 100),
        )

        split_ratio = frac / current_fraction

        # --- Select Subsampling Strategy ---
        if subsample_strategy == "stratified":
            # This logic prevents crashes when a class has only one sample.
            stratify_for_split = df_current["stratify_key"].copy()
            class_counts = stratify_for_split.value_counts()
            rare_classes = class_counts[class_counts < 2].index

            stratify_argument = stratify_for_split
            if not rare_classes.empty:
                logger.warning(
                    f"Found {len(rare_classes)} class(es) with only 1 sample. "
                    "Grouping them into a '_RARE_' category for this split."
                )
                is_rare = stratify_for_split.isin(rare_classes)
                stratify_for_split.loc[is_rare] = "_RARE_"

                if stratify_for_split.value_counts().get("_RARE_", 0) < 2:
                    logger.error(
                        "Cannot stratify: rare class group has fewer than 2 members. "
                        "Falling back to a non-stratified split for this step."
                    )
                    stratify_argument = None

            _, df_subset = train_test_split(
                df_current,
                test_size=split_ratio,
                random_state=seed,
                stratify=stratify_argument,
            )
        elif subsample_strategy == "uniform":
            target_size = int(len(df_current) * split_ratio)
            
            # Calculate inverse frequency weights
            class_frequencies = df_current["stratify_key"].value_counts()
            weights = 1 / df_current["stratify_key"].map(class_frequencies)
            
            logger.info(
                f"Performing weighted sampling to get a more uniform "
                f"distribution. Target size: {target_size}"
            )

            df_subset = df_current.sample(
                n=target_size,
                weights=weights,
                random_state=seed,
                replace=False,  # Ensure we don't sample the same item twice
            )
        else:
            raise ValueError(f"Unknown strategy: {subsample_strategy}")

        subsets[frac] = df_subset
        df_current = df_subset.copy()
        current_fraction = frac

    # --- Save the Subsets ---
    output_dir = Path(config.paths.data_dir) / "splits" / "hpo_subsets"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving subsets to: {output_dir}")

    for frac, df_subset in subsets.items():
        percent = int(frac * 100)
        output_filename = f"{input_path.stem}_{percent}_percent.csv"
        output_path = output_dir / output_filename

        df_to_save = df_subset.drop(columns=["stratify_key"] + disease_cols)
        df_to_save.to_csv(output_path, index=False)
        logger.info(f"   - Saved {output_path} (size: {len(df_to_save)})")

    # --- Verification of Disease Distribution ---
    logger.info("\n--- Verifying Disease Distribution ---")
    original_dist = df_full[disease_cols].mean().rename("original")
    verification_results = [original_dist]

    for frac, df_subset in subsets.items():
        subset_dist = df_subset[disease_cols].mean().rename(
            f"{int(frac*100)}%_{subsample_strategy}"
        )
        verification_results.append(subset_dist)

    dist_df = pd.concat(verification_results, axis=1)
    logger.info("Comparison of disease prevalence across subsets:")
    print(dist_df.to_string())


def main():
    """Main function to parse arguments and initiate subset creation."""
    parser = argparse.ArgumentParser(
        description=(
            "Create nested training data subsets for hyperparameter optimization."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input CSV file to be split (e.g., 'data/splits/train.csv').",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        required=True,
        help=("A list of fractions for the subsets, e.g., --fractions 0.5 0.2"),
    )
    parser.add_argument(
        "--subsample-strategy",
        type=str,
        default="stratified",
        choices=["stratified", "uniform"],
        help=(
            "The strategy for creating subsets:\n"
            " - 'stratified': (Default) Keep original label distribution.\n"
            " - 'uniform': Undersample majority classes for a more balanced "
            "label distribution."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    create_training_subsets(
        config, args.input_file, args.fractions, args.subsample_strategy
    )


if __name__ == "__main__":
    main()