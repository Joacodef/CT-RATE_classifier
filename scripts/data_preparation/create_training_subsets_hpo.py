import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

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
):
    """
    Creates nested, multi-label stratified subsets from a training data file.

    This function loads a list of volume identifiers, merges them with the full
    set of pathology labels, and then uses Iterative Stratification to generate
    the nested subsets. This approach is robust for multi-label data and
    ensures that the distribution of each pathology is preserved across subsets.

    Args:
        config: The loaded configuration object.
        input_file (str): Path to the input CSV file containing 'VolumeName'.
        fractions (list[float]): A list of fractions for the subsets
                                 (e.g., [0.5, 0.2, 0.05]).
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
    df_full = pd.merge(df_volumes, df_labels, on="VolumeName", how="inner")

    # --- Multi-Label Stratification Setup ---
    disease_cols = config.pathologies.columns
    if any(col not in df_full.columns for col in disease_cols):
        logger.error(
            "One or more pathology columns from the config are missing in the dataframe."
        )
        return

    logger.info(
        f"Using multi-label iterative stratification based on columns: {disease_cols}"
    )

    # --- Create Nested Subsets using Iterative Stratification ---
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

        # The goal is to create a nested subset. For example, to get a 20% subset from a 50% subset,
        # we need to take 40% of the current data (0.2 / 0.5 = 0.4).
        # iterative_train_test_split splits into train and test. We want our new subset to be the 'test' set.
        subset_size_ratio = frac / current_fraction

        # We will split the indices of the dataframe to avoid data duplication issues.
        # We pass an array of indices (0, 1, 2, ...) as X and the labels as y.
        indices = np.arange(df_current.shape[0]).reshape(-1, 1)
        y_labels = df_current[disease_cols].to_numpy()

        try:
            # The function returns: X_train, y_train, X_test, y_test
            # We want X_test, which contains the *indices* for our new subset.
            # My previous error was capturing the fourth element (y_test) instead of the third (X_test).
            _, _, subset_indices, _ = iterative_train_test_split(
                indices, y_labels, test_size=subset_size_ratio
            )

            # We use these indices to select the correct rows from the original dataframe.
            df_subset = df_current.iloc[subset_indices.flatten()]

        except ValueError as e:
            logger.error(
                f"Stratification failed for fraction {frac}. This can happen if "
                "the dataset is too small for the requested split. "
                f"Error: {e}"
            )
            return


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

        df_to_save = df_subset[["VolumeName"]]
        df_to_save.to_csv(output_path, index=False)
        logger.info(f"   - Saved {output_path} (size: {len(df_to_save)})")

    # --- Verification of Disease Distribution ---
    logger.info("\n--- Verifying Disease Distribution ---")
    original_dist = df_full[disease_cols].mean().rename("original")
    verification_results = [original_dist]

    for frac, df_subset in subsets.items():
        subset_dist = df_subset[disease_cols].mean().rename(
            f"{int(frac*100)}%"
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
    args = parser.parse_args()

    config = load_config(args.config)
    create_training_subsets(
        config, args.input_file, args.fractions
    )


if __name__ == "__main__":
    main()