import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to the Python path to allow importing from 'config'
project_root = Path(__file__).resolve().parents[1]
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
    config, input_file: str, fractions: list[float]
):
    """
    Creates stratified, nested subsets from a given training data file.

    The stratification is performed based on the disease classes defined in the
    project configuration to ensure that the distribution of diseases is
    preserved in the subsets. Subsets are nested, meaning smaller percentage
    subsets are created from larger ones. This function includes robust
    handling for rare classes with single samples to prevent crashing.

    Args:
        config: The loaded configuration object.
        input_file (str): Path to the input CSV file.
        fractions (list[float]): A list of fractions for the subsets
                                 (e.g., [0.5, 0.2, 0.05]).
    """
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found at: {input_path}")
        return

    logger.info(f"Loading full training data from: {input_path}")
    df_full = pd.read_csv(input_path)

    # --- Multi-Label Stratification Setup ---
    disease_cols = config.pathologies.columns
    seed = config.training.seed  # Get seed from the loaded config
    missing_cols = [
        col for col in disease_cols if col not in df_full.columns
    ]
    if missing_cols:
        logger.error(
            "Stratification columns not found in dataframe: %s", missing_cols
        )
        return

    logger.info(f"Stratifying based on disease columns: {disease_cols}")
    # Create a single stratification key from the combination of disease labels
    df_full["stratify_key"] = (
        df_full[disease_cols].astype(str).agg("-".join, axis=1)
    )

    # --- Create Nested Subsets ---
    # Sort fractions in descending order to create nested subsets correctly
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

        # Calculate the proportion to split from the current dataframe
        split_ratio = frac / current_fraction

        # --- Handle Singleton Classes for Stratification ---
        # This logic prevents crashes when a class has only one sample.
        stratify_for_split = df_current["stratify_key"].copy()
        class_counts = stratify_for_split.value_counts()
        rare_classes = class_counts[class_counts < 2].index

        stratify_argument = stratify_for_split
        # If there are any classes with only one member, group them.
        if not rare_classes.empty:
            logger.warning(
                f"Found {len(rare_classes)} class(es) with only 1 sample. "
                "Grouping them into a '_RARE_' category for this split."
            )
            is_rare = stratify_for_split.isin(rare_classes)
            stratify_for_split.loc[is_rare] = '_RARE_'

            # If the new '_RARE_' group itself has < 2 members, stratification
            # is impossible. Fall back to a non-stratified split for this step.
            if stratify_for_split.value_counts().get('_RARE_') < 2:
                logger.error(
                    "Cannot stratify: rare class group has fewer than 2 members. "
                    "Falling back to a non-stratified split for this step. "
                    "Distribution may be slightly skewed."
                )
                stratify_argument = None
            else:
                stratify_argument = stratify_for_split

        # The 'test' set from the split becomes our new, smaller subset
        _, df_subset = train_test_split(
            df_current,
            test_size=split_ratio,
            random_state=seed,
            stratify=stratify_argument,
        )

        subsets[frac] = df_subset
        # The next split will be from this new, smaller subset
        df_current = df_subset.copy()
        current_fraction = frac

    # --- Save the Subsets ---
    output_dir = input_path.parent
    logger.info(f"Saving subsets to: {output_dir}")

    for frac, df_subset in subsets.items():
        percent = int(frac * 100)
        output_filename = f"{input_path.stem}_{percent}_percent.csv"
        output_path = output_dir / output_filename

        # Remove the temporary stratification key before saving
        df_to_save = df_subset.drop(columns=["stratify_key"])
        df_to_save.to_csv(output_path, index=False)
        logger.info(f"   - Saved {output_path} (size: {len(df_to_save)})")

    # --- Verification of Disease Distribution ---
    logger.info("\n--- Verifying Disease Distribution ---")

    original_dist = df_full[disease_cols].mean().rename("original")
    verification_results = [original_dist]

    for frac, df_subset in subsets.items():
        subset_dist = df_subset[disease_cols].mean().rename(
            f"{int(frac*100)}%_subset"
        )
        verification_results.append(subset_dist)

    dist_df = pd.concat(verification_results, axis=1)
    logger.info("Comparison of disease prevalence across subsets:")
    print(dist_df.to_string())

    if fractions:
        smallest_frac = min(fractions)
        if smallest_frac in subsets:
            smallest_subset_dist = subsets[smallest_frac][disease_cols].mean()
            diff = (original_dist - smallest_subset_dist).abs().sum()

            logger.info(
                "\nAbsolute sum of differences in distribution between original and "
                "%d%% subset: %.4f",
                int(smallest_frac * 100),
                diff,
            )
            # A small threshold for acceptable difference
            if diff < 0.1:
                logger.info("Stratification appears to be working correctly.")
            else:
                logger.warning(
                    "Large difference in distribution detected. "
                    "Stratification might not be optimal."
                )


def main():
    """Main function to parse arguments and initiate subset creation."""
    parser = argparse.ArgumentParser(
        description=(
            "Create stratified, nested training data subsets for "
            "hyperparameter optimization."
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
        help=(
            "A list of fractions for the subsets, "
            "e.g., --fractions 0.5 0.2 0.05"
        ),
    )
    args = parser.parse_args()

    # Load configuration from the specified YAML file
    config = load_config(args.config)
    
    # Pass the loaded config object to the subset creation function
    create_training_subsets(config, args.input_file, args.fractions)


if __name__ == "__main__":
    main()