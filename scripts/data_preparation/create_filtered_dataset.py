import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from src.config.config import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_name_from_path(path_str: str) -> str:
    """
    Takes any string (path or filename) and returns a clean,
    extension-free, whitespace-free filename.
    """
    if not isinstance(path_str, str) or not path_str:
        return ""
    filename = Path(path_str).name
    return filename.replace(".nii.gz", "").replace(".nii", "").strip()


def natural_sort_key(s: str):
    """
    Creates a key for natural sorting.
    "train_10_a_1" -> ["train_", 10, "_a_", 1]
    "train_2_a_1" -> ["train_", 2, "_a_", 1]
    Python then correctly sorts 2 before 10.
    """
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)
    ]


def create_filtered_dataset(config):
    """
    Generates a filtered list of volume names by excluding specified scans
    using correct normalization and sequential, mutually exclusive logic.
    """
    logger.info("Starting dataset filtering process...")

    # --- 1. Load all available volumes ---
    try:
        train_df = pd.read_csv(config.paths.metadata.train)
        valid_df = pd.read_csv(config.paths.metadata.valid)
        all_volumes_df = pd.concat([train_df, valid_df], ignore_index=True)
        all_volume_names_raw = (
            all_volumes_df["VolumeName"].dropna().unique().tolist()
        )
        logger.info(
            f"Loaded {len(all_volume_names_raw)} unique raw volume names from metadata."
        )
    except FileNotFoundError as e:
        logger.error(
            f"Metadata file not found: {e}. Please check the paths defined in your config file."
        )
        return

    # --- 2. Load EXCLUSION lists ---
    exact_match_exclusions = set()
    try:
        with open(config.paths.exclusion_files.brain_scans, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
            exact_match_exclusions.update({normalize_name_from_path(p) for p in paths})
        logger.info(
            f"Loaded {len(paths)} brain scan volumes for exact-match exclusion."
        )
    except FileNotFoundError:
        logger.warning(
            f"File not found: {config.paths.exclusion_files.brain_scans}. Skipping."
        )

    try:
        with open(config.paths.exclusion_files.missing_z, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
            exact_match_exclusions.update({normalize_name_from_path(p) for p in paths})
        logger.info(
            f"Loaded {len(paths)} missing z-space volumes for exact-match exclusion."
        )
    except FileNotFoundError:
        logger.warning(
            f"File not found: {config.paths.exclusion_files.missing_z}. Skipping."
        )

    logger.info(
        f"Total unique volumes for exact-match exclusion: {len(exact_match_exclusions)}"
    )

    manual_label_prefixes = set()
    try:
        manual_exclude_df = pd.read_csv(config.paths.exclusion_files.manual_labels)
        manual_label_prefixes = {
            name.strip() for name in manual_exclude_df["VolumeName"].dropna()
        }
        logger.info(
            f"Loaded {len(manual_label_prefixes)} prefixes for prefix-based exclusion."
        )
    except FileNotFoundError:
        logger.warning(
            f"File not found: {config.paths.exclusion_files.manual_labels}. Skipping."
        )

    # --- 3. Perform Sequential Filtering ---
    final_filtered_list = []
    removed_by_exact = 0
    removed_by_prefix = 0

    prefixes_tuple = tuple(manual_label_prefixes) if manual_label_prefixes else None

    for raw_name in all_volume_names_raw:
        normalized_name_for_check = normalize_name_from_path(raw_name)

        if normalized_name_for_check in exact_match_exclusions:
            removed_by_exact += 1
            continue

        if prefixes_tuple and raw_name.startswith(prefixes_tuple):
            removed_by_prefix += 1
            continue

        final_filtered_list.append(raw_name)

    logger.info(f"Removed {removed_by_exact} volumes by exact match.")
    logger.info(
        f"Removed {removed_by_prefix} volumes by prefix match (from the remainder)."
    )

    # --- 4. Final Count and Save ---

    # Sort the list using the natural sort key
    final_filtered_list.sort(key=natural_sort_key)
    logger.info("Final list sorted numerically.")

    logger.info(
        f"Total volumes remaining after all filtering: {len(final_filtered_list)}"
    )

    expected_final_count = (
        len(all_volume_names_raw) - removed_by_exact - removed_by_prefix
    )
    logger.info(
        f"Verification: {len(all_volume_names_raw)} - {removed_by_exact} - {removed_by_prefix} = {expected_final_count}"
    )
    if len(final_filtered_list) != expected_final_count:
        logger.error("LOGIC ERROR: Final count does not match verification count!")

    output_df = pd.DataFrame(final_filtered_list, columns=["VolumeName"])
    # Use full_dataset_csv if available, otherwise fall back to the old key
    if hasattr(config.paths, "full_dataset_csv"):
        output_path = config.paths.full_dataset_csv
    else:
        output_path = config.paths.data_dir / config.paths.output_filename

    output_df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved the filtered volume list to: {output_path}")


def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(
        description="Create a filtered dataset CSV from a config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {args.config}")
        return
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return

    create_filtered_dataset(config)


if __name__ == "__main__":
    main()