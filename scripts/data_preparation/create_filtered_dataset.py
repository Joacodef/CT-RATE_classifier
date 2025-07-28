import argparse
import logging
import re
from pathlib import Path
import sys

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config.config import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_patient_id(volume_name: str) -> str:
    """
    Extracts the patient ID from a volume name string.
    Assumes the format "split_patientID_scanID_reconstructionID".
    Example: "train_123_a_1" -> "123"
    """
    if not isinstance(volume_name, str) or not volume_name:
        return ""
    parts = volume_name.split("_")
    if len(parts) > 1:
        return parts[1]
    return ""


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
    Generates a filtered list of volumes by excluding all scans from any patient
    who appears in the exclusion lists, preventing data leakage.
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

    # --- 2. Identify all PATIENT IDs to exclude ---
    patient_ids_to_exclude = set()

    # From brain_scans.txt and missing_z_space.txt
    exclusion_files = {
        "brain_scans": config.paths.data_dir / config.paths.exclusion_files.brain_scans,
        "missing_z": config.paths.data_dir / config.paths.exclusion_files.missing_z,
    }

    for name, path in exclusion_files.items():
        try:
            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                normalized_names = {normalize_name_from_path(p) for p in lines}
                patient_ids = {get_patient_id(n) for n in normalized_names}
                patient_ids_to_exclude.update(patient_ids)
                logger.info(
                    f"Found {len(patient_ids)} unique patient IDs to exclude from {name} file."
                )
        except FileNotFoundError:
            logger.warning(f"File not found: {path}. Skipping.")

    # From manual_labels.csv
    try:
        manual_exclude_df = pd.read_csv(config.paths.data_dir / config.paths.exclusion_files.manual_labels)
        manual_volume_names = {
            name.strip() for name in manual_exclude_df["VolumeName"].dropna()
        }
        patient_ids = {get_patient_id(n) for n in manual_volume_names}
        patient_ids_to_exclude.update(patient_ids)
        logger.info(
            f"Found {len(patient_ids)} unique patient IDs to exclude from manual labels."
        )
    except FileNotFoundError:
        logger.warning(
            f"File not found: {config.paths.exclusion_files.manual_labels}. Skipping."
        )

    # Remove any empty strings that might have resulted from parsing errors
    patient_ids_to_exclude.discard("")

    logger.info(
        f"Total unique patients to exclude across all lists: {len(patient_ids_to_exclude)}"
    )

    # --- 3. Perform Patient-Level Filtering ---
    final_filtered_list = []
    removed_count = 0

    for raw_name in all_volume_names_raw:
        patient_id = get_patient_id(raw_name)
        if patient_id in patient_ids_to_exclude:
            removed_count += 1
            continue
        final_filtered_list.append(raw_name)

    logger.info(f"Removed {removed_count} volumes belonging to excluded patients.")

    # --- 4. Final Count and Save ---
    final_filtered_list.sort(key=natural_sort_key)
    logger.info("Final list sorted numerically.")

    logger.info(
        f"Total volumes remaining after all filtering: {len(final_filtered_list)}"
    )

    expected_final_count = len(all_volume_names_raw) - removed_count
    logger.info(
        f"Verification: {len(all_volume_names_raw)} - {removed_count} = {expected_final_count}"
    )
    if len(final_filtered_list) != expected_final_count:
        logger.error("LOGIC ERROR: Final count does not match verification count!")

    output_df = pd.DataFrame(final_filtered_list, columns=["VolumeName"])
    output_path = Path(config.paths.data_dir) / config.paths.output_filename
    
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