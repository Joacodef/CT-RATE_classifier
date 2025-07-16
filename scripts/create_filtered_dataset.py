import pandas as pd
import logging
from pathlib import Path

# --- Configuration ---
BASE_DATA_DIR = Path("E:/ProyectoRN/data") 
METADATA_DIR = BASE_DATA_DIR / "metadata"
EXCLUSION_DIR = BASE_DATA_DIR

TRAIN_METADATA_PATH = METADATA_DIR / "train_metadata.csv"
VALID_METADATA_PATH = METADATA_DIR / "valid_metadata.csv"

MANUAL_LABELS_PATH = EXCLUSION_DIR / "all_manual_labels.csv"
BRAIN_SCANS_PATH = EXCLUSION_DIR / "brain_scans.txt"
MISSING_Z_PATH = EXCLUSION_DIR / "missing_z_space.txt"

OUTPUT_FILENAME = BASE_DATA_DIR / "filtered_master_list.csv"


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_name_from_path(path_str: str) -> str:
    """
    Takes any string (path or filename) and returns a clean,
    extension-free, whitespace-free filename.
    """
    if not isinstance(path_str, str) or not path_str:
        return ""
    # Extract the base filename from the path
    filename = Path(path_str).name
    # Clean and return it
    return filename.replace(".nii.gz", "").replace(".nii", "").strip()

def create_filtered_dataset():
    """
    Generates a filtered list of volume names by excluding specified scans
    using correct normalization and sequential, mutually exclusive logic.
    """
    logger.info("Starting dataset filtering process...")
    
    # --- 1. Load all available volumes ---
    try:
        train_df = pd.read_csv(TRAIN_METADATA_PATH)
        valid_df = pd.read_csv(VALID_METADATA_PATH)
        all_volumes_df = pd.concat([train_df, valid_df], ignore_index=True)
        # Get a clean, unique list of the original VolumeNames
        all_volume_names_raw = all_volumes_df['VolumeName'].dropna().unique().tolist()
        logger.info(f"Loaded {len(all_volume_names_raw)} unique raw volume names from metadata.")
    except FileNotFoundError as e:
        logger.error(f"Metadata file not found: {e}. Please check the paths defined in the script.")
        return

    # --- 2. Load EXCLUSION lists ---
    
    # Load lists for EXACT match (brain scans, missing z)
    exact_match_exclusions = set()
    try:
        with open(BRAIN_SCANS_PATH, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
            exact_match_exclusions.update({normalize_name_from_path(p) for p in paths})
        logger.info(f"Loaded {len(paths)} brain scan volumes for exact-match exclusion.")
    except FileNotFoundError:
        logger.warning(f"File not found: {BRAIN_SCANS_PATH}. Skipping.")
        
    try:
        with open(MISSING_Z_PATH, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
            exact_match_exclusions.update({normalize_name_from_path(p) for p in paths})
        logger.info(f"Loaded {len(paths)} missing z-space volumes for exact-match exclusion.")
    except FileNotFoundError:
        logger.warning(f"File not found: {MISSING_Z_PATH}. Skipping.")
    
    logger.info(f"Total unique volumes for exact-match exclusion: {len(exact_match_exclusions)}")

    # Load list for PREFIX match (manual labels)
    manual_label_prefixes = set()
    try:
        manual_exclude_df = pd.read_csv(MANUAL_LABELS_PATH)
        # Prefixes should just be stripped of whitespace
        manual_label_prefixes = {name.strip() for name in manual_exclude_df['VolumeName'].dropna()}
        logger.info(f"Loaded {len(manual_label_prefixes)} prefixes for prefix-based exclusion.")
    except FileNotFoundError:
        logger.warning(f"File not found: {MANUAL_LABELS_PATH}. Skipping.")

    # --- 3. Perform Sequential Filtering ---
    final_filtered_list = []
    removed_by_exact = 0
    removed_by_prefix = 0
    
    prefixes_tuple = tuple(manual_label_prefixes) if manual_label_prefixes else None

    for raw_name in all_volume_names_raw:
        # Normalize the name from the main list for the exact-match check
        normalized_name_for_check = normalize_name_from_path(raw_name)

        # Check 1: Is the normalized name in the exact-match exclusion set?
        if normalized_name_for_check in exact_match_exclusions:
            removed_by_exact += 1
            continue  # Skip this name and do not check the next condition

        # Check 2 (only if Check 1 failed): Does the raw name start with an exclusion prefix?
        if prefixes_tuple and raw_name.startswith(prefixes_tuple):
            removed_by_prefix += 1
            continue  # Skip this name

        # If the name passes all checks, keep it
        final_filtered_list.append(raw_name)

    logger.info(f"Removed {removed_by_exact} volumes by exact match.")
    logger.info(f"Removed {removed_by_prefix} volumes by prefix match (from the remainder).")
    
    # --- 4. Final Count and Save ---
    final_filtered_list.sort()
    logger.info(f"Total volumes remaining after all filtering: {len(final_filtered_list)}")
    
    # Verification step
    expected_final_count = len(all_volume_names_raw) - removed_by_exact - removed_by_prefix
    logger.info(f"Verification: {len(all_volume_names_raw)} - {removed_by_exact} - {removed_by_prefix} = {expected_final_count}")
    if len(final_filtered_list) != expected_final_count:
        logger.error("LOGIC ERROR: Final count does not match verification count!")

    output_df = pd.DataFrame(final_filtered_list, columns=['VolumeName'])
    output_df.to_csv(OUTPUT_FILENAME, index=False)
    logger.info(f"Successfully saved the filtered volume list to: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    create_filtered_dataset()