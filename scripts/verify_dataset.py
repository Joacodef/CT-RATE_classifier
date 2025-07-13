# scripts/verify_dataset.py

import argparse
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys

# Add the project root to the Python path to allow importing from 'src'
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

try:
    # Correctly import from src.config.config
    from src.config.config import load_config
    from src.data.utils import get_dynamic_image_path
except ImportError as e:
    print(f"Error: Failed to import project modules. Make sure you are running this script "
          f"from the root directory of the project.\nDetails: {e}")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_dataset_integrity(dataset_name: str, csv_path: Path, img_dir: Path, structure: str):
    """
    Checks if all volumes listed in a CSV file exist in the target directory.

    Args:
        dataset_name (str): The name of the dataset being checked (e.g., 'Training', 'Validation').
        csv_path (Path): Path to the dataset CSV file (e.g., train.csv).
                         The CSV must contain a 'VolumeName' column.
        img_dir (Path): Path to the base directory containing the CT scan files.
        structure (str): The directory structure, either 'flat' or 'nested'.
    """
    header = f" Verifying {dataset_name} Dataset "
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("=" * len(header))

    if not csv_path.is_file():
        logger.error(f"CSV file not found at: {csv_path}")
        return

    if not img_dir.is_dir():
        logger.error(f"Image directory not found at: {img_dir}")
        return

    logger.info(f"Loading dataset list from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        if 'VolumeName' not in df.columns:
            logger.error("CSV file must contain a 'VolumeName' column.")
            return
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return

    volume_ids = df['VolumeName'].tolist()
    total_files = len(volume_ids)
    missing_files = []

    logger.info(f"Checking for {total_files} files in '{img_dir}' using '{structure}' structure...")

    # Iterate through all volume IDs with a progress bar
    for volume_id in tqdm(volume_ids, desc=f"Verifying {dataset_name} files"):
        # Use the centralized utility function to construct the path
        try:
            file_path = get_dynamic_image_path(
                base_dir=img_dir,
                volume_name=str(volume_id),
                dir_structure=structure
            )

            if not file_path.exists():
                missing_files.append(str(file_path))
        except ValueError as e:
            logger.error(f"Error processing volume '{volume_id}': {e}")
            continue

    # --- Report Results ---
    logger.info(f"\n--- {dataset_name} Verification Complete ---")
    logger.info(f"Total files to check: {total_files}")
    logger.info(f"Files found: {total_files - len(missing_files)}")
    logger.info(f"Files missing: {len(missing_files)}")
    
    if missing_files:
        logger.warning("The following files are missing:")
        for missing_file in missing_files:
            logger.warning(f"-> {missing_file}")
    else:
        logger.info(f"🎉 All {dataset_name} dataset files were found successfully!")
    logger.info("") # Add a blank line for spacing


def main(config_path: str):
    """
    Main function to load configuration and run verification for all specified datasets.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    """
    logger.info(f"Loading project configuration from: {config_path}")
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Verify the training dataset
    verify_dataset_integrity(
        dataset_name="Training",
        csv_path=config.paths.data_subsets.train,
        img_dir=config.paths.train_img_dir,
        structure=config.paths.dir_structure
    )

    # Verify the validation dataset
    verify_dataset_integrity(
        dataset_name="Validation",
        csv_path=config.paths.data_subsets.valid,
        img_dir=config.paths.valid_img_dir,
        structure=config.paths.dir_structure
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Verify the integrity of a CT dataset by checking for missing files."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to the YAML configuration file. Defaults to "configs/config.yaml".'
    )
    args = parser.parse_args()
    main(args.config)