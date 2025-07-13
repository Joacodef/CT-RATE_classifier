# scripts/verify_dataset.py

import argparse
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

try:
    from src.config.config import load_config
    from src.data.utils import get_dynamic_image_path
except ImportError as e:
    print(f"Error: Failed to import project modules. Run this script from the project root.\nDetails: {e}")
    sys.exit(1)

# --- Logging Configuration ---
log_directory = project_root / "logs"
log_directory.mkdir(exist_ok=True)
# Match log file name to script name
log_file_path = log_directory / "verify_dataset.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'), # Overwrite log each run
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
disable_progress_bars()


def find_missing_files(csv_path: Path, img_dir: Path, structure: str) -> list[str]:
    """Scans a dataset definition and returns a list of missing volume names."""
    if not csv_path.is_file():
        logger.error(f"CSV file not found: {csv_path}")
        return []
    
    df = pd.read_csv(csv_path)
    missing_volumes = []
    
    for _, row in df.iterrows():
        volume_name = row['VolumeName']
        expected_path = get_dynamic_image_path(
            base_dir=img_dir,
            volume_name=str(volume_name),
            dir_structure=structure
        )
        if not expected_path.exists():
            missing_volumes.append(str(volume_name))
            
    return missing_volumes


def download_worker(volume_name: str, config: object) -> str:
    """
    Worker function to download and then copy a single file from Hugging Face.
    """
    cfg_downloads = config.downloads
    
    try:
        # Construct the repository subfolder path from the volume name
        parts = volume_name.split('_')
        if len(parts) >= 3:
            split = parts[0]
            folder1 = f"{parts[0]}_{parts[1]}"
            folder2 = f"{parts[0]}_{parts[1]}_{parts[2]}"
            repo_subfolder = f"dataset/{split}_fixed/{folder1}/{folder2}"
        else:
            split = parts[0] if parts else 'unknown_split'
            repo_subfolder = f"dataset/{split}"
            logging.warning(f"Volume '{volume_name}' using base subfolder '{repo_subfolder}'.")

        # Sanitize the filename to ensure it ends correctly
        volume_stem = volume_name.replace(".nii.gz", "").replace(".nii", "")
        correct_filename = f"{volume_stem}.nii.gz"

        # Step 1: Download the file to the HF cache
        logger.info(f"Downloading {correct_filename}...")
        temp_file_path = hf_hub_download(
            repo_id=cfg_downloads.repo_id,
            repo_type='dataset',
            token=cfg_downloads.hf_token,
            subfolder=repo_subfolder,
            filename=correct_filename,
            cache_dir=cfg_downloads.hf_cache_dir,
            resume_download=True,
        )

        # Step 2: Determine the correct final destination
        img_dir = config.paths.train_img_dir if "train" in volume_name else config.paths.valid_img_dir
        final_destination = get_dynamic_image_path(
            base_dir=img_dir,
            volume_name=volume_name,
            dir_structure=config.paths.dir_structure
        )

        # Step 3: Copy the downloaded file to its final destination
        logger.info(f"Copying {correct_filename} to {final_destination}")
        final_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(temp_file_path, final_destination)

        logger.info(f"Successfully placed {volume_name} at {final_destination}")
        return f"OK: {volume_name}"

    except Exception as e:
        error_message = f"Failed during processing of {volume_name}: {e}"
        logger.error(error_message, exc_info=True)
        return f"FAIL: {volume_name} | Error: {e}"


def main(config_path: str):
    """Main orchestrator function."""
    logger.info("--- Starting Dataset Verification and Download Script ---")
    config = load_config(config_path)

    logger.info("Step 1: Checking for missing files...")
    missing_train_files = find_missing_files(
        config.paths.data_subsets.train, config.paths.train_img_dir, config.paths.dir_structure
    )
    missing_valid_files = find_missing_files(
        config.paths.data_subsets.valid, config.paths.valid_img_dir, config.paths.dir_structure
    )
    all_missing_files = missing_train_files + missing_valid_files
    
    if not all_missing_files:
        logger.info("All dataset files are already present. No download needed.")
        return

    logger.info("-" * 50)
    logger.info(f"Found {len(missing_train_files)} missing training files.")
    logger.info(f"Found {len(missing_valid_files)} missing validation files.")
    logger.info(f"Total missing files: {len(all_missing_files)}")
    
    try:
        user_input = input("Do you want to download these missing files? (Y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        logger.info("\nDownload cancelled by user.")
        return

    if user_input != 'y':
        logger.info("Download cancelled by user.")
        return

    logger.info(f"Step 2: Starting parallel download of {len(all_missing_files)} files...")
    download_task = partial(download_worker, config=config)
    
    with ThreadPoolExecutor(max_workers=config.downloads.max_workers) as executor:
        results = list(tqdm(
            executor.map(download_task, all_missing_files),
            total=len(all_missing_files),
            desc="Downloading files"
        ))

    success_count = sum(1 for r in results if r.startswith("OK"))
    fail_count = len(results) - success_count
    logger.info("\n--- Download Summary ---")
    logger.info(f"Download complete. {success_count} successful, {fail_count} failed.")
    logger.info(f"Check the log file '{log_file_path.resolve()}' for detailed results.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check for and download missing files for the CT-RATE dataset."
    )
    parser.add_argument(
        '--config', type=str, default='configs/config.yaml', help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()
    main(args.config)