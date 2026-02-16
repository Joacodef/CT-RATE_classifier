import argparse
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config import load_config
from src.data.utils import get_dynamic_image_path

# --- Logging Configuration ---
log_directory = project_root / "logs"
log_directory.mkdir(exist_ok=True)
log_file_path = log_directory / "verify_and_download.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
disable_progress_bars()


def find_missing_files(csv_path: Path, img_dir: Path, structure: str) -> list[str]:
    """
    Scans a dataset definition CSV and returns a list of volume names
    for files that are not found on the local disk.
    """
    if not csv_path.is_file():
        logger.error(f"Dataset CSV file not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    return [
        str(row['VolumeName'])
        for _, row in df.iterrows()
        if not get_dynamic_image_path(img_dir, str(row['VolumeName']), structure).exists()
    ]


def download_worker(volume_name: str, config: SimpleNamespace, hf_cache_dir: Path) -> str:
    """
    Worker function to download a single volume from Hugging Face.

    Args:
        volume_name: The name of the volume to download.
        config: A namespace object containing download and path configurations.
        hf_cache_dir: The path to the temporary cache directory for Hugging Face downloads.
    """
    cfg_downloads = config.downloads
    try:
        parts = volume_name.split('_')
        repo_subfolder = f"dataset/{parts[0]}_fixed/{parts[0]}_{parts[1]}/{parts[0]}_{parts[1]}_{parts[2]}" if len(parts) >= 3 else f"dataset/{parts[0]}"
        correct_filename = f"{volume_name.replace('.nii.gz', '')}.nii.gz"
        
        logger.debug(f"Downloading {volume_name} to temporary cache: {hf_cache_dir}")
        
        temp_file_path = hf_hub_download(
            repo_id=cfg_downloads.repo_id,
            repo_type='dataset',
            token=cfg_downloads.hf_token,
            subfolder=repo_subfolder,
            filename=correct_filename,
            cache_dir=hf_cache_dir,
            resume_download=True
        )

        final_destination = get_dynamic_image_path(
            config.paths.img_dir, volume_name, config.paths.dir_structure
        )
        
        final_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(temp_file_path, final_destination)
        return f"OK: {volume_name}"
    except Exception as e:
        logger.error(f"Failed during processing of {volume_name}: {e}", exc_info=False)
        return f"FAIL: {volume_name}"


def main(config_path: str, subset: str = "train"):
    """Main orchestrator function that finds and downloads missing dataset files."""
    logger.info("--- Starting Dataset Verification and Download Script ---")
    config = load_config(config_path)

    data_subsets = config.paths.data_subsets
    if subset == "train":
        dataset_paths = [Path(data_subsets.train)]
    elif subset == "valid":
        dataset_paths = [Path(data_subsets.valid)]
    else:  # subset == "all"
        dataset_paths = [Path(data_subsets.train), Path(data_subsets.valid)]

    missing_files_set = set()
    for dataset_path in dataset_paths:
        logger.info(f"Using dataset definition from: {dataset_path}")
        missing_for_split = find_missing_files(
            dataset_path, Path(config.paths.img_dir), config.paths.dir_structure
        )
        missing_files_set.update(missing_for_split)

    all_missing_files = sorted(missing_files_set)
    
    # Define and create a temporary directory for HF downloads
    temp_hf_cache_dir = Path(config.paths.data_dir) / "temp_hf_download_cache"
    temp_hf_cache_dir.mkdir(exist_ok=True)

    try:
        if all_missing_files:
            logger.info(f"\nFound {len(all_missing_files)} total missing files.")
            if input("Do you want to download them? (Y/N): ").strip().lower() == 'y':
                task = partial(download_worker, config=config, hf_cache_dir=temp_hf_cache_dir)
                with ThreadPoolExecutor(max_workers=config.downloads.max_workers) as executor:
                    list(tqdm(executor.map(task, all_missing_files), total=len(all_missing_files), desc="Downloading"))
            else:
                logger.info("Download declined by user.")
        else:
            logger.info("All dataset files specified in the CSV are already present.")

    except (EOFError, KeyboardInterrupt):
        logger.info("\nDownload cancelled by user.")
    finally:
        # Ensure the temporary cache is always cleaned up
        logger.info(f"Cleaning up temporary Hugging Face cache directory: {temp_hf_cache_dir}")
        shutil.rmtree(temp_hf_cache_dir)
        logger.info("Temporary cache cleanup complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Verify that all dataset files exist and download any missing ones."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to the YAML config file.'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='train',
        choices=['train', 'valid', 'all'],
        help="Which split to verify/download: 'train', 'valid', or 'all'."
    )
    args = parser.parse_args()
    
    main(args.config, args.subset)