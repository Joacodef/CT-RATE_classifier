# scripts/verify_dataset.py

import argparse
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

# --- MONAI Imports ---
from monai.data import PersistentDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, Orientationd, Spacingd,
    ScaleIntensityRanged, Resized, EnsureTyped
)

import torch.multiprocessing

if sys.platform == "linux":
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

try:
    from src.config import load_config
    from src.data.dataset import CTMetadataDataset
    from src.data.utils import get_dynamic_image_path
    # Import the new, centralized cache utilities
    from src.data.cache_utils import (
        get_or_create_cache_subdirectory,
        deterministic_json_hash,
        worker_init_fn
    )
except ImportError as e:
    print(f"Error: Failed to import project modules. Run this script from the project root.\nDetails: {e}")
    sys.exit(1)

# --- Logging Configuration ---
log_directory = project_root / "logs"
log_directory.mkdir(exist_ok=True)
log_file_path = log_directory / "verify_dataset.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
# Suppress progress bars from Hugging Face downloader
disable_progress_bars()


# --- Core Script Functions ---

def find_missing_files(csv_path: Path, img_dir: Path, structure: str) -> list[str]:
    """Scans a dataset definition and returns a list of missing volume names."""
    if not csv_path.is_file():
        logger.error(f"CSV file not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    missing_volumes = [
        str(row['VolumeName'])
        for _, row in df.iterrows()
        if not get_dynamic_image_path(
            img_dir, str(row['VolumeName']), structure
        ).exists()
    ]
    return missing_volumes

def download_worker(volume_name: str, config: SimpleNamespace) -> str:
    """Worker function to download and then copy a single file from Hugging Face."""
    cfg_downloads = config.downloads
    try:
        parts = volume_name.split('_')
        repo_subfolder = f"dataset/{parts[0]}_fixed/{parts[0]}_{parts[1]}/{parts[0]}_{parts[1]}_{parts[2]}" if len(parts) >= 3 else f"dataset/{parts[0]}"
        correct_filename = f"{volume_name.replace('.nii.gz', '')}.nii.gz"
        
        logger.info(f"Downloading {correct_filename}...")
        temp_file_path = hf_hub_download(
            repo_id=cfg_downloads.repo_id, repo_type='dataset', token=cfg_downloads.hf_token,
            subfolder=repo_subfolder, filename=correct_filename, cache_dir=cfg_downloads.hf_cache_dir,
            resume_download=True
        )

        img_dir = config.paths.train_img_dir if "train" in volume_name else config.paths.valid_img_dir
        final_destination = get_dynamic_image_path(img_dir, volume_name, config.paths.dir_structure)
        
        logger.info(f"Copying {correct_filename} to {final_destination}")
        final_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(temp_file_path, final_destination)
        return f"OK: {volume_name}"
    except Exception as e:
        logger.error(f"Failed during processing of {volume_name}: {e}", exc_info=True)
        return f"FAIL: {volume_name}"

def precache_dataset(config: SimpleNamespace):
    """
    Generates a MONAI PersistentDataset cache by iterating through all items.
    This serves as a robust check for corrupt files and leverages PersistentDataset's
    own efficient caching logic to skip existing items.
    """
    logger.info("\n--- Starting Dataset Pre-caching Process ---")

    # Define the preprocessing pipeline once.
    preprocess_transforms = Compose([
        LoadImaged(keys="image", image_only=True, ensure_channel_first=True, reader="NibabelReader"),
        Orientationd(keys="image", axcodes=config.image_processing.orientation_axcodes),
        Spacingd(keys="image", pixdim=config.image_processing.target_spacing, mode="bilinear"),
        ScaleIntensityRanged(
            keys="image", a_min=config.image_processing.clip_hu_min,
            a_max=config.image_processing.clip_hu_max, b_min=0.0, b_max=1.0, clip=True
        ),
        Resized(keys="image", spatial_size=config.image_processing.target_shape_dhw, mode="area"),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32)
    ])

    for split in ["train", "valid"]:
        logger.info(f"\nProcessing '{split}' split for caching...")
        df_path = getattr(config.paths.data_subsets, split)
        img_dir = getattr(config.paths, f"{split}_img_dir")

        if not df_path.exists():
            logger.warning(f"Dataframe for '{split}' split not found at {df_path}. Skipping.")
            continue

        base_ds = CTMetadataDataset(
            dataframe=pd.read_csv(df_path), img_dir=img_dir,
            pathology_columns=config.pathologies.columns, path_mode=config.paths.dir_structure
        )

        cache_dir = get_or_create_cache_subdirectory(config.paths.cache_dir, preprocess_transforms, split)

        # Create a PersistentDataset with the *full* dataset.
        # This lets MONAI handle the check for existing files, which is more reliable.
        persistent_ds = PersistentDataset(
            data=base_ds,
            transform=preprocess_transforms,
            cache_dir=cache_dir,
            hash_func=deterministic_json_hash,
            hash_transform=deterministic_json_hash
        )

        # Using num_workers=0 is crucial for getting clear errors on corrupt files.
        loader = DataLoader(
            persistent_ds,
            batch_size=1,
            shuffle=False,
            # Use the number of workers from your config for consistency
            num_workers=config.training.num_workers,
            # Add the worker_init_fn for compatibility with PersistentDataset
            worker_init_fn=worker_init_fn
        )
        
        # This progress bar will now accurately reflect MONAI's process.
        # It will be fast for cached items and slow for new ones.
        progress_bar = tqdm(loader, desc=f"Verifying and Caching {split} data", unit="volume")

        try:
            for item in progress_bar:
                # We can add a description to see which volume is being processed.
                # The item dictionary from the dataloader contains the metadata.
                vol_name = item.get("VolumeName", ["Unknown"])[0]
                progress_bar.set_postfix_str(f"Volume: {vol_name}")
        except RuntimeError as e:
            logger.error(f"\n\n--- A corrupt file was detected during caching! ---")
            logger.error(f"The process failed while trying to load a volume.")
            logger.error("The last volume name shown in the progress bar is likely the corrupt one.")
            logger.error(f"Original error: {e}\n", exc_info=True)
            return  # Stop the caching process
        except Exception as e:
            logger.error(f"\n\nAn unexpected error occurred during caching: {e}", exc_info=True)
            return

    logger.info("\n--- Caching process completed successfully! ---")

def main(config_path: str, generate_cache: bool):
    """Main orchestrator function."""
    logger.info("--- Starting Dataset Verification and Download Script ---")
    config = load_config(config_path)

    logger.info("Step 1: Checking for missing files...")
    all_missing_files = find_missing_files(
        config.paths.data_subsets.train, config.paths.train_img_dir, config.paths.dir_structure
    ) + find_missing_files(
        config.paths.data_subsets.valid, config.paths.valid_img_dir, config.paths.dir_structure
    )

    if all_missing_files:
        logger.info(f"\nFound {len(all_missing_files)} total missing files.")
        try:
            if input("Do you want to download these missing files? (Y/N): ").strip().lower() != 'y':
                logger.info("Download cancelled.")
            else:
                logger.info(f"Step 2: Starting parallel download...")
                task = partial(download_worker, config=config)
                with ThreadPoolExecutor(max_workers=config.downloads.max_workers) as executor:
                    list(tqdm(executor.map(task, all_missing_files), total=len(all_missing_files), desc="Downloading"))
        except (EOFError, KeyboardInterrupt):
            logger.info("\nDownload cancelled by user.")
    else:
        logger.info("All dataset files are already present.")

    if generate_cache:
        final_missing = find_missing_files(
            config.paths.data_subsets.train, config.paths.train_img_dir, config.paths.dir_structure
        ) + find_missing_files(
            config.paths.data_subsets.valid, config.paths.valid_img_dir, config.paths.dir_structure
        )
        if final_missing:
            logger.error("\nCannot generate cache because some files are still missing.")
            return

        try:
            if input("\nAll files present. Generate cache now? (Y/N): ").strip().lower() == 'y':
                precache_dataset(config)
            else:
                logger.info("Caching cancelled by user.")
        except (EOFError, KeyboardInterrupt):
            logger.info("\nCaching cancelled by user.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check, download, and optionally pre-cache the CT-RATE dataset.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--generate-cache', action='store_true', help='If set, generates the MONAI cache after verifying files.')
    args = parser.parse_args()
    main(args.config, args.generate_cache)