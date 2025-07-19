# scripts/verify_dataset.py

import argparse
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import functools

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

# --- MONAI Imports ---
from monai.data import PersistentDataset, DataLoader

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


from src.config import load_config
from src.data.dataset import CTMetadataDataset
from src.data.utils import get_dynamic_image_path
from src.data.cache_utils import (
    get_or_create_cache_subdirectory,
    deterministic_hash,
    worker_init_fn  # Import the worker_init_fn
)
from src.data.transforms import get_preprocessing_transforms


# --- Logging Configuration ---
log_directory = project_root / "logs"
log_directory.mkdir(exist_ok=True)
log_file_path = log_directory / "verify_dataset.log"

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

# Apply a global patch to torch.load for the main process.
# This is necessary to deserialize MONAI's MetaTensors.
# The worker_init_fn will handle this for worker processes.
# logger.info("Applying global patch to torch.load to allow MetaTensor deserialization.")
torch.load = functools.partial(torch.load, weights_only=False)

# --- Error-Handling Dataset Wrapper ---

class VerifyingDataset(torch.utils.data.Dataset):
    """
    A wrapper for PersistentDataset that safely handles errors during item access.
    This is used to pre-cache a dataset using multiple workers in a DataLoader,
    allowing the main process to identify and log corrupt or problematic files
    without crashing the entire process.
    """
    def __init__(self, base_dataset, transform, cache_dir):
        """Initializes the VerifyingDataset."""
        # The underlying dataset that loads and transforms data, caching the results.
        self.persistent_ds = PersistentDataset(
            data=base_dataset,
            transform=transform,    
            cache_dir=cache_dir,
            hash_func=deterministic_hash
        )
        # A reference to the base dataset is kept to access original metadata (e.g., VolumeName)
        # even if the transform fails early.
        self.base_ds = base_dataset

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return len(self.persistent_ds)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves an item from the PersistentDataset. If an exception occurs
        during loading or transformation, it catches the error and returns a
        dictionary containing error information instead of crashing the worker.
        """
        try:
            # Attempt to get the (potentially cached) transformed item.
            return self.persistent_ds[idx]
        except Exception as e:
            # If an error occurs (e.g., corrupt file, transform issue),
            # retrieve the original volume name from the base dataset.
            volume_name = self.base_ds[idx]['volume_name']
            # Return a dictionary indicating failure. This will be handled by the main loop.
            return {"VolumeName": volume_name, "error": e}

# --- Collate Function for DataLoader ---

def identity_collate(batch):
    """
    A simple collate function that just returns the item as-is.
    This is necessary because when batch_size=None, the default `list_data_collate`
    expects a list of items, but it receives a single dictionary, causing a KeyError.
    It is defined at the top level to be pickleable for multiprocessing.
    """
    return batch

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
        if not get_dynamic_image_path(img_dir, str(row['VolumeName']), structure).exists()
    ]
    return missing_volumes

def download_worker(volume_name: str, config: SimpleNamespace) -> str:
    """Worker function to download and then copy a single file from Hugging Face."""
    cfg_downloads = config.downloads
    try:
        parts = volume_name.split('_')
        repo_subfolder = f"dataset/{parts[0]}_fixed/{parts[0]}_{parts[1]}/{parts[0]}_{parts[1]}_{parts[2]}" if len(parts) >= 3 else f"dataset/{parts[0]}"
        correct_filename = f"{volume_name.replace('.nii.gz', '')}.nii.gz"
        
        temp_file_path = hf_hub_download(
            repo_id=cfg_downloads.repo_id, repo_type='dataset', token=cfg_downloads.hf_token,
            subfolder=repo_subfolder, filename=correct_filename, cache_dir=cfg_downloads.hf_cache_dir,
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



def precache_dataset(config: SimpleNamespace, full_df: pd.DataFrame, num_shards: int, shard_id: int, num_workers: int):
    """
    Verifies and pre-caches existing files in a dataset shard using parallel workers.

    This function filters the dataframe to include only existing files, then uses a
    shared, centralized preprocessing pipeline to create a self-contained,
    label-free cache of the processed image data.
    """
    logger.info(
        f"\n--- Starting Unified Dataset Verification and Caching ---\n"
        f"Shard: {shard_id + 1}/{num_shards} | Parallel Workers: {num_workers}"
    )

    # --- Filter dataframe for files that actually exist on disk ---
    logger.info("Scanning for existing files specified in the input dataframe...")
    existing_rows = [
        row for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Checking file existence")
        if get_dynamic_image_path(config.paths.img_dir, str(row['VolumeName']), config.paths.dir_structure).exists()
    ]

    if not existing_rows:
        logger.warning("No existing files from the provided list were found. Nothing to cache.")
        return

    existing_files_df = pd.DataFrame(existing_rows)
    logger.info(f"Found {len(existing_files_df)} existing files out of {len(full_df)} total entries listed.")

    # Get the centralized, reusable preprocessing pipeline.
    preprocess_transforms = get_preprocessing_transforms(config)

    # --- Sharding Logic ---
    total_files = len(existing_files_df)
    shard_size = (total_files + num_shards - 1) // num_shards
    start_index = shard_id * shard_size
    end_index = min(start_index + shard_size, total_files)

    if start_index >= total_files:
        logger.info(f"No files for shard {shard_id + 1}. Skipping.")
        return

    shard_df = existing_files_df.iloc[start_index:end_index].copy()
    logger.info(f"[Shard {shard_id + 1}/{num_shards}] Processing {len(shard_df)} files from index {start_index} to {end_index-1}.")

    # Use the label-agnostic dataset. No dummy labels are needed.
    base_ds = CTMetadataDataset(
        dataframe=shard_df,
        img_dir=config.paths.img_dir,
        path_mode=config.paths.dir_structure
    )

    cache_dir = get_or_create_cache_subdirectory(
        config.paths.cache_dir, preprocess_transforms, split="unified"
    )

    # Use the error-handling wrapper dataset.
    verifying_ds = VerifyingDataset(
        base_dataset=base_ds,
        transform=preprocess_transforms,
        cache_dir=cache_dir
    )

    # Use DataLoader to parallelize the caching process.
    data_loader = DataLoader(
        verifying_ds,
        batch_size=None,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        collate_fn=identity_collate
    )

    corrupt_items = []
    # Iterate through the loader to trigger caching for each item.
    progress_bar = tqdm(data_loader, desc=f"Verifying Dataset (Shard {shard_id + 1}/{num_shards})", total=len(verifying_ds))
    for item in progress_bar:
        if item and isinstance(item, dict) and 'error' in item:
            logger.error(
                f"\n[Shard {shard_id + 1}/{num_shards}] Detected corrupt file while analyzing "
                f"'{item['VolumeName']}'. Reason: {item['error']}"
            )
            corrupt_items.append(item['VolumeName'])

    logger.info(f"\n--- Verification and Caching Summary (Shard {shard_id + 1}/{num_shards}) ---")
    if not corrupt_items:
        logger.info(f"All {len(verifying_ds)} files in Shard {shard_id + 1} were verified/cached successfully!")
    else:
        logger.warning(
            f"{len(verifying_ds) - len(corrupt_items)} files were verified/cached successfully. "
            f"Found {len(corrupt_items)} corrupt or unreadable files in Shard {shard_id + 1}:"
        )
        for f in corrupt_items:
            logger.warning(f"  - {f}")


def main(config_path: str, generate_cache: bool, num_shards: int, shard_id: int, num_workers: int):
    """Main orchestrator function."""
    if generate_cache and not (0 <= shard_id < num_shards):
        logger.error(f"Error: --shard-id ({shard_id}) must be between 0 and --num-shards-1 ({num_shards - 1}).")
        return

    logger.info("--- Starting Dataset Verification Script ---")
    config = load_config(config_path)

    # Determine the dataset list to use from config.
    full_dataset_path = Path(config.paths.data_dir) / config.paths.data_subsets.train
    logger.info(f"Using dataset definition from: {full_dataset_path}")

    if not generate_cache:
        all_missing_files = find_missing_files(
            full_dataset_path, config.paths.img_dir, config.paths.dir_structure
        )
        if all_missing_files:
            logger.info(f"\nFound {len(all_missing_files)} total missing files.")            
            try:
                if input("Do you want to download them? (Y/N): ").strip().lower() == 'y':
                    task = partial(download_worker, config=config)
                    with ThreadPoolExecutor(max_workers=config.downloads.max_workers) as executor:
                        list(tqdm(executor.map(task, all_missing_files), total=len(all_missing_files), desc="Downloading"))
            except (EOFError, KeyboardInterrupt):
                logger.info("\nDownload cancelled.")
        else:
            logger.info("All dataset files are already present.")
        return

    if generate_cache:
        if not full_dataset_path.exists():
            logger.error(f"Full dataset CSV not found at {full_dataset_path}. Please check config.")
            return
        full_df = pd.read_csv(full_dataset_path)
        precache_dataset(config, full_df, num_shards, shard_id, num_workers)


if __name__ == '__main__':
    # This check is essential for multiprocessing on Windows.
    # It prevents child processes from re-executing the main script's code.
    if sys.platform == "win32":
        torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Verify, download, and cache the CT-RATE dataset.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the YAML config file.')
    parser.add_argument('--generate-cache', action='store_true', help='Run the verification and caching process.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers for caching.')
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of parallel shards to divide the dataset into.')
    parser.add_argument('--shard-id', type=int, default=0, help='The ID of this specific shard to process (0-indexed).')
    args = parser.parse_args()
    
    main(args.config, args.generate_cache, args.num_shards, args.shard_id, args.num_workers)