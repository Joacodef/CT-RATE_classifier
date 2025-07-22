# scripts/generate_cache.py

import argparse
import functools
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# --- MONAI Imports ---
from monai.data import DataLoader, PersistentDataset

# --- Project Setup ---
# Ensures that the script can find modules in the 'src' directory.
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import load_config
from src.data.cache_utils import (
    deterministic_hash,
    get_or_create_cache_subdirectory,
    worker_init_fn
)
from src.data.dataset import CTMetadataDataset
from src.data.transforms import get_preprocessing_transforms
from src.data.utils import get_dynamic_image_path

# --- Logging Configuration ---
# Sets up logging to both a file and the console for clear progress tracking.
log_directory = project_root / "logs"
log_directory.mkdir(exist_ok=True)
log_file_path = log_directory / "generate_cache.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Apply a global patch to torch.load for the main process.
# This is necessary to deserialize MONAI's MetaTensors.
# The worker_init_fn will handle this for worker processes.
torch.load = functools.partial(torch.load, weights_only=False)


# --- Error-Handling Dataset Wrapper ---

class CachingDataset(torch.utils.data.Dataset):
    """
    A wrapper for PersistentDataset that safely handles errors during item access.
    This is used to pre-cache a dataset using multiple workers in a DataLoader,
    allowing the main process to identify and log corrupt or problematic files
    without crashing the entire caching process.
    """
    def __init__(self, base_dataset, transform, cache_dir):
        """
        Initializes the CachingDataset.

        Args:
            base_dataset: The underlying dataset to be cached.
            transform: The MONAI transforms to apply to each item.
            cache_dir: The directory where cached items will be stored.
        """
        # The underlying dataset that loads and transforms data, caching the results.
        self.persistent_ds = PersistentDataset(
            data=base_dataset,
            transform=transform,
            cache_dir=cache_dir,
            hash_func=deterministic_hash
        )
        # A reference to the base dataset is kept to access original metadata
        # in case an error occurs during the transform pipeline.
        self.base_ds = base_dataset

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return len(self.persistent_ds)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves an item from the PersistentDataset. If an exception occurs
        during loading or transformation, it catches the error and returns a
        dictionary containing error information instead of raising the exception.
        """
        try:
            # Attempt to get the (potentially cached) transformed item.
            return self.persistent_ds[idx]
        except Exception as e:
            # If an error occurs, retrieve the original volume name.
            volume_name = self.base_ds[idx]['volume_name']
            # Return a dictionary indicating failure for this item.
            return {"VolumeName": volume_name, "error": e}


# --- Collate Function for DataLoader ---

def identity_collate(batch):
    """
    A simple collate function that returns the batch as-is.
    This is used when batch_size is None in the DataLoader, where each 'batch'
    is a single item. It prevents the default collate function from attempting
    to stack items, which is not desired during the caching process.
    """
    return batch


# --- Core Script Functions ---

def precache_dataset(config, full_df, num_shards, shard_id, num_workers):
    """
    Verifies and pre-caches existing files in a dataset shard using parallel workers.

    Args:
        config: The application configuration object.
        full_df: A pandas DataFrame containing the full list of dataset items.
        num_shards: The total number of shards the dataset is divided into.
        shard_id: The specific shard index (0-indexed) this run will process.
        num_workers: The number of worker processes for the DataLoader.
    """
    logger.info(
        f"\n--- Starting Dataset Caching ---\n"
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
    logger.info(f"Found {len(existing_files_df)} existing files to process.")

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

    base_ds = CTMetadataDataset(
        dataframe=shard_df,
        img_dir=config.paths.img_dir,
        path_mode=config.paths.dir_structure
    )

    cache_dir = get_or_create_cache_subdirectory(
        config.paths.cache_dir, preprocess_transforms, split="unified"
    )

    # Use the error-handling wrapper dataset.
    caching_ds = CachingDataset(
        base_dataset=base_ds,
        transform=preprocess_transforms,
        cache_dir=cache_dir
    )

    data_loader = DataLoader(
        caching_ds,
        batch_size=None,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        collate_fn=identity_collate
    )

    corrupt_items = []
    # Iterate through the loader to trigger caching for each item.
    progress_bar = tqdm(data_loader, desc=f"Caching Dataset (Shard {shard_id + 1}/{num_shards})", total=len(caching_ds))
    for item in progress_bar:
        if item and isinstance(item, dict) and 'error' in item:
            logger.error(
                f"\n[Shard {shard_id + 1}/{num_shards}] Detected corrupt file while processing "
                f"'{item['VolumeName']}'. Reason: {item['error']}"
            )
            corrupt_items.append(item['VolumeName'])

    logger.info(f"\n--- Caching Summary (Shard {shard_id + 1}/{num_shards}) ---")
    if not corrupt_items:
        logger.info(f"All {len(caching_ds)} files in Shard {shard_id + 1} were cached successfully!")
    else:
        logger.warning(
            f"{len(caching_ds) - len(corrupt_items)} files were cached successfully. "
            f"Found {len(corrupt_items)} corrupt or unreadable files:"
        )
        for f in corrupt_items:
            logger.warning(f"  - {f}")


def main(config_path: str, num_shards: int, shard_id: int, num_workers: int):
    """Main orchestrator function."""
    if not (0 <= shard_id < num_shards):
        logger.error(f"Error: --shard-id ({shard_id}) must be between 0 and --num-shards-1 ({num_shards - 1}).")
        return

    config = load_config(config_path)

    full_dataset_path = Path(config.paths.data_dir) / config.paths.data_subsets.train
    logger.info(f"Using dataset definition from: {full_dataset_path}")

    if not full_dataset_path.exists():
        logger.error(f"Dataset CSV not found at {full_dataset_path}. Please check config or run the download script.")
        return

    full_df = pd.read_csv(full_dataset_path)
    precache_dataset(config, full_df, num_shards, shard_id, num_workers)


if __name__ == '__main__':
    # This check is essential for multiprocessing on Windows.
    if sys.platform == "win32":
        torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Generate a pre-processed cache for the CT-RATE dataset.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the YAML config file.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers for caching.')
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of parallel shards to divide the dataset into.')
    parser.add_argument('--shard-id', type=int, default=0, help='The ID of this specific shard to process (0-indexed).')
    args = parser.parse_args()
    
    main(args.config, args.num_shards, args.shard_id, args.num_workers)