# scripts/cache_management/generate_cache.py

import argparse
import functools
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# --- MONAI Imports ---
from monai.data import DataLoader, PersistentDataset

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[2]
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
from scripts.data_preparation.verify_and_download import download_worker

# --- Logging Configuration ---
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

torch.load = functools.partial(torch.load, weights_only=False)

class CachingDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform, cache_dir):
        self.persistent_ds = PersistentDataset(
            data=base_dataset, transform=transform,
            cache_dir=cache_dir, hash_func=deterministic_hash
        )
        self.base_ds = base_dataset

    def __len__(self) -> int:
        return len(self.persistent_ds)

    def __getitem__(self, idx: int) -> dict:
        try:
            return self.persistent_ds[idx]
        except Exception as e:
            volume_name = self.base_ds[idx]['volume_name']
            return {"VolumeName": volume_name, "error": e}

def identity_collate(batch):
    return batch

def analyze_cache_state(volumes_df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    if not cache_dir.exists():
        logger.info("Cache directory does not exist. All volumes are missing.")
        return volumes_df
    
    missing_volumes = []
    logger.info(f"Analyzing cache state in: {cache_dir}")
    for _, row in tqdm(volumes_df.iterrows(), total=len(volumes_df), desc="Checking cache status"):
        data_item = {"volume_name": row['VolumeName']}
        hashed_filename_str = deterministic_hash(data_item).decode('utf-8')
        cache_filepath = cache_dir / f"{hashed_filename_str}.pt"
        if not cache_filepath.exists():
            missing_volumes.append(row.to_dict())
    return pd.DataFrame(missing_volumes) if missing_volumes else pd.DataFrame()

def process_in_batches(config, files_df: pd.DataFrame, batch_size: int, num_workers: int):
    num_batches = (len(files_df) + batch_size - 1) // batch_size
    logger.info(f"Processing {len(files_df)} files in {num_batches} batches of size {batch_size}.")

    preprocess_transforms = get_preprocessing_transforms(config)
    cache_dir = get_or_create_cache_subdirectory(
        Path(config.paths.cache_dir), preprocess_transforms, split="unified"
    )
    
    # Create a single parent directory for all temporary batch caches
    main_temp_hf_dir = Path(config.paths.data_dir) / "temp_hf_batch_cache"
    main_temp_hf_dir.mkdir(exist_ok=True)

    try:
        for i in range(num_batches):
            batch_num = i + 1
            logger.info(f"\n--- Starting Batch {batch_num}/{num_batches} ---")
            
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_df = files_df.iloc[start_idx:end_idx]

            # Create a dedicated temporary cache for this specific batch
            batch_hf_cache_dir = main_temp_hf_dir / f"batch_{batch_num}"
            batch_hf_cache_dir.mkdir(exist_ok=True)

            volumes_to_download = [
                row['VolumeName'] for _, row in batch_df.iterrows()
                if not get_dynamic_image_path(Path(config.paths.img_dir), row['VolumeName'], config.paths.dir_structure).exists()
            ]
            
            downloaded_volume_names = set()
            if volumes_to_download:
                logger.info(f"[Batch {batch_num}] Downloading {len(volumes_to_download)} files...")
                with ThreadPoolExecutor(max_workers=config.downloads.max_workers) as executor:
                    task = functools.partial(download_worker, config=config, hf_cache_dir=batch_hf_cache_dir)
                    results = list(tqdm(executor.map(task, volumes_to_download), total=len(volumes_to_download), desc=f"Downloading Batch {batch_num}"))
                for vol_name, res in zip(volumes_to_download, results):
                    if "OK" in res:
                        downloaded_volume_names.add(vol_name)
            else:
                logger.info(f"[Batch {batch_num}] All raw files for this batch are already present locally. No downloads needed.")

            logger.info(f"[Batch {batch_num}] Caching {len(batch_df)} files...")
            base_ds = CTMetadataDataset(
                dataframe=batch_df, img_dir=Path(config.paths.img_dir),
                path_mode=config.paths.dir_structure
            )
            caching_ds = CachingDataset(
                base_dataset=base_ds, transform=preprocess_transforms,
                cache_dir=cache_dir
            )
            data_loader = DataLoader(
                caching_ds, batch_size=None, num_workers=num_workers,
                worker_init_fn=worker_init_fn, collate_fn=identity_collate
            )

            successfully_cached_volumes = set()
            progress_bar = tqdm(enumerate(data_loader), desc=f"Caching Batch {batch_num}", total=len(caching_ds))
            for j, item in progress_bar:
                if item and isinstance(item, dict) and 'error' not in item:
                    original_volume_name = base_ds[j]['volume_name']
                    successfully_cached_volumes.add(original_volume_name)

            files_to_clean = downloaded_volume_names.intersection(successfully_cached_volumes)
            if files_to_clean:
                logger.info(f"[Batch {batch_num}] Cleaning up {len(files_to_clean)} downloaded raw NIfTI files...")
                cleaned_count = 0
                for volume_name in files_to_clean:
                    try:
                        file_path_to_delete = get_dynamic_image_path(Path(config.paths.img_dir), volume_name, config.paths.dir_structure)
                        if file_path_to_delete.exists():
                            os.remove(file_path_to_delete)
                            cleaned_count += 1
                    except OSError as e:
                        logger.error(f"Failed to delete {volume_name}: {e}")
                logger.info(f"[Batch {batch_num}] Cleanup complete. Deleted {cleaned_count} files.")
            else:
                logger.info(f"[Batch {batch_num}] No downloaded files to clean up for this batch.")
            
            # Clean up the temporary Hugging Face cache for the current batch
            logger.info(f"[Batch {batch_num}] Cleaning up temporary Hugging Face cache: {batch_hf_cache_dir}")
            shutil.rmtree(batch_hf_cache_dir)
    
    finally:
        # Final cleanup of the parent temporary directory
        if main_temp_hf_dir.exists():
            logger.info(f"Cleaning up main temporary cache directory: {main_temp_hf_dir}")
            shutil.rmtree(main_temp_hf_dir)


def cleanup_existing_raw_files(config, full_df: pd.DataFrame, cache_dir: Path):
    """
    Scans for and deletes local raw NIfTI files if a corresponding cache file exists.
    """
    logger.info("\n--- Starting Final Cleanup of Existing Raw Files ---")
    
    cleaned_count = 0
    volumes_to_check = full_df['VolumeName'].tolist()
    
    for volume_name in tqdm(volumes_to_check, desc="Cleaning up existing files"):
        raw_file_path = get_dynamic_image_path(Path(config.paths.img_dir), volume_name, config.paths.dir_structure)
        if not raw_file_path.exists():
            continue

        data_item = {"volume_name": volume_name}
        hashed_filename_str = deterministic_hash(data_item).decode('utf-8')
        cache_filepath = cache_dir / f"{hashed_filename_str}.pt"

        if cache_filepath.exists():
            try:
                os.remove(raw_file_path)
                logger.debug(f"Deleted existing raw file with cache: {raw_file_path}")
                cleaned_count += 1
            except OSError as e:
                logger.error(f"Failed to delete existing raw file {volume_name}: {e}")
    logger.info(f"Cleanup complete. Deleted {cleaned_count} raw NIfTI files that were already cached.")


def main(config_path: str, num_workers: int, batch_size: int, clean_local: bool):
    config = load_config(config_path)

    train_csv_path = Path(config.paths.data_dir) / config.paths.data_subsets.train
    valid_csv_path = Path(config.paths.data_dir) / config.paths.data_subsets.valid
    df_train = pd.read_csv(train_csv_path)
    df_valid = pd.read_csv(valid_csv_path)
    full_df = pd.concat([df_train, df_valid], ignore_index=True).drop_duplicates(subset=['VolumeName']).reset_index(drop=True)
    logger.info(f"Loaded a total of {len(full_df)} unique volumes for cache analysis.")

    preprocess_transforms = get_preprocessing_transforms(config)
    cache_dir = get_or_create_cache_subdirectory(Path(config.paths.cache_dir), preprocess_transforms, split="unified")
    missing_files_df = analyze_cache_state(full_df, cache_dir)
    
    num_missing = len(missing_files_df)
    logger.info("\n--- Cache Analysis Complete ---")
    logger.info(f"Total volumes required: {len(full_df)}")
    logger.info(f"Existing cached files:  {len(full_df) - num_missing}")
    logger.info(f"Missing cache files:    {num_missing}")
    
    if num_missing > 0:
        try:
            if input("\nProceed with processing missing files in batches? (Y/N): ").strip().lower() != 'y':
                logger.info("Process declined by user.")
            else:
                process_in_batches(config, missing_files_df, batch_size, num_workers)
        except (EOFError, KeyboardInterrupt):
            logger.info("\nProcess cancelled by user.")
    else:
        logger.info("Cache is already complete. Nothing to do for missing files.")

    if clean_local:
        cleanup_existing_raw_files(config, full_df, cache_dir)
    else:
        logger.info("Skipping final cleanup of existing raw files. Use --clean-local-raw-files to enable.")

if __name__ == '__main__':
    if sys.platform == "win32":
        torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Intelligently generate a pre-processed dataset cache in batches.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the YAML config file.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel workers for caching.')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of files to download and cache in each batch.')
    parser.add_argument(
        '--clean-local-raw-files',
        action='store_true',
        help="If set, delete any local raw NIfTI files that already have a corresponding cache file."
    )
    args = parser.parse_args()
    
    main(args.config, args.num_workers, args.batch_size, args.clean_local_raw_files)