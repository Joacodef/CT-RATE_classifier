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
from monai.transforms import (
    Compose, LoadImaged, Orientationd, Spacingd,
    ScaleIntensityRanged, Resized, EnsureTyped
)

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

try:
    from src.config import load_config
    from src.data.dataset import CTMetadataDataset
    from src.data.utils import get_dynamic_image_path
    from src.data.cache_utils import (
        get_or_create_cache_subdirectory,
        deterministic_json_hash
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
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
disable_progress_bars()

logger.info("Applying global patch to torch.load to allow MetaTensor deserialization.")
torch.load = functools.partial(torch.load, weights_only=False)

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

# In scripts/verify_dataset.py

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

        # Use the single unified image directory for all files
        final_destination = get_dynamic_image_path(
            config.paths.img_dir, volume_name, config.paths.dir_structure
        )
        
        final_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(temp_file_path, final_destination)
        return f"OK: {volume_name}"
    except Exception as e:
        logger.error(f"Failed during processing of {volume_name}: {e}", exc_info=False)
        return f"FAIL: {volume_name}"


def precache_dataset(config: SimpleNamespace, num_shards: int, shard_id: int):
    """
    Verifies all files in the unified dataset for corruption by leveraging
    PersistentDataset's internal logic, which correctly identifies cached items.
    """
    logger.info(f"\n--- Starting Unified Dataset Verification and Caching (Shard {shard_id + 1}/{num_shards}) ---")

    # This transform is only used for getting the cache path hash.
    path_defining_transform = Compose([
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

    df_path = Path(config.paths.data_dir) / config.paths.full_dataset_csv
    if not df_path.exists():
        logger.error(f"Full dataset CSV not found at {df_path}. Please check config.")
        return

    full_df = pd.read_csv(df_path)
    
    # --- Sharding Logic ---
    total_files = len(full_df)
    if total_files == 0:
        logger.warning("Dataset is empty. Nothing to cache.")
        return
        
    shard_size = (total_files + num_shards - 1) // num_shards
    start_index = shard_id * shard_size
    end_index = min(start_index + shard_size, total_files)
    
    if start_index >= total_files:
        logger.info(f"No files for shard {shard_id + 1}. Skipping.")
        return

    shard_df = full_df.iloc[start_index:end_index]
    logger.info(f"[Shard {shard_id + 1}/{num_shards}] Processing {len(shard_df)} files from index {start_index} to {end_index}.")

    # Create a dummy labels dataframe as it's required by the dataset class
    dummy_labels_df = pd.DataFrame({
        'VolumeName': shard_df['VolumeName'],
        config.pathologies.columns[0]: 0
    })

    base_ds = CTMetadataDataset(
        dataframe=dummy_labels_df,
        img_dir=config.paths.img_dir,
        pathology_columns=[config.pathologies.columns[0]],
        path_mode=config.paths.dir_structure
    )
    
    cache_dir = get_or_create_cache_subdirectory(
        config.paths.cache_dir, path_defining_transform, split="unified"
    )
    
    verifying_ds = PersistentDataset(
        data=base_ds, transform=path_defining_transform, cache_dir=cache_dir,
        hash_func=deterministic_json_hash, hash_transform=deterministic_json_hash
    )

    corrupt_items = []
    for i in tqdm(range(len(verifying_ds)), desc=f"Verifying Dataset (Shard {shard_id + 1}/{num_shards})"):
        try:
            _ = verifying_ds[i]
        except Exception as e:
            vol_name = base_ds[i].get("VolumeName", "Unknown")
            logger.error(
                f"\n[Shard {shard_id + 1}/{num_shards}] Detected corrupt file while analyzing '{vol_name}'. Reason: {e}"
            )
            corrupt_items.append(vol_name)

    logger.info(f"\n--- Verification and Caching Summary (Shard {shard_id + 1}/{num_shards}) ---")
    if not corrupt_items:
        logger.info(f"All files in Shard {shard_id + 1} were verified successfully!")
    else:
        logger.warning(f"Found {len(corrupt_items)} corrupt files in Shard {shard_id + 1}:")
        for f in corrupt_items:
            logger.warning(f"  - {f}")



def main(config_path: str, generate_cache: bool, num_shards: int, shard_id: int):
    """Main orchestrator function."""
    if generate_cache and not (0 <= shard_id < num_shards):
        logger.error(f"Error: --shard-id ({shard_id}) must be between 0 and --num-shards-1 ({num_shards - 1}).")
        return

    logger.info("--- Starting Dataset Verification Script ---")
    config = load_config(config_path)

    # Use the unified full_dataset_csv for all operations
    full_dataset_path = Path(config.paths.data_dir) / config.paths.data_subsets.train

    logger.info(f"Using dataset CSV at: {full_dataset_path}")

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
        precache_dataset(config, num_shards, shard_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify, download, and cache the CT-RATE dataset.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the YAML config file.')
    parser.add_argument('--generate-cache', action='store_true', help='Run the verification and caching process.')
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of parallel shards to divide the dataset into.')
    parser.add_argument('--shard-id', type=int, default=0, help='The ID of this specific shard to process (0-indexed).')
    args = parser.parse_args()
    
    main(args.config, args.generate_cache, args.num_shards, args.shard_id)