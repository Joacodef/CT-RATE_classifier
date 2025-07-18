# scripts/verify_cache_integrity.py
import argparse
import logging
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import torch
from tqdm import tqdm

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import load_config
from src.data.cache_utils import get_or_create_cache_subdirectory
from src.utils.logging_config import setup_logging

# --- MONAI Imports ---
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    Resized,
    EnsureTyped,
)

# --- Logging Configuration ---
setup_logging(log_file=project_root / "logs" / "cache_verification.log")
logger = logging.getLogger(__name__)


def get_cache_defining_transform(config: SimpleNamespace) -> Compose:
    """
    Recreates the exact transform pipeline used for caching in verify_dataset.py.
    """
    return Compose(
        [
            LoadImaged(
                keys="image",
                image_only=True,
                ensure_channel_first=True,
                reader="NibabelReader",
            ),
            Orientationd(keys="image", axcodes=config.image_processing.orientation_axcodes),
            Spacingd(
                keys="image",
                pixdim=config.image_processing.target_spacing,
                mode="bilinear",
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=config.image_processing.clip_hu_min,
                a_max=config.image_processing.clip_hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(
                keys="image",
                spatial_size=config.image_processing.target_shape_dhw,
                mode="area",
            ),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )


def verify_file_integrity(file_path: Path) -> Tuple[Path, bool]:
    """
    Checks if a single .pt file can be loaded by torch.
    """
    try:
        # --- THIS IS THE FIX ---
        # Explicitly set weights_only=False to allow loading MONAI MetaTensors
        # on newer versions of PyTorch (>=2.6).
        torch.load(file_path, map_location="cpu", weights_only=False)
        return file_path, True
    except (RuntimeError, EOFError, zipfile.BadZipFile, pickle.UnpicklingError) as e:
        # Catches common errors for corrupted torch/zip files and unpickling issues.
        logger.debug(f"Corruption detected in {file_path}: {e}")
        return file_path, False


def verify_cache_integrity(config_path: str, fix: bool, workers: int):
    """
    Scans a MONAI cache directory to find and optionally delete corrupt files.
    """
    try:
        logger.info(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        base_cache_dir = config.paths.cache_dir

        if not base_cache_dir.is_dir():
            logger.info(
                f"Base cache directory not found at {base_cache_dir}. Nothing to do."
            )
            return

        path_defining_transform = get_cache_defining_transform(config)
        unified_cache_dir = get_or_create_cache_subdirectory(
            base_cache_dir, path_defining_transform, split="unified"
        )
        logger.info(f"Target cache directory identified: {unified_cache_dir}")

        if not unified_cache_dir.is_dir():
            logger.info("Cache directory does not exist. No files to verify.")
            return

        files_to_check = list(unified_cache_dir.glob("*.pt"))
        if not files_to_check:
            logger.info("No .pt files found in cache directory. Exiting.")
            return

        logger.info(
            f"Verifying {len(files_to_check)} files using {workers} worker processes..."
        )

        corrupted_files: List[Path] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_file = {
                executor.submit(verify_file_integrity, file): file
                for file in files_to_check
            }
            progress = tqdm(
                as_completed(future_to_file),
                total=len(files_to_check),
                desc="Verifying Cache",
            )
            for future in progress:
                # We need to handle exceptions that might be raised from the future
                try:
                    file_path, is_valid = future.result()
                    if not is_valid:
                        corrupted_files.append(file_path)
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Unexpected error processing {file_path}: {e}")
                    corrupted_files.append(file_path)


        logger.info("Verification complete.")

        if not corrupted_files:
            logger.info(
                f"Success! All {len(files_to_check)} checked files are valid."
            )
            return

        logger.warning(
            f"Found {len(corrupted_files)} corrupted or unreadable file(s) out of {len(files_to_check)} total."
        )
        for f in corrupted_files:
            logger.warning(f"  - Corrupted: {f}")

        if fix:
            logger.info("Applying fix: Deleting corrupted files...")
            deleted_count = 0
            for f in corrupted_files:
                try:
                    os.remove(f)
                    logger.info(f"  - Deleted: {f}")
                    deleted_count += 1
                except OSError as e:
                    logger.error(f"Failed to delete {f}: {e}")
            logger.info(f"Successfully deleted {deleted_count} file(s).")
        else:
            logger.info("Dry-run complete. No files were deleted.")
            logger.info("--> To delete these files, run this script again with the --fix flag.")

    except Exception as e:
        logger.critical(
            f"An unexpected error occurred during cache verification: {e}",
            exc_info=True,
        )

def main():
    """Main function to parse arguments and run the verification."""
    parser = argparse.ArgumentParser(
        description="Verify integrity of MONAI cache files and delete corrupted ones.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="If set, permanently delete any corrupted files that are found.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes for checking files. Defaults to CPU count.",
    )
    args = parser.parse_args()

    verify_cache_integrity(args.config, args.fix, args.workers)


if __name__ == "__main__":
    if sys.platform == "win32":
        torch.multiprocessing.freeze_support()
    # Import pickle here for the except block
    import pickle
    main()