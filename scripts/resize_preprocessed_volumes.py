# scripts/resize_preprocessed_volumes.py
"""
Resize already-preprocessed CT volumes to smaller dimensions.

This script is designed to work with volumes that have already been preprocessed
(oriented, intensity-scaled to [0,1], etc.) and only applies resizing to reduce
their dimensions and file size.

Usage:
    python scripts/resize_preprocessed_volumes.py \
        --input_dir /path/to/preprocessed/images \
        --output_dir /path/to/resized/images \
        --target_shape 96 192 192
"""

import argparse
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import nibabel as nib
from typing import Tuple

# MONAI imports
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    EnsureTyped,
    SaveImaged
)

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('resize_volumes.log')
    ]
)
logger = logging.getLogger(__name__)


def create_resize_pipeline(
    target_shape_dhw: Tuple[int, int, int],
    output_dir: Path
) -> Compose:
    """
    Creates a minimal MONAI pipeline for resizing preprocessed volumes.
    
    Args:
        target_shape_dhw: Target volume shape as (depth, height, width)
        output_dir: Directory where resized volumes will be saved
        
    Returns:
        MONAI Compose transform pipeline
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transforms = [
        # Load the preprocessed volume
        LoadImaged(
            keys="image",
            image_only=True,
            ensure_channel_first=False,
            reader="NibabelReader"
        ),
        # Ensure channel dimension is present
        EnsureChannelFirstd(
            keys="image",
            channel_dim="no_channel"
        ),
        # Resize to target dimensions
        Resized(
            keys="image",
            spatial_size=target_shape_dhw,
            mode="trilinear",  # Use trilinear for smooth interpolation
            align_corners=True
        ),
        # Save the resized volume
        SaveImaged(
            keys="image",
            output_dir=output_dir,
            output_postfix="",
            resample=False,
            separate_folder=False,
            print_log=False,
            output_dtype=np.float32
        ),
        # Ensure output type
        EnsureTyped(
            keys="image",
            dtype=torch.float32,
            track_meta=False
        )
    ]
    
    return Compose(transforms)


def process_single_volume(
    volume_path: Path,
    pipeline: Compose,
    dry_run: bool = False
) -> bool:
    """
    Process a single volume through the resize pipeline.
    
    Args:
        volume_path: Path to the volume file
        pipeline: MONAI transform pipeline
        dry_run: If True, only log what would be done without processing
        
    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would process: {volume_path.name}")
        return True
        
    try:
        # Create data dictionary for MONAI
        data_dict = {"image": str(volume_path)}
        
        # Apply the pipeline
        _ = pipeline(data_dict)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {volume_path.name}: {str(e)}")
        return False


def calculate_size_reduction(
    original_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int, int]
) -> dict:
    """Calculate and return size reduction statistics."""
    original_voxels = np.prod(original_shape)
    target_voxels = np.prod(target_shape)
    reduction_factor = original_voxels / target_voxels
    reduction_percent = (1 - target_voxels / original_voxels) * 100
    
    return {
        "original_voxels": original_voxels,
        "target_voxels": target_voxels,
        "reduction_factor": reduction_factor,
        "reduction_percent": reduction_percent
    }


def main():
    parser = argparse.ArgumentParser(
        description="Resize already-preprocessed CT volumes to smaller dimensions"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing preprocessed NIfTI files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where resized volumes will be saved"
    )
    parser.add_argument(
        "--target_shape",
        type=int,
        nargs=3,
        default=[96, 192, 192],
        help="Target shape as 'depth height width' (default: 96 192 192)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.nii.gz",
        help="File pattern to match (default: *.nii.gz)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be processed without actually doing it"
    )
    parser.add_argument(
        "--sample_original_shape",
        action="store_true",
        help="Load a sample file to show original shape and size reduction"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Find all matching files
    volume_files = list(args.input_dir.glob(args.pattern))
    if not volume_files:
        logger.error(f"No files matching pattern '{args.pattern}' found in {args.input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(volume_files)} files to process")
    
    # If requested, show size reduction info
    if args.sample_original_shape and volume_files:
        sample_file = volume_files[0]
        logger.info(f"\nAnalyzing sample file: {sample_file.name}")
        try:
            # Load the file header to get shape without loading full data
            img = nib.load(sample_file)
            original_shape = img.shape[:3] if len(img.shape) > 3 else img.shape
            
            logger.info(f"Original shape: {original_shape}")
            logger.info(f"Target shape: {tuple(args.target_shape)}")
            
            stats = calculate_size_reduction(original_shape, tuple(args.target_shape))
            logger.info(f"Size reduction: {stats['reduction_percent']:.1f}% "
                       f"(factor of {stats['reduction_factor']:.2f})")
            logger.info(f"Voxel count: {stats['original_voxels']:,} -> {stats['target_voxels']:,}")
            
            # Estimate file size reduction (rough approximation)
            original_size_mb = sample_file.stat().st_size / (1024 * 1024)
            estimated_new_size_mb = original_size_mb / stats['reduction_factor']
            logger.info(f"Estimated file size: {original_size_mb:.1f} MB -> {estimated_new_size_mb:.1f} MB\n")
            
        except Exception as e:
            logger.warning(f"Could not analyze sample file: {e}")
    
    if args.dry_run:
        logger.info("\n--- DRY RUN MODE ---")
        logger.info(f"Would resize {len(volume_files)} files")
        logger.info(f"Target shape: {args.target_shape}")
        logger.info(f"Output directory: {args.output_dir}")
        return
    
    # Create the resize pipeline
    logger.info(f"\nCreating resize pipeline with target shape: {args.target_shape}")
    pipeline = create_resize_pipeline(
        target_shape_dhw=tuple(args.target_shape),
        output_dir=args.output_dir
    )
    
    # Process all volumes
    logger.info(f"Starting resize process...")
    successful = 0
    failed = 0
    
    for volume_path in tqdm(volume_files, desc="Resizing volumes"):
        if process_single_volume(volume_path, pipeline):
            successful += 1
        else:
            failed += 1
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("RESIZE COMPLETE")
    logger.info("="*50)
    logger.info(f"Successfully resized: {successful} volumes")
    if failed > 0:
        logger.warning(f"Failed to resize: {failed} volumes")
    logger.info(f"Output directory: {args.output_dir}")
    
    # List a few output files as confirmation
    output_files = list(args.output_dir.glob("*.nii.gz"))[:5]
    if output_files:
        logger.info("\nSample output files created:")
        for f in output_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  - {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()