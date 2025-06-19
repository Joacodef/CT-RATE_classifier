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
            output_postfix="resized",
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


def get_expected_output_filename(input_path: Path, output_dir: Path) -> Path:
    """Get the expected output filename for a given input file."""
    # MONAI's SaveImaged adds the postfix before the extension
    stem = input_path.stem.replace('.nii', '')  # Remove .nii if present
    return output_dir / f"{stem}resized.nii.gz"


def get_average_file_size(output_dir: Path, pattern: str = "*.nii.gz", sample_size: int = 10) -> float:
    """
    Calculate the average file size in the output directory.
    
    Args:
        output_dir: Directory to check
        pattern: File pattern to match
        sample_size: Maximum number of files to sample for average
        
    Returns:
        Average file size in MB, or 0 if no files found
    """
    files = list(output_dir.glob(pattern))
    if not files:
        return 0.0
    
    # Sample up to sample_size files for efficiency
    sample_files = files[:sample_size] if len(files) > sample_size else files
    sizes_mb = [f.stat().st_size / (1024 * 1024) for f in sample_files]
    
    return sum(sizes_mb) / len(sizes_mb)


def check_if_already_processed(input_path: Path, output_dir: Path, min_size_mb: float = 1.0, size_tolerance: float = 0.8) -> bool:
    """
    Check if a file has already been successfully processed.
    
    Args:
        input_path: Path to the input file
        output_dir: Output directory
        min_size_mb: Minimum file size in MB to consider valid (used as fallback)
        size_tolerance: Minimum fraction of average size to consider valid (e.g., 0.8 = 80%)
        
    Returns:
        True if the file appears to be already processed
    """
    expected_output = get_expected_output_filename(input_path, output_dir)
    
    if expected_output.exists():
        size_mb = expected_output.stat().st_size / (1024 * 1024)
        
        # Try to get average size from other files
        avg_size_mb = get_average_file_size(output_dir, sample_size=20)
        
        if avg_size_mb > 0:
            # Use comparison with average size
            expected_min_size = avg_size_mb * size_tolerance
            if size_mb >= expected_min_size:
                return True
            else:
                # File exists but seems too small compared to others
                logger.warning(f"Found incomplete output file ({size_mb:.1f} MB vs expected ≥{expected_min_size:.1f} MB): {expected_output.name}")
                expected_output.unlink()  # Remove the incomplete file
                return False
        else:
            # No other files to compare against, use absolute minimum
            if size_mb >= min_size_mb:
                return True
            else:
                logger.warning(f"Found incomplete output file (only {size_mb:.1f} MB): {expected_output.name}")
                expected_output.unlink()  # Remove the incomplete file
                return False
    return False


def save_progress(progress_file: Path, processed_files: set):
    """Save the list of successfully processed files."""
    with open(progress_file, 'w') as f:
        for file_path in sorted(processed_files):
            f.write(f"{file_path}\n")


def load_progress(progress_file: Path) -> set:
    """Load the list of previously processed files."""
    if not progress_file.exists():
        return set()
    
    processed = set()
    with open(progress_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                processed.add(line)
    return processed


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run, skipping already processed files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all files, even if they already exist"
    )
    parser.add_argument(
        "--size_tolerance",
        type=float,
        default=0.8,
        help="Minimum fraction of average file size to consider valid (default: 0.8 = 80%%)"
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
    
    # Check existing output files if resuming
    if args.resume and args.output_dir.exists():
        avg_existing_size = get_average_file_size(args.output_dir)
        if avg_existing_size > 0:
            logger.info(f"Average size of existing output files: {avg_existing_size:.1f} MB")
            logger.info(f"Files smaller than {avg_existing_size * args.size_tolerance:.1f} MB "
                       f"({args.size_tolerance*100:.0f}% of average) will be considered incomplete\n")
    
    if args.dry_run:
        logger.info("\n--- DRY RUN MODE ---")
        logger.info(f"Would resize {len(volume_files)} files")
        logger.info(f"Target shape: {args.target_shape}")
        logger.info(f"Output directory: {args.output_dir}")
        
        if args.resume:
            # Check how many would be skipped
            would_skip = sum(1 for v in volume_files 
                           if check_if_already_processed(v, args.output_dir, size_tolerance=args.size_tolerance))
            if would_skip > 0:
                logger.info(f"Would skip {would_skip} already processed files")
                logger.info(f"Would process {len(volume_files) - would_skip} new files")
        return
    
    # Create the resize pipeline
    logger.info(f"\nCreating resize pipeline with target shape: {args.target_shape}")
    pipeline = create_resize_pipeline(
        target_shape_dhw=tuple(args.target_shape),
        output_dir=args.output_dir
    )
    
    # Setup progress tracking
    progress_file = args.output_dir / ".resize_progress.txt"
    processed_files = load_progress(progress_file) if args.resume and not args.force else set()
    
    # Filter out already processed files if resuming
    if args.resume and not args.force:
        files_to_process = []
        skipped = 0
        
        for volume_path in volume_files:
            if str(volume_path) in processed_files or check_if_already_processed(volume_path, args.output_dir, size_tolerance=args.size_tolerance):
                skipped += 1
                processed_files.add(str(volume_path))
            else:
                files_to_process.append(volume_path)
        
        if skipped > 0:
            logger.info(f"Resuming: Skipping {skipped} already processed files")
            save_progress(progress_file, processed_files)
    else:
        files_to_process = volume_files
        if args.force:
            logger.info("Force mode: Will reprocess all files")
    
    # Process all volumes
    logger.info(f"Starting resize process for {len(files_to_process)} files...")
    successful = 0
    failed = 0
    
    try:
        for volume_path in tqdm(files_to_process, desc="Resizing volumes"):
            if process_single_volume(volume_path, pipeline):
                successful += 1
                processed_files.add(str(volume_path))
                
                # Save progress every 10 files
                if successful % 10 == 0:
                    save_progress(progress_file, processed_files)
            else:
                failed += 1
                
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user!")
        logger.info("Saving progress...")
        save_progress(progress_file, processed_files)
        logger.info(f"Progress saved. Processed {successful} files so far.")
        logger.info(f"To resume, run the same command with --resume")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        logger.info("Saving progress before exit...")
        save_progress(progress_file, processed_files)
        raise
    finally:
        # Final save of progress
        save_progress(progress_file, processed_files)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("RESIZE COMPLETE")
    logger.info("="*50)
    logger.info(f"Successfully resized: {successful} volumes")
    if failed > 0:
        logger.warning(f"Failed to resize: {failed} volumes")
    logger.info(f"Total processed (including resumed): {len(processed_files)} volumes")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Clean up progress file if all files were processed successfully
    if failed == 0 and len(processed_files) == len(volume_files):
        progress_file.unlink(missing_ok=True)
        logger.info("All files processed successfully. Progress file removed.")
    else:
        logger.info(f"Progress saved to: {progress_file}")
    
    # List a few output files as confirmation
    output_files = list(args.output_dir.glob("*.nii.gz"))
    if output_files:
        logger.info("\nOutput file statistics:")
        sizes_mb = [f.stat().st_size / (1024 * 1024) for f in output_files]
        avg_size = sum(sizes_mb) / len(sizes_mb)
        std_size = np.std(sizes_mb) if len(sizes_mb) > 1 else 0
        
        logger.info(f"  - Count: {len(output_files)} files")
        logger.info(f"  - Average size: {avg_size:.1f} MB (±{std_size:.1f} MB)")
        logger.info(f"  - Min size: {min(sizes_mb):.1f} MB")
        logger.info(f"  - Max size: {max(sizes_mb):.1f} MB")
        logger.info(f"  - Total size: {sum(sizes_mb):.1f} MB ({sum(sizes_mb)/1024:.2f} GB)")
        
        logger.info("\nSample output files:")
        for f in output_files[:5]:
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  - {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()