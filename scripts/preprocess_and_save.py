# scripts/preprocess_and_save.py
"""
scripts.preprocess_and_save
===========================

This script provides a command-line interface to preprocess the CT scan dataset
and save the transformed volumes to a specified directory on disk.

It leverages the CTDataset3D class and the configured MONAI preprocessing pipeline.
By iterating through the dataset, it triggers the `SaveImaged` transform that was
conditionally added to the pipeline.

This is particularly useful for creating a complete, pre-processed version of the
dataset for analysis, inspection, or faster loading in alternative workflows.

Usage:
    python scripts/preprocess_and_save.py \
        --dataframe_path /path/to/your/dataset_info.csv \
        --img_dir /path/to/your/nifti_images \
        --output_dir /path/to/save/transformed_images
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add the project root to the Python path to allow for absolute imports from
# other modules like 'config' and 'data'.
# Add project root and src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from config import config
from data.dataset import CTDataset3D

def main():
    """
    Main function to run the data preprocessing and saving script.
    
    Parses command-line arguments, sets up the dataset, and iterates through
    it to trigger the saving of preprocessed images.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess CT scan data and save transformed volumes to disk."
    )
    parser.add_argument(
        '--dataframe_path',
        type=Path,
        required=True,
        help='Path to the CSV file containing the dataset information (e.g., VolumeName, labels).'
    )
    parser.add_argument(
        '--img_dir',
        type=Path,
        required=True,
        help='Path to the directory containing the raw NIfTI (.nii.gz) image files.'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Path to the directory where transformed images will be saved.'
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory is ready at: {args.output_dir}")

    # Load the dataframe
    dataframe = pd.read_csv(args.dataframe_path)
    print(f"Loaded dataframe with {len(dataframe)} records.")

    # Instantiate the dataset.
    # CRITICAL: use_cache is set to False to ensure the preprocessing pipeline
    # (including the saving transform) is executed for every item, rather than
    # loading from a pre-existing cache.
    dataset = CTDataset3D(
        dataframe=dataframe,
        img_dir=args.img_dir,
        pathology_columns=config.PATHOLOGY_COLUMNS,
        target_spacing_xyz=config.TARGET_SPACING_XYZ,
        target_shape_dhw=config.TARGET_SHAPE_DHW,
        clip_hu_min=config.CLIP_HU_MIN,
        clip_hu_max=config.CLIP_HU_MAX,
        orientation_axcodes=config.ORIENTATION_AXCODES,
        use_cache=False,  # Disable cache to force processing and saving
        save_transformed_path=args.output_dir  # Provide the output path
    )
    
    print(f"Starting preprocessing of {len(dataset)} scans...")
    
    # Iterate through the entire dataset. Accessing each item triggers the 
    # __getitem__ method, which applies the transformation pipeline. Because
    # we provided a 'save_transformed_path', the SaveImaged transform will
    # execute and save the file.
    for i in tqdm(range(len(dataset)), desc="Preprocessing and Saving Scans"):
        dataset[i]

    print(f"Preprocessing and saving complete. Transformed files are in {args.output_dir}")

if __name__ == '__main__':
    main()