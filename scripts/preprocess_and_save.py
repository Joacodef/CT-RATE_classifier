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
# Add the project root to the Python path to allow for absolute imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import load_config
from src.data.dataset import CTDataset3D

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
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file defining the preprocessing parameters.'
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
    parser.add_argument("--input_path_mode", type=str, default="nested", choices=["nested", "flat"], help="Path generation mode for input files.")
    parser.add_argument("--output_path_mode", type=str, default="flat", choices=["nested", "flat"], help="Path generation mode for output files.")

    args = parser.parse_args()

    # Load configuration from the specified YAML file
    config = load_config(args.config)

    # Ensure the output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory is ready at: {args.output_dir}")

    # Load the dataframe
    dataframe = pd.read_csv(args.dataframe_path)
    print(f"Loaded dataframe with {len(dataframe)} records.")

    # Instantiate the dataset using parameters from the loaded config
    # CRITICAL: use_cache is set to False to ensure the preprocessing pipeline
    # (including the saving transform) is executed for every item.
    dataset = CTDataset3D(
        dataframe=dataframe,
        img_dir=args.img_dir,
        pathology_columns=config.pathologies.columns,
        target_spacing_xyz=config.image_processing.target_spacing,
        target_shape_dhw=config.image_processing.target_shape_dhw,
        clip_hu_min=config.image_processing.clip_hu_min,
        clip_hu_max=config.image_processing.clip_hu_max,
        orientation_axcodes=config.image_processing.orientation_axcodes,
        use_cache=False,  # Disable cache to force processing and saving
        save_transformed_path=args.output_dir,  # Provide the output path
        path_mode=args.input_path_mode,
        output_path_mode=args.output_path_mode
    )

    print(f"Starting preprocessing of {len(dataset)} scans...")

    # Iterate through the entire dataset. Accessing each item triggers the
    # __getitem__ method, which applies the transformation pipeline. Because
    # we provided a 'save_transformed_path', the SaveImaged transform will
    # execute and save the file.
    for i in tqdm(range(len(dataset)), desc="Preprocessing and Saving Scans"):
        # The __getitem__ call here triggers the processing and saving
        _ = dataset[i]

    print(f"\nPreprocessing and saving complete. Transformed files are in {args.output_dir}")

if __name__ == '__main__':
    main()