# scripts/cache_management/map_hashes_to_volumes.py

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --- Project Setup ---
# Ensures that the script can find modules in the 'src' directory.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data.cache_utils import deterministic_hash

# --- Logging Configuration ---
# Basic logging setup for status updates.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def create_hash_map(csv_path: Path) -> dict[str, str]:
    """
    Generates a mapping from hashed filenames to original volume names.

    This function reads a dataset CSV, computes the cache hash for each
    volume name using the project's deterministic hashing function, and
    returns a dictionary that serves as a reverse lookup table.

    Args:
        csv_path: Path to the dataset CSV file containing 'VolumeName' column.

    Returns:
        A dictionary where keys are hashed filenames (as hex strings without
        the '.pt' extension) and values are the original volume names.
    """
    if not csv_path.is_file():
        logger.error(f"Dataset CSV file not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    if 'VolumeName' not in df.columns:
        logger.error(f"'VolumeName' column not found in {csv_path}")
        return {}

    hash_map = {}
    logger.info(f"Generating hash map for {len(df)} volumes from {csv_path.name}...")

    for volume_name in tqdm(df['VolumeName'], desc="Mapping Hashes"):
        # The data item must be in a dictionary to match the input
        # format expected by the deterministic_hash function.
        data_item = {"volume_name": volume_name}
        
        # Compute the hash, convert the resulting bytes to a hex string.
        hashed_filename_bytes = deterministic_hash(data_item)
        hashed_filename_hex = hashed_filename_bytes.hex()
        
        hash_map[hashed_filename_hex] = volume_name

    return hash_map


def main(csv_path: str, output_path: str):
    """
    Main function to generate and save the hash-to-volume map.
    """
    logger.info("--- Starting Hash to Volume Name Mapper ---")
    
    input_file = Path(csv_path)
    output_file = Path(output_path)

    hash_map = create_hash_map(input_file)

    if hash_map:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            # Save the map as a nicely formatted JSON file.
            json.dump(hash_map, f, indent=4)
        logger.info(f"Successfully created hash map with {len(hash_map)} entries.")
        logger.info(f"Map saved to: {output_file}")
    else:
        logger.error("Failed to create hash map. No output file was generated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a JSON map from cached filenames (hashes) to original volume names."
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        required=True,
        help="Path to the dataset CSV file (e.g., 'data/processed/train_dataset.csv')."
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='hash_to_volume_map.json',
        help="Path to save the output JSON map file."
    )
    args = parser.parse_args()

    main(args.csv_path, args.output_path)