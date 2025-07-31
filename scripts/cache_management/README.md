# Cache Management Scripts (`/scripts/cache_management`)

This directory contains scripts dedicated to creating, maintaining, and debugging the MONAI `PersistentDataset` cache.

## What is Caching?

In this project, caching is the process of pre-computing the results of the data preprocessing pipeline. Each raw 3D volume is loaded, resampled, resized, and normalized, and the final tensor is saved to disk. During training, the `DataLoader` can then fetch these pre-processed tensors directly, bypassing the computationally expensive preprocessing steps.

The cache directory's name is determined by a hash of the preprocessing transforms, so a new cache is automatically created if any image processing settings (like target spacing or shape) are changed.

-----

## Script Descriptions

### `generate_cache.py`

This is the primary script for building the cache. It first analyzes the required dataset and the existing cache to identify only the volumes that are missing. It then processes these missing files in batches, where for each batch it will:

1.  Download the required raw NIfTI files from Hugging Face.
2.  Use a MONAI `PersistentDataset` and a multi-worker `DataLoader` to apply the preprocessing transforms and save the resulting tensors to the persistent cache directory.
3.  Clean up the downloaded raw NIfTI files for that batch after they have been successfully cached.

This batch-based approach allows for efficient cache generation without requiring all raw data to be stored locally at once.

**Usage:**

```bash
# Generate the cache for the entire dataset using 8 worker processes
# and a batch size of 100 for downloads.
python scripts/cache_management/generate_cache.py \
    --config configs/config.yaml \
    --num-workers 8 \
    --batch-size 100
```

-----

### `verify_cache_integrity.py`

This is a maintenance utility for checking the health of an existing cache. If the caching process was interrupted, some files might be corrupted. This script scans the cache directory and uses a pool of worker processes to attempt to load every `.pt` file, reporting any that are unreadable.

**Usage:**

```bash
# Perform a dry-run to identify corrupted files without deleting them
python scripts/cache_management/verify_cache_integrity.py --config configs/config.yaml

# Find and permanently delete any corrupted files
python scripts/cache_management/verify_cache_integrity.py --config configs/config.yaml --fix
```

-----

### `map_hashes_to_volumes.py`

This is a debugging tool. Since MONAI's `PersistentDataset` uses a hash of the item's metadata for the filename, the cache directory is filled with files like `a1b2c3d4.pt`. This script creates a human-readable JSON file that maps these cryptic hashes back to their original `VolumeName`.

**Usage:**

```bash
# Create the map using the master dataset list
python scripts/cache_management/map_hashes_to_volumes.py \
    --csv-path data/filtered_master_list.csv \
    --output-path hash_map.json
```