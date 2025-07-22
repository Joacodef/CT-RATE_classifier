# Cache Management Scripts (`/scripts/cache_management`)

This directory contains scripts dedicated to creating, maintaining, and debugging the MONAI `PersistentDataset` cache.

## What is Caching?

In this project, caching is the process of pre-computing the results of the entire data preprocessing pipeline. Each raw 3D volume is loaded, resampled, resized, and normalized, and the final tensor is saved to disk. During training, the data loader can then fetch these pre-processed tensors directly from the disk, bypassing the computationally expensive preprocessing steps and significantly accelerating the training loop.

The cache directory's name is determined by a hash of the preprocessing transforms, so a new cache is automatically created if you change any image processing settings (like target spacing or shape).

---

## Script Descriptions

### `generate_cache.py`

This is the primary script for building the cache. It iterates through the entire dataset, applies the preprocessing transforms defined in the configuration, and saves each resulting tensor to the cache directory. The script is designed to be run in parallel on multiple machines or instances by using sharding.

**Usage:**

```bash
# Generate the cache for the entire dataset using 8 worker processes
python scripts/cache_management/generate_cache.py \
    --config configs/config.yaml \
    --num-workers 8

# Generate the cache using 2 machines (shards)
# On machine 1:
python scripts/cache_management/generate_cache.py \
    --config configs/config.yaml \
    --num-shards 2 \
    --shard-id 0

# On machine 2:
python scripts/cache_management/generate_cache.py \
    --config configs/config.yaml \
    --num-shards 2 \
    --shard-id 1
````

-----

### `verify_cache_integrity.py`

This is a maintenance utility for checking the health of an existing cache. If the caching process was interrupted, some files might be corrupted. This script scans the cache directory, attempts to load every `.pt` file, and reports any that are unreadable.

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