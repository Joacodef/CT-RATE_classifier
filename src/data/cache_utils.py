# Standard library imports
import json
import hashlib
import inspect
import functools
import logging
from pathlib import Path

# Third-party imports
import numpy as np
import torch
from monai.transforms import Compose

from src.data.transforms import KeyCleanerD

# Get logger
logger = logging.getLogger(__name__)

def json_serial_converter(o):
    """
    Custom JSON converter for types that are not serializable by default.
    Handles Path objects, numpy arrays, torch Tensors/dtypes, and other special types.
    """
    
    if isinstance(o, KeyCleanerD):
        # Serialize the class name and its important parameters
        return {"class": o.__class__.__name__, "keys_to_keep": sorted(list(o.keys_to_keep))}
    
    if isinstance(o, type):
        return o.__name__
        
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (torch.dtype, np.dtype)):
        return str(o)
    if isinstance(o, np.random.RandomState):
        # The state of a random generator is complex and not meant for serialization.
        # We only need to know that a RandomState object exists for the hash.
        return "np.random.RandomState object"
    
    # If the type is not recognized, raise the default error.
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def worker_init_fn(worker_id):
    """
    Initialization function for DataLoader workers.

    This function overrides torch.load within each worker process to force
    `weights_only=False`. This is a necessary workaround for PyTorch versions
    where this argument defaults to True, which prevents MONAI's PersistentDataset
    from loading cached complex objects (like MetaTensors) that are not just model weights.
    """
    # Create a new version of torch.load that has weights_only=False by default.
    # This ensures that our custom objects can be unpickled by the worker processes.
    custom_load = functools.partial(torch.load, weights_only=False)
    
    # Replace the original torch.load with our custom version for this worker.
    torch.load = custom_load

import inspect


def get_transform_params(obj):
    """
    Recursively gets all public, non-callable parameters of a MONAI transform
    for stable hashing and JSON serialization. This uses `inspect` for a robust
    approach to extracting the configuration of a transform pipeline.
    """
    # --- Base Cases: Handle non-decomposable objects ---
    if isinstance(obj, (list, tuple)):
        # If we get a list, process each item recursively.
        return [get_transform_params(item) for item in obj]

    # Stop recursion if the object is not a MONAI transform or a related object.
    # The final conversion will be handled by the JSON serializer.
    if not hasattr(obj, '__class__') or ("monai.transforms" not in str(obj.__class__) and "monai.data" not in str(obj.__class__)):
        return obj

    # --- Recursive Case: Process MONAI transform objects ---
    params = {"class": obj.__class__.__name__}

    # Use inspect.getmembers to reliably find all attributes and properties.
    for name, value in inspect.getmembers(obj):
        # Skip private, dunder, and method attributes.
        if name.startswith('_') or inspect.ismethod(value) or inspect.isfunction(value):
            continue

        # Attributes in this list are excluded from the hash. They are either
        # internal state variables (like 'R' for random state), dynamic properties
        # ('lazy'), or irrelevant metadata ('VERSION') that would make the
        # cache hash unstable across runs.
        if name in ['f', 'g', 'R', 'lazy', 'backend', 'end_pending', 'progress', 'VERSION']:
            continue
        
        # Recursively process the attribute's value.
        params[name] = get_transform_params(value)

    return params


def deterministic_hash(item_to_hash: any) -> bytes:
    """
    Creates a deterministic MD5 hash for an object, crucial for MONAI's cache.

    This function has two distinct behaviors required by PersistentDataset:
    1. Hashing a data item: If the input is a dictionary containing 'volume_name',
       it hashes ONLY the volume_name. This ensures each cached file has a
       stable, predictable name based on its unique identifier.
    2. Hashing a transform pipeline: For any other input (like the transform config),
       it serializes the entire object to a stable JSON string and hashes that.
       This generates the unique hash for the cache *directory* itself.
    """
    # Check if the item is a data dictionary. Use 'volume_name' (lowercase v).
    if isinstance(item_to_hash, dict) and "volume_name" in item_to_hash:
        # Case 1: Hashing a data item. Use only the volume name.
        item_str = str(item_to_hash["volume_name"])

    else:
        # Case 2: Hashing a list of transforms or another object.
        item_str = json.dumps(item_to_hash, sort_keys=True, default=json_serial_converter)

    # Return the hash as bytes.
    return hashlib.md5(item_str.encode('utf-8')).hexdigest().encode('utf-8')



def get_or_create_cache_subdirectory(base_cache_dir: Path, transforms: Compose, split: str) -> Path:
    """
    Determines the correct cache subdirectory based on a hash of transform parameters.
    If the directory doesn't exist, it is created.
    """
    # First, get the complete, serializable dictionary of transform parameters.
    transform_params = get_transform_params(transforms)

    # Serialize the parameters into a stable JSON string.
    transform_params_str = json.dumps(transform_params, sort_keys=True, default=json_serial_converter)

    # Now, generate the hash FROM this string using the new hasher.
    config_hash = deterministic_hash(transform_params_str).decode('utf-8')

    # Construct the path for the specific cache subdirectory.
    cache_path = base_cache_dir / config_hash
    params_file = cache_path / "cache_params.json"

    # If the specific cache directory already exists, use it.
    if cache_path.exists():
        logger.info(f"Found existing cache for '{split}' split with current configuration.")
        logger.info(f"Using cache directory: {cache_path}")
        return cache_path

    # If the cache does not exist, create it.
    logger.info(f"No existing cache found for '{split}' split with this configuration.")
    logger.info(f"Creating new cache directory: {cache_path}")
    
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Save the same dictionary to the JSON file.
        with open(params_file, 'w') as f:
            json.dump(transform_params, f, indent=4, default=json_serial_converter)
            
        logger.info(f"Saved cache parameters to {params_file}")

    except OSError as e:
        logger.error(f"Failed to create cache directory {cache_path}: {e}")
        raise

    return cache_path