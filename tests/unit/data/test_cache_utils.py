# Standard library imports
import json
import hashlib
import functools
from pathlib import Path
import sys

# Third-party imports
import pytest
import numpy as np
import torch
from monai.transforms import Compose, RandFlipd, EnsureTyped

# Add the project root to the Python path to allow src imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Local application imports
from src.data.cache_utils import (
    json_serial_converter,
    get_transform_params,
    deterministic_hash,
    get_or_create_cache_subdirectory,
    worker_init_fn
)


# --- Fixtures ---

class MockSimpleTransform:
    """A mock transform class to simulate a MONAI transform for testing."""
    def __init__(self, public_param="default", _private_param="secret"):
        self.public_param = public_param
        self._private_param = _private_param
        self.R = np.random.RandomState(42)  # A random state, should be excluded

    def __call__(self, data):
        return data

    def a_method(self):
        return "method_should_be_excluded"
    
MockSimpleTransform.__module__ = "monai.transforms.mocking"

# --- Test Cases ---

@pytest.mark.parametrize("input_obj, expected_output", [
    # The expected path is now generated dynamically to be OS-agnostic
    (Path("/tmp/test"), str(Path("/tmp/test"))),
    (torch.tensor([1, 2]), [1, 2]),
    (np.array([3, 4]), [3, 4]),
    (torch.float32, "torch.float32"),
    (np.int64, "int64"),
    (np.random.RandomState(1), "np.random.RandomState object"),
    (int, "int"),
])
def test_json_serial_converter_supported_types(input_obj, expected_output):
    """
    Tests that the JSON converter correctly serializes various supported object types.
    """
    assert json_serial_converter(input_obj) == expected_output

def test_json_serial_converter_unsupported_type():
    """
    Tests that the JSON converter raises a TypeError for unsupported objects.
    """
    class Unsupported:
        pass
    with pytest.raises(TypeError):
        json_serial_converter(Unsupported())

def test_get_transform_params_simple_object():
    """
    Tests that get_transform_params returns simple, non-transform objects as-is.
    """
    assert get_transform_params(123) == 123
    assert get_transform_params("hello") == "hello"

def test_get_transform_params_mock_transform():
    """
    Tests parameter extraction from a single mock transform object.
    It should only capture public, non-callable attributes.
    """
    transform = MockSimpleTransform(public_param="value")
    params = get_transform_params(transform)
    expected = {
        "class": "MockSimpleTransform",
        "public_param": "value"
    }
    assert params == expected

def test_get_transform_params_composed_transforms():
    """
    Tests parameter extraction from a MONAI Compose object.
    """
    composed_transform = Compose([
        MockSimpleTransform("first"),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        EnsureTyped(keys=["image"])
    ])
    params = get_transform_params(composed_transform)

    assert params["class"] == "Compose"
    assert len(params["transforms"]) == 3
    assert params["transforms"][0]["class"] == "MockSimpleTransform"
    assert params["transforms"][0]["public_param"] == "first"
    assert params["transforms"][1]["class"] == "RandFlipd"
    assert params["transforms"][1]["prob"] == 0.5
    assert params["transforms"][2]["class"] == "EnsureTyped"

def test_deterministic_hash_data_item():
    """
    Tests that the hash function correctly hashes a data dictionary
    based ONLY on its 'volume_name'.
    """
    item = {"volume_name": "volume_001.nii.gz", "image": "path", "label": [1]}
    expected_hash = hashlib.md5(b"volume_001.nii.gz").hexdigest().encode('utf-8')
    assert deterministic_hash(item) == expected_hash

def test_deterministic_hash_config_object():
    """
    Tests that the hash function correctly hashes a configuration object
    based on its sorted JSON representation.
    """
    config1 = {"b": 2, "a": 1}
    config2 = {"a": 1, "b": 2}  # Same content, different order
    
    # Hashes should be identical due to sorted serialization
    assert deterministic_hash(config1) == deterministic_hash(config2)
    
    # Manual verification for one case
    json_str = json.dumps(config1, sort_keys=True, default=json_serial_converter)
    expected_hash = hashlib.md5(json_str.encode('utf-8')).hexdigest().encode('utf-8')
    assert deterministic_hash(config1) == expected_hash

def test_get_or_create_cache_subdirectory(tmp_path):
    """
    Tests the creation and retrieval of a cache subdirectory.
    Verifies file system side-effects and consistent path generation.
    """
    base_cache_dir = tmp_path
    transforms = Compose([RandFlipd(keys=["image"], prob=0.5)])
    split_name = "train"

    # --- First Call: Create the cache ---
    cache_path_1 = get_or_create_cache_subdirectory(base_cache_dir, transforms, split_name)
    params_file = cache_path_1 / "cache_params.json"

    assert cache_path_1.exists()
    assert cache_path_1.is_dir()
    assert params_file.exists()

    # Verify the contents of the saved parameters file
    with open(params_file, 'r') as f:
        saved_params = json.load(f)
    
    expected_params = get_transform_params(transforms)
    assert saved_params == expected_params

    # --- Second Call: Retrieve the same cache ---
    cache_path_2 = get_or_create_cache_subdirectory(base_cache_dir, transforms, split_name)

    # Should return the exact same path
    assert cache_path_2 == cache_path_1

def test_worker_init_fn():
    """
    Tests that the worker_init_fn correctly patches torch.load.
    """
    original_torch_load = torch.load
    try:
        # Apply the worker init function in the main thread for testing
        worker_init_fn(0)
        
        # Check if torch.load is now a partial function
        assert isinstance(torch.load, functools.partial)
        
        # Check if the partial has the correct keyword argument set
        assert torch.load.keywords == {'weights_only': False}

    finally:
        # ALWAYS restore the original torch.load to avoid side-effects
        torch.load = original_torch_load
        
    # Verify that the original function is restored
    assert torch.load is original_torch_load