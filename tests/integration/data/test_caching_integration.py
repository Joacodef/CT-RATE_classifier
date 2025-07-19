# tests/integration/data/test_caching_integration.py
"""
Integration test for the full data caching pipeline.

This test verifies that the combination of CTMetadataDataset, PersistentDataset,
and LabelAttacherDataset correctly processes and caches data, ensuring the
cache itself is self-contained and free of labels.
"""

import sys
from pathlib import Path
import pytest
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from types import SimpleNamespace

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Imports for the components to be tested
from src.data.dataset import CTMetadataDataset, LabelAttacherDataset
from src.data.transforms import get_preprocessing_transforms
from src.data.cache_utils import get_or_create_cache_subdirectory, deterministic_hash
from monai.data import PersistentDataset

# --- Test Fixtures ---

@pytest.fixture
def mock_config() -> SimpleNamespace:
    """Provides a mock config object with necessary parameters for transforms."""
    config = SimpleNamespace(
        image_processing=SimpleNamespace(
            orientation_axcodes="RAS",
            target_spacing=(1.5, 1.5, 2.0),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            target_shape_dhw=(64, 64, 64)
        ),
        torch_dtype=torch.float32
    )
    return config

@pytest.fixture
def temp_data_setup(tmp_path: Path):
    """Creates a temporary directory structure with mock data and returns all paths."""
    base_dir = tmp_path
    img_dir = base_dir / "images"
    cache_dir = base_dir / "cache"
    img_dir.mkdir()
    cache_dir.mkdir()

    # Create mock NIfTI file
    volume_name = "train_01_001.nii.gz"
    subject_session = "train_01"
    subject_session_scan = "train_01_001"
    scan_dir = img_dir / subject_session / subject_session_scan
    scan_dir.mkdir(parents=True)
    file_path = scan_dir / volume_name
    
    nifti_img = nib.Nifti1Image(np.random.rand(128, 128, 128).astype(np.float32), np.eye(4))
    nib.save(nifti_img, file_path)

    # Create mock DataFrame
    df = pd.DataFrame({
        'VolumeName': [volume_name],
        'Pathology1': [1],
        'Pathology2': [0]
    })
    
    return {
        "img_dir": img_dir,
        "cache_dir": cache_dir,
        "dataframe": df,
        "pathology_columns": ["Pathology1", "Pathology2"]
    }

# --- Test Class ---

class TestEndToEndCaching:
    """Tests the full data pipeline from file path to labeled, cached data."""

    def test_pipeline_creates_valid_label_free_cache(self, temp_data_setup, mock_config):
        """
        Tests the entire data pipeline, asserting that:
        1. The final output dictionary is correctly formed.
        2. The cache is created.
        3. The cached file contains ONLY the image and its metadata, NOT the label.
        """
        # --- 1. Setup all components of the new data pipeline ---
        
        # Get the preprocessing transforms
        preprocess_transforms = get_preprocessing_transforms(mock_config)
        
        # Determine the exact cache path that will be created
        split_cache_dir = get_or_create_cache_subdirectory(
            temp_data_setup["cache_dir"], preprocess_transforms, split="integration_test"
        )
        
        # Create the label-agnostic dataset pointing to the raw NIfTI file
        base_ds = CTMetadataDataset(
            dataframe=temp_data_setup["dataframe"],
            img_dir=temp_data_setup["img_dir"],
            path_mode="nested"
        )
        
        # Create the persistent dataset which applies transforms and caches the result
        image_source = PersistentDataset(
            data=base_ds,
            transform=preprocess_transforms,
            cache_dir=split_cache_dir,
            hash_func=deterministic_hash
        )
        
        # Create the label attacher to merge cached data with labels
        final_dataset = LabelAttacherDataset(
            image_source=image_source,
            labels_df=temp_data_setup["dataframe"],
            pathology_columns=temp_data_setup["pathology_columns"]
        )
        
        # --- 2. Trigger caching by accessing the item ---
        
        # This first access will be slow as it processes and saves to disk
        item = final_dataset[0]
        
        # Assert that the final item returned by the pipeline is correctly formed
        assert "image" in item
        assert "label" in item
        assert "image_meta_dict" in item
        assert item["image"].shape[1:] == mock_config.image_processing.target_shape_dhw
        assert len(item["label"]) == len(temp_data_setup["pathology_columns"])
        
        # This second access should be fast as it reads from the cache
        _ = final_dataset[0]

        # --- 3. Verify the cache contents (the new core assertion) ---
        
        # Find the created cache file. There should be one .pt file and one .json file.
        cached_files = list(split_cache_dir.glob("*.pt"))
        assert len(cached_files) == 1
        cached_file_path = cached_files[0]

        # Directly load the cached data object from disk
        cached_data = torch.load(cached_file_path)

        # Assert that the cached object is a dictionary
        assert isinstance(cached_data, dict)
        
        # CRUCIAL ASSERTION: The cached dictionary must ONLY contain the processed
        # image and its metadata, proving our pipeline is working correctly.
        assert set(cached_data.keys()) == {"image", "image_meta_dict"}
        
        # Explicitly assert that no labels were saved to the cache
        assert "label" not in cached_data
        assert "volume_name" not in cached_data