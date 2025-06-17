"""
Unit tests for data/dataset.py
Tests cover:
- Dataset initialization and configuration
- Cache functionality (loading and saving) with MONAI-processed data
- Data loading and MONAI preprocessing integration
- Augmentation pipeline (applied after MONAI)
- Error handling for missing files
- Label extraction and formatting
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call

import pytest
import numpy as np
import pandas as pd
import torch
import nibabel as nib

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Import project modules
from src.data.dataset import CTDataset3D
from src.data.preprocessing import create_monai_preprocessing_pipeline
from src.data.utils import get_dynamic_image_path
from monai.transforms import Compose as MonaiCompose # For type checking

# === Module-level fixtures ===
@pytest.fixture
def mock_dataframe():
    """Create a mock dataframe with pathology labels"""
    data = {
        'VolumeName': ['volume1.nii.gz', 'volume2.nii.gz', 'volume3.nii.gz'],
        'Lung nodule': [1, 0, 1],
        'Emphysema': [0, 1, 0],
        'Atelectasis': [1, 1, 0],
        'Pleural effusion': [0, 0, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def pathology_columns():
    """Return test pathology columns"""
    return ['Lung nodule', 'Emphysema', 'Atelectasis', 'Pleural effusion']

@pytest.fixture
def temp_img_dir(tmp_path: Path) -> Path: # Use pytest's tmp_path fixture
    """Create temporary directory with mock NIfTI files"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create mock NIfTI files
    for i in range(1, 4): # volume1, volume2, volume3
        volume_name = f"volume{i}.nii.gz"
        # Create a small test volume, ensure data type is suitable for MONAI processing
        data = np.random.rand(32, 32, 40).astype(np.float32) * 1000 # X, Y, Z
        affine = np.diag([1.0, 1.0, 1.0, 1.0]) # Simple affine
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, img_dir / volume_name)
    return img_dir

@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path: # Use pytest's tmp_path fixture
    """Create temporary cache directory"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir

@pytest.fixture
def default_dataset_params(temp_img_dir, pathology_columns, temp_cache_dir):
    """Provides default parameters for CTDataset3D initialization."""
    return {
        "img_dir": temp_img_dir,
        "pathology_columns": pathology_columns,
        "target_spacing_xyz": np.array([1.0, 1.0, 1.0]),
        "target_shape_dhw": (32, 64, 64), # D, H, W
        "clip_hu_min": -1000.0,
        "clip_hu_max": 1000.0,
        "orientation_axcodes": "LPS",
        "use_cache": False, # Default to no cache for most tests unless specified
        "cache_dir": temp_cache_dir,
        "augment": False
    }

# === Test Classes ===
class TestCTDataset3D:
    """Tests for the CTDataset3D class with MONAI integration."""

    def test_initialization_basic(self, mock_dataframe, default_dataset_params):
        """Test basic dataset initialization with MONAI pipeline creation."""
        params = default_dataset_params.copy()
        dataset = CTDataset3D(dataframe=mock_dataframe, **params)

        assert len(dataset) == 3
        assert dataset.img_dir == params["img_dir"]
        assert dataset.pathology_columns == params["pathology_columns"]
        assert not dataset.use_cache
        assert not dataset.augment
        # Verify MONAI pipeline is created
        assert hasattr(dataset, 'monai_preprocessing_pipeline')
        assert isinstance(dataset.monai_preprocessing_pipeline, MonaiCompose)

    def test_initialization_with_cache(self, mock_dataframe, default_dataset_params, temp_cache_dir):
        """Test dataset initialization with caching enabled."""
        params = default_dataset_params.copy()
        params["use_cache"] = True
        params["cache_dir"] = temp_cache_dir # Ensure cache_dir is properly passed

        dataset = CTDataset3D(dataframe=mock_dataframe, **params)

        assert dataset.use_cache
        assert dataset.cache_dir == temp_cache_dir
        assert temp_cache_dir.exists()

    def test_initialization_cache_without_dir_raises_error(self, mock_dataframe, default_dataset_params):
        """Test that enabling cache without cache_dir raises ValueError."""
        params = default_dataset_params.copy()
        params["use_cache"] = True
        params["cache_dir"] = None # Set cache_dir to None to trigger error

        with pytest.raises(ValueError, match="Cache directory must be provided if use_cache is True."):
            CTDataset3D(dataframe=mock_dataframe, **params)

    def test_len_method(self, mock_dataframe, default_dataset_params):
        """Test __len__ returns correct dataset size."""
        dataset = CTDataset3D(dataframe=mock_dataframe, **default_dataset_params)
        assert len(dataset) == len(mock_dataframe)

    def test_getitem_basic_monai_processed(self, mock_dataframe, default_dataset_params):
        """Test basic item retrieval with MONAI preprocessing (no caching)."""
        params = default_dataset_params.copy()
        dataset = CTDataset3D(dataframe=mock_dataframe, **params)

        sample = dataset[0]

        assert 'pixel_values' in sample
        assert 'labels' in sample
        assert 'volume_name' in sample

        assert isinstance(sample['pixel_values'], torch.Tensor)
        assert isinstance(sample['labels'], torch.Tensor)
        assert isinstance(sample['volume_name'], str)

        # Check shape based on target_shape_dhw
        expected_shape = (1, *params["target_shape_dhw"]) # C, D, H, W
        assert sample['pixel_values'].shape == expected_shape
        # Check dtype
        assert sample['pixel_values'].dtype == torch.float32
        # Check normalization (output of ScaleIntensityRanged should be [0,1])
        assert sample['pixel_values'].min() >= 0.0
        assert sample['pixel_values'].max() <= 1.0

        assert sample['labels'].shape == (len(params["pathology_columns"]),)
        assert sample['volume_name'] == 'volume1.nii.gz'

        expected_labels = torch.tensor([1.0, 0.0, 1.0, 0.0]) # Based on mock_dataframe and pathology_columns
        assert torch.allclose(sample['labels'], expected_labels)

    def test_getitem_with_missing_file(self, mock_dataframe, default_dataset_params):
        """Test handling of missing volume files; should return zero tensor."""
        params = default_dataset_params.copy()
        # Add a non-existent file to a copy of the dataframe
        df_with_missing = mock_dataframe.copy()
        missing_row_data = {'VolumeName': 'missing_volume.nii.gz'}
        for col in params["pathology_columns"]:
            missing_row_data[col] = 0 # Add dummy label data
        missing_row_df = pd.DataFrame([missing_row_data])
        df_with_missing = pd.concat([df_with_missing, missing_row_df], ignore_index=True)

        dataset = CTDataset3D(dataframe=df_with_missing, **params)
        
        sample = dataset[3] # Index of the missing file

        assert torch.all(sample['pixel_values'] == 0.0)
        expected_shape = (1, *params["target_shape_dhw"])
        assert sample['pixel_values'].shape == expected_shape

    # Patch the path to preprocess_ct_volume_monai if it's called inside CTDataset3D
    # The relevant path for patching depends on how it's imported in data/dataset.py
    # Assuming 'from .preprocessing import preprocess_ct_volume_monai'
    @patch('src.data.dataset.preprocess_ct_volume_monai')
    def test_cache_save_and_load(self, mock_preprocess_monai, mock_dataframe, default_dataset_params, temp_cache_dir):
        """Test that caching saves and loads MONAI-processed data correctly."""
        params = default_dataset_params.copy()
        params["use_cache"] = True
        params["cache_dir"] = temp_cache_dir

        # Define a return value for the mocked MONAI preprocessing
        # This tensor should match what the MONAI pipeline would output
        mock_processed_tensor = torch.rand(1, *params["target_shape_dhw"], dtype=torch.float32)
        mock_preprocess_monai.return_value = mock_processed_tensor
        
        dataset = CTDataset3D(dataframe=mock_dataframe, **params)

        # First access: should call preprocess, save to cache
        sample1 = dataset[0]
        mock_preprocess_monai.assert_called_once() # Should be called as cache is initially empty
        # Get the arguments preprocess_ct_volume_monai was called with
        args_call, kwargs_call = mock_preprocess_monai.call_args
        assert kwargs_call['nii_path'] == get_dynamic_image_path(params["img_dir"], mock_dataframe.iloc[0]['VolumeName'])
        assert kwargs_call['target_shape_dhw'] == params["target_shape_dhw"]
        assert isinstance(kwargs_call['preprocessing_pipeline'], MonaiCompose)


        # Check that the cache file was created
        cache_path = dataset._get_cache_path(0) # Use the dataset's internal method for cache path
        assert cache_path.exists()

        # Second access: should load from cache, not call preprocess
        mock_preprocess_monai.reset_mock() # Reset mock for the second call test
        
        # To ensure _load_from_cache is tested, we can spy on it
        with patch.object(dataset, '_load_from_cache', wraps=dataset._load_from_cache) as mock_load_cache_spy:
            sample2 = dataset[0]
            mock_load_cache_spy.assert_called_once_with(0) # Check it was called with the correct index

        mock_preprocess_monai.assert_not_called() # Should NOT be called for the second access

        assert torch.allclose(sample1['pixel_values'], sample2['pixel_values'])
        assert torch.allclose(sample1['labels'], sample2['labels'])
        assert sample1['volume_name'] == sample2['volume_name']

    def test_cache_corrupted_file_handling(self, mock_dataframe, default_dataset_params, temp_cache_dir):
        """Test handling of corrupted cache files, ensuring reprocessing and re-caching."""
        params = default_dataset_params.copy()
        params["use_cache"] = True
        params["cache_dir"] = temp_cache_dir

        dataset = CTDataset3D(dataframe=mock_dataframe, **params)

        # Manually create a corrupted cache file for index 0
        cache_path = dataset._get_cache_path(0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            f.write(b"corrupted cache data") # Write some invalid data

        assert cache_path.exists(), "Corrupted cache file should exist initially."

        # Expected tensor after reprocessing
        expected_reprocessed_tensor = torch.rand(1, *params["target_shape_dhw"])

        # Patch preprocess_ct_volume_monai to monitor its calls and return a known tensor
        with patch('src.data.dataset.preprocess_ct_volume_monai') as mock_re_preprocess:
            mock_re_preprocess.return_value = expected_reprocessed_tensor

            # Accessing the item should trigger cache load error, reprocessing, and re-caching
            sample_after_corruption = dataset[0]

            # 1. Assert that preprocessing was called (due to corrupted cache)
            mock_re_preprocess.assert_called_once()

            # 2. Assert that the cache file now exists (it was re-saved with good data)
            assert cache_path.exists(), "Cache file should exist after reprocessing and re-caching."

            # 3. Assert that the returned sample contains the reprocessed data
            assert torch.allclose(sample_after_corruption['pixel_values'], expected_reprocessed_tensor), \
                "Returned sample does not contain the correctly reprocessed pixel values."

            # 4. Assert that the re-saved cache file can be loaded and contains the correct data
            try:
                reloaded_cached_sample = torch.load(cache_path, map_location='cpu')
                assert 'pixel_values' in reloaded_cached_sample
                assert torch.allclose(reloaded_cached_sample['pixel_values'], expected_reprocessed_tensor), \
                    "Re-saved cache file does not contain the correctly reprocessed pixel values."
            except Exception as e:
                pytest.fail(f"Failed to load or verify the re-saved cache file: {e}")
            
            # Additional check: Access again, should now load directly from the valid cache
            # without calling preprocessing.
            mock_re_preprocess.reset_mock() # Reset the mock before the next access
            with patch.object(dataset, '_load_from_cache', wraps=dataset._load_from_cache) as spy_load_from_cache:
                sample_final_load = dataset[0]
                spy_load_from_cache.assert_called_once() # Should load from cache
                mock_re_preprocess.assert_not_called() # Should NOT reprocess
                assert torch.allclose(sample_final_load['pixel_values'], expected_reprocessed_tensor)


    def test_augmentation_applied(self, mock_dataframe, default_dataset_params):
        """Test that _apply_augmentation is called when augment=True."""
        params = default_dataset_params.copy()
        params["augment"] = True
        
        dataset = CTDataset3D(dataframe=mock_dataframe, **params)
        
        # Spy on _apply_augmentation
        with patch.object(dataset, '_apply_augmentation', wraps=dataset._apply_augmentation) as mock_aug_spy:
            _ = dataset[0] # Access an item to trigger the process
            mock_aug_spy.assert_called_once()

    def test_augmentation_transforms(self, mock_dataframe, default_dataset_params):
        """Test the _apply_augmentation method itself (isolated test)."""
        params = default_dataset_params.copy()
        dataset = CTDataset3D(dataframe=mock_dataframe, **params) # Augment is False here by default from params
        
        # Create a dummy tensor that mimics output from MONAI pipeline
        test_volume = torch.rand(1, *params["target_shape_dhw"], dtype=torch.float32) * 0.5 + 0.25 # Values between 0.25 and 0.75
        
        # Call augmentation manually (usually called internally if dataset.augment is True)
        torch.manual_seed(42) # For reproducibility of random ops in augmentation
        augmented_volume = dataset._apply_augmentation(test_volume.clone())
        
        assert augmented_volume.shape == test_volume.shape
        assert augmented_volume.min() >= 0.0 # Augmentations should keep it in [0,1]
        assert augmented_volume.max() <= 1.0
        # Check if it's actually different (stochastic augmentations)
        # This might fail if all random checks in augmentation happen to not apply anything
        # A more robust check would be to set random seeds and verify specific transformations
        # For now, just check it didn't error out and shape/range are maintained
        # If specific augmentations are guaranteed, check for their effects.

    @patch('src.data.dataset.preprocess_ct_volume_monai')
    def test_augmentation_with_cache(self, mock_preprocess_monai, mock_dataframe, default_dataset_params, temp_cache_dir):
        """Test that augmentation is applied after loading from cache if augment=True."""
        params = default_dataset_params.copy()
        params["use_cache"] = True
        params["cache_dir"] = temp_cache_dir
        params["augment"] = True # Enable augmentation

        # Mock the preprocessing to return a predictable, non-augmented tensor
        original_tensor = torch.rand(1, *params["target_shape_dhw"])
        mock_preprocess_monai.return_value = original_tensor.clone()

        dataset = CTDataset3D(dataframe=mock_dataframe, **params)

        # First access: preprocesses, saves original_tensor to cache, then augments for output
        with patch.object(dataset, '_apply_augmentation', wraps=dataset._apply_augmentation) as mock_aug_spy1:
            sample1 = dataset[0]
            mock_aug_spy1.assert_called_once() # Augmentation applied
            # Ensure the output is different from the cached (original) tensor if augmentation is effective
            if not torch.allclose(sample1['pixel_values'], original_tensor):
                pass # Augmentation had an effect
            else:
                # This might happen if augmentation is stochastic and by chance did nothing
                # Consider a deterministic part of _apply_augmentation for this test if possible
                pass


        mock_preprocess_monai.assert_called_once() # Preprocessing called only once

        # Second access: loads original_tensor from cache, then augments for output
        mock_preprocess_monai.reset_mock()
        with patch.object(dataset, '_apply_augmentation', wraps=dataset._apply_augmentation) as mock_aug_spy2:
            sample2 = dataset[0] # Should load 'original_tensor' from cache and then augment
            mock_aug_spy2.assert_called_once()

        mock_preprocess_monai.assert_not_called() # Not called again
        # sample1['pixel_values'] and sample2['pixel_values'] might be different due to stochastic augmentation
        # The key is that _apply_augmentation was called in both cases.


    def test_preprocessing_parameters_passed_to_pipeline_creation(self, mock_dataframe, default_dataset_params):
        """Test that CTDataset3D passes correct parameters to create_monai_preprocessing_pipeline."""
        params_to_check = default_dataset_params.copy()
        # Use unique values to ensure they are correctly passed
        params_to_check["target_spacing_xyz"] = np.array([1.1, 1.2, 1.3])
        params_to_check["target_shape_dhw"] = (33, 65, 66)
        params_to_check["clip_hu_min"] = -900.0
        params_to_check["clip_hu_max"] = 900.0
        params_to_check["orientation_axcodes"] = "RAS"

        # Patch 'create_monai_preprocessing_pipeline' where it's imported by 'data.dataset'
        with patch('src.data.dataset.create_monai_preprocessing_pipeline', return_value=Mock(spec=MonaiCompose)) as mock_create_pipeline:
            _ = CTDataset3D(dataframe=mock_dataframe, **params_to_check) # Initialize dataset
            
            mock_create_pipeline.assert_called_once()
            # Retrieve the arguments it was called with
            args, kwargs = mock_create_pipeline.call_args
            
            # Check keyword arguments
            assert np.array_equal(kwargs['target_spacing_xyz'], params_to_check["target_spacing_xyz"])
            assert kwargs['target_shape_dhw'] == params_to_check["target_shape_dhw"]
            assert kwargs['clip_hu_min'] == params_to_check["clip_hu_min"]
            assert kwargs['clip_hu_max'] == params_to_check["clip_hu_max"]
            assert kwargs['orientation_axcodes'] == params_to_check["orientation_axcodes"]

    # Dynamic path and label dtype tests are largely unaffected by MONAI change itself,
    # as they happen before/after the core image processing.
    # We assume get_dynamic_image_path and label extraction work as before.

    def test_dynamic_path_handling(self, mock_dataframe, default_dataset_params, temp_img_dir):
        """Test handling of hierarchical directory structure using get_dynamic_image_path."""
        params = default_dataset_params.copy()
        # Create a specific nested structure for one of the files
        # volume_entry = mock_dataframe.iloc[0] # Not strictly needed for this test logic
        # vol_name_parts = volume_entry['VolumeName'].replace(".nii.gz","").split("_") 
        
        hierarchical_name = "SUBJ01_SESSA_T1scan.nii.gz"

        expected_path_from_util = get_dynamic_image_path(params["img_dir"], hierarchical_name)
        expected_subdir = expected_path_from_util.parent
        expected_subdir.mkdir(parents=True, exist_ok=True)

        # Create the NIfTI file there with data that will not be entirely clipped to zero
        # by ScaleIntensityRanged (e.g., values within clip_hu_min and clip_hu_max)
        # Example: data ranging from approx -500 HU to 1000 HU
        # clip_hu_min = -1000, clip_hu_max = 1000.
        # Data like (np.random.rand() * 2000 - 1000) would span this range.
        # Or ensure some values are, for example, at 0 HU.
        test_nifti_data = (np.random.rand(32, 32, 32) * 1500.0 - 500.0).astype(np.float32) # Approx range [-500, 1000)
        # Ensure at least one voxel will be non-zero after scaling to [0,1]
        # For example, set a voxel to be 0 HU, which maps to 0.5 after scaling.
        test_nifti_data[0,0,0] = 0.0 
        
        nib.save(nib.Nifti1Image(test_nifti_data, np.eye(4)), expected_path_from_util)
        assert expected_path_from_util.exists(), "NIfTI file for dynamic path test was not saved correctly."


        df_hierarchical = mock_dataframe.copy()
        df_hierarchical.loc[0, 'VolumeName'] = hierarchical_name # Update one entry to use this name
        
        dataset = CTDataset3D(dataframe=df_hierarchical, **params)
        
        sample = dataset[0] # This should now use the hierarchical path
        assert sample['volume_name'] == hierarchical_name
        
        # Check that pixel_values are not the zero tensor (i.e., file was found and processed
        # with some non-zero resulting values)
        assert not torch.all(sample['pixel_values'] == 0.0), \
            "Processed pixel values are all zero, check NIfTI data range and MONAI's ScaleIntensityRanged."
        # Further check: ensure values are within [0,1] as expected from MONAI pipeline
        assert sample['pixel_values'].min() >= 0.0, "Pixel values should be >= 0.0"
        assert sample['pixel_values'].max() <= 1.0, "Pixel values should be <= 1.0"

# Integration tests with DataLoader
class TestDatasetIntegration:
    """Integration tests for CTDataset3D with DataLoader"""
    
    @pytest.fixture
    def small_dataset_for_loader(self, mock_dataframe, default_dataset_params):
        """Create a small dataset instance for DataLoader integration tests."""
        return CTDataset3D(dataframe=mock_dataframe, **default_dataset_params)
    
    def test_dataloader_integration(self, small_dataset_for_loader, default_dataset_params):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader # Local import for clarity
        
        dataloader = DataLoader(
            small_dataset_for_loader,
            batch_size=2,
            shuffle=True,
            num_workers=0 # Set to 0 for simplicity in testing
        )
        
        batch_count = 0
        expected_pixel_shape_no_batch = (1, *default_dataset_params["target_shape_dhw"]) # C, D, H, W
        
        for batch in dataloader:
            batch_count += 1
            assert 'pixel_values' in batch
            assert 'labels' in batch
            assert 'volume_name' in batch # Is a list of strings
            
            assert batch['pixel_values'].dim() == 5 # B, C, D, H, W
            assert batch['pixel_values'].shape[1:] == expected_pixel_shape_no_batch
            assert batch['labels'].dim() == 2 # B, NumPathologies
            assert isinstance(batch['volume_name'], list) # DataLoader collates strings into a list
        
        # 3 samples, batch_size 2 -> 2 batches (first batch size 2, second batch size 1)
        assert batch_count == 2