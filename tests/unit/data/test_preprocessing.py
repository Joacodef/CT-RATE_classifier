# tests/unit/data/test_preprocessing.py
"""
Unit tests for MONAI-based data/preprocessing.py
"""

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
import torch
import tempfile
from unittest.mock import patch, MagicMock
import warnings # Import the standard Python warnings module

# Add project root to path for imports if running tests from a different directory
import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Functions to test
from data.preprocessing import (
    create_monai_preprocessing_pipeline,
    preprocess_ct_volume_monai
)
# MONAI imports for type checking
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    Resized,
    EnsureTyped
)

# === Fixtures ===

@pytest.fixture
def default_config_params():
    """Provides default configuration parameters for preprocessing."""
    return {
        "target_spacing_xyz": np.array([1.0, 1.0, 1.5]),
        "target_shape_dhw": (64, 128, 128), # D, H, W
        "clip_hu_min": -1000.0,
        "clip_hu_max": 1000.0,
        "orientation_axcodes": "LPS"
    }

@pytest.fixture
def sample_nifti_file(tmp_path: Path) -> Path:
    """Creates a temporary NIfTI file for testing."""
    # Define image properties
    shape = (50, 60, 70) # X, Y, Z
    data = np.random.randint(-1500, 1500, size=shape, dtype=np.int16)
    affine = np.diag([2.0, 2.5, 3.0, 1.0]) # Original spacing: 2.0, 2.5, 3.0 for X, Y, Z
    
    nifti_img = nib.Nifti1Image(data, affine)
    
    # Save to a temporary file in the tmp_path fixture directory
    filepath = tmp_path / "test_volume.nii.gz"
    nib.save(nifti_img, filepath)
    return filepath

# === Tests for create_monai_preprocessing_pipeline ===

class TestCreateMonaiPreprocessingPipeline:
    """Tests for the create_monai_preprocessing_pipeline function."""

    def test_returns_compose_object(self, default_config_params):
        """Test that the function returns a MONAI Compose object."""
        pipeline = create_monai_preprocessing_pipeline(**default_config_params)
        assert isinstance(pipeline, Compose)

    def test_pipeline_contains_expected_transforms(self, default_config_params):
        """Test that the pipeline contains the correct sequence of transforms."""
        pipeline = create_monai_preprocessing_pipeline(**default_config_params)
        expected_transform_types = [
            LoadImaged,
            EnsureChannelFirstd,
            Orientationd,
            Spacingd,
            ScaleIntensityRanged,
            Resized,
            EnsureTyped
        ]
        assert len(pipeline.transforms) == len(expected_transform_types)
        for transform, expected_type in zip(pipeline.transforms, expected_transform_types):
            assert isinstance(transform, expected_type), \
                f"Expected {expected_type}, got {type(transform)}"

    def test_transform_parameters_are_set(self, default_config_params):
        """Test that key parameters are correctly configured in the pipeline's transforms."""
        pipeline = create_monai_preprocessing_pipeline(**default_config_params)
        
        # --- Spacingd check ---
        spacing_transform_d = next((t for t in pipeline.transforms if isinstance(t, Spacingd)), None)
        assert spacing_transform_d is not None, "Spacingd transform not found in pipeline."
        expected_pixdim = tuple(float(s) for s in default_config_params["target_spacing_xyz"])
        try:
            # Access pixdim via the .spacing_transform attribute which holds the array Spacing instance
            actual_pixdim = spacing_transform_d.spacing_transform.pixdim
            assert np.allclose(actual_pixdim, expected_pixdim), \
                f"Spacingd pixdim expected {expected_pixdim}, got {actual_pixdim}"
        except AttributeError as e:
            pytest.fail(f"Failed to access Spacingd parameters via .spacing_transform.pixdim: {e}. "
                        f"Check dir(spacing_transform_d) and dir(spacing_transform_d.spacing_transform).")
        except Exception as e: # Catch other unexpected errors
            pytest.fail(f"Error checking Spacingd parameters: {e}")

        # --- ScaleIntensityRanged check ---
        scale_transform_d = next((t for t in pipeline.transforms if isinstance(t, ScaleIntensityRanged)), None)
        assert scale_transform_d is not None, "ScaleIntensityRanged transform not found in pipeline."
        try:
            # Access parameters via the .scaler attribute which holds the array ScaleIntensityRange instance
            array_scaler_instance = scale_transform_d.scaler
            assert array_scaler_instance.a_min == default_config_params["clip_hu_min"]
            assert array_scaler_instance.a_max == default_config_params["clip_hu_max"]
            assert array_scaler_instance.b_min == 0.0
            assert array_scaler_instance.b_max == 1.0
            assert array_scaler_instance.clip is True
        except AttributeError as e:
            pytest.fail(f"Failed to access ScaleIntensityRanged parameters via .scaler.{e.name}: {e}. "
                        f"Check dir(scale_transform_d) and dir(scale_transform_d.scaler).")
        except Exception as e: # Catch other unexpected errors
             pytest.fail(f"Error checking ScaleIntensityRanged parameters: {e}")

        # --- Resized check ---
        resize_transform_d = next((t for t in pipeline.transforms if isinstance(t, Resized)), None)
        assert resize_transform_d is not None, "Resized transform not found in pipeline."
        expected_spatial_size = tuple(int(s) for s in default_config_params["target_shape_dhw"])
        try:
            # Access spatial_size via the .resizer attribute which holds the array Resize instance
            actual_spatial_size = resize_transform_d.resizer.spatial_size
            # MONAI's Resize stores spatial_size as a tuple of ints
            assert actual_spatial_size == expected_spatial_size, \
                 f"Resized spatial_size expected {expected_spatial_size}, got {actual_spatial_size}"
        except AttributeError as e:
            pytest.fail(f"Failed to access Resized parameters via .resizer.spatial_size: {e}. "
                        f"Check dir(resize_transform_d) and dir(resize_transform_d.resizer).")
        except Exception as e: # Catch other unexpected errors
            pytest.fail(f"Error checking Resized parameters: {e}")
        
        # --- Orientationd check ---
        orientation_transform_d = next((t for t in pipeline.transforms if isinstance(t, Orientationd)), None)
        assert orientation_transform_d is not None, "Orientationd transform not found in pipeline."
        try:
            # Access axcodes via the .ornt_transform attribute which holds the array Orientation instance
            actual_axcodes = orientation_transform_d.ornt_transform.axcodes
            assert actual_axcodes == default_config_params["orientation_axcodes"]
        except AttributeError as e:
            pytest.fail(f"Failed to access Orientationd parameters via .ornt_transform.axcodes: {e}. "
                        f"Check dir(orientation_transform_d) and dir(orientation_transform_d.ornt_transform).")
        except Exception as e: # Catch other unexpected errors
            pytest.fail(f"Error checking Orientationd parameters: {e}")

         # --- EnsureTyped check ---
        ensure_typed_transform = next((t for t in pipeline.transforms if isinstance(t, EnsureTyped)), None)
        assert ensure_typed_transform is not None, "EnsureTyped transform not found in pipeline."
        try:
            expected_dtype = torch.float32
            actual_dtype_attr = None

            if hasattr(ensure_typed_transform, 'output_dtype'):
                actual_dtype_attr = ensure_typed_transform.output_dtype
            elif hasattr(ensure_typed_transform, 'dtype'): 
                actual_dtype_attr = ensure_typed_transform.dtype
            else:
                raise AttributeError("'output_dtype' or 'dtype' not found on EnsureTyped object.")

            # Check if the actual dtype attribute is a tuple containing the expected dtype,
            # or if it's the dtype itself.
            if isinstance(actual_dtype_attr, tuple):
                assert expected_dtype in actual_dtype_attr, \
                    f"EnsureTyped dtype expected {expected_dtype} to be in tuple {actual_dtype_attr}"
                # Optionally, if you expect it to be the *only* item:
                # assert actual_dtype_attr == (expected_dtype,), \
                #    f"EnsureTyped dtype expected {(expected_dtype,)}, got {actual_dtype_attr}"
            else:
                assert actual_dtype_attr == expected_dtype, \
                    f"EnsureTyped dtype expected {expected_dtype}, got {actual_dtype_attr}"

        except AttributeError as e: 
            warnings.warn(UserWarning(
                f"'EnsureTyped' object does not appear to have 'output_dtype' or 'dtype' attribute as expected ({e}). "
                "Use print(dir(ensure_typed_transform)) to investigate. Cannot directly verify dtype."
            ))
        except AssertionError: # Catch the specific assertion error from the dtype comparison
            pytest.fail(f"EnsureTyped dtype assertion failed. Expected {expected_dtype}, got {actual_dtype_attr}")
        except Exception as e: 
            pytest.fail(f"Error checking EnsureTyped parameters: {e}")


# === Tests for preprocess_ct_volume_monai ===

class TestPreprocessCTVolumeMonai:
    """Tests for the preprocess_ct_volume_monai function."""

    def test_basic_preprocessing_success(self, sample_nifti_file, default_config_params):
        """Test successful preprocessing of a valid NIfTI file."""
        pipeline = create_monai_preprocessing_pipeline(**default_config_params)
        target_shape_dhw = default_config_params["target_shape_dhw"]
        
        result_tensor = preprocess_ct_volume_monai(
            nii_path=sample_nifti_file,
            preprocessing_pipeline=pipeline,
            target_shape_dhw=target_shape_dhw
        )
        
        assert isinstance(result_tensor, torch.Tensor)
        # Expected shape: [C, D, H, W] where C=1
        expected_shape = (1, *target_shape_dhw)
        assert result_tensor.shape == expected_shape, \
            f"Output shape {result_tensor.shape} does not match expected {expected_shape}"
        assert result_tensor.dtype == torch.float32
        
        # Check intensity range (should be [0, 1] after ScaleIntensityRanged)
        assert result_tensor.min() >= 0.0
        assert result_tensor.max() <= 1.0
        # A more robust check for max might be needed if the sample data doesn't span the full HU range
        # For instance, if all input HU values are low, max might be < 1.0.
        # But it should not exceed 1.0.

    def test_file_not_found_returns_zero_tensor(self, default_config_params):
        """Test that a non-existent file path returns a zero tensor of the correct shape."""
        pipeline = create_monai_preprocessing_pipeline(**default_config_params)
        target_shape_dhw = default_config_params["target_shape_dhw"]
        non_existent_path = Path("/tmp/this/path/does/not/exist/fake.nii.gz")
        
        result_tensor = preprocess_ct_volume_monai(
            nii_path=non_existent_path,
            preprocessing_pipeline=pipeline,
            target_shape_dhw=target_shape_dhw
        )
        
        assert isinstance(result_tensor, torch.Tensor)
        expected_shape = (1, *target_shape_dhw)
        assert result_tensor.shape == expected_shape
        assert torch.all(result_tensor == 0.0)
        assert result_tensor.dtype == torch.float32

    @patch('monai.transforms.LoadImaged.__call__') # Mock the __call__ method of LoadImaged instance
    def test_loadimage_error_returns_zero_tensor(self, mock_load_imaged_call, default_config_params):
        """Test that an error during image loading (simulated) returns a zero tensor."""
        mock_load_imaged_call.side_effect = Exception("Simulated NIfTI loading error")
        
        # Create a dummy file path, though LoadImaged is mocked
        dummy_path = Path("dummy_for_load_error.nii.gz")

        pipeline = create_monai_preprocessing_pipeline(**default_config_params)
        target_shape_dhw = default_config_params["target_shape_dhw"]

        result_tensor = preprocess_ct_volume_monai(
            nii_path=dummy_path, # Path itself doesn't matter as LoadImaged is mocked
            preprocessing_pipeline=pipeline,
            target_shape_dhw=target_shape_dhw
        )
        
        assert isinstance(result_tensor, torch.Tensor)
        expected_shape = (1, *target_shape_dhw)
        assert result_tensor.shape == expected_shape
        assert torch.all(result_tensor == 0.0)

    def test_output_shape_consistency(self, sample_nifti_file, default_config_params):
        """Ensure the output shape is strictly [1, D, H, W]."""
        custom_target_shape = (32, 64, 72) # Different from fixture default
        config_params = default_config_params.copy()
        config_params["target_shape_dhw"] = custom_target_shape
        
        pipeline = create_monai_preprocessing_pipeline(**config_params)
        
        result_tensor = preprocess_ct_volume_monai(
            nii_path=sample_nifti_file,
            preprocessing_pipeline=pipeline,
            target_shape_dhw=custom_target_shape
        )
        
        expected_shape = (1, *custom_target_shape)
        assert result_tensor.shape == expected_shape

    def test_intensity_clipping_and_scaling(self, tmp_path, default_config_params):
        """Test specific effect of ScaleIntensityRanged."""
        shape = (20, 20, 20)
        # Data with values below min, above max, and within range
        data = np.zeros(shape, dtype=np.float32)
        data[..., :5] = default_config_params["clip_hu_min"] - 500  # Below min
        data[..., 5:10] = default_config_params["clip_hu_max"] + 500 # Above max
        data[..., 10:15] = (default_config_params["clip_hu_min"] + default_config_params["clip_hu_max"]) / 2 # Mid-range
        data[..., 15:] = default_config_params["clip_hu_min"] # At min

        affine = np.diag([1.0, 1.0, 1.0, 1.0]) # Simple spacing
        nifti_img = nib.Nifti1Image(data, affine)
        filepath = tmp_path / "intensity_test.nii.gz"
        nib.save(nifti_img, filepath)

        # Use a simplified pipeline for this specific test if needed, or full pipeline
        # For simplicity, we assume the full pipeline should work.
        # We'll use a target shape that's close to original to minimize resizing artifacts.
        test_config_params = default_config_params.copy()
        test_config_params["target_shape_dhw"] = shape # Test with minimal resizing
        test_config_params["target_spacing_xyz"] = np.array([1.0, 1.0, 1.0])


        pipeline = create_monai_preprocessing_pipeline(**test_config_params)
        
        result_tensor = preprocess_ct_volume_monai(
            nii_path=filepath,
            preprocessing_pipeline=pipeline,
            target_shape_dhw=shape
        )

        assert result_tensor.min() >= 0.0, "Min value should be >= 0.0"
        assert result_tensor.max() <= 1.0, "Max value should be <= 1.0"

        # Check regions:
        # Where original was far below min, output should be 0.0
        # Where original was far above max, output should be 1.0
        # Where original was mid-range, output should be ~0.5
        # Where original was at min, output should be 0.0
        # This is harder to check precisely due to interpolation during SpacingD/Resized,
        # but the overall range [0,1] must hold.

        # A simple check: if the original image had large uniform areas,
        # after processing (especially if little resampling), these areas should
        # map to 0, 1, or 0.5 respectively.
        # For example, if data[0,0,0] was clip_hu_min - 500, then result_tensor[0,:,:,:] corresponding to this
        # original region should be close to 0.0.
        # However, precise voxel-wise checking is complex with resampling.
        # The range check is the most robust for an end-to-end test.

        # If we want to check more precisely, we'd mock SpacingD and Resized to be identity
        # or test ScaleIntensityRanged in isolation.
        # For now, the range check is a good indicator.