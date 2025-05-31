"""
Unit tests for data/preprocessing.py

Tests cover:
- Basic preprocessing functionality
- Edge cases and error handling
- Memory-efficient resize operations
"""

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
import torch
import tempfile
import os
from unittest.mock import Mock, patch

# Add project root to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent) + "/data")

from preprocessing import preprocess_ct_volume, resize_volume


class TestPreprocessCTVolume:
    """Tests for the main preprocess_ct_volume function"""
    
    @pytest.fixture
    def sample_volume_data(self):
        """Create a simple synthetic CT volume for testing"""
        # Create a 32x32x32 volume with values in HU range
        volume = np.random.randint(-1000, 1000, size=(32, 32, 32), dtype=np.int16)
        # Add some structure (a sphere in the center)
        center = np.array([16, 16, 16])
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                    if dist < 10:
                        volume[i, j, k] = 100  # Soft tissue HU value
        return volume
    
    @pytest.fixture
    def sample_nifti_file(self, sample_volume_data, tmp_path):
        """Create a temporary NIfTI file for testing"""
        # Create NIfTI image with specific spacing
        affine = np.eye(4)
        affine[0, 0] = 2.0  # 2mm spacing in X
        affine[1, 1] = 2.0  # 2mm spacing in Y
        affine[2, 2] = 2.0  # 2mm spacing in Z
        
        nifti_img = nib.Nifti1Image(sample_volume_data, affine)
        
        # Save to temporary file
        filepath = tmp_path / "test_volume.nii.gz"
        nib.save(nifti_img, filepath)
        
        return filepath
    
    def test_basic_preprocessing(self, sample_nifti_file):
        """Test basic preprocessing with valid input"""
        # Define parameters
        target_spacing = np.array([1.0, 1.0, 1.0])
        target_shape = (32, 32, 32)
        clip_min = -1000
        clip_max = 1000
        
        # Run preprocessing
        result = preprocess_ct_volume(
            sample_nifti_file,
            target_spacing,
            target_shape,
            clip_min,
            clip_max
        )
        
        # Verify output
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 32, 32)  # [C, D, H, W]
        assert result.dtype == torch.float32
        
    def test_output_value_range(self, sample_nifti_file):
        """Verify that output values are normalized to [0, 1]"""
        target_spacing = np.array([1.0, 1.0, 1.0])
        target_shape = (32, 32, 32)
        clip_min = -1000
        clip_max = 1000
        
        result = preprocess_ct_volume(
            sample_nifti_file,
            target_spacing,
            target_shape,
            clip_min,
            clip_max
        )
        
        # Check value range
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        
    def test_clipping_extreme_values(self, tmp_path):
        """Test that HU values outside clip range are properly clipped"""
        # Create volume with extreme values
        extreme_volume = np.array([
            [[-2000, -1500], [500, 1500]],
            [[0, 100], [2000, 3000]]
        ], dtype=np.float32)
        extreme_volume = np.repeat(extreme_volume[:, :, :, np.newaxis], 16, axis=3)
        extreme_volume = np.repeat(extreme_volume[:, :, np.newaxis, :], 16, axis=2)
        
        # Create NIfTI
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(extreme_volume, affine)
        filepath = tmp_path / "extreme_volume.nii.gz"
        nib.save(nifti_img, filepath)
        
        # Process
        result = preprocess_ct_volume(
            filepath,
            np.array([1.0, 1.0, 1.0]),
            (16, 16, 16),
            clip_hu_min=-1000,
            clip_hu_max=1000
        )
        
        # Verify clipping worked
        # Original values were -2000 to 3000, should be clipped to -1000 to 1000
        # Then normalized to 0 to 1
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        
    def test_spacing_resampling(self, tmp_path):
        """Test that spacing resampling works correctly"""
        # Create volume with known spacing
        volume = np.ones((20, 20, 20), dtype=np.float32) * 100
        
        # 4mm spacing
        affine = np.eye(4)
        affine[0, 0] = 4.0
        affine[1, 1] = 4.0
        affine[2, 2] = 4.0
        
        nifti_img = nib.Nifti1Image(volume, affine)
        filepath = tmp_path / "spacing_test.nii.gz"
        nib.save(nifti_img, filepath)
        
        # Resample to 2mm spacing
        result = preprocess_ct_volume(
            filepath,
            target_spacing_xyz=np.array([2.0, 2.0, 2.0]),
            target_shape_dhw=(40, 40, 40),  # Should be ~40x40x40 after resampling
            clip_hu_min=-1000,
            clip_hu_max=1000
        )
        
        # Check shape after resampling and resizing
        assert result.shape == (1, 40, 40, 40)
        
    def test_transpose_dimensions(self, sample_nifti_file):
        """Test that dimensions are transposed correctly from XYZ to DHW"""
        # Create a volume with distinct dimensions
        volume = np.zeros((10, 20, 30), dtype=np.float32)  # X, Y, Z
        
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(volume, affine)
        filepath = sample_nifti_file.parent / "transpose_test.nii.gz"
        nib.save(nifti_img, filepath)
        
        # Process without resizing to check transpose
        result = preprocess_ct_volume(
            filepath,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(30, 20, 10),  # Z, Y, X (transposed)
            clip_hu_min=-1000,
            clip_hu_max=1000
        )
        
        assert result.shape == (1, 30, 20, 10)  # [C, D, H, W]
        
    def test_file_not_found(self):
        """Test handling of non-existent file"""
        non_existent = Path("/path/to/nowhere/file.nii.gz")
        
        result = preprocess_ct_volume(
            non_existent,
            np.array([1.0, 1.0, 1.0]),
            (32, 32, 32),
            -1000,
            1000
        )
        
        # Should return zero tensor on error
        assert result.shape == (1, 32, 32, 32)
        assert torch.all(result == 0.0)
        
    @patch('nibabel.load')
    def test_corrupted_file(self, mock_nib_load):
        """Test handling of corrupted NIfTI file"""
        # Mock nibabel.load to raise an exception
        mock_nib_load.side_effect = Exception("Corrupted file")
        
        result = preprocess_ct_volume(
            Path("dummy.nii.gz"),
            np.array([1.0, 1.0, 1.0]),
            (32, 32, 32),
            -1000,
            1000
        )
        
        # Should return zero tensor on error
        assert result.shape == (1, 32, 32, 32)
        assert torch.all(result == 0.0)


class TestResizeVolume:
    """Tests for the resize_volume function"""
    
    def test_same_shape_returns_original(self):
        """If input and target shapes are the same, return original array"""
        data = np.random.rand(32, 32, 32).astype(np.float32)
        result = resize_volume(data, (32, 32, 32))
        
        # Should be the exact same array
        assert np.array_equal(data, result)
        
    def test_basic_upsampling(self):
        """Test upsampling from smaller to larger volume"""
        data = np.ones((16, 16, 16), dtype=np.float32)
        result = resize_volume(data, (32, 32, 32))
        
        assert result.shape == (32, 32, 32)
        assert result.dtype == np.float32
        # Values should still be around 1.0 (interpolation)
        assert np.allclose(result.mean(), 1.0, atol=0.1)
        
    def test_basic_downsampling(self):
        """Test downsampling from larger to smaller volume"""
        data = np.ones((64, 64, 64), dtype=np.float32)
        result = resize_volume(data, (32, 32, 32))
        
        assert result.shape == (32, 32, 32)
        assert result.dtype == np.float32
        # Values should still be around 1.0
        assert np.allclose(result.mean(), 1.0, atol=0.1)
        
    def test_non_uniform_scaling(self):
        """Test resizing with different scaling factors per dimension"""
        data = np.ones((10, 20, 30), dtype=np.float32)
        result = resize_volume(data, (20, 20, 20))
        
        assert result.shape == (20, 20, 20)
        
    @patch('scipy.ndimage.zoom')
    def test_memory_error_fallback(self, mock_zoom):
        """Test fallback mechanism when scipy.zoom raises MemoryError"""
        # First call raises MemoryError, forcing fallback
        mock_zoom.side_effect = MemoryError("Out of memory")
        
        data = np.random.rand(32, 32, 32).astype(np.float32)
        result = resize_volume(data, (64, 64, 64))
        
        # Should still produce correct shape using fallback method
        assert result.shape == (64, 64, 64)
        assert result.dtype == np.float32
        
    @patch('preprocessing.zoom')
    def test_unexpected_error_fallback(self, mock_zoom):
        """Test fallback mechanism for unexpected errors"""
        # Simulate unexpected error
        mock_zoom.side_effect = RuntimeError("Unexpected error")
        
        data = np.random.rand(32, 32, 32).astype(np.float32)
        result = resize_volume(data, (64, 64, 64))
        
        # Should return zeros as emergency fallback
        assert result.shape == (64, 64, 64)
        assert np.all(result == 0.0)
        
    def test_floating_point_zoom_factors(self):
        """Test that floating point errors in zoom factors are handled"""
        # This can cause issues if zoom results in slightly wrong shape
        data = np.ones((33, 33, 33), dtype=np.float32)
        result = resize_volume(data, (32, 32, 32))
        
        # Should have exact target shape
        assert result.shape == (32, 32, 32)
        
    @pytest.mark.parametrize("input_shape,target_shape", [
        ((16, 16, 16), (32, 32, 32)),    # 2x upsampling
        ((32, 32, 32), (16, 16, 16)),    # 2x downsampling
        ((10, 20, 30), (30, 20, 10)),    # Mixed scaling
        ((17, 23, 29), (32, 32, 32)),    # Prime numbers to standard
    ])
    def test_various_resize_scenarios(self, input_shape, target_shape):
        """Test resize with various input/output shape combinations"""
        data = np.random.rand(*input_shape).astype(np.float32)
        result = resize_volume(data, target_shape)
        
        assert result.shape == target_shape
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))  # No NaN values
        

class TestEdgeCases:
    """Additional edge case tests"""
    
    def test_zero_volume(self, tmp_path):
        """Test processing of volume with all zeros"""
        zero_volume = np.zeros((32, 32, 32), dtype=np.float32)
        
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(zero_volume, affine)
        filepath = tmp_path / "zero_volume.nii.gz"
        nib.save(nifti_img, filepath)
        
        result = preprocess_ct_volume(
            filepath,
            np.array([1.0, 1.0, 1.0]),
            (32, 32, 32),
            -1000,
            1000
        )
        
        # Should normalize to 0.5 (since -1000 -> 0, 1000 -> 1, 0 -> 0.5)
        expected_value = (0 - (-1000)) / (1000 - (-1000))
        assert np.allclose(result.numpy(), expected_value, atol=0.01)
        
    def test_single_slice_volume(self, tmp_path):
        """Test handling of volume with single slice"""
        single_slice = np.ones((64, 64, 1), dtype=np.float32) * 100
        
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(single_slice, affine)
        filepath = tmp_path / "single_slice.nii.gz"
        nib.save(nifti_img, filepath)
        
        result = preprocess_ct_volume(
            filepath,
            np.array([1.0, 1.0, 1.0]),
            (32, 32, 32),
            -1000,
            1000
        )
        
        # Should handle single slice gracefully
        assert result.shape == (1, 32, 32, 32)


class TestPerformance:
    """Performance-related tests"""
    
    def test_large_volume_memory_usage(self, tmp_path):
        """Test that memory usage is reasonable for larger volumes"""
        # Create a 256x256x256 volume
        large_volume = np.random.randint(-1000, 1000, size=(256, 256, 256), dtype=np.int16)
        
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(large_volume, affine)
        filepath = tmp_path / "large_volume.nii.gz"
        nib.save(nifti_img, filepath)
        
        # This should complete without memory errors
        result = preprocess_ct_volume(
            filepath,
            np.array([2.0, 2.0, 2.0]),
            (128, 128, 128),
            -1000,
            1000
        )
        
        assert result.shape == (1, 128, 128, 128)