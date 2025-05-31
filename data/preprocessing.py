import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
from pathlib import Path
import torch
import gc

import logging
logger = logging.getLogger(__name__)


def preprocess_ct_volume(
    nii_path: Path,
    target_spacing_xyz: np.ndarray,
    target_shape_dhw: tuple,
    clip_hu_min: float = -1000,
    clip_hu_max: float = 1000
) -> torch.Tensor:
    """
    Preprocess CT volume for 3D classification with memory-efficient operations.
    
    This function loads a NIfTI CT scan, resamples it to uniform spacing,
    clips Hounsfield Unit (HU) values to a specified range, normalizes the
    intensities, and resizes to a fixed shape suitable for neural network input.
    
    Args:
        nii_path: Path to the input NIfTI file
        target_spacing_xyz: Desired voxel spacing in mm as [x, y, z] array
        target_shape_dhw: Target volume shape as (depth, height, width)
        clip_hu_min: Minimum HU value for clipping (default: -1000 for air)
        clip_hu_max: Maximum HU value for clipping (default: 1000 for bone)
    
    Returns:
        torch.Tensor: Preprocessed volume with shape [1, D, H, W] where 1 is
                     the channel dimension required for 3D CNNs
    
    Note:
        - Uses float32 throughout to balance precision and memory usage
        - Employs linear interpolation (order=1) for speed
        - Returns zero tensor on error to maintain batch processing stability
    """
    
    try:
        # Load NIfTI file with explicit float32 to prevent memory bloat
        nii_img = nib.load(str(nii_path))
        data = nii_img.get_fdata(dtype=np.float32)
        
        # Extract voxel spacing from NIfTI header (in mm)
        original_spacing_xyz = np.array(nii_img.header.get_zooms()[:3])
        
        # Calculate zoom factors for each axis to achieve target spacing
        # zoom_factor > 1 means upsampling, < 1 means downsampling
        zoom_factors_xyz = original_spacing_xyz / target_spacing_xyz
        
        # Resample volume to isotropic spacing if needed
        # Skip if already at target spacing (within 1% tolerance)
        if not np.allclose(zoom_factors_xyz, 1.0, atol=0.01):
            # Linear interpolation (order=1) provides good balance between
            # quality and speed for medical images
            # prefilter=False saves memory by skipping spline prefiltering
            data = zoom(data, zoom_factors_xyz, order=1, mode='nearest', prefilter=False)
        
        # Clip HU values to focus on relevant tissue range
        # -1000 HU (air) to 1000 HU (dense bone) covers most diagnostic needs
        data = np.clip(data, clip_hu_min, clip_hu_max)
        
        # Normalize to [0, 1] range for neural network compatibility
        data = (data - clip_hu_min) / (clip_hu_max - clip_hu_min)
        
        # Reorder axes from (X, Y, Z) to (Z, Y, X) format
        # This makes Z the first dimension (depth in medical convention)
        data = np.transpose(data, (2, 1, 0))
        
        # Resize to fixed shape for batch processing
        # Uses memory-efficient implementation with fallback options
        data = resize_volume(data, target_shape_dhw)
        
        # Convert to PyTorch tensor with channel dimension
        # Shape becomes [1, D, H, W] for 3D convolution compatibility
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        
        # Explicitly free memory from large numpy array
        del data
        gc.collect()
        
        return tensor
        
    except Exception as e:
        logger.error(f"Error processing {nii_path}: {e}")
        # Return zero-filled tensor to maintain batch integrity
        # This allows training to continue even if one sample fails
        return torch.zeros((1, *target_shape_dhw), dtype=torch.float32)


def resize_volume(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resize 3D volume to target shape with multiple fallback strategies.
    
    This function attempts progressively simpler resizing methods if memory
    errors occur, ensuring processing can continue even on limited hardware.
    
    Args:
        data: Input 3D numpy array
        target_shape: Desired output shape as (depth, height, width)
    
    Returns:
        np.ndarray: Resized volume with exact target_shape
    
    Strategies (in order):
        1. scipy.ndimage.zoom with linear interpolation
        2. Simple nearest-neighbor sampling using numpy indexing
        3. Zero-filled array (emergency fallback)
    """
    current_shape = np.array(data.shape)
    target_shape = np.array(target_shape)
    
    # Fast path: no resizing needed
    if np.array_equal(current_shape, target_shape):
        return data
    
    # Calculate zoom factors for each dimension
    zoom_factors = target_shape.astype(float) / current_shape.astype(float)
    
    # Primary method: scipy zoom with memory-efficient settings
    try:
        # order=1: linear interpolation (faster than cubic)
        # mode='constant': pad with zeros at boundaries
        # prefilter=False: skip spline prefiltering to save memory
        data_resized = zoom(data, zoom_factors, order=1, mode='constant', 
                          cval=0, prefilter=False)
        
        # Handle floating-point rounding errors in zoom calculation
        # Ensure output has exact target shape
        if data_resized.shape != tuple(target_shape):
            result = np.zeros(target_shape, dtype=data.dtype)
            
            # Copy valid region (handles both over and undersized results)
            min_shape = np.minimum(data_resized.shape, target_shape)
            result[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                data_resized[:min_shape[0], :min_shape[1], :min_shape[2]]
            
            return result
        
        return data_resized
        
    except MemoryError:
        logger.warning("Memory error during resize, using fallback method")
        # Fallback method: Simple nearest-neighbor sampling
        # Much more memory efficient but lower quality
        result = np.zeros(target_shape, dtype=data.dtype)
        
        # Pre-calculate sampling indices for each dimension
        # This avoids repeated calculations in loops
        depth_indices = np.round(np.linspace(0, current_shape[0] - 1, target_shape[0])).astype(int)
        height_indices = np.round(np.linspace(0, current_shape[1] - 1, target_shape[1])).astype(int)
        width_indices = np.round(np.linspace(0, current_shape[2] - 1, target_shape[2])).astype(int)
        
        # Vectorized indexing for efficiency
        # Process one depth slice at a time to manage memory
        for i, d_idx in enumerate(depth_indices):
            for j, h_idx in enumerate(height_indices):
                result[i, j, :] = data[d_idx, h_idx, width_indices]
        
        return result
    
    except Exception as e:
        logger.error(f"Unexpected error during resize: {e}")
        # Emergency fallback: return zero-filled volume
        # This ensures the pipeline doesn't crash completely
        logger.warning("Using zero-filled fallback due to resize error")
        return np.zeros(target_shape, dtype=data.dtype)