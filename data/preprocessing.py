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
    """Preprocess CT volume for 3D classification with improved memory handling"""
    
    try:
        # Load NIfTI file
        nii_img = nib.load(str(nii_path))
        data = nii_img.get_fdata(dtype=np.float32)
        
        # Get original spacing
        original_spacing_xyz = np.array(nii_img.header.get_zooms()[:3])
        
        # Calculate zoom factors
        zoom_factors_xyz = original_spacing_xyz / target_spacing_xyz
        
        # Resample if needed
        if not np.allclose(zoom_factors_xyz, 1.0, atol=0.01):
            # Use order=1 (linear) for speed and memory efficiency
            data = zoom(data, zoom_factors_xyz, order=1, mode='nearest', prefilter=False)
        
        # Clip HU values
        data = np.clip(data, clip_hu_min, clip_hu_max)
        
        # Normalize to [0, 1]
        data = (data - clip_hu_min) / (clip_hu_max - clip_hu_min)
        
        # Transpose to (D, H, W) format
        data = np.transpose(data, (2, 1, 0))
        
        # Resize to target shape
        data = resize_volume_robust(data, target_shape_dhw)
        
        # Convert to tensor and add channel dimension
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]
        
        # Explicitly delete large arrays
        del data
        gc.collect()
        
        return tensor
        
    except Exception as e:
        logger.error(f"Error processing {nii_path}: {e}")
        # Return zero tensor as fallback
        return torch.zeros((1, *target_shape_dhw), dtype=torch.float32)



def resize_volume_robust(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Robust volume resizing with improved memory efficiency"""
    current_shape = np.array(data.shape)
    target_shape = np.array(target_shape)
    
    # Direct resize if shapes match
    if np.array_equal(current_shape, target_shape):
        return data
    
    # Calculate zoom factors
    zoom_factors = target_shape.astype(float) / current_shape.astype(float)
    
    # Apply zoom with memory-efficient settings
    try:
        data_resized = zoom(data, zoom_factors, order=1, mode='constant', 
                          cval=0, prefilter=False)
        
        # Ensure exact target shape (handle floating point errors)
        if data_resized.shape != tuple(target_shape):
            result = np.zeros(target_shape, dtype=data.dtype)
            
            # Calculate overlap region
            min_shape = np.minimum(data_resized.shape, target_shape)
            result[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                data_resized[:min_shape[0], :min_shape[1], :min_shape[2]]
            
            return result
        
        return data_resized
        
    except MemoryError:
        logger.warning("Memory error during resize, using fallback method")
        # Fallback: simple nearest neighbor sampling
        result = np.zeros(target_shape, dtype=data.dtype)
        
        # Calculate sampling indices for each dimension
        depth_indices = np.round(np.linspace(0, current_shape[0] - 1, target_shape[0])).astype(int)
        height_indices = np.round(np.linspace(0, current_shape[1] - 1, target_shape[1])).astype(int)
        width_indices = np.round(np.linspace(0, current_shape[2] - 1, target_shape[2])).astype(int)
        
        # Use vectorized indexing instead of nested loops
        for i, d_idx in enumerate(depth_indices):
            for j, h_idx in enumerate(height_indices):
                result[i, j, :] = data[d_idx, h_idx, width_indices]
        
        return result
    
    except Exception as e:
        logger.error(f"Unexpected error during resize: {e}")
        # Emergency fallback: return zeros
        logger.warning("Using zero-filled fallback due to resize error")
        return np.zeros(target_shape, dtype=data.dtype)