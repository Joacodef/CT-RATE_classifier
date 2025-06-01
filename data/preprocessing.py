# data/preprocessing.py

import numpy as np
import torch
from pathlib import Path
import logging # Standard library logging
import gc # Garbage collection

# MONAI imports
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    Resized,
    EnsureTyped
    # CropForegroundd was commented out, keeping it that way unless specified
)
from monai.data import MetaTensor

logger = logging.getLogger(__name__)

def create_monai_preprocessing_pipeline(
    target_spacing_xyz: np.ndarray,
    target_shape_dhw: tuple,
    clip_hu_min: float,
    clip_hu_max: float,
    orientation_axcodes: str = "LPS"
) -> Compose:
    """
    Creates a MONAI preprocessing pipeline.

    Args:
        target_spacing_xyz: Desired voxel spacing in mm as [x, y, z] array.
        target_shape_dhw: Target volume shape as (depth, height, width).
        clip_hu_min: Minimum Hounsfield Unit (HU) value for clipping.
        clip_hu_max: Maximum Hounsfield Unit (HU) value for clipping.
        orientation_axcodes: Target orientation for the image data (e.g., "LPS", "RAS").

    Returns:
        monai.transforms.Compose: The MONAI preprocessing pipeline.
    """
    pixdim = tuple(float(s) for s in target_spacing_xyz)
    spatial_size = tuple(int(s) for s in target_shape_dhw)

    transforms = [
        LoadImaged(keys="image", image_only=True, ensure_channel_first=False, reader="NibabelReader"),
        EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
        Orientationd(keys="image", axcodes=orientation_axcodes),
        Spacingd(
            keys="image",
            pixdim=pixdim,
            mode="bilinear",
            align_corners=True # align_corners is generally fine with bilinear/trilinear for Spacingd
        ),
        ScaleIntensityRanged(
            keys="image",
            a_min=float(clip_hu_min),
            a_max=float(clip_hu_max),
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        # CropForegroundd(keys="image", source_key="image", k_divisible=tuple(int(s/4) for s in spatial_size)),
        Resized(
            keys="image",
            spatial_size=spatial_size,
            mode="area"
            # align_corners=True, # REMOVED: Incompatible with mode="area"
        ),
        EnsureTyped(keys="image", dtype=torch.float32, track_meta=False)
    ]
    return Compose(transforms)

def preprocess_ct_volume_monai(
    nii_path: Path,
    preprocessing_pipeline: Compose,
    target_shape_dhw: tuple
) -> torch.Tensor:
    data_dict = {"image": str(nii_path)}
    try:
        processed_data_dict = preprocessing_pipeline(data_dict)
        processed_tensor = processed_data_dict["image"]

        if isinstance(processed_tensor, MetaTensor):
            processed_tensor = processed_tensor.as_tensor()

        if processed_tensor.ndim == 3:
            processed_tensor = processed_tensor.unsqueeze(0)
        elif processed_tensor.ndim == 4 and processed_tensor.shape[0] != 1 :
            logger.warning(f"Processed tensor for {nii_path} has unexpected channel dimension: {processed_tensor.shape}. Attempting to fix.")
            if processed_tensor.shape[0] > 1:
                processed_tensor = processed_tensor[0, ...].unsqueeze(0)
        
        # This check is important for verifying the pipeline's final output shape.
        # If an error occurs in the pipeline, the except block below handles returning a zero tensor.
        # This block handles cases where the pipeline runs without crashing but produces an unexpected shape.
        if processed_tensor.shape[1:] != target_shape_dhw or processed_tensor.shape[0] !=1:
            logger.error(f"Shape mismatch after MONAI processing: actual {processed_tensor.shape} vs expected (1, {target_shape_dhw}). Path: {nii_path}. This might occur if an error was suppressed or if pipeline logic is flawed.")
            # Fallback to zero tensor if shape is incorrect, even if no direct crash in pipeline.
            # This indicates a logic error or unexpected behavior in the transforms.
            return torch.zeros((1, *target_shape_dhw), dtype=torch.float32) # Ensure zero tensor is returned

        gc.collect()
        return processed_tensor
    except Exception as e:
        logger.error(f"Error processing {nii_path} with MONAI: {type(e).__name__} - {e}", exc_info=True)
        return torch.zeros((1, *target_shape_dhw), dtype=torch.float32)