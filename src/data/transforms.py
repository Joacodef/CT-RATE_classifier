# src/data/transforms.py
"""
This module defines the transformation pipelines for data preprocessing.

Centralizing the transforms ensures that the same preprocessing is applied
consistently, whether during dataset verification, training, or inference.
This is crucial for the integrity of the caching mechanism.
"""

from typing import List, Any, Dict, Hashable

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    Resized,
    EnsureTyped,
    MapTransform
)

class KeyCleanerD(MapTransform):
    """
    A MONAI-style dictionary transform to keep only a specified set of keys.
    """
    def __init__(self, keys_to_keep: List[str], allow_missing_keys: bool = False):
        super().__init__(keys_to_keep, allow_missing_keys)
        self.keys_to_keep = set(keys_to_keep)

    def __call__(self, data: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        return {key: data[key] for key in self.keys_to_keep if key in data}

    def get_transform_info(self) -> Dict[str, Any]:
        """Returns a dictionary of parameters for serialization."""
        return {
            "class": self.__class__.__name__,
            "keys_to_keep": sorted(list(self.keys_to_keep))
        }


def get_preprocessing_transforms(config: Any) -> Compose:
    """
    Builds and returns the MONAI preprocessing transform pipeline.

    This pipeline is responsible for loading a NIfTI file and preparing it for
    caching. It loads the image and its metadata, applies spatial and intensity
    transformations, and cleans the resulting dictionary to ensure the cache
    is self-contained and minimal.

    Args:
        config: A configuration object (e.g., a SimpleNamespace) containing
                parameters from the config file, such as target spacing and shape.

    Returns:
        A MONAI Compose object representing the preprocessing pipeline.
    """
    # These are the only keys that will be saved to the persistent cache.
    final_keys = ["image", "image_meta_dict"]

    return Compose([
        # Load image data and metadata (affine, etc.). image_only=False is critical.
        LoadImaged(keys="image", image_only=False, ensure_channel_first=True, reader="NibabelReader"),

        # Apply spatial transformations which require the affine from the metadata.
        Orientationd(keys="image", axcodes=config.image_processing.orientation_axcodes),
        Spacingd(keys="image", pixdim=config.image_processing.target_spacing, mode="bilinear"),
        
        # Apply intensity and shape transformations.
        ScaleIntensityRanged(
            keys="image",
            a_min=config.image_processing.clip_hu_min,
            a_max=config.image_processing.clip_hu_max,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        Resized(keys="image", spatial_size=config.image_processing.target_shape_dhw, mode="area"),
        
        # Ensure correct data types.
        EnsureTyped(keys="image", dtype=config.torch_dtype),

        # Clean the dictionary, keeping only the self-contained, processed data.
        KeyCleanerD(keys_to_keep=final_keys)
    ])