# data/dataset.py
import torch
from torch.utils.data import Dataset
# Updated import: using MONAI-based preprocessing
from .preprocessing import create_monai_preprocessing_pipeline, preprocess_ct_volume_monai
from .utils import get_dynamic_image_path
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import logging # Standard library logging
logger = logging.getLogger(__name__) # Standard library logger

class CTDataset3D(Dataset):
    """
    Represents a 3D CT scan dataset for pathology classification.
    This class handles loading, preprocessing (using MONAI), caching, and augmenting CT scan volumes.
    It is designed to work with PyTorch's DataLoader for efficient batch processing.
    """

    def __init__(self, dataframe: pd.DataFrame, img_dir: Path,
                 pathology_columns: List[str], target_spacing_xyz: np.ndarray,
                 target_shape_dhw: Tuple[int, int, int], clip_hu_min: float, clip_hu_max: float,
                 use_cache: bool = True, cache_dir: Optional[Path] = None,
                 augment: bool = False, orientation_axcodes: str = "LPS", path_mode: str = 'nested',
                 save_transformed_path: Optional[Path] = None):
        """
        Initializes the CTDataset3D instance.

        Args:
            dataframe: Pandas DataFrame containing metadata for the dataset,
                       including volume names and pathology labels.
            img_dir: Path to the directory containing CT scan image files (e.g., .nii.gz).
            pathology_columns: List of column names in the DataFrame that represent pathology labels.
            target_spacing_xyz: Numpy array specifying the desired voxel spacing (x, y, z)
                                for resampling.
            target_shape_dhw: Tuple specifying the desired shape (depth, height, width)
                                  for the processed volumes.
            clip_hu_min: Minimum Hounsfield Unit (HU) value for windowing.
            clip_hu_max: Maximum Hounsfield Unit (HU) value for windowing.
            use_cache: Boolean indicating whether to use caching for preprocessed samples.
                       Defaults to True.
            cache_dir: Path to the directory for storing cached samples.
                       Required if use_cache is True. Defaults to None.
            augment: Boolean indicating whether to apply data augmentation.
                     Defaults to False.
            orientation_axcodes: Target orientation for MONAI's Orientationd transform (e.g., "LPS", "RAS").
                                 Defaults to "LPS".
            save_transformed_path (Optional[Path]): If provided, the directory path where
                                                     the transformed images will be saved.
                                                     Defaults to None.
        """

        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = Path(img_dir) # Ensures img_dir is a Path object
        self.pathology_columns = pathology_columns
        # Parameters for MONAI pipeline are stored directly
        self.target_spacing_xyz = np.asarray(target_spacing_xyz)
        self.target_shape_dhw = tuple(target_shape_dhw)
        self.clip_hu_min = float(clip_hu_min)
        self.clip_hu_max = float(clip_hu_max)
        self.orientation_axcodes = orientation_axcodes
        self.path_mode = path_mode 
        self.save_transformed_path = save_transformed_path
        

        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augment = augment

        if self.use_cache:
            if not self.cache_dir:
                # Logs an error if caching is enabled but no cache directory is provided.
                logger.error("Cache directory must be provided if use_cache is True.")
                raise ValueError("Cache directory must be provided if use_cache is True.")
            # Creates the cache directory if it does not already exist.
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the MONAI preprocessing pipeline
        # This pipeline is created once and reused for all samples.
        self.monai_preprocessing_pipeline = create_monai_preprocessing_pipeline(
            target_spacing_xyz=self.target_spacing_xyz,
            target_shape_dhw=self.target_shape_dhw,
            clip_hu_min=self.clip_hu_min,
            clip_hu_max=self.clip_hu_max,
            orientation_axcodes=self.orientation_axcodes,
            save_transformed_path=self.save_transformed_path # Step 2.2: Pass argument to pipeline creator
        )

        logger.info(f"Dataset initialized with {len(self.dataframe)} samples. Caching: {self.use_cache}, Augmentation: {self.augment}. MONAI pipeline created.")

    def _get_cache_path(self, idx: int) -> Path:
        """
        Generates the file path for a cached sample.
        """
        volume_name = self.dataframe.iloc[idx]['VolumeName']
        cleaned_name = volume_name.replace(".nii.gz", "").replace(".nii", "")
        return self.cache_dir / f"sample_monai_{cleaned_name}_{idx:06d}.pt"

    def _load_from_cache(self, idx: int) -> Optional[Dict[str, any]]:
        """
        Attempts to load a preprocessed sample from the cache.
        If loading fails (e.g., corrupted file), it attempts to delete the corrupted cache file.
        """
        if not self.use_cache or not self.cache_dir:
            return None

        cache_path = self._get_cache_path(idx)
        if cache_path.exists():
            try:
                logger.debug(f"Loading sample {idx} from cache: {cache_path}")
                return torch.load(cache_path, map_location='cpu')
            except Exception as e:
                logger.warning(f"CACHE_LOAD_ERROR: Error loading sample {idx} from cache {cache_path}: {type(e).__name__} - {e}. Attempting to remove corrupted file.")
                if cache_path.exists():
                    logger.info(f"CACHE_DELETE_ATTEMPT: Corrupted file {cache_path} exists. Attempting unlink.")
                    try:
                        cache_path.unlink()
                        if not cache_path.exists():
                            logger.info(f"CACHE_DELETE_SUCCESS: Successfully removed corrupted cache file: {cache_path}")
                        else:
                            logger.error(f"CACHE_DELETE_FAILURE_POST_UNLINK: File {cache_path} still exists immediately after unlink attempt.")
                    except OSError as unlink_e:
                        logger.error(f"CACHE_DELETE_OSERROR: Failed to remove corrupted cache file {cache_path} due to OSError: {type(unlink_e).__name__} - {unlink_e}")
                        if hasattr(unlink_e, 'winerror'):
                            logger.error(f"CACHE_DELETE_OSERROR_WINCODE: Windows error code: {unlink_e.winerror}")
                else:
                    logger.info(f"CACHE_DELETE_SKIP: Corrupted cache file {cache_path} was already removed or did not exist before unlink attempt in except block.")
        return None

    def _save_to_cache(self, idx: int, sample: Dict[str, any]):
        """
        Saves a preprocessed sample to the cache.
        """
        if not self.use_cache or not self.cache_dir:
            return

        cache_path = self._get_cache_path(idx)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample, cache_path)
            logger.debug(f"Saved sample {idx} to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving sample {idx} to cache {cache_path}: {e}")

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Retrieves a sample from the dataset by index.
        """
        # NOTE: When saving transformed files, caching should ideally be disabled
        # to ensure the save operation is triggered for each item.
        cached_sample = self._load_from_cache(idx)
        if cached_sample is not None:
            if self.augment and 'pixel_values' in cached_sample:
                cached_sample['pixel_values'] = self._apply_augmentation(cached_sample['pixel_values'].clone())
            return cached_sample

        row = self.dataframe.iloc[idx]
        volume_name = row['VolumeName']
        labels_data = row[self.pathology_columns].values.astype(np.float32)
        labels = torch.tensor(labels_data, dtype=torch.float32)
        img_path = get_dynamic_image_path(self.img_dir, volume_name, mode=self.path_mode)

        pixel_values: torch.Tensor
        if not img_path or not img_path.exists():
            logger.error(f"Volume not found: {img_path} for index {idx}. Returning zero tensor.")
            pixel_values = torch.zeros((1, *self.target_shape_dhw), dtype=torch.float32)
        else:
            # The monai_preprocessing_pipeline now includes the saving transform if the path was provided
            pixel_values = preprocess_ct_volume_monai(
                nii_path=img_path,
                preprocessing_pipeline=self.monai_preprocessing_pipeline,
                target_shape_dhw=self.target_shape_dhw
            )

        sample = {
            "pixel_values": pixel_values,
            "labels": labels,
            "volume_name": str(volume_name)
        }

        self._save_to_cache(idx, sample)

        if self.augment:
            sample["pixel_values"] = self._apply_augmentation(sample["pixel_values"].clone())

        return sample

    def _apply_augmentation(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Applies a series of random augmentations to the input volume.
        """
        volume = volume.float()
        if torch.rand(1).item() > 0.5:
            volume = torch.flip(volume, dims=[1])
            logger.debug("Applied depth flip augmentation.")
        if torch.rand(1).item() > 0.5:
            volume = torch.flip(volume, dims=[3])
            logger.debug("Applied width flip augmentation.")
        if torch.rand(1).item() > 0.7:
            noise = torch.randn_like(volume) * 0.01
            volume = volume + noise
            volume = torch.clamp(volume, 0.0, 1.0)
            logger.debug("Applied noise augmentation.")
        if torch.rand(1).item() > 0.5:
            shift = (torch.rand(1).item() - 0.5) * 0.1
            volume = volume + shift
            volume = torch.clamp(volume, 0.0, 1.0)
            logger.debug(f"Applied intensity shift: {shift:.4f}")
        return volume