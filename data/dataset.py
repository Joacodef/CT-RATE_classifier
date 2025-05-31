import torch
from torch.utils.data import Dataset
from .preprocessing import preprocess_ct_volume
from .utils import get_dynamic_image_path
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import logging
logger = logging.getLogger(__name__)

class CTDataset3D(Dataset):
    """
    Represents a 3D CT scan dataset for pathology classification.
    This class handles loading, preprocessing, caching, and augmenting CT scan volumes.
    It is designed to work with PyTorch's DataLoader for efficient batch processing.
    """

    def __init__(self, dataframe: pd.DataFrame, img_dir: Path,
                 pathology_columns: List[str], target_spacing_xyz: np.ndarray,
                 target_shape_dhw: Tuple[int, int, int], clip_hu_min: float, clip_hu_max: float,
                 use_cache: bool = True, cache_dir: Optional[Path] = None,
                 augment: bool = False):
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
        """

        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = Path(img_dir) # Ensures img_dir is a Path object
        self.pathology_columns = pathology_columns
        self.target_spacing_xyz = np.asarray(target_spacing_xyz) # Ensures it's a numpy array
        self.target_shape_dhw = tuple(target_shape_dhw) # Ensures it's a tuple
        self.clip_hu_min = float(clip_hu_min)
        self.clip_hu_max = float(clip_hu_max)
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

        logger.info(f"Dataset initialized with {len(self.dataframe)} samples. Caching: {self.use_cache}, Augmentation: {self.augment}")

    def _get_cache_path(self, idx: int) -> Path:
        """
        Generates the file path for a cached sample.

        The cache path is constructed using the volume name and index to ensure uniqueness.
        It replaces '.nii.gz' to create a cleaner filename for the PyTorch tensor file.

        Args:
            idx: The index of the sample in the dataset.

        Returns:
            Path object representing the full path to the cached file.
        """
        # Retrieves the volume name from the dataframe for the given index.
        volume_name = self.dataframe.iloc[idx]['VolumeName']
        # Cleans the volume name by removing common NIfTI file extensions.
        cleaned_name = volume_name.replace(".nii.gz", "").replace(".nii", "")
        return self.cache_dir / f"sample_{cleaned_name}_{idx:06d}.pt"

    def _load_from_cache(self, idx: int) -> Optional[Dict[str, any]]:
        """
        Attempts to load a preprocessed sample from the cache.

        Args:
            idx: The index of the sample in the dataset.

        Returns:
            A dictionary containing the cached sample if found and valid, otherwise None.
            Handles potential errors during cache loading, such as corrupted files.
        """
        if not self.use_cache or not self.cache_dir:
            return None

        cache_path = self._get_cache_path(idx)
        if cache_path.exists():
            try:
                # Loads the sample from the .pt file. map_location='cpu' ensures
                # the tensor is loaded to CPU memory, which is generally safer
                # before moving to a specific device later if needed.
                logger.debug(f"Loading sample {idx} from cache: {cache_path}")
                return torch.load(cache_path, map_location='cpu')
            except Exception as e:
                # Logs a warning and removes the corrupted cache file to prevent future errors.
                logger.warning(f"Error loading sample {idx} from cache {cache_path}: {e}. Removing corrupted file.")
                cache_path.unlink(missing_ok=True) # missing_ok=True prevents error if file was already deleted
        return None

    def _save_to_cache(self, idx: int, sample: Dict[str, any]):
        """
        Saves a preprocessed sample to the cache.

        The sample is saved as a PyTorch tensor file (.pt).

        Args:
            idx: The index of the sample in the dataset.
            sample: The preprocessed sample dictionary to be cached.
        """
        if not self.use_cache or not self.cache_dir:
            return

        cache_path = self._get_cache_path(idx)
        try:
            # Ensures the parent directory exists before attempting to save.
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample, cache_path)
            logger.debug(f"Saved sample {idx} to cache: {cache_path}")
        except Exception as e:
            # Logs a warning if saving to cache fails, but does not interrupt the process.
            # This could be due to disk space issues or permissions.
            logger.warning(f"Error saving sample {idx} to cache {cache_path}: {e}")

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        This is a required method for PyTorch Datasets.
        """
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Retrieves a sample from the dataset by index.

        This method first attempts to load the sample from cache. If not found or
        caching is disabled, it loads and preprocesses the CT scan volume.
        Augmentation is applied if enabled, after potential caching of the
        original preprocessed sample.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing the processed 'pixel_values' (the 3D image tensor),
            'labels' (the pathology labels tensor), and 'volume_name' (string).
        """
        # Attempts to load the sample from cache.
        cached_sample = self._load_from_cache(idx)
        if cached_sample is not None:
            # If augmentation is enabled and the sample is from cache,
            # applies augmentation to the cached pixel values.
            # This ensures that augmentations are applied on-the-fly even for cached items.
            if self.augment and 'pixel_values' in cached_sample:
                # Clones the tensor before augmentation if it might be used elsewhere,
                # or if augmentation is in-place. Here, _apply_augmentation returns a new tensor.
                cached_sample['pixel_values'] = self._apply_augmentation(cached_sample['pixel_values'].clone())
            return cached_sample

        # If not in cache or caching is disabled, proceeds to load and process the sample.
        row = self.dataframe.iloc[idx]
        volume_name = row['VolumeName']

        # Extracts pathology labels from the dataframe row and converts them to a float tensor.
        labels_data = row[self.pathology_columns].values.astype(np.float32)
        labels = torch.tensor(labels_data, dtype=torch.float32)

        # Constructs the full path to the image file.
        # get_dynamic_image_path might handle cases where images are in subdirectories or have varying names.
        img_path = get_dynamic_image_path(self.img_dir, volume_name)

        pixel_values: torch.Tensor # Type hint for clarity
        if not img_path or not img_path.exists(): # Added check for None from get_dynamic_image_path
            # Logs an error and returns a zero tensor if the image file is not found.
            # This prevents crashes and allows the training to continue, though it might affect model performance.
            logger.error(f"Volume not found: {img_path} for index {idx}. Returning zero tensor.")
            # Creates a tensor of zeros with the expected shape (channels, depth, height, width).
            # Assuming single channel for CT scans.
            pixel_values = torch.zeros((1, *self.target_shape_dhw), dtype=torch.float32)
        else:
            # Preprocesses the CT volume using the specified parameters.
            # This typically involves loading, resampling, windowing, normalizing, and converting to a tensor.
            pixel_values = preprocess_ct_volume(
                img_path,
                self.target_spacing_xyz,
                self.target_shape_dhw,
                self.clip_hu_min,
                self.clip_hu_max
            )

        sample = {
            "pixel_values": pixel_values,
            "labels": labels,
            "volume_name": str(volume_name) # Ensures volume_name is a string
        }

        # Saves the non-augmented version of the sample to cache if caching is enabled.
        # This allows faster retrieval later and separates caching from augmentation.
        self._save_to_cache(idx, sample)

        # Applies augmentation if enabled.
        # Augmentation is applied after caching the original preprocessed sample.
        if self.augment:
            # Clones the tensor before augmentation to avoid modifying the cached version
            # if _apply_augmentation modifies in-place or if the original tensor is needed.
            sample["pixel_values"] = self._apply_augmentation(sample["pixel_values"].clone())

        return sample

    def _apply_augmentation(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Applies a series of random augmentations to the input volume.

        Augmentations include random flips along depth and width axes,
        addition of Gaussian noise, and random intensity shifts.
        These help improve model generalization by exposing it to more varied data.

        Args:
            volume: A PyTorch tensor representing the 3D CT scan volume,
                    expected shape (channels, depth, height, width).

        Returns:
            The augmented volume as a PyTorch tensor.
        """
        # Ensures volume is a floating point tensor for augmentations.
        volume = volume.float()

        # Randomly flips the volume along the depth axis (dim=1 for C,D,H,W).
        if torch.rand(1).item() > 0.5:
            volume = torch.flip(volume, dims=[1])
            logger.debug("Applied depth flip augmentation.")

        # Randomly flips the volume along the width axis (dim=3 for C,D,H,W).
        if torch.rand(1).item() > 0.5:
            volume = torch.flip(volume, dims=[3])
            logger.debug("Applied width flip augmentation.")

        # Adds a small amount of Gaussian noise to the volume.
        # This simulates scanner noise and can improve robustness.
        # Applied with a 30% probability.
        if torch.rand(1).item() > 0.7: # Original probability was 0.7, meaning 30% chance to add noise
            noise = torch.randn_like(volume) * 0.01 # Small noise standard deviation
            volume = volume + noise
            # Clamps the volume to the typical [0, 1] range after adding noise.
            volume = torch.clamp(volume, 0.0, 1.0)
            logger.debug("Applied noise augmentation.")

        # Randomly shifts the intensity of the volume.
        # This simulates variations in image intensity.
        if torch.rand(1).item() > 0.5:
            # Generates a random shift factor between -0.05 and +0.05.
            shift = (torch.rand(1).item() - 0.5) * 0.1 # Shift range +/- 5%
            volume = volume + shift
            # Clamps the volume to the [0, 1] range after intensity shift.
            volume = torch.clamp(volume, 0.0, 1.0)
            logger.debug(f"Applied intensity shift: {shift:.4f}")

        return volume