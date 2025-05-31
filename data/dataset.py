import torch
from torch.utils.data import Dataset
from .preprocessing import preprocess_ct_volume
from .utils import get_dynamic_image_path
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

import logging
logger = logging.getLogger(__name__)

class CTDataset3D(Dataset):
    """3D CT Dataset for pathology classification with improved error handling"""
    
    def __init__(self, dataframe: pd.DataFrame, img_dir: Path,
                 pathology_columns: list, target_spacing_xyz: np.ndarray,
                 target_shape_dhw: tuple, clip_hu_min: float, clip_hu_max: float,
                 use_cache: bool = True, cache_dir: Optional[Path] = None,
                 augment: bool = False):
        
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.pathology_columns = pathology_columns
        self.target_spacing_xyz = target_spacing_xyz
        self.target_shape_dhw = target_shape_dhw
        self.clip_hu_min = clip_hu_min
        self.clip_hu_max = clip_hu_max
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.augment = augment
        
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Dataset initialized with {len(self.dataframe)} samples")
        
    def _get_cache_path(self, idx: int) -> Path:
        name = self.dataframe.iloc[idx]['VolumeName'].replace(".nii.gz", "")
        return self.cache_dir / f"sample_{name}_{idx:06d}.pt"
    
    def _load_from_cache(self, idx: int) -> Optional[dict]:
        if not self.use_cache or not self.cache_dir:
            return None
            
        cache_path = self._get_cache_path(idx)
        if cache_path.exists():
            try:
                return torch.load(cache_path, map_location='cpu')
            except Exception as e:
                logger.warning(f"Error loading cache {cache_path}: {e}")
                cache_path.unlink(missing_ok=True)
        return None
    
    def _save_to_cache(self, idx: int, sample: dict):
        if not self.use_cache or not self.cache_dir:
            return
            
        cache_path = self._get_cache_path(idx)
        try:
            torch.save(sample, cache_path)
        except Exception as e:
            logger.warning(f"Error saving to cache {cache_path}: {e}")
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> dict:
        # Try loading from cache first
        cached_sample = self._load_from_cache(idx)
        if cached_sample is not None:
            if self.augment and 'pixel_values' in cached_sample:
                cached_sample['pixel_values'] = self._apply_augmentation(cached_sample['pixel_values'])
            return cached_sample
        
        # Load and process sample
        row = self.dataframe.iloc[idx]
        volume_name = row['VolumeName']
        
        # Get labels
        labels = torch.tensor(
            row[self.pathology_columns].values.astype(np.float32),
            dtype=torch.float32
        )
        
        # Get image path and load volume
        img_path = get_dynamic_image_path(self.img_dir, volume_name)
        
        if not img_path.exists():
            logger.error(f"Volume not found: {img_path}")
            pixel_values = torch.zeros((1, *self.target_shape_dhw), dtype=torch.float32)
        else:
            pixel_values = preprocess_ct_volume(
                img_path, self.target_spacing_xyz, self.target_shape_dhw,
                self.clip_hu_min, self.clip_hu_max
            )
        
        sample = {
            "pixel_values": pixel_values,
            "labels": labels,
            "volume_name": volume_name
        }
        
        # Save to cache before augmentation
        self._save_to_cache(idx, sample)
        
        # Apply augmentation if enabled
        if self.augment:
            sample["pixel_values"] = self._apply_augmentation(sample["pixel_values"])
        
        return sample

    def _apply_augmentation(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply simple but effective augmentations"""
        # Random flip along depth axis
        if torch.rand(1) > 0.5:
            volume = torch.flip(volume, dims=[1])  # Flip depth
        
        # Random flip along width axis  
        if torch.rand(1) > 0.5:
            volume = torch.flip(volume, dims=[3])  # Flip width
            
        # Add small amount of noise
        if torch.rand(1) > 0.7:
            noise = torch.randn_like(volume) * 0.01
            volume = volume + noise
            volume = torch.clamp(volume, 0, 1)
        
        # Random intensity shift
        if torch.rand(1) > 0.5:
            shift = (torch.rand(1) - 0.5) * 0.1
            volume = volume + shift
            volume = torch.clamp(volume, 0, 1)
        
        return volume
