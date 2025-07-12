# src/data/dataset.py
"""
This module defines the dataset classes used for loading and preparing CT scan data
for the training pipeline. It follows a modular approach in line with MONAI best practices.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.utils import get_dynamic_image_path

# Get a logger for this module
logger = logging.getLogger(__name__)

class CTMetadataDataset(Dataset):
    """
    A PyTorch Dataset that provides metadata for CT scans.

    This dataset's primary role is to return the file path of a CT volume and its
    corresponding multi-label pathology vector for a given index. It does not load
    or perform any transformations on the image data itself. This separation of
    concerns makes it compatible with MONAI's data handling components like
    PersistentDataset, which manage data loading and caching.

    Attributes:
        dataframe (pd.DataFrame): The DataFrame containing volume names and labels.
        img_dir (Path): The base directory where the CT scan volumes are stored.
        pathology_columns (List[str]): A list of column names in the DataFrame
            that represent the pathology labels.
        path_mode (str): The directory structure mode ('flat' or 'nested') used
            to locate the image files.
    """
    def __init__(self, dataframe: pd.DataFrame, img_dir: Path,
                 pathology_columns: List[str], path_mode: str = "nested"):
        """
        Initializes the CTMetadataDataset.

        Args:
            dataframe (pd.DataFrame): DataFrame with volume names and pathology labels.
            img_dir (Path): The root directory for the image volumes.
            pathology_columns (List[str]): The names of the label columns.
            path_mode (str): The directory structure type. Can be 'flat' or 'nested'.
                'flat': All images are directly inside img_dir.
                'nested': Images are in subdirectories based on their name.
        """
        self.dataframe = dataframe
        self.img_dir = Path(img_dir)
        self.pathology_columns = pathology_columns
        self.path_mode = path_mode

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the metadata for a single sample.

        This method fetches the volume name and labels for the given index,
        constructs the full path to the NIfTI file, and returns them in a
        dictionary. This dictionary is the standard format expected by MONAI's
        downstream transform and loading components.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A dictionary containing:
            - 'image' (Path): The absolute path to the CT volume file.
            - 'label' (torch.Tensor): A float tensor of the pathology labels.
            - 'volume_name' (str): The name of the volume file.
        """
        row = self.dataframe.iloc[idx]
        volume_name = row['VolumeName']

        # Construct the full path to the image using the original volume name
        nii_path = get_dynamic_image_path(self.img_dir, volume_name, self.path_mode)

        if not nii_path.exists():
            logger.warning(
                f"Volume not found: {nii_path} for index {idx}. "
                "The 'image' key will point to a non-existent path."
            )

        labels = torch.tensor(row[self.pathology_columns].values.astype(float), dtype=torch.float32)

        return {
            "image": nii_path,
            "label": labels,
            "volume_name": volume_name
        }


class ApplyTransforms(Dataset):
    """
    A wrapper dataset that applies a set of transforms to the output of another dataset.

    This class is useful for applying transformations, such as data augmentation,
    on-the-fly after data has been loaded from a base dataset (which could be a
    caching dataset like MONAI's PersistentDataset).

    Attributes:
        data (Dataset): The base dataset to wrap.
        transform (Callable): A callable (e.g., a MONAI Compose object) that takes
            a data dictionary and returns a transformed version.
    """
    def __init__(self, data: Dataset, transform: Callable):
        """
        Initializes the ApplyTransforms wrapper.

        Args:
            data (Dataset): The base dataset that provides the initial data dict.
            transform (Callable): The function or Compose object to apply to the data.
        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        """Returns the length of the base dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves an item from the base dataset and applies the transform.

        Args:
            idx (int): The index of the sample to retrieve and transform.

        Returns:
            The transformed data dictionary.
        """
        item = self.data[idx]
        return self.transform(item)