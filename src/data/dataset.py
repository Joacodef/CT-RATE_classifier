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
    A PyTorch Dataset that provides the file path for a given CT scan.

    This dataset's sole responsibility is to resolve and return the file path
    for a given index based on a dataframe of volume names. It is completely
    agnostic to labels and image content, making it a clean source for a
    caching pipeline that should not be concerned with metadata.
    """
    def __init__(self, dataframe: pd.DataFrame, img_dir: Path, path_mode: str = "nested"):
        """
        Initializes the CTMetadataDataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing at least a 'VolumeName' column.
            img_dir (Path): The root directory for the image volumes.
            path_mode (str): The directory structure type ('flat' or 'nested').
        """
        self.dataframe = dataframe
        self.img_dir = Path(img_dir)
        self.path_mode = path_mode

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the metadata for a single sample.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A dictionary containing:
            - 'image' (Path): The absolute path to the CT volume file.
            - 'volume_name' (str): The name of the volume file.
        """
        row = self.dataframe.iloc[idx]
        volume_name = row['VolumeName']

        # Construct the full path to the image.
        nii_path = get_dynamic_image_path(self.img_dir, volume_name, self.path_mode)

        # if not nii_path.exists():
        #     logger.warning(
        #         f"Volume not found: {nii_path} for index {idx}. "
        #         "The 'image' key will point to a non-existent path."
        #     )

        return {
            "image": nii_path,
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
    

class LabelAttacherDataset(Dataset):
    """
    A wrapper dataset that attaches labels to data from a source dataset.

    This class is designed to work with a caching pipeline. It takes an
    'image_source' dataset (e.g., a MONAI CacheDataset) that yields processed,
    label-free image data, and a pandas DataFrame that holds the labels.
    It merges them together on-the-fly.
    """
    def __init__(self, image_source: Dataset, labels_df: pd.DataFrame, pathology_columns: List[str]):
        """
        Initializes the LabelAttacherDataset.

        Args:
            image_source (Dataset): The base dataset that provides the processed
                image data (e.g., a tensor and its metadata).
            labels_df (pd.DataFrame): The DataFrame containing the labels.
            pathology_columns (List[str]): The names of the label columns in the DataFrame.
        """
        self.image_source = image_source
        self.labels_df = labels_df
        self.pathology_columns = pathology_columns

    def __len__(self) -> int:
        """Returns the length of the source dataset."""
        return len(self.image_source)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves an item, attaches its label, and returns the combined dictionary.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A dictionary containing the image data from the source, plus the
            'label' and 'volume_name' from the labels DataFrame.
        """
        # Get the processed image and metadata from the caching pipeline.
        data_dict = self.image_source[idx]

        # Get the corresponding label information from the DataFrame.
        label_row = self.labels_df.iloc[idx]
        labels = torch.tensor(label_row[self.pathology_columns].values.astype(float), dtype=torch.float32)

        # Add the label to the dictionary.
        data_dict['label'] = labels
        
        # The volume_name should already be in data_dict, but we can ensure it is.
        # This also serves as a sanity check.
        data_dict['volume_name'] = label_row['VolumeName']

        return data_dict