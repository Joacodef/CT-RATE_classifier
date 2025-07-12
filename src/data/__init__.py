# src/data/__init__.py
"""
Initializes the data module for the CT-RATE project.

This file makes key classes from the data handling modules available for easier
importing elsewhere in the project. By managing imports here, we can simplify
access to classes like `CTMetadataDataset` and `ApplyTransforms`, which are
central to creating and processing datasets within the training pipeline.

Exposed Classes:
- CTMetadataDataset: A PyTorch Dataset that loads CT scan paths and their
  corresponding multi-label pathology data from a pandas DataFrame. It serves
  as the base dataset before any MONAI transforms are applied.
- ApplyTransforms: A wrapper class that applies a set of transformations to
  the output of a base dataset, allowing for the separation of preprocessing
  and augmentation steps, which is particularly useful when using caching.
"""

from .dataset import CTMetadataDataset, ApplyTransforms

__all__ = [
    "CTMetadataDataset",
    "ApplyTransforms"
]