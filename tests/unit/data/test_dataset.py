# tests/unit/data/test_dataset.py
"""
Unit tests for src/data/dataset.py

This module contains tests for the dataset-related classes.
- CTMetadataDataset: Tests that it correctly maps an index to a file path and label.
- ApplyTransforms: Tests that it correctly wraps a dataset and applies a transform.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np
import pandas as pd
import torch
import nibabel as nib

# Add the project root to the Python path to allow src imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Import the classes to be tested
from src.data.dataset import CTMetadataDataset, ApplyTransforms

# --- Fixtures ---

@pytest.fixture
def mock_dataframe() -> pd.DataFrame:
    """Creates a mock dataframe with pathology labels."""
    data = {
        'VolumeName': ['volume1.nii.gz', 'volume2.nii.gz'],
        'Cardiomegaly': [1, 0],
        'Atelectasis': [0, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def pathology_columns() -> list[str]:
    """Returns a list of pathology column names."""
    return ['Cardiomegaly', 'Atelectasis']

@pytest.fixture
def temp_img_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory with mock NIfTI files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create mock NIfTI files with the correct "__transformed" suffix
    for i in range(1, 3):
        # FIX: Create the filename that the code will actually look for.
        volume_name = f"volume{i}__transformed.nii.gz"
        data = np.random.rand(10, 10, 10).astype(np.float32)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, img_dir / volume_name)
    return img_dir

# --- Test Classes ---

class TestCTMetadataDataset:
    """Tests for the CTMetadataDataset class."""

    def test_initialization(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test basic dataset initialization."""
        dataset = CTMetadataDataset(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            path_mode="flat"
        )
        assert len(dataset.dataframe) == 2
        assert dataset.img_dir == temp_img_dir
        assert dataset.pathology_columns == pathology_columns

    def test_len_method(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test that __len__ returns the correct dataset size."""
        dataset = CTMetadataDataset(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            path_mode="flat"
        )
        assert len(dataset) == len(mock_dataframe)

    def test_getitem_returns_correct_data(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test that __getitem__ returns a dictionary with the correct path and labels."""
        dataset = CTMetadataDataset(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            path_mode="flat"
        )
        
        # Test the first item
        item = dataset[0]
        
        # FIX: The expected path now includes the "__transformed" suffix to match the code's behavior.
        expected_path = temp_img_dir / "volume1__transformed.nii.gz"
        expected_label = torch.tensor([1, 0], dtype=torch.float32)
        
        assert isinstance(item, dict)
        assert "image" in item
        assert "label" in item
        assert "volume_name" in item
        
        assert item["image"] == expected_path
        assert torch.all(torch.eq(item["label"], expected_label))
        assert item["volume_name"] == "volume1.nii.gz" # The volume name itself should remain original

class TestApplyTransforms:
    """Tests for the ApplyTransforms wrapper class."""

    def test_transform_is_applied(self):
        """Test that the transform is correctly applied to the base dataset's output."""
        # 1. Create a mock for the base dataset
        mock_base_dataset = MagicMock()
        base_item = {"image": "path/to/image", "label": torch.tensor([1.0])}
        mock_base_dataset.__getitem__.return_value = base_item
        mock_base_dataset.__len__.return_value = 1

        # 2. Create a mock for the transform
        mock_transform = MagicMock()
        transformed_item = {"image": torch.rand(1, 64, 64, 64), "label": torch.tensor([1.0])}
        mock_transform.return_value = transformed_item
        
        # 3. Initialize ApplyTransforms with the mocks
        wrapped_dataset = ApplyTransforms(data=mock_base_dataset, transform=mock_transform)
        
        assert len(wrapped_dataset) == 1

        # 4. Access an item to trigger the process
        result = wrapped_dataset[0]

        # 5. Assert that the mocks were used as expected
        mock_base_dataset.__getitem__.assert_called_once_with(0)
        mock_transform.assert_called_once_with(base_item)
        assert result == transformed_item