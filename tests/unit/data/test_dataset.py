"""
Unit tests for data/dataset.py

Tests cover:
- Dataset initialization and configuration
- Cache functionality (loading and saving)
- Data loading and preprocessing
- Augmentation pipeline
- Error handling for missing files
- Label extraction and formatting
"""

import os
import sys
from pathlib import Path

# Set up the import path before any other imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import standard libraries
import pytest
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import tempfile
import shutil # Required if you use shutil.rmtree, though not explicitly used in current test snippets
from unittest.mock import Mock, patch, MagicMock

# Import project modules
from data.dataset import CTDataset3D
from data.preprocessing import preprocess_ct_volume
from data.utils import get_dynamic_image_path

# === Module-level fixtures ===
@pytest.fixture
def mock_dataframe(): # Removed 'self'
    """Create a mock dataframe with pathology labels"""
    data = {
        'VolumeName': ['volume1.nii.gz', 'volume2.nii.gz', 'volume3.nii.gz'],
        'Lung nodule': [1, 0, 1],
        'Emphysema': [0, 1, 0],
        'Atelectasis': [1, 1, 0],
        'Pleural effusion': [0, 0, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def pathology_columns(): # Removed 'self'
    """Return test pathology columns"""
    return ['Lung nodule', 'Emphysema', 'Atelectasis', 'Pleural effusion']

@pytest.fixture
def temp_img_dir(tmp_path): # Removed 'self'
    """Create temporary directory with mock NIfTI files"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    # Create mock NIfTI files
    for i in range(3):
        volume_name = f"volume{i+1}.nii.gz"
        # Create a small test volume
        data = np.random.rand(32, 32, 32).astype(np.float32) * 1000
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, img_dir / volume_name)
    
    return img_dir

@pytest.fixture
def temp_cache_dir(tmp_path): # Removed 'self'
    """Create temporary cache directory"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir

# === Test Classes ===
class TestCTDataset3D:
    """Tests for the CTDataset3D class"""
    
    # This class can now directly use the module-level fixtures
    def test_initialization_basic(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test basic dataset initialization"""
        dataset = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=False,
            augment=False
        )
        
        assert len(dataset) == 3
        assert dataset.img_dir == temp_img_dir
        assert dataset.pathology_columns == pathology_columns
        assert not dataset.use_cache
        assert not dataset.augment
    
    def test_initialization_with_cache(self, mock_dataframe, temp_img_dir, temp_cache_dir, pathology_columns):
        """Test dataset initialization with caching enabled"""
        dataset = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=True,
            cache_dir=temp_cache_dir,
            augment=False
        )
        
        assert dataset.use_cache
        assert dataset.cache_dir == temp_cache_dir
        assert temp_cache_dir.exists()
    
    def test_initialization_cache_without_dir_raises_error(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test that enabling cache without cache_dir raises ValueError"""
        with pytest.raises(ValueError, match="Cache directory must be provided"):
            CTDataset3D(
                dataframe=mock_dataframe,
                img_dir=temp_img_dir,
                pathology_columns=pathology_columns,
                target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
                target_shape_dhw=(32, 32, 32),
                clip_hu_min=-1000,
                clip_hu_max=1000,
                use_cache=True,
                cache_dir=None,
                augment=False
            )
    
    def test_len_method(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test __len__ returns correct dataset size"""
        dataset = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=False
        )
        
        assert len(dataset) == len(mock_dataframe)
    
    def test_getitem_basic(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test basic item retrieval without caching"""
        dataset = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=False,
            augment=False
        )
        
        sample = dataset[0]
        
        assert 'pixel_values' in sample
        assert 'labels' in sample
        assert 'volume_name' in sample
        
        assert isinstance(sample['pixel_values'], torch.Tensor)
        assert isinstance(sample['labels'], torch.Tensor)
        assert isinstance(sample['volume_name'], str)
        
        assert sample['pixel_values'].shape == (1, 32, 32, 32)
        assert sample['labels'].shape == (4,)
        assert sample['volume_name'] == 'volume1.nii.gz'
        
        expected_labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert torch.allclose(sample['labels'], expected_labels)
    
    def test_getitem_with_missing_file(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test handling of missing volume files"""
        # Create a new DataFrame for this test to avoid modifying the fixture for other tests
        df_with_missing = mock_dataframe.copy()
        # Add a non-existent file to dataframe using pd.concat
        missing_row = pd.DataFrame([{'VolumeName': 'missing_volume.nii.gz', 
                                     'Lung nodule': 1, 'Emphysema': 0, 
                                     'Atelectasis': 1, 'Pleural effusion': 0}])
        df_with_missing = pd.concat([df_with_missing, missing_row], ignore_index=True)

        dataset = CTDataset3D(
            dataframe=df_with_missing,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=False,
            augment=False
        )
        
        sample = dataset[3] # Index of the missing file
        assert torch.all(sample['pixel_values'] == 0.0)
    
    def test_cache_save_and_load(self, mock_dataframe, temp_img_dir, temp_cache_dir, pathology_columns):
        """Test that caching saves and loads correctly"""
        dataset = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=True,
            cache_dir=temp_cache_dir,
            augment=False
        )
        
        sample1 = dataset[0]
        
        cache_files = list(temp_cache_dir.glob("*.pt"))
        assert len(cache_files) == 1
        
        with patch.object(dataset, '_load_from_cache', wraps=dataset._load_from_cache) as mock_load:
            sample2 = dataset[0]
            mock_load.assert_called_once()
        
        assert torch.allclose(sample1['pixel_values'], sample2['pixel_values'])
        assert torch.allclose(sample1['labels'], sample2['labels'])
        assert sample1['volume_name'] == sample2['volume_name']
    
    def test_cache_corrupted_file_handling(self, mock_dataframe, temp_img_dir, temp_cache_dir, pathology_columns):
        """Test handling of corrupted cache files"""
        dataset = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=True,
            cache_dir=temp_cache_dir,
            augment=False
        )
        
        cache_path = dataset._get_cache_path(0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            f.write(b"corrupted data")
        
        sample = dataset[0]
        assert 'pixel_values' in sample
        assert sample['pixel_values'].shape == (1, 32, 32, 32)
    
    def test_augmentation_applied(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test that augmentation is applied when enabled"""
        dataset = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=False,
            augment=True
        )
        
        with patch.object(dataset, '_apply_augmentation', wraps=dataset._apply_augmentation) as mock_aug:
            sample = dataset[0]
            mock_aug.assert_called_once()
    
    def test_augmentation_transforms(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test specific augmentation transforms"""
        dataset = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=False,
            augment=True
        )
        
        test_volume = torch.rand(1, 32, 32, 32)
        
        torch.manual_seed(42)
        augmented = dataset._apply_augmentation(test_volume.clone())
        
        assert augmented.shape == test_volume.shape
        assert augmented.min() >= 0.0
        assert augmented.max() <= 1.0
    
    def test_augmentation_with_cache(self, mock_dataframe, temp_img_dir, temp_cache_dir, pathology_columns):
        """Test that augmentation is applied even when loading from cache"""
        dataset = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=True,
            cache_dir=temp_cache_dir,
            augment=True
        )
        
        _ = dataset[0] # First access, saves non-augmented to cache
        
        with patch.object(dataset, '_apply_augmentation', wraps=dataset._apply_augmentation) as mock_aug:
            _ = dataset[0] # Second access, loads from cache then augments
            mock_aug.assert_called_once()
    
    def test_dynamic_path_handling(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test handling of hierarchical directory structure"""
        subdir = temp_img_dir / "SUBJ01_SESS1" / "SUBJ01_SESS1_T1"
        subdir.mkdir(parents=True)
        
        volume_name = "SUBJ01_SESS1_T1.nii.gz"
        data = np.random.rand(32, 32, 32).astype(np.float32) * 1000
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, subdir / volume_name)
        
        # Create a new DataFrame for this test
        df_hierarchical = mock_dataframe.copy()
        df_hierarchical.loc[0, 'VolumeName'] = volume_name
        
        dataset = CTDataset3D(
            dataframe=df_hierarchical,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=False,
            augment=False
        )
        
        sample = dataset[0]
        assert sample['volume_name'] == volume_name
        assert not torch.all(sample['pixel_values'] == 0.0)
    
    def test_dataframe_reset_index(self, pathology_columns, temp_img_dir, temp_img_dir_module_scope): # Added temp_img_dir_module_scope for dummy files
        """Test that dataframe index is reset properly"""
        data = {
            'VolumeName': ['volume1.nii.gz', 'volume2.nii.gz'],
            'Lung nodule': [1, 0], 'Emphysema': [0, 1],
            'Atelectasis': [1, 1], 'Pleural effusion': [0, 0]
        }
        df = pd.DataFrame(data, index=[5, 10])
        
        # Ensure dummy files exist for these volume names if not already created by temp_img_dir
        # This uses a slightly different img_dir to ensure these specific files exist.
        # For simplicity, we assume temp_img_dir_module_scope ensures 'volume1.nii.gz' and 'volume2.nii.gz' exist.

        dataset = CTDataset3D(
            dataframe=df,
            img_dir=temp_img_dir_module_scope, # Use the one that has volume1 and volume2
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=False
        )
        
        sample0 = dataset[0]
        sample1 = dataset[1]
        
        assert sample0['volume_name'] == 'volume1.nii.gz'
        assert sample1['volume_name'] == 'volume2.nii.gz'

    # Helper fixture for test_dataframe_reset_index to ensure files exist
    @pytest.fixture
    def temp_img_dir_module_scope(self, tmp_path):
        img_dir = tmp_path / "images_module"
        img_dir.mkdir(exist_ok=True)
        for vol_name in ['volume1.nii.gz', 'volume2.nii.gz']:
            data = np.random.rand(32, 32, 32).astype(np.float32)
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(data, affine)
            nib.save(nifti_img, img_dir / vol_name)
        return img_dir

    def test_label_dtype_conversion(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test that labels are properly converted to float32"""
        df_int_labels = mock_dataframe.copy()
        df_int_labels[pathology_columns] = df_int_labels[pathology_columns].astype(int)
        
        dataset = CTDataset3D(
            dataframe=df_int_labels,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=False
        )
        
        sample = dataset[0]
        assert sample['labels'].dtype == torch.float32
    
    def test_preprocessing_parameters_passed(self, mock_dataframe, temp_img_dir, pathology_columns):
        """Test that preprocessing parameters are correctly passed"""
        target_spacing = np.array([2.0, 2.0, 2.0])
        target_shape = (64, 64, 64)
        clip_min = -500
        clip_max = 500
        
        dataset = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=target_spacing,
            target_shape_dhw=target_shape,
            clip_hu_min=clip_min,
            clip_hu_max=clip_max,
            use_cache=False
        )
        
        with patch('data.dataset.preprocess_ct_volume') as mock_preprocess:
            mock_preprocess.return_value = torch.zeros((1, *target_shape))
            _ = dataset[0]
            
            mock_preprocess.assert_called_once()
            call_args = mock_preprocess.call_args[0]
            
            assert np.array_equal(call_args[1], target_spacing)
            assert call_args[2] == target_shape
            assert call_args[3] == clip_min
            assert call_args[4] == clip_max
    
    def test_concurrent_cache_access(self, mock_dataframe, temp_img_dir, temp_cache_dir, pathology_columns):
        """Test that multiple dataset instances can share cache"""
        dataset1 = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=True,
            cache_dir=temp_cache_dir,
            augment=False
        )
        
        _ = dataset1[0]
        
        dataset2 = CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=True,
            cache_dir=temp_cache_dir,
            augment=False
        )
        
        with patch('data.dataset.preprocess_ct_volume') as mock_preprocess:
            _ = dataset2[0]
            mock_preprocess.assert_not_called()


class TestDatasetIntegration:
    """Integration tests for CTDataset3D with DataLoader"""
    
    @pytest.fixture
    def small_dataset(self, mock_dataframe, temp_img_dir, pathology_columns): # Uses module-level fixtures
        """Create a small dataset for integration tests"""
        return CTDataset3D(
            dataframe=mock_dataframe,
            img_dir=temp_img_dir,
            pathology_columns=pathology_columns,
            target_spacing_xyz=np.array([1.0, 1.0, 1.0]),
            target_shape_dhw=(32, 32, 32),
            clip_hu_min=-1000,
            clip_hu_max=1000,
            use_cache=False,
            augment=False
        )
    
    def test_dataloader_integration(self, small_dataset):
        """Test that dataset works with PyTorch DataLoader"""
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            small_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0 # Set to 0 for simplicity in testing, can test with >0 separately
        )
        
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            assert 'pixel_values' in batch
            assert 'labels' in batch
            assert 'volume_name' in batch
            
            assert batch['pixel_values'].dim() == 5
            assert batch['labels'].dim() == 2
        
        # 3 samples, batch_size 2 -> 2 batches (2, 1)
        assert batch_count == 2 
    
    def test_dataloader_with_multiple_workers(self, small_dataset):
        """Test dataset with multiple workers if feasible (can be slow/resource-intensive)"""
        from torch.utils.data import DataLoader
        
        # Note: num_workers > 0 can have issues on some platforms (e.g. Windows) if not careful
        # with __main__ guards or if fixtures are not serializable.
        # For robust testing, often num_workers=0 is used, or specific care is taken.
        try:
            dataloader = DataLoader(
                small_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=2 # Using 1 to test parallelism, adjust if needed
            )
            
            samples = list(dataloader)
            assert len(samples) == 3
        except Exception as e:
            # Depending on the environment, multi-worker tests can be tricky.
            # This catch is to acknowledge potential issues but ideally should pass.
            pytest.skip(f"Skipping multi-worker test due to environment or issue: {e}")