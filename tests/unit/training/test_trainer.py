"""
Unit tests for src/training/trainer.py

This module contains tests for the trainer functions, including model creation,
data preparation, and the main training orchestration logic.
"""

import sys
import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, call, ANY

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.training.trainer import (
    create_model,
    load_and_prepare_data,
    train_model,
    worker_init_fn 
)
from src.data.cache_utils import deterministic_hash
from src.models.resnet3d import resnet18_3d
from src.models.densenet3d import densenet121_3d
from src.models.vit3d import vit_small_3d


# --- Fixtures ---

@pytest.fixture
def mock_config(tmp_path: Path) -> SimpleNamespace:
    """Creates a mock SimpleNamespace config object for testing."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create nested namespaces to mimic the real config structure
    config = SimpleNamespace(
        model=SimpleNamespace(
            type="resnet3d",
            variant="18",
            vit_specific=SimpleNamespace(patch_size=(16, 16, 16))
        ),
        paths=SimpleNamespace(
            output_dir=output_dir,
            cache_dir=output_dir / "cache",
            data_dir=data_dir,
            # This is the key change: a single, unified image directory
            img_dir=data_dir / "images",
            dir_structure="flat",
            data_subsets=SimpleNamespace(
                train=data_dir / "train_vols.csv",
                valid=data_dir / "valid_vols.csv"
            ),
            # This is the key change: added the 'all' attribute
            labels=SimpleNamespace(
                all=data_dir / "all_labels.csv",
                # The 'train' and 'valid' labels are now legacy but kept for non-breaking tests
                train=data_dir / "train_labels.csv",
                valid=data_dir / "valid_labels.csv"
            )
        ),
        pathologies=SimpleNamespace(columns=["Cardiomegaly", "Atelectasis"]),
        image_processing=SimpleNamespace(
            target_shape_dhw=(64, 64, 64),
            target_spacing=(1.5, 1.5, 1.5),
            clip_hu_min=-1000,
            clip_hu_max=400,
            orientation_axcodes="RAS"
        ),
        loss_function=SimpleNamespace(
            type="BCEWithLogitsLoss",
            focal_loss=SimpleNamespace(alpha=0.25, gamma=2.0)
        ),
        training=SimpleNamespace(
            num_epochs=2, # Reduced for faster tests
            batch_size=2,
            learning_rate=1e-4,
            weight_decay=1e-5,
            num_workers=0,
            pin_memory=False,
            early_stopping_patience=3, # Prevent early stop
            resume_from_checkpoint=None,
            gradient_accumulation_steps=1,
            augment=True,
            prefetch_factor=2
        ),
        optimization=SimpleNamespace(
            gradient_checkpointing=False,
            mixed_precision=False,
            use_bf16=False
        ),
        cache=SimpleNamespace(
            use_cache=False,
            memory_rate=1.0,
        ),
        torch_dtype=torch.float32,
        wandb=SimpleNamespace(enabled=False, project=None, run_name=None, resume=None, group=None)
    )
    # Create the unified image directory
    config.paths.img_dir.mkdir(exist_ok=True)
    return config


# --- Test Classes ---

class TestCreateModel:
    """Tests the `create_model` factory function."""

    def test_create_resnet(self, mock_config):
        mock_config.model.type = "resnet3d"
        mock_config.model.variant = "18"
        model = create_model(mock_config)
        assert isinstance(model, nn.Module)

    def test_create_densenet(self, mock_config):
        mock_config.model.type = "densenet3d"
        mock_config.model.variant = "121"
        model = create_model(mock_config)
        assert isinstance(model, nn.Module)

    def test_create_vit(self, mock_config):
        mock_config.model.type = "vit3d"
        mock_config.model.variant = "small"
        model = create_model(mock_config)
        assert isinstance(model, nn.Module)

    def test_create_unknown_model_raises_error(self, mock_config):
        mock_config.model.type = "unknown_model"
        with pytest.raises(ValueError, match="Unknown model type: unknown_model"):
            create_model(mock_config)


class TestLoadAndPrepareData:
    """Tests the `load_and_prepare_data` function."""

    @pytest.fixture
    def setup_mock_csv_files(self, mock_config):
        """Helper to create mock CSV files for testing."""
        # Create dummy volume lists for the split
        pd.DataFrame({'VolumeName': [f'vol_{i}' for i in range(5)]}).to_csv(
            mock_config.paths.data_subsets.train, index=False
        )
        pd.DataFrame({'VolumeName': [f'vol_{i}' for i in range(5, 8)]}).to_csv(
            mock_config.paths.data_subsets.valid, index=False
        )

        # Create the unified 'all_labels.csv' file
        all_labels = pd.DataFrame({
            'VolumeName': [f'vol_{i}' for i in range(8)],
            'Cardiomegaly': [1, 0, 1, 0, 1, 1, 0, 0],
            'Atelectasis': [0, 1, 1, 0, 0, 0, 1, 1]
        })
        all_labels.to_csv(mock_config.paths.labels.all, index=False)


    def test_successful_loading(self, mock_config, setup_mock_csv_files):
        """Tests that data is correctly split and merged from unified labels."""
        train_df, valid_df = load_and_prepare_data(mock_config)
        assert not train_df.empty
        assert not valid_df.empty
        assert len(train_df) == 5
        assert len(valid_df) == 3
        assert 'Cardiomegaly' in train_df.columns
        assert train_df['Cardiomegaly'].dtype == int
        # Check that the correct volumes are in each dataframe
        assert train_df['VolumeName'].tolist() == [f'vol_{i}' for i in range(5)]
        assert valid_df['VolumeName'].tolist() == [f'vol_{i}' for i in range(5, 8)]

    def test_missing_csv_raises_error(self, mock_config):
        """Tests that a FileNotFoundError is raised if any CSV is missing."""
        # Intentionally do not call setup_mock_csv_files
        with pytest.raises(FileNotFoundError):
            load_and_prepare_data(mock_config)

    def test_empty_dataframe_raises_error(self, mock_config, setup_mock_csv_files):
        """Tests that a ValueError is raised if a split results in an empty dataframe."""
        # Create an empty volume list to cause an empty merged dataframe
        pd.DataFrame({'VolumeName': []}).to_csv(
            mock_config.paths.data_subsets.train, index=False
        )
        with pytest.raises(ValueError, match="Training or validation dataframe is empty"):
            load_and_prepare_data(mock_config)


class TestTrainModel:
    """Tests the main `train_model` orchestration function."""

    @pytest.mark.parametrize("use_cache", [True, False])
    @patch('src.training.trainer.get_preprocessing_transforms')
    @patch('src.training.trainer.LabelAttacherDataset')
    @patch('src.training.trainer.generate_final_report')
    @patch('src.training.trainer.save_checkpoint')
    @patch('src.training.trainer.validate_epoch') # Keep this mock
    @patch('src.training.trainer.train_epoch')
    @patch('src.training.trainer.create_model')
    @patch('src.training.trainer.load_and_prepare_data')
    @patch('src.training.trainer.wandb.init')
    @patch('src.training.trainer.DataLoader')
    @patch('src.training.trainer.Dataset')
    @patch('src.training.trainer.CacheDataset')
    @patch('src.training.trainer.PersistentDataset')
    @patch('src.training.trainer.CTMetadataDataset')
    @patch('src.training.trainer.ApplyTransforms')
    def test_train_model_happy_path(
        self, mock_apply_transforms, mock_ctmetadata_dataset, mock_persistent_dataset,
        mock_cache_ds, mock_monai_dataset, mock_dataloader, mock_wandb_init,
        mock_load_data, mock_create_model, mock_train_epoch,
        mock_validate_epoch, mock_save_checkpoint,
        mock_generate_report, mock_label_attacher, mock_get_transforms,
        mock_config, use_cache
    ):
        """
        Tests the 'happy path' of train_model, ensuring the new decoupled data
        pipeline is constructed correctly for both cached and non-cached scenarios.
        """
        # --- Config Setup ---
        mock_config.cache.use_cache = use_cache
        mock_config.training.augment = True

        # --- Mock Return Values ---
        mock_load_data.return_value = (
            pd.DataFrame({'VolumeName': ['train_vol_1'], 'Cardiomegaly': [1], 'Atelectasis': [0]}),
            pd.DataFrame({'VolumeName': ['valid_vol_1'], 'Cardiomegaly': [0], 'Atelectasis': [1]})
        )
        mock_model = MagicMock(spec=nn.Module); mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model
        mock_model.parameters.return_value = nn.Linear(1, 1).parameters()
        mock_train_epoch.return_value = 0.5

        mock_validate_epoch.return_value = (0.4, {'roc_auc_macro': 0.85, 'f1_macro': 0.75})

        from monai.transforms import Compose
        mock_get_transforms.return_value = Compose([])

        # --- Mock Data Pipeline Setup ---
        mock_base_train_ds, mock_base_valid_ds = MagicMock(), MagicMock()
        mock_ctmetadata_dataset.side_effect = [mock_base_train_ds, mock_base_valid_ds]

        mock_labeled_train, mock_labeled_valid = MagicMock(), MagicMock()
        mock_label_attacher.side_effect = [mock_labeled_train, mock_labeled_valid]
        
        mock_augmented_train = MagicMock()
        mock_apply_transforms.return_value = mock_augmented_train

        if use_cache:
            mock_persistent_train, mock_persistent_valid = MagicMock(), MagicMock()
            mock_persistent_dataset.side_effect = [mock_persistent_train, mock_persistent_valid]
            
            mock_cached_train, mock_cached_valid = MagicMock(), MagicMock()
            mock_cache_ds.side_effect = [mock_cached_train, mock_cached_valid]
        else:
            mock_preprocessed_train, mock_preprocessed_valid = MagicMock(), MagicMock()
            mock_monai_dataset.side_effect = [mock_preprocessed_train, mock_preprocessed_valid]

        # --- EXECUTION ---
        train_model(mock_config)

        # --- ASSERTIONS ---
        mock_load_data.assert_called_once_with(mock_config)
        mock_get_transforms.assert_called_once_with(mock_config)

        mock_ctmetadata_dataset.assert_has_calls([
            call(dataframe=ANY, img_dir=mock_config.paths.img_dir, path_mode=ANY),
            call(dataframe=ANY, img_dir=mock_config.paths.img_dir, path_mode=ANY)
        ])
        
        if use_cache:
            mock_label_attacher.assert_has_calls([
                call(image_source=mock_cached_train, labels_df=ANY, pathology_columns=ANY),
                call(image_source=mock_cached_valid, labels_df=ANY, pathology_columns=ANY)
            ])
        else:
            mock_label_attacher.assert_has_calls([
                call(image_source=mock_preprocessed_train, labels_df=ANY, pathology_columns=ANY),
                call(image_source=mock_preprocessed_valid, labels_df=ANY, pathology_columns=ANY)
            ])

        # Assert that augmentations are applied only to the training dataset
        mock_apply_transforms.assert_called_once_with(
            data=mock_labeled_train, transform=ANY
        )

        # Assert DataLoaders get the correct final datasets
        mock_dataloader.assert_has_calls([
           call(
               mock_augmented_train, # Check that the augmented dataset is used
               batch_size=mock_config.training.batch_size,
               shuffle=True,
               num_workers=mock_config.training.num_workers,
               pin_memory=mock_config.training.pin_memory,
               persistent_workers=mock_config.training.num_workers > 0,
               prefetch_factor=mock_config.training.prefetch_factor if mock_config.training.num_workers > 0 else None,
               worker_init_fn=worker_init_fn
           ),
           call(
               mock_labeled_valid, # Validation set should not be augmented
               batch_size=mock_config.training.batch_size,
               shuffle=False,
               num_workers=mock_config.training.num_workers,
               pin_memory=mock_config.training.pin_memory,
               persistent_workers=mock_config.training.num_workers > 0,
               prefetch_factor=mock_config.training.prefetch_factor if mock_config.training.num_workers > 0 else None,
               worker_init_fn=worker_init_fn
           )
       ], any_order=False)