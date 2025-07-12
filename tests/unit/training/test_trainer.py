# tests/unit/training/test_trainer.py
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
    train_model
)
from src.models.resnet3d import resnet18_3d
from src.models.densenet3d import densenet121_3d
from src.models.vit3d import vit_small_3d


# --- Fixtures ---

@pytest.fixture
def mock_config(tmp_path: Path) -> SimpleNamespace:
    """Creates a mock SimpleNamespace config object for testing."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    
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
            data_dir=tmp_path / "data",
            train_img_dir=tmp_path / "data" / "train",
            valid_img_dir=tmp_path / "data" / "valid",
            dir_structure="flat",
            data_subsets=SimpleNamespace(
                train=tmp_path / "train_vols.csv",
                valid=tmp_path / "valid_vols.csv"
            ),
            labels=SimpleNamespace(
                train=tmp_path / "train_labels.csv",
                valid=tmp_path / "valid_labels.csv"
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
            num_epochs=3,
            batch_size=2,
            learning_rate=1e-4,
            weight_decay=1e-5,
            num_workers=0,
            pin_memory=False,
            early_stopping_patience=2,
            early_stopping_metric='roc_auc_macro',
            resume_from_checkpoint=None,
            gradient_accumulation_steps=1,
            augment=True  # Ensure augmentation path is tested
        ),
        optimization=SimpleNamespace(
            gradient_checkpointing=False,
            mixed_precision=False,
            use_bf16=False
        ),
        cache=SimpleNamespace(use_cache=False),
        # Added: wandb config to prevent attribute error during tests
        wandb=SimpleNamespace(enabled=False, project=None, run_name=None, resume=None, group=None)
    )
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
        # Create dummy CSV files
        pd.DataFrame({'VolumeName': [f'vol_{i}' for i in range(5)]}).to_csv(
            mock_config.paths.data_subsets.train, index=False
        )
        pd.DataFrame({'VolumeName': [f'vol_{i}' for i in range(5, 8)]}).to_csv(
            mock_config.paths.data_subsets.valid, index=False
        )
        train_labels = pd.DataFrame({
            'VolumeName': [f'vol_{i}' for i in range(5)],
            'Cardiomegaly': [1, 0, 1, 0, 1],
            'Atelectasis': [0, 1, 1, 0, 0]
        })
        train_labels.to_csv(mock_config.paths.labels.train, index=False)
        
        valid_labels = pd.DataFrame({
            'VolumeName': [f'vol_{i}' for i in range(5, 8)],
            'Cardiomegaly': [1, 0, 0],
            'Atelectasis': [0, 1, 1]
        })
        valid_labels.to_csv(mock_config.paths.labels.valid, index=False)


    def test_successful_loading(self, mock_config, setup_mock_csv_files):
        train_df, valid_df = load_and_prepare_data(mock_config)
        assert not train_df.empty
        assert not valid_df.empty
        assert len(train_df) == 5
        assert len(valid_df) == 3
        assert 'Cardiomegaly' in train_df.columns
        assert train_df['Cardiomegaly'].dtype == int

    def test_missing_csv_raises_error(self, mock_config):
        with pytest.raises(FileNotFoundError):
            load_and_prepare_data(mock_config)

    def test_empty_dataframe_raises_error(self, mock_config, setup_mock_csv_files):
        # Create an empty volume list to cause an empty merged dataframe
        pd.DataFrame({'VolumeName': []}).to_csv(
            mock_config.paths.data_subsets.train, index=False
        )
        with pytest.raises(ValueError, match="Training or validation dataframe is empty"):
            load_and_prepare_data(mock_config)


class TestTrainModel:
    """Tests the main `train_model` orchestration function."""

    @pytest.mark.parametrize("use_cache", [True, False])
    @patch('src.training.trainer.generate_final_report')
    @patch('src.training.trainer.save_checkpoint')
    @patch('src.training.trainer.compute_metrics')
    @patch('src.training.trainer.validate_epoch')
    @patch('src.training.trainer.train_epoch')
    @patch('src.training.trainer.create_model')
    @patch('src.training.trainer.load_and_prepare_data')
    @patch('src.training.trainer.wandb.init')
    # Patches for the new MONAI data pipeline
    @patch('src.training.trainer.DataLoader')
    @patch('src.training.trainer.PersistentDataset')
    @patch('src.training.trainer.Dataset')
    @patch('src.training.trainer.CTMetadataDataset')
    @patch('src.training.trainer.ApplyTransforms')
    def test_train_model_happy_path(
        self, mock_apply_transforms, mock_ctmetadata_dataset, mock_monai_dataset, 
        mock_persistent_dataset, mock_dataloader, mock_wandb_init, 
        mock_load_data, mock_create_model, mock_train_epoch, 
        mock_validate_epoch, mock_compute_metrics, mock_save_checkpoint, 
        mock_generate_report, mock_config, use_cache
    ):
        """
        Tests the 'happy path' of train_model, ensuring all components are called
        as expected, for both cached and non-cached scenarios.
        """
        # --- Config Setup ---
        mock_config.cache.use_cache = use_cache
        mock_config.training.augment = True  # Ensure augmentation path is covered

        # --- Mock Setup ---
        mock_load_data.return_value = (
            pd.DataFrame({'VolumeName': ['train_vol_1'], 'Cardiomegaly': [1], 'Atelectasis': [0]}),
            pd.DataFrame({'VolumeName': ['valid_vol_1'], 'Cardiomegaly': [0], 'Atelectasis': [1]})
        )
        mock_model = MagicMock(spec=nn.Module)
        mock_model.parameters.return_value = nn.Linear(1, 1).parameters()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model
        
        mock_train_epoch.return_value = 0.5
        mock_validate_epoch.return_value = (0.4, np.array([[0.1]]), np.array([[0]]))
        
        metrics_sequence = [
            {'roc_auc_macro': 0.80, 'f1_macro': 0.70, 'loss': 0.40},
            {'roc_auc_macro': 0.85, 'f1_macro': 0.72, 'loss': 0.35},
            {'roc_auc_macro': 0.82, 'f1_macro': 0.71, 'loss': 0.38},
        ]
        mock_compute_metrics.side_effect = metrics_sequence
        
        # --- Mock Data Pipeline Setup ---
        mock_base_train_ds = MagicMock()
        mock_base_valid_ds = MagicMock()
        mock_ctmetadata_dataset.side_effect = [mock_base_train_ds, mock_base_valid_ds]

        mock_final_train_ds = MagicMock()
        mock_final_valid_ds = MagicMock()

        if use_cache:
            mock_persistent_dataset.side_effect = [mock_final_train_ds, mock_final_valid_ds]
            mock_augmented_ds = MagicMock()
            mock_apply_transforms.return_value = mock_augmented_ds
        else:
            mock_monai_dataset.side_effect = [mock_final_train_ds, mock_final_valid_ds]
        
        # --- Execution ---
        trained_model, history = train_model(mock_config)

        # --- Assertions ---
        mock_load_data.assert_called_once_with(mock_config)
        mock_create_model.assert_called_once_with(mock_config)
        
        mock_ctmetadata_dataset.assert_has_calls([
            call(dataframe=ANY, img_dir=mock_config.paths.train_img_dir, pathology_columns=ANY, path_mode=ANY),
            call(dataframe=ANY, img_dir=mock_config.paths.valid_img_dir, pathology_columns=ANY, path_mode=ANY)
        ])

        if use_cache:
            mock_persistent_dataset.assert_has_calls([
                call(data=mock_base_train_ds, transform=ANY, cache_dir=mock_config.paths.cache_dir / "train"),
                call(data=mock_base_valid_ds, transform=ANY, cache_dir=mock_config.paths.cache_dir / "valid")
            ])
            mock_monai_dataset.assert_not_called()
            mock_apply_transforms.assert_called_once_with(mock_final_train_ds, ANY)
            mock_dataloader.assert_has_calls([
                call(mock_augmented_ds, batch_size=ANY, shuffle=True, num_workers=ANY, pin_memory=ANY, persistent_workers=ANY),
                call(mock_final_valid_ds, batch_size=ANY, shuffle=False, num_workers=ANY, pin_memory=ANY, persistent_workers=ANY)
            ])
        else:
            mock_persistent_dataset.assert_not_called()
            mock_monai_dataset.assert_has_calls([
                call(data=mock_base_train_ds, transform=ANY),
                call(data=mock_base_valid_ds, transform=ANY)
            ])
            mock_apply_transforms.assert_not_called()
            mock_dataloader.assert_has_calls([
                call(mock_final_train_ds, batch_size=ANY, shuffle=True, num_workers=ANY, pin_memory=ANY, persistent_workers=ANY),
                call(mock_final_valid_ds, batch_size=ANY, shuffle=False, num_workers=ANY, pin_memory=ANY, persistent_workers=ANY)
            ])

        assert mock_train_epoch.call_count == mock_config.training.num_epochs
        assert mock_validate_epoch.call_count == mock_config.training.num_epochs
        
        assert len(history['metrics']) == mock_config.training.num_epochs
        assert history['metrics'][-1]['roc_auc_macro'] == 0.82
        
        # Checkpoint saves: 3 for 'last_model' + 2 for 'best_model' + 1 for 'final_model'
        assert mock_save_checkpoint.call_count == 6
        best_call = call(
            mock_model, ANY, ANY, 1,
            metrics_sequence[1],
            mock_config.paths.output_dir / 'best_model.pth'
        )
        mock_save_checkpoint.assert_has_calls([best_call], any_order=True)

        mock_generate_report.assert_called_once()