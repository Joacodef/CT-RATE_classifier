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
    train_epoch,
    validate_epoch,
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
            train_img_dir=Path("path/to/train/img"),
            valid_img_dir=Path("path/to/valid/img"),
            data_subsets=SimpleNamespace(
                selected_train_volumes=tmp_path / "train_vols.csv",
                selected_valid_volumes=tmp_path / "valid_vols.csv"
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
            gradient_accumulation_steps=1,
            resume_from_checkpoint=None
        ),
        optimization=SimpleNamespace(
            gradient_checkpointing=False,
            mixed_precision=False,
            use_bf16=False
        ),
        cache=SimpleNamespace(use_cache=False)
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
            mock_config.paths.data_subsets.selected_train_volumes, index=False
        )
        pd.DataFrame({'VolumeName': [f'vol_{i}' for i in range(5, 8)]}).to_csv(
            mock_config.paths.data_subsets.selected_valid_volumes, index=False
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
            mock_config.paths.data_subsets.selected_train_volumes, index=False
        )
        with pytest.raises(ValueError, match="Training or validation dataframe is empty"):
            load_and_prepare_data(mock_config)


class TestTrainModel:
    """Tests the main `train_model` orchestration function."""

    @patch('src.training.trainer.generate_final_report')
    @patch('src.training.trainer.save_checkpoint')
    @patch('src.training.trainer.compute_metrics')
    @patch('src.training.trainer.validate_epoch')
    @patch('src.training.trainer.train_epoch')
    @patch('src.training.trainer.create_model')
    @patch('src.training.trainer.load_and_prepare_data')
    @patch('src.training.trainer.wandb.init')
    def test_train_model_happy_path(
        self, mock_wandb_init, mock_load_data, mock_create_model,
        mock_train_epoch, mock_validate_epoch, mock_compute_metrics,
        mock_save_checkpoint, mock_generate_report, mock_config
    ):
        """
        Tests the 'happy path' of train_model, ensuring all components are called
        as expected over a few epochs.
        """
        # --- Mock Setup ---
        # Mock data loading to return dummy dataframes
        mock_load_data.return_value = (
            pd.DataFrame({'VolumeName': ['train_vol_1'], 'Cardiomegaly': [1], 'Atelectasis': [0]}),
            pd.DataFrame({'VolumeName': ['valid_vol_1'], 'Cardiomegaly': [0], 'Atelectasis': [1]})
        )
        # Mock model creation to provide valid parameters to the optimizer
        mock_model = MagicMock(spec=nn.Module)
        # Create a real mini-model to source parameters from
        dummy_param_model = nn.Linear(1, 1) 
        mock_model.parameters.return_value = dummy_param_model.parameters()
        mock_model.to.return_value = mock_model # Ensure that model.to(device) returns the mock itself
        mock_create_model.return_value = mock_model
        
        # Mock epoch loops to return dummy values
        mock_train_epoch.return_value = 0.5  # Mock train loss
        mock_validate_epoch.return_value = (0.4, np.array([[0.1]]), np.array([[0]])) # loss, preds, labels
        
        # Mock metrics calculation to return increasing AUC
        mock_compute_metrics.side_effect = [
            {'roc_auc_macro': 0.80, 'f1_macro': 0.7},
            {'roc_auc_macro': 0.85, 'f1_macro': 0.72}, # Improvement
            {'roc_auc_macro': 0.82, 'f1_macro': 0.71},
        ]
        
        # --- Execution ---
        trained_model, history = train_model(mock_config)

        # --- Assertions ---
        # Check that setup functions were called
        mock_load_data.assert_called_once_with(mock_config)
        mock_create_model.assert_called_once_with(mock_config)
        
        # Check that training ran for the configured number of epochs
        assert mock_train_epoch.call_count == mock_config.training.num_epochs
        assert mock_validate_epoch.call_count == mock_config.training.num_epochs
        assert mock_compute_metrics.call_count == mock_config.training.num_epochs
        
        # Check that history was recorded correctly
        assert len(history['train_loss']) == mock_config.training.num_epochs
        assert history['metrics'][-1]['roc_auc_macro'] == 0.82
        
        # Check that checkpoints were saved
        # best_model.pth (epoch 2), checkpoint_epoch_5, last_checkpoint (all 3 epochs)
        # In our 3-epoch run: best (epoch 2), last (epoch 1, 2, 3)
        assert mock_save_checkpoint.call_count >= 1 + mock_config.training.num_epochs
        
        # Check that the best model was saved on the correct epoch (epoch 2, index 1)
        best_call = call(
            mock_model, ANY, ANY, 1, 
            {'roc_auc_macro': 0.85, 'f1_macro': 0.72}, 
            mock_config.paths.output_dir / 'best_model.pth'
        )
        mock_save_checkpoint.assert_has_calls([best_call], any_order=True)

        # Check that final report was generated
        mock_generate_report.assert_called_once()