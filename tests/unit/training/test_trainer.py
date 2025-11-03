"""
Unit tests for src/training/trainer.py

This module contains tests for the trainer functions, including model creation,
data preparation, and the main training orchestration logic.
"""

import sys
import math
import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, call, ANY
from torch.utils.data import DataLoader, Dataset

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

import src.training.trainer as trainer_module

from src.training.trainer import (
    create_model,
    generate_wandb_run_name,
    load_and_prepare_data,
    train_model,
    train_epoch,
    validate_epoch,
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


class _ToyBatchDataset(Dataset):
    """Deterministic toy dataset yielding small regression batches."""

    def __init__(self, num_samples: int, feature_dim: int):
        self.num_samples = num_samples
        self.feature_dim = feature_dim

    def __len__(self) -> int:  # pragma: no cover - thin wrapper
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        image = torch.full((self.feature_dim,), float(idx + 1) / 100.0)
        label = torch.tensor([(idx % 2)], dtype=torch.float32)
        return {"image": image, "label": label}


class _CountingSGD(torch.optim.SGD):
    """SGD optimizer that records how many step calls occur."""

    def __init__(self, params, lr: float):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _DummyScaler:
    """Minimal GradScaler stand-in for exercising AMP branches."""

    def __init__(self):
        self.scale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, loss):
        self.scale_calls += 1

        class _ScaledLoss:
            def __init__(self_inner, wrapped_loss):
                self_inner.loss = wrapped_loss

            def backward(self_inner):
                self_inner.loss.backward()

        return _ScaledLoss(loss)

    def step(self, optimizer):
        self.step_calls += 1
        optimizer.step()

    def update(self):
        self.update_calls += 1


class _SimpleLoader:
    """Lightweight iterable mimicking a PyTorch DataLoader."""

    def __init__(self, batches: list[dict]):
        self._batches = batches

    def __iter__(self):  # pragma: no cover - simple generator
        for batch in self._batches:
            yield batch

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._batches)


class _SequentialLogitModel(nn.Module):
    """Model stub that yields a predefined sequence of logits per forward."""

    def __init__(self, logits_per_batch: list[torch.Tensor]):
        super().__init__()
        self.logits_per_batch = logits_per_batch
        self.call_index = 0

    def forward(self, _inputs):  # pragma: no cover - trivial mapping
        logits = self.logits_per_batch[self.call_index]
        self.call_index += 1
        return logits


class TestTrainEpoch:
    """Exercises gradient accumulation and AMP execution paths."""

    def _build_dataloader(self, num_samples: int = 6, batch_size: int = 2, feature_dim: int = 3):
        dataset = _ToyBatchDataset(num_samples=num_samples, feature_dim=feature_dim)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def test_gradient_accumulation_respects_step_schedule(self):
        dataloader = self._build_dataloader(num_samples=6, batch_size=2, feature_dim=3)
        device = torch.device('cpu')
        model = nn.Linear(3, 1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)

        grad_steps = 2

        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=None,
            device=device,
            epoch=0,
            total_epochs=1,
            gradient_accumulation_steps=grad_steps,
            use_amp=False,
            use_bf16=False,
        )

        num_batches = len(dataloader)
        expected_steps = math.ceil(num_batches / grad_steps)
        assert optimizer.step_calls == expected_steps
        assert isinstance(avg_loss, float)

    @pytest.mark.parametrize("use_bf16,expected_dtype", [
        (False, torch.float16),
        (True, torch.bfloat16),
    ])
    def test_autocast_uses_expected_dtype(self, use_bf16, expected_dtype):
        dataloader = self._build_dataloader(num_samples=4, batch_size=2, feature_dim=3)
        device = torch.device('cpu')
        model = nn.Linear(3, 1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = _CountingSGD(model.parameters(), lr=0.1)
        scaler = _DummyScaler()

        class _AutocastRecorder:
            calls = []

            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                _AutocastRecorder.calls.append(self.kwargs)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        _AutocastRecorder.calls.clear()

        with patch('src.training.trainer.torch.cuda.is_bf16_supported', return_value=True), \
             patch('src.training.trainer.torch.amp.autocast', new=_AutocastRecorder):
            train_epoch(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=0,
                total_epochs=1,
                gradient_accumulation_steps=1,
                use_amp=True,
                use_bf16=use_bf16,
            )

        assert optimizer.step_calls == len(dataloader)
        assert scaler.scale_calls == len(dataloader)
        assert scaler.step_calls == len(dataloader)
        assert scaler.update_calls == len(dataloader)
        assert len(_AutocastRecorder.calls) == len(dataloader)
        for call_kwargs in _AutocastRecorder.calls:
            assert call_kwargs['device_type'] == 'cuda'
            assert call_kwargs['dtype'] == expected_dtype


class TestValidateEpoch:
    """Validates metric computation and lifecycle management."""

    _PATHOLOGIES = ["Cardiomegaly", "Atelectasis"]

    def _build_batches_and_logits(self):
        labels_batch1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        labels_batch2 = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

        logits_batch1 = torch.tensor([[20.0, -20.0], [-20.0, 20.0]], dtype=torch.float32)
        logits_batch2 = torch.tensor([[20.0, 20.0], [-20.0, -20.0]], dtype=torch.float32)

        image_shape = (labels_batch1.size(0), 1, 2, 2, 2)
        batches = [
            {"image": torch.zeros(image_shape, dtype=torch.float32), "label": labels_batch1},
            {"image": torch.zeros(image_shape, dtype=torch.float32), "label": labels_batch2},
        ]
        logits = [logits_batch1, logits_batch2]
        return batches, logits

    def test_metrics_are_perfect_for_matching_logits(self):
        batches, logits = self._build_batches_and_logits()
        loader = _SimpleLoader(batches)
        model = _SequentialLogitModel([logit.clone() for logit in logits])
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device('cpu')

        avg_loss, metrics_dict = validate_epoch(model, loader, criterion, device, self._PATHOLOGIES)

        assert avg_loss == pytest.approx(0.0, abs=1e-5)

        expected_keys = {'roc_auc_macro', 'roc_auc_micro', 'f1_macro', 'f1_micro'}
        expected_keys.update(f"{name}_auc" for name in self._PATHOLOGIES)
        assert expected_keys.issubset(metrics_dict.keys())

        assert metrics_dict['roc_auc_macro'] == pytest.approx(1.0)
        assert metrics_dict['roc_auc_micro'] == pytest.approx(1.0)
        assert metrics_dict['f1_macro'] == pytest.approx(1.0)
        assert metrics_dict['f1_micro'] == pytest.approx(1.0)
        for pathology in self._PATHOLOGIES:
            assert metrics_dict[f"{pathology}_auc"] == pytest.approx(1.0)

    def test_metric_objects_are_reset(self):
        batches, logits = self._build_batches_and_logits()
        loader = _SimpleLoader(batches)
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device('cpu')

        with patch.object(trainer_module.ROCAUCMetric, "reset", autospec=True) as mock_auc_reset, \
             patch.object(trainer_module.FBetaScore, "reset", autospec=True) as mock_f1_reset:
            model = _SequentialLogitModel([logit.clone() for logit in logits])
            validate_epoch(model, loader, criterion, device, self._PATHOLOGIES)
            first_auc_resets = mock_auc_reset.call_count
            first_f1_resets = mock_f1_reset.call_count

            model_second = _SequentialLogitModel([logit.clone() for logit in logits])
            validate_epoch(model_second, loader, criterion, device, self._PATHOLOGIES)

        assert first_auc_resets >= 1
        assert mock_auc_reset.call_count - first_auc_resets >= 1
        assert first_f1_resets >= 1
        assert mock_f1_reset.call_count - first_f1_resets >= 1


class TestGenerateWandbRunName:
    """Validates W&B run name generation logic."""

    def _base_config(self):
        return SimpleNamespace(
            model=SimpleNamespace(type="resnet3d", variant="18"),
            workflow=SimpleNamespace(mode="end-to-end"),
            paths=SimpleNamespace(
                data_subsets=SimpleNamespace(train="/data/splits/train_fold_0.csv"),
            ),
            training=SimpleNamespace(learning_rate=1e-4, batch_size=2, augment=True),
            cache=SimpleNamespace(use_cache=True),
            optimization=SimpleNamespace(mixed_precision=False),
        )

    def test_sanitization_and_fallbacks(self):
        config = self._base_config()
        config.model.type = "ViT 3D"
        config.model.variant = "small@beta"
        config.workflow.mode = "Feat*Mode"
        config.paths.data_subsets.train = "/path/to/data/train_split (set).csv"

        name = generate_wandb_run_name(config)

        assert "VIT-3D" in name
        assert "SMALL-BETA" in name
        assert "feat-mode" in name
        assert "train_split--set" in name
        assert name.startswith("VIT-3D-SMALL-BETA_feat-mode_train_split--set_")
        signature_part = name.split("_")[-1]
        assert len(signature_part) == 4

    def test_signature_hash_changes_with_payload(self):
        config = self._base_config()
        name_a = generate_wandb_run_name(config)

        config.training.learning_rate = 5e-5
        name_b = generate_wandb_run_name(config)

        config.training.learning_rate = 1e-4
        config.training.batch_size = 4
        name_c = generate_wandb_run_name(config)

        hash_a = name_a.split("_")[-1]
        hash_b = name_b.split("_")[-1]
        hash_c = name_c.split("_")[-1]

        assert hash_a != hash_b
        assert hash_a != hash_c
        assert hash_b != hash_c