import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.training.utils import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)


class DummyScaler:
    """Minimal GradScaler replacement for checkpoint tests."""

    def __init__(self, state=None):
        self._state = state or {"scale": 1.0}

    def state_dict(self):
        return copy.deepcopy(self._state)

    def load_state_dict(self, state):
        self._state = copy.deepcopy(state)


class TestEarlyStopping:
    """Covers EarlyStopping modes and min_delta handling."""

    def test_min_mode_respects_min_delta_and_patience(self):
        stopper = EarlyStopping(patience=2, min_delta=0.1, mode="min")

        assert stopper(1.0) is False
        assert stopper(0.95) is False  # delta below threshold; counter = 1
        # Second insufficient improvement should trigger early stop
        assert stopper(0.96) is True
        assert stopper.early_stop is True
        assert stopper.best_value == 1.0

    def test_max_mode_improvement_resets_counter(self):
        stopper = EarlyStopping(patience=2, min_delta=0.05, mode="max")

        assert stopper(0.5) is False
        assert stopper(0.53) is False  # below min_delta improvement; counter = 1
        assert stopper(0.56) is False  # improvement resets counter
        assert stopper.counter == 0
        assert stopper.best_value == pytest.approx(0.56)
        # Consecutive non-improvements should exhaust patience
        assert stopper(0.56) is False  # identical value -> counter = 1
        assert stopper(0.56) is True   # patience exhausted


class TestCheckpointHelpers:
    """Tests for save/load checkpoint utilities."""

    def _setup_model_optimizer(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # Perform a dummy update to create non-default optimizer state
        dummy_input = torch.tensor([[1.0, -1.0]])
        target = torch.tensor([[0.5]])
        loss = nn.MSELoss()(model(dummy_input), target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return model, optimizer

    def test_checkpoint_round_trip_with_scaler(self, tmp_path: Path):
        model, optimizer = self._setup_model_optimizer()
        saved_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        saved_optim_state = copy.deepcopy(optimizer.state_dict())

        scaler = DummyScaler({"scale": 3.14})
        metrics = {"roc_auc_macro": 0.82}
        ckpt_path = tmp_path / "best_model.pth"

        save_checkpoint(model, optimizer, scaler, epoch=5, best_metrics=metrics, checkpoint_path=ckpt_path)

        # Mutate states to ensure load restores them
        for param in model.parameters():
            param.data.zero_()
        optimizer.state = {}
        scaler.load_state_dict({"scale": -1})

        loaded_epoch, loaded_metrics = load_checkpoint(ckpt_path, model, optimizer, scaler)

        assert loaded_epoch == 5
        assert loaded_metrics == metrics
        for key, tensor in saved_model_state.items():
            assert torch.allclose(model.state_dict()[key], tensor)
        assert optimizer.state_dict() == saved_optim_state
        assert scaler.state_dict() == {"scale": 3.14}

    def test_checkpoint_round_trip_without_scaler(self, tmp_path: Path):
        model, optimizer = self._setup_model_optimizer()
        metrics = {"loss": 0.4}
        ckpt_path = tmp_path / "final_model.pth"

        save_checkpoint(model, optimizer, scaler=None, epoch=3, best_metrics=metrics, checkpoint_path=ckpt_path)

        # Trash current state before loading
        for param in model.parameters():
            param.data.add_(10.0)

        loaded_epoch, loaded_metrics = load_checkpoint(ckpt_path, model, optimizer, scaler=None)
        assert loaded_epoch == 3
        assert loaded_metrics == metrics

    def test_find_latest_checkpoint_prefers_last_checkpoint(self, tmp_path: Path):
        last_ckpt = tmp_path / "last_checkpoint.pth"
        last_ckpt.write_text("last")
        (tmp_path / "checkpoint_epoch_4.pth").write_text("older")
        (tmp_path / "checkpoint_epoch_7.pth").write_text("newer")

        assert find_latest_checkpoint(tmp_path) == last_ckpt

    def test_find_latest_checkpoint_highest_epoch_when_no_last(self, tmp_path: Path):
        (tmp_path / "checkpoint_epoch_2.pth").write_text("two")
        (tmp_path / "checkpoint_epoch_10.pth").write_text("ten")
        (tmp_path / "checkpoint_epoch_5.pth").write_text("five")

        latest = find_latest_checkpoint(tmp_path)
        assert latest == tmp_path / "checkpoint_epoch_10.pth"

    def test_find_latest_checkpoint_returns_none_when_empty(self, tmp_path: Path):
        assert find_latest_checkpoint(tmp_path) is None
