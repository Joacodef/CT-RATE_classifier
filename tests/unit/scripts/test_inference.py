import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from scripts.inference import CTInference


@pytest.fixture
def base_config():
    return SimpleNamespace(
        model=SimpleNamespace(type="resnet3d", variant="18"),
        optimization=SimpleNamespace(gradient_checkpointing=True),
        image_processing=SimpleNamespace(
            orientation_axcodes=("L", "P", "S"),
            target_spacing=(1.0, 1.0, 1.0),
            clip_hu_min=-1000.0,
            clip_hu_max=1000.0,
            target_shape_dhw=(16, 16, 16),
        ),
        pathologies=SimpleNamespace(columns=["p1", "p2"]),
    )


class DummyCompose:
    """Minimal stand-in for MONAI Compose used in CTInference."""

    def __init__(self, _transforms):
        self.transforms = _transforms

    def __call__(self, _data):
        return {"image": torch.zeros(1, 2, 2, 2)}


@pytest.fixture(autouse=True)
def patch_monai_components(monkeypatch):
    monkeypatch.setattr("scripts.inference.Compose", lambda transforms: DummyCompose(transforms))


@pytest.fixture
def config_with_two_pathologies(base_config):
    return base_config


@pytest.fixture
def dummy_model():
    model = torch.nn.Linear(4, 2)
    return model


@pytest.fixture
def inference_instance(base_config, monkeypatch, dummy_model):
    monkeypatch.setattr("scripts.inference.create_model", lambda cfg: dummy_model)
    instance = object.__new__(CTInference)
    instance.config = base_config
    instance.device = torch.device("cpu")
    instance.logger = MagicMock()
    instance.monai_pipeline = DummyCompose([])
    instance.model = dummy_model.eval()
    return instance


class TestLoadModel:
    def test_load_model_from_checkpoint_dict(self, base_config, tmp_path, monkeypatch, dummy_model):
        monkeypatch.setattr("scripts.inference.create_model", lambda cfg: dummy_model)
        checkpoint_path = tmp_path / "model_with_meta.pth"
        torch.save({"model_state_dict": dummy_model.state_dict(), "epoch": 9}, checkpoint_path)

        instance = object.__new__(CTInference)
        instance.config = base_config
        instance.device = torch.device("cpu")
        instance.logger = MagicMock()

        loaded = CTInference._load_model(instance, str(checkpoint_path))

        assert loaded is dummy_model
        assert not loaded.training
        assert instance.config.optimization.gradient_checkpointing is False
        assert all(torch.equal(a, b) for a, b in zip(dummy_model.state_dict().values(), loaded.state_dict().values()))

    def test_load_model_from_state_dict(self, base_config, tmp_path, monkeypatch, dummy_model):
        monkeypatch.setattr("scripts.inference.create_model", lambda cfg: dummy_model)
        checkpoint_path = tmp_path / "model_state_only.pth"
        torch.save(dummy_model.state_dict(), checkpoint_path)

        instance = object.__new__(CTInference)
        instance.config = base_config
        instance.device = torch.device("cpu")
        instance.logger = MagicMock()

        loaded = CTInference._load_model(instance, str(checkpoint_path))

        assert loaded is dummy_model
        assert not loaded.training
        assert instance.config.optimization.gradient_checkpointing is False
        assert all(torch.equal(a, b) for a, b in zip(dummy_model.state_dict().values(), loaded.state_dict().values()))


class TestPredictVolume:
    def test_predict_volume_missing_file_raises(self, inference_instance):
        nonexistent = "nonexistent_volume.nii.gz"

        with pytest.raises(FileNotFoundError):
            inference_instance.predict_volume(nonexistent)


class TestPredictBatch:
    def test_predict_batch_handles_failures_and_writes_csv(self, inference_instance, tmp_path):
        successful_result = {
            "volume_path": "vol_success.nii.gz",
            "predictions": {
                "p1": {"probability": 0.9, "prediction": 1},
                "p2": {"probability": 0.1, "prediction": 0},
            },
        }

        def fake_predict_volume(path):
            if "fail" in path:
                raise RuntimeError("failed")
            return successful_result

        inference_instance.predict_volume = MagicMock(side_effect=lambda path: fake_predict_volume(path))
        output_csv = tmp_path / "batch" / "results.csv"

        df = inference_instance.predict_batch(["vol_success.nii.gz", "vol_fail.nii.gz"], str(output_csv))

        assert len(df) == 2
        first_row = df.iloc[0]
        assert first_row["p1_probability"] == pytest.approx(0.9)
        assert first_row["p2_prediction"] == 0

        second_row = df.iloc[1]
        assert np.isnan(second_row["p1_probability"])
        assert np.isnan(second_row["p2_prediction"])

        assert output_csv.exists()
        saved = pd.read_csv(output_csv)
        pd.testing.assert_frame_equal(saved, df)