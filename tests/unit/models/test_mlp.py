import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.mlp import create_mlp_classifier, _discover_feature_file


class TestDiscoverFeatureFile:
    def test_prefers_train_subdirectory(self, tmp_path: Path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        train_dir = feature_dir / "train"
        train_dir.mkdir()

        root_feature = feature_dir / "root_feat.pt"
        torch.save(torch.randn(8), root_feature)
        train_feature = train_dir / "train_feat.pt"
        torch.save(torch.randn(8), train_feature)

        discovered = _discover_feature_file(feature_dir)

        assert discovered == train_feature


class TestCreateMlpClassifier:
    def _base_config(self, feature_dir: Path, hidden_dims=None, dropout_probs=None, num_classes: int = 3):
        model_cfg = SimpleNamespace()
        if hidden_dims is not None:
            model_cfg.hidden_dims = hidden_dims
        if dropout_probs is not None:
            model_cfg.dropout_probs = dropout_probs

        config = SimpleNamespace(
            workflow=SimpleNamespace(feature_config=SimpleNamespace(feature_dir=str(feature_dir))),
            pathologies=SimpleNamespace(columns=[f"c{i}" for i in range(num_classes)]),
            model=model_cfg,
        )
        return config

    def _write_feature(self, directory: Path, filename: str, tensor: torch.Tensor) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        feature_path = directory / filename
        torch.save(tensor, feature_path)
        return feature_path

    def test_scalar_dropout_replicates_per_hidden_layer(self, tmp_path: Path):
        feature_dir = tmp_path / "features"
        self._write_feature(feature_dir / "train", "sample.pt", torch.randn(16))

        config = self._base_config(feature_dir, hidden_dims=[128, 64, 32], dropout_probs=0.2, num_classes=4)
        model = create_mlp_classifier(config)

        layers = list(model.children())
        linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
        dropout_layers = [layer for layer in layers if isinstance(layer, nn.Dropout)]

        assert [layer.out_features for layer in linear_layers[:-1]] == [128, 64, 32]
        assert linear_layers[-1].out_features == 4
        assert len(dropout_layers) == 3
        assert all(dropout.p == pytest.approx(0.2) for dropout in dropout_layers)

    def test_defaults_used_when_dropout_not_provided(self, tmp_path: Path):
        feature_dir = tmp_path / "features"
        self._write_feature(feature_dir, "flat_sample.pt", torch.randn(32))

        config = self._base_config(feature_dir, hidden_dims=None, dropout_probs=None, num_classes=2)
        model = create_mlp_classifier(config)

        layers = list(model.children())
        linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
        dropout_layers = [layer for layer in layers if isinstance(layer, nn.Dropout)]

        assert [layer.out_features for layer in linear_layers[:-1]] == [512, 256]
        assert linear_layers[-1].out_features == 2
        assert [dropout.p for dropout in dropout_layers] == [pytest.approx(0.5), pytest.approx(0.3)]
