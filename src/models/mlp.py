"""MLP-based classifier for feature workflows."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _determine_feature_dimension(sample_feature: Any) -> int:
    """Infer the flattened feature dimension from a saved tensor-like object."""
    if isinstance(sample_feature, torch.Tensor):
        return sample_feature.numel()

    if hasattr(sample_feature, "tensor") and isinstance(getattr(sample_feature, "tensor"), torch.Tensor):
        return sample_feature.tensor.numel()

    if isinstance(sample_feature, dict):
        for key in ("features", "feature", "embedding", "tensor"):
            value = sample_feature.get(key)
            if isinstance(value, torch.Tensor):
                return value.numel()
        tensor_vals = [v for v in sample_feature.values() if isinstance(v, torch.Tensor)]
        if tensor_vals:
            return tensor_vals[0].numel()
        raise ValueError("Loaded feature dict does not contain a tensor-like entry.")

    try:
        import numpy as _np  # local import to avoid hard dependency
        if isinstance(sample_feature, _np.ndarray):
            return int(sample_feature.size)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Unsupported feature format encountered during dimension inference.") from exc

    raise ValueError("Unsupported feature format encountered during dimension inference.")


def _discover_feature_file(feature_dir: Path) -> Path:
    """Locate a representative feature file within the provided directory."""
    train_sub = feature_dir / "train"
    if train_sub.exists() and any(train_sub.glob("*.pt")):
        logger.info("Using features from subfolder: %s", train_sub)
        return next(train_sub.glob("*.pt"))

    flattened = list(feature_dir.glob("*.pt"))
    if flattened:
        logger.info("Using flattened feature directory: %s", feature_dir)
        return flattened[0]

    raise FileNotFoundError(f"No feature files found in {feature_dir}.")


def _safe_torch_load(path: Path) -> Any:
    """Safely load a torch serialized object, retrying without weights-only mode when needed."""
    try:
        return torch.load(path)
    except Exception as load_exc:  # pragma: no cover - defensive branch
        try:
            import _pickle as _p
            if isinstance(load_exc, _p.UnpicklingError) or "Weights only load failed" in str(load_exc):
                logger.warning(
                    "torch.load weights-only unpickling failed for %s. Retrying with weights_only=False (unsafe).",
                    path,
                )
                return torch.load(path, weights_only=False)
        except Exception:
            pass
        raise load_exc


def create_mlp_classifier(config: SimpleNamespace) -> nn.Module:
    """Create an MLP classifier sized to saved feature embeddings."""
    feature_dir = Path(config.workflow.feature_config.feature_dir)
    try:
        sample_path = _discover_feature_file(feature_dir)
    except StopIteration as exc:  # pragma: no cover - defensive
        raise FileNotFoundError(
            f"No feature files found in {feature_dir}. Please run the feature generation script first."
        ) from exc

    sample_feature = _safe_torch_load(sample_path)
    in_features = _determine_feature_dimension(sample_feature)
    logger.info("Determined input feature dimension from sample: %d", in_features)

    num_classes = len(config.pathologies.columns)

    model = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "MLP classifier created with %s total parameters and %s trainable parameters.",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    return model
