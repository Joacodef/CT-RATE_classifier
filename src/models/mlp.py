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

    default_hidden_dims = [512, 256]
    hidden_dims = getattr(config.model, "hidden_dims", default_hidden_dims)
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    hidden_dims = list(hidden_dims) if hidden_dims else list(default_hidden_dims)

    default_dropouts = [0.5, 0.3]
    dropout_cfg = getattr(config.model, "dropout_probs", default_dropouts)
    if isinstance(dropout_cfg, (float, int)):
        dropout_rates = [float(dropout_cfg)] * len(hidden_dims)
    else:
        dropout_rates = list(dropout_cfg) if dropout_cfg else []

    if not dropout_rates:
        dropout_rates = [default_dropouts[min(i, len(default_dropouts) - 1)] for i in range(len(hidden_dims))]
    elif len(dropout_rates) < len(hidden_dims):
        dropout_rates.extend(
            [dropout_rates[-1]] * (len(hidden_dims) - len(dropout_rates))
        )
    elif len(dropout_rates) > len(hidden_dims):
        dropout_rates = dropout_rates[: len(hidden_dims)]

    layers: list[nn.Module] = []
    in_dim = in_features
    for idx, hidden_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        dropout_rate = float(dropout_rates[idx]) if idx < len(dropout_rates) else 0.0
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, num_classes))
    model = nn.Sequential(*layers)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "MLP architecture configured with hidden_dims=%s and dropouts=%s",
        hidden_dims,
        dropout_rates,
    )
    logger.info(
        "MLP classifier created with %s total parameters and %s trainable parameters.",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    return model
