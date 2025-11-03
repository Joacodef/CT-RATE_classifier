"""Logistic regression classifier for feature workflows."""

from pathlib import Path
from types import SimpleNamespace
import logging

import torch.nn as nn

from .mlp import _discover_feature_file, _safe_torch_load, _determine_feature_dimension

logger = logging.getLogger(__name__)


def create_logistic_classifier(config: SimpleNamespace) -> nn.Module:
    """Create a single linear layer sized to saved feature embeddings."""
    feature_dir = Path(config.workflow.feature_config.feature_dir)
    sample_path = _discover_feature_file(feature_dir)
    sample_feature = _safe_torch_load(sample_path)
    in_features = _determine_feature_dimension(sample_feature)
    logger.info("Determined input feature dimension from sample: %d", in_features)

    num_classes = len(config.pathologies.columns)
    layer = nn.Linear(in_features, num_classes)
    logger.info(
        "Logistic classifier created with %s total parameters.",
        f"{sum(p.numel() for p in layer.parameters()):,}",
    )
    return layer
