# ============================================================================
# models/__init__.py
# ============================================================================
"""
Neural network models for CT 3D classification
"""

from .resnet3d import (
    resnet18_3d,
    resnet34_3d,
)
from .densenet3d import (
    DenseNet3D,
    densenet121_3d,
    densenet169_3d,
    densenet201_3d,
    densenet161_3d,
    densenet_small_3d,
    densenet_tiny_3d
)
from .vit3d import (
    VisionTransformer3D,
    vit_tiny_3d,
    vit_small_3d,
    vit_base_3d,
    vit_large_3d
)
from .mlp import create_mlp_classifier
from .logistic import create_logistic_classifier
from monai.losses import FocalLoss

__all__ = [
    # ResNet models
    'ResNet3D',
    'resnet18_3d',
    'resnet34_3d',
    
    # DenseNet models
    'DenseNet3D',
    'densenet121_3d',
    'densenet169_3d', 
    'densenet201_3d',
    'densenet161_3d',
    'densenet_small_3d',
    'densenet_tiny_3d',
    
    # Vision Transformer models
    'VisionTransformer3D',
    'vit_tiny_3d',
    'vit_small_3d',
    'vit_base_3d',
    'vit_large_3d',
    
    # Loss functions
    'FocalLoss',

    # Feature classifiers
    'create_mlp_classifier',
    'create_logistic_classifier'
]