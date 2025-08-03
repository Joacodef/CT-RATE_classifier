import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet
from typing import Any, Tuple
from torch.utils.checkpoint import checkpoint_sequential

class DenseNet3D(nn.Module):
    """
    A 3D DenseNet model for volumetric classification. This class wraps the
    MONAI DenseNet implementation, using it as a feature extractor and
    appending a custom classification head. It supports gradient
    checkpointing to reduce VRAM usage during training.
    """
    def __init__(self, growth_rate: int, block_config: Tuple[int, ...],
                 num_init_features: int, num_classes: int, use_checkpointing: bool = False, **kwargs):
        super().__init__()
        
        # Store the gradient checkpointing flag.
        self.use_checkpointing = use_checkpointing

        # The MONAI DenseNet backbone, used for feature extraction.
        self.monai_densenet_base = DenseNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,  # This will be overridden by the custom classifier
            init_features=num_init_features,
            growth_rate=growth_rate,
            block_config=block_config
        )
        
        # Determine the number of features output by the MONAI backbone.
        num_features = self.monai_densenet_base.features[-1].num_features

        # The custom classification head.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features using the MONAI backbone's feature sequence.
        if self.use_checkpointing and self.training:
            # Use checkpoint_sequential for efficient gradient checkpointing.
            segments = len(self.monai_densenet_base.features)
            x = checkpoint_sequential(self.monai_densenet_base.features, segments, x)
        else:
            # Execute the feature extractor normally without checkpointing.
            x = self.monai_densenet_base.features(x)

        out = F.relu(x, inplace=True)
        
        # Pass features through the custom classification head.
        out = self.classifier(out)
        return out

def densenet121_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a 3D DenseNet-121 model."""
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 24, 16),
                      num_init_features=64, num_classes=num_classes,
                      use_checkpointing=use_checkpointing, **kwargs)

def densenet169_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a 3D DenseNet-169 model."""
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 32, 32),
                      num_init_features=64, num_classes=num_classes,
                      use_checkpointing=use_checkpointing, **kwargs)

def densenet201_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a 3D DenseNet-201 model."""
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 48, 32),
                      num_init_features=64, num_classes=num_classes,
                      use_checkpointing=use_checkpointing, **kwargs)

def densenet161_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a 3D DenseNet-161 model."""
    return DenseNet3D(growth_rate=48, block_config=(6, 12, 36, 24),
                      num_init_features=96, num_classes=num_classes,
                      use_checkpointing=use_checkpointing, **kwargs)

def densenet_small_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a small 3D DenseNet model."""
    return DenseNet3D(growth_rate=16, block_config=(4, 8, 12, 8),
                      num_init_features=32, num_classes=num_classes,
                      use_checkpointing=use_checkpointing, **kwargs)

def densenet_tiny_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a tiny 3D DenseNet model."""
    return DenseNet3D(growth_rate=12, block_config=(3, 6, 9, 6),
                      num_init_features=24, num_classes=num_classes,
                      use_checkpointing=use_checkpointing, **kwargs)