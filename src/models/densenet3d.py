import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any, Tuple
from torch.utils.checkpoint import checkpoint_sequential

class _DenseLayer(nn.Module):
    """
    A single layer within a DenseBlock, adapted for 3D. It performs a
    BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3) sequence.
    """
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float):
        super(_DenseLayer, self).__init__()
        # Bottleneck layer (1x1 conv)
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        # Composite function (3x3 conv)
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)

    def forward(self, prev_features: torch.Tensor) -> torch.Tensor:
        # Concatenate previous features
        concated_features = torch.cat(prev_features, 1)
        # Pass through bottleneck layer
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        # Pass through composite function
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    """
    A block of multiple DenseLayers, where the input to each layer is the
    concatenation of all previous layer outputs within the block.
    """
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int,
                 growth_rate: int, drop_rate: float):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    """
    A transition layer between two DenseBlocks. It reduces the number of
    feature maps and performs spatial downsampling.
    """
    def __init__(self, num_input_features: int, num_output_features: int):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3D(nn.Module):
    """
    A self-contained 3D DenseNet model for volumetric classification.
    This implementation is adapted for 3D from the official PyTorch model and
    does not depend on MONAI for its core architecture. It supports gradient
    checkpointing to reduce VRAM usage during training.
    """
    def __init__(self, growth_rate: int = 32, block_config: Tuple[int, ...] = (6, 12, 24, 16),
                 num_init_features: int = 64, bn_size: int = 4, drop_rate: float = 0,
                 num_classes: int = 18, use_checkpointing: bool = False):
        super(DenseNet3D, self).__init__()
        self.use_checkpointing = use_checkpointing

        # --- Network Stem ---
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # --- Main Body (DenseBlocks and Transitions) ---
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features using the backbone.
        if self.use_checkpointing and self.training:
            # Use checkpoint_sequential for efficient gradient checkpointing.
            # We checkpoint each component of the features sequence (conv, norm, block, transition, etc.)
            segments = len(self.features)
            features_out = checkpoint_sequential(self.features, segments, x)
        else:
            features_out = self.features(x)

        # Final activation and classification
        out = F.relu(features_out, inplace=True)
        out = self.classifier(out)
        return out

def densenet121_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a 3D DenseNet-121 model."""
    return DenseNet3D(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                      num_classes=num_classes, use_checkpointing=use_checkpointing, **kwargs)

def densenet169_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a 3D DenseNet-169 model."""
    return DenseNet3D(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                      num_classes=num_classes, use_checkpointing=use_checkpointing, **kwargs)

def densenet201_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a 3D DenseNet-201 model."""
    return DenseNet3D(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                      num_classes=num_classes, use_checkpointing=use_checkpointing, **kwargs)

def densenet161_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a 3D DenseNet-161 model."""
    return DenseNet3D(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                      num_classes=num_classes, use_checkpointing=use_checkpointing, **kwargs)

def densenet_small_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a small 3D DenseNet model for lower memory usage."""
    return DenseNet3D(num_init_features=32, growth_rate=16, block_config=(4, 8, 12, 8),
                      num_classes=num_classes, use_checkpointing=use_checkpointing, **kwargs)

def densenet_tiny_3d(num_classes: int = 18, use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """Constructs a tiny 3D DenseNet model for minimal memory usage."""
    return DenseNet3D(num_init_features=24, growth_rate=12, block_config=(3, 6, 9, 6),
                      num_classes=num_classes, use_checkpointing=use_checkpointing, **kwargs)