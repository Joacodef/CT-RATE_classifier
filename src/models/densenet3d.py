import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any, List, Tuple
import torch.utils.checkpoint as cp


class _DenseLayer(nn.Module):
    """
    A single 3D DenseNet layer, which includes a bottleneck and a 3x3x3 convolution.

    This layer takes all previous feature maps as input, processes them through a
    bottleneck to reduce computational cost, and then produces a new feature map
    (the "growth rate") which is concatenated to the input for the next layer.
    """
    
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, 
                 drop_rate: float, memory_efficient: bool = False) -> None:
        super().__init__()

        # Bottleneck layer: 1x1x1 Conv to reduce the number of feature-maps.        
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate,
                              kernel_size=1, stride=1, bias=False)
        
        # Main convolutional layer: 3x3x3 Conv to produce new features.
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient
    
    def bn_function(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        The bottleneck function, which is the part of the layer that can be
        checkpointed for memory savings. It concatenates features and passes
        them through the first convolution.
        """
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output
    
    def any_requires_grad(self, input: List[torch.Tensor]) -> bool:
        """Check if any tensor in a list requires a gradient."""
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the dense layer. It supports a memory-efficient
        path using gradient checkpointing.
        """
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        
        # If memory_efficient is enabled, use gradient checkpointing on the
        # bottleneck function. This saves memory by not storing intermediate
        # activations for the backward pass.
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)
        
        # The second convolution is applied to produce the new feature map.
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        return new_features
    
    def call_checkpoint_bottleneck(self, input: List[torch.Tensor]) -> torch.Tensor:
        """A wrapper to apply checkpointing to the bottleneck function."""
        def closure(*inputs):
            return self.bn_function(inputs)
        return cp.checkpoint(closure, *input)


class _DenseBlock(nn.ModuleDict):
    """
    A block of multiple dense layers. In a dense block, the feature maps of all
    preceding layers are used as inputs for all subsequent layers.
    """
    
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int,
                 growth_rate: int, drop_rate: float, memory_efficient: bool = False) -> None:
        super().__init__()
        
        for i in range(num_layers):
            layer = _DenseLayer(
                # The input to each new layer is the sum of the initial features
                # and the features created by all previous layers in the block.
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module(f'denselayer{i + 1}', layer)
    
    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for the dense block, which iteratively builds upon
        a list of feature maps.
        """
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """
    A transition layer between dense blocks. It reduces the number of channels
    and downsamples the spatial dimensions (height, width, depth).
    """
    
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        # 1x1x1 Conv to halve the number of channels.
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                         kernel_size=1, stride=1, bias=False))
        # Average pooling to halve the spatial dimensions.
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3D(nn.Module):
    """3D DenseNet for CT volume classification
    
    Args:
        growth_rate (int): how many filters to add each layer (k in paper)
        block_config (list): how many layers in each pooling block
        num_init_features (int): the number of filters to learn in the first convolution layer
        bn_size (int): multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float): dropout rate after each dense layer
        num_classes (int): number of classification classes
        memory_efficient (bool): If True, uses checkpointing inside the dense layers.
        use_checkpointing (bool): If True, uses gradient checkpointing on entire dense blocks.
    """
    
    def __init__(self, growth_rate: int = 32, block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
                 num_init_features: int = 64, bn_size: int = 4, drop_rate: float = 0.0,
                 num_classes: int = 18, memory_efficient: bool = False,
                 use_checkpointing: bool = False) -> None:
        super().__init__()
        
        self.use_checkpointing = use_checkpointing
        
        # The initial convolution layer that processes the input volume.
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Each dense block followed by a transition layer.
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                   num_output_features=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2
        
        # Final batch norm after the last dense block.
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        
        # The final classification head of the network.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Official weight initialization from the torchvision repository.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The main forward pass, with an optional checkpointed path."""
        if self.use_checkpointing and self.training:
            # The checkpointed path trades compute for memory by not storing
            # intermediate activations for the entire feature extractor.
            features = self._forward_features_checkpointed(x)
        else:
            features = self.features(x)
        
        out = F.relu(features, inplace=True)
        out = self.classifier(out)
        return out
    
    def _forward_features_checkpointed(self, x: torch.Tensor) -> torch.Tensor:
        """
        An alternative forward pass for the feature extractor that applies
        gradient checkpointing to each dense block. This is highly memory
        efficient during training.
        """
        
        # Initial layers are run normally.
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        
        # Each dense block is wrapped in a checkpoint call.
        x = cp.checkpoint(self.features.denseblock1, x)
        x = self.features.transition1(x)
        
        x = cp.checkpoint(self.features.denseblock2, x)
        x = self.features.transition2(x)
        
        x = cp.checkpoint(self.features.denseblock3, x)
        x = self.features.transition3(x)
        
        x = cp.checkpoint(self.features.denseblock4, x)
        x = self.features.norm5(x)
        
        return x


def densenet121_3d(num_classes: int = 18, memory_efficient: bool = False, 
                   use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """DenseNet-121 model for 3D volumes
    
    Args:
        num_classes: number of output classes
        memory_efficient: use memory efficient implementation inside layers
        use_checkpointing: use gradient checkpointing on entire blocks
    """
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 24, 16), 
                      num_init_features=64, num_classes=num_classes,
                      memory_efficient=memory_efficient, 
                      use_checkpointing=use_checkpointing, **kwargs)


def densenet169_3d(num_classes: int = 18, memory_efficient: bool = False,
                   use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """DenseNet-169 model for 3D volumes
    
    Args:
        num_classes: number of output classes
        memory_efficient: use memory efficient implementation inside layers
        use_checkpointing: use gradient checkpointing on entire blocks
    """
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 32, 32),
                      num_init_features=64, num_classes=num_classes,
                      memory_efficient=memory_efficient,
                      use_checkpointing=use_checkpointing, **kwargs)


def densenet201_3d(num_classes: int = 18, memory_efficient: bool = False,
                   use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """DenseNet-201 model for 3D volumes
    
    Args:
        num_classes: number of output classes
        memory_efficient: use memory efficient implementation inside layers
        use_checkpointing: use gradient checkpointing on entire blocks
    """
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 48, 32),
                      num_init_features=64, num_classes=num_classes,
                      memory_efficient=memory_efficient,
                      use_checkpointing=use_checkpointing, **kwargs)


def densenet161_3d(num_classes: int = 18, memory_efficient: bool = False,
                   use_checkpointing: bool = False, **kwargs: Any) -> DenseNet3D:
    """DenseNet-161 model for 3D volumes
    
    Args:
        num_classes: number of output classes
        memory_efficient: use memory efficient implementation inside layers
        use_checkpointing: use gradient checkpointing on entire blocks
    """
    return DenseNet3D(growth_rate=48, block_config=(6, 12, 36, 24),
                      num_init_features=96, num_classes=num_classes,
                      memory_efficient=memory_efficient,
                      use_checkpointing=use_checkpointing, **kwargs)


# Compact versions for memory-constrained environments
def densenet_small_3d(num_classes: int = 18, memory_efficient: bool = True,
                      use_checkpointing: bool = True, **kwargs: Any) -> DenseNet3D:
    """Small DenseNet model optimized for memory efficiency
    
    Args:
        num_classes: number of output classes
        memory_efficient: use memory efficient implementation inside layers
        use_checkpointing: use gradient checkpointing on entire blocks
    """
    return DenseNet3D(growth_rate=16, block_config=(4, 8, 12, 8),
                      num_init_features=32, num_classes=num_classes,
                      memory_efficient=memory_efficient,
                      use_checkpointing=use_checkpointing, **kwargs)


def densenet_tiny_3d(num_classes: int = 18, memory_efficient: bool = True,
                     use_checkpointing: bool = True, **kwargs: Any) -> DenseNet3D:
    """Tiny DenseNet model for very limited memory environments
    
    Args:
        num_classes: number of output classes
        memory_efficient: use memory efficient implementation inside layers
        use_checkpointing: use gradient checkpointing on entire blocks
    """
    return DenseNet3D(growth_rate=12, block_config=(3, 6, 9, 6),
                      num_init_features=24, num_classes=num_classes,
                      memory_efficient=memory_efficient,
                      use_checkpointing=use_checkpointing, **kwargs)