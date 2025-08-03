import torch
import torch.nn as nn
from monai.networks.nets import ResNet
from torch.utils.checkpoint import checkpoint_sequential

class ResNet3D(nn.Module):
    """
    A 3D ResNet model for volumetric data classification. This class
    leverages MONAI's ResNet implementation as a backbone and attaches a
    custom fully connected head for classification. It supports gradient
    checkpointing to reduce VRAM usage during training.
    """
    def __init__(self, layers, num_classes=18, use_checkpointing=False):
        super().__init__()
        
        # Store the gradient checkpointing flag.
        self.use_checkpointing = use_checkpointing

        # Select the ResNet architecture based on the 'layers' argument.
        if layers == [2, 2, 2, 2]: # ResNet-18 configuration
            monai_layers = [2, 2, 2, 2]
            monai_block = 'basic'
        elif layers == [3, 4, 6, 3]: # ResNet-34 configuration
            monai_layers = [3, 4, 6, 3]
            monai_block = 'basic'
        else:
            raise ValueError("Unsupported ResNet architecture specified by layers.")

        # The MONAI ResNet backbone, configured to output feature maps.
        self.monai_resnet_base = ResNet(
            block=monai_block,
            layers=monai_layers,
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            feed_forward=False
        )

        # Consolidate residual layers for efficient checkpointing.
        self.residual_layers = nn.Sequential(
            self.monai_resnet_base.layer1,
            self.monai_resnet_base.layer2,
            self.monai_resnet_base.layer3,
            self.monai_resnet_base.layer4
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # The classification head, composed of linear layers and dropout.
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # --- Stem ---
        # Process input through the initial layers of the ResNet backbone.
        x = self.monai_resnet_base.conv1(x)
        x = self.monai_resnet_base.bn1(x)
        x = self.monai_resnet_base.act(x)
        x = self.monai_resnet_base.maxpool(x)

        # --- Residual Layers ---
        # Apply checkpointing to the main residual layers to save memory.
        # This is active only during training.
        if self.use_checkpointing and self.training:
            x = checkpoint_sequential(self.residual_layers, 4, x)
        else:
            x = self.residual_layers(x)
        
        # --- Classification Head ---
        # Flatten the feature map and pass it through the classification head.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet18_3d(num_classes=18, use_checkpointing=False):
    """Constructs a 3D ResNet-18 model."""
    return ResNet3D([2, 2, 2, 2], num_classes=num_classes, 
                   use_checkpointing=use_checkpointing)

def resnet34_3d(num_classes=18, use_checkpointing=False):
    """Constructs a 3D ResNet-34 model."""
    return ResNet3D([3, 4, 6, 3], num_classes=num_classes,
                   use_checkpointing=use_checkpointing)