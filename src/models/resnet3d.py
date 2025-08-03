import torch
import torch.nn as nn
from monai.networks.nets import ResNet

class ResNet3D(nn.Module):
    """
    A 3D ResNet model for volumetric data classification. This class
    leverages MONAI's ResNet implementation as a backbone and attaches a
    custom fully connected head for classification.
    """
    def __init__(self, block, layers, num_classes=18, **kwargs):
        super().__init__()
        
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
        
        # The classification head, composed of linear layers and dropout.
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features from the MONAI ResNet backbone.
        x = self.monai_resnet_base(x)
        
        # Flatten the feature map and pass it through the classification head.
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet18_3d(num_classes=18, **kwargs):
    """Constructs a 3D ResNet-18 model."""
    # The 'block' parameter is symbolic for model instantiation.
    return ResNet3D(None, [2, 2, 2, 2], num_classes=num_classes)

def resnet34_3d(num_classes=18, **kwargs):
    """Constructs a 3D ResNet-34 model."""
    # The 'block' parameter is symbolic for model instantiation.
    return ResNet3D(None, [3, 4, 6, 3], num_classes=num_classes)