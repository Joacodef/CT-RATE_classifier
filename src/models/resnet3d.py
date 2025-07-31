import torch
import torch.nn as nn
from monai.losses import FocalLoss

class BasicBlock3D(nn.Module):
    """
    A basic 3D residual block, which forms the building block of ResNet-18 and ResNet-34.
    It consists of two 3x3x3 convolutions and a residual (or identity) connection.
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # The first convolutional layer of the block.
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # The second convolutional layer of the block.
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        # The downsample layer is used for the residual connection when dimensions change.
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        # The identity is the original input to the block.
        identity = x
        
        # Main path of the residual block.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # If the spatial dimensions or channel counts are changing, the identity
        # must be downsampled to match the output shape for addition.
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # The core of the residual block: add the identity to the output.
        out += identity
        out = self.relu(out)
        
        return out
    
class ResNet3D(nn.Module):
    """3D ResNet for CT classification, adapted for single-channel volumetric input."""
    
    def __init__(self, block, layers, num_classes=18, zero_init_residual=False, 
                 use_checkpointing=False):
        super().__init__()
        self.inplanes = 64
        self.use_checkpointing = use_checkpointing
        
        # --- Network Stem ---
        # The initial layers that process the raw input volume.
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # --- Residual Layers ---
        # The main body of the network, composed of sequential residual blocks.
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # --- Classification Head ---
        # Final layers that produce the classification output.
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * block.expansion, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Zero-initialize the last BN in each residual branch, so that the residual
        # branch starts with zeros, and each residual block behaves like an identity.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock3D):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """Helper function to create a layer of multiple residual blocks."""
        downsample = None
        # A downsample connection is needed if the stride is not 1 (spatial downsampling)
        # or if the number of input channels does not match the output channels.
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
            
        layers = []
        # The first block in a layer handles downsampling.
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # Subsequent blocks in the layer have no downsampling.
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # --- Stem ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # --- Residual Layers ---
        # If checkpointing is enabled, wrap each layer in a checkpoint call.
        # This saves memory by not storing intermediate activations, at the cost
        # of re-computing them during the backward pass.
        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.layer1, x)
            x = torch.utils.checkpoint.checkpoint(self.layer2, x)
            x = torch.utils.checkpoint.checkpoint(self.layer3, x)
            x = torch.utils.checkpoint.checkpoint(self.layer4, x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
        # --- Classification Head ---
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet18_3d(num_classes=18, use_checkpointing=False):
    """Constructs a 3D ResNet-18 model."""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=num_classes, 
                   use_checkpointing=use_checkpointing)

def resnet34_3d(num_classes=18, use_checkpointing=False):
    """Constructs a 3D ResNet-34 model."""
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], num_classes=num_classes,
                   use_checkpointing=use_checkpointing)