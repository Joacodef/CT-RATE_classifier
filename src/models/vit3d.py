import torch
import torch.nn as nn
from monai.networks.nets import ViT
from typing import Any

class VisionTransformer3D(nn.Module):
    """
    3D Vision Transformer for volume classification. This class wraps the
    MONAI ViT implementation, using it as a backbone and attaching a
    custom classification head.
    
    Args:
        volume_size: input volume dimensions (D, H, W)
        patch_size: patch dimensions for splitting the volume
        in_channels: number of input channels (1 for CT)
        num_classes: number of classification classes
        embed_dim: embedding dimension
        depth: number of transformer blocks
        num_heads: number of attention heads
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        qkv_bias: whether to use bias in qkv projection
        drop_rate: dropout rate
        attn_drop_rate: attention dropout rate
    """
    def __init__(self, volume_size=(96, 224, 224), patch_size=(16, 16, 16),
                 in_channels=1, num_classes=18, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., **kwargs):
        super().__init__()

        self.monai_vit_base = ViT(
            in_channels=in_channels,
            img_size=volume_size,
            patch_size=patch_size,
            hidden_size=embed_dim,
            mlp_dim=int(embed_dim * mlp_ratio),
            num_layers=depth,
            num_heads=num_heads,
            classification=True, # Use MONAI's classification structure
            num_classes=embed_dim, # Output embedding for the head
            dropout_rate=drop_rate,
            qkv_bias=qkv_bias,
            post_activation=None # No activation in the backbone's head
        )

        # Replace MONAI's classification head with our custom one
        self.monai_vit_base.classification_head = nn.Identity()

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), # Add final normalization
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features from the MONAI ViT backbone.
        # The MONAI ViT's forward with classification=True already handles
        # CLS token extraction.
        x, _ = self.monai_vit_base(x)
        
        # Pass the features through the custom classification head.
        x = self.head(x)
        return x

def vit_tiny_3d(num_classes=18, **kwargs: Any) -> VisionTransformer3D:
    """Constructs a Tiny 3D ViT model."""
    model_kwargs = {'embed_dim': 192, 'depth': 12, 'num_heads': 3, 'mlp_ratio': 4.0, 'num_classes': num_classes}
    model_kwargs.update(kwargs)
    return VisionTransformer3D(**model_kwargs)

def vit_small_3d(num_classes=18, **kwargs: Any) -> VisionTransformer3D:
    """Constructs a Small 3D ViT model."""
    model_kwargs = {'embed_dim': 384, 'depth': 12, 'num_heads': 6, 'mlp_ratio': 4.0, 'num_classes': num_classes}
    model_kwargs.update(kwargs)
    return VisionTransformer3D(**model_kwargs)

def vit_base_3d(num_classes=18, **kwargs: Any) -> VisionTransformer3D:
    """Constructs a Base 3D ViT model."""
    model_kwargs = {'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4.0, 'num_classes': num_classes}
    model_kwargs.update(kwargs)
    return VisionTransformer3D(**model_kwargs)

def vit_large_3d(num_classes=18, **kwargs: Any) -> VisionTransformer3D:
    """Constructs a Large 3D ViT model."""
    model_kwargs = {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4.0, 'num_classes': num_classes}
    model_kwargs.update(kwargs)
    return VisionTransformer3D(**model_kwargs)