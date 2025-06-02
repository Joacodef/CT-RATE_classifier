"""
3D Vision Transformer (ViT) implementation for CT volume classification

Based on "An Image is Worth 16x16 Words" but adapted for 3D medical imaging.
Includes memory optimization techniques for handling large 3D volumes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding layer"""
    
    def __init__(self, volume_size=(96, 224, 224), patch_size=(16, 16, 16), 
                 in_channels=1, embed_dim=768):
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.n_patches = (volume_size[0] // patch_size[0]) * \
                        (volume_size[1] // patch_size[1]) * \
                        (volume_size[2] // patch_size[2])
        
        self.proj = nn.Conv3d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x shape: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = rearrange(x, 'b e d h w -> b (d h w) e')
        return x


class MultiHeadAttention3D(nn.Module):
    """Multi-head self-attention for 3D data"""
    
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block for transformer"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, 
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                         attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer3D(nn.Module):
    """3D Vision Transformer for CT volume classification
    
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
        use_checkpointing: whether to use gradient checkpointing
    """
    
    def __init__(self, volume_size=(96, 224, 224), patch_size=(16, 16, 16), 
                 in_channels=1, num_classes=18, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., use_checkpointing=False):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim # This will be set by the final kwargs
        self.use_checkpointing = use_checkpointing
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(volume_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate
            ) for _ in range(depth)
        ])
        
        # Normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        if self.use_checkpointing and self.training:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x)
        else:
            for block in self.blocks:
                x = block(x)
        
        # Normalize
        x = self.norm(x)
        
        # Extract cls token
        return x[:, 0]
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_tiny_3d(num_classes=18, use_checkpointing=False, **kwargs):
    """Tiny ViT model for 3D volumes (faster training, less memory)"""
    # Define default parameters for this variant
    model_kwargs = {
        'embed_dim': 192, 
        'depth': 12, 
        'num_heads': 3, 
        'mlp_ratio': 4.0,
        'num_classes': num_classes, 
        'use_checkpointing': use_checkpointing
    }
    # Allow kwargs passed from create_model to override these defaults
    model_kwargs.update(kwargs)
    return VisionTransformer3D(**model_kwargs)


def vit_small_3d(num_classes=18, use_checkpointing=False, **kwargs):
    """Small ViT model for 3D volumes"""
    # Define default parameters for this variant
    model_kwargs = {
        'embed_dim': 384, 
        'depth': 12, 
        'num_heads': 6, 
        'mlp_ratio': 4.0,
        'num_classes': num_classes, 
        'use_checkpointing': use_checkpointing
    }
    # Allow kwargs passed from create_model to override these defaults
    model_kwargs.update(kwargs)
    return VisionTransformer3D(**model_kwargs)


def vit_base_3d(num_classes=18, use_checkpointing=False, **kwargs):
    """Base ViT model for 3D volumes"""
    # Define default parameters for this variant
    model_kwargs = {
        'embed_dim': 768, 
        'depth': 12, 
        'num_heads': 12, 
        'mlp_ratio': 4.0,
        'num_classes': num_classes, 
        'use_checkpointing': use_checkpointing
    }
    # Allow kwargs passed from create_model to override these defaults
    model_kwargs.update(kwargs)
    return VisionTransformer3D(**model_kwargs)


def vit_large_3d(num_classes=18, use_checkpointing=False, **kwargs):
    """Large ViT model for 3D volumes (requires significant memory)"""
    # Define default parameters for this variant
    model_kwargs = {
        'embed_dim': 1024, 
        'depth': 24, 
        'num_heads': 16, 
        'mlp_ratio': 4.0,
        'num_classes': num_classes, 
        'use_checkpointing': use_checkpointing
    }
    # Allow kwargs passed from create_model to override these defaults
    model_kwargs.update(kwargs)
    return VisionTransformer3D(**model_kwargs)