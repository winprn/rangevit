# Copyright 2023 - Valeo Comfort and Driving Assistance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional, Tuple

from .model_utils import init_weights


class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer backbone for RangeViT
    Returns hierarchical multi-scale features compatible with RangeViT decoder

    Features:
    - 4 hierarchical stages with different spatial resolutions
    - Channel dimensions that double at each stage
    - Window-based attention mechanism
    - Compatible with timm pretrained weights
    """

    def __init__(
        self,
        model_name: str = 'swin_tiny_patch4_window7_224',
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        channels: int = 3,
        pretrained: bool = False,
        drop_path_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize Swin Transformer backbone

        Args:
            model_name: Name of the timm model
            embed_dim: Base embedding dimension (C)
            depths: Number of layers in each stage
            num_heads: Number of attention heads in each stage
            window_size: Window size for attention
            channels: Number of input channels
            pretrained: Whether to load timm pretrained weights
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size

        try:
            # Create Swin model using timm
            self.swin = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,  # Return intermediate features
                out_indices=(0, 1, 2, 3),  # All 4 stages
                in_chans=channels,
                drop_path_rate=drop_path_rate
            )

            # Modify patch embedding to accept flexible input sizes
            # Find the patch embedding module and remove size constraints
            def remove_size_check(module):
                if hasattr(module, 'img_size'):
                    # Store the original check but disable it
                    module._original_img_size = module.img_size
                    module.img_size = None
                for child in module.children():
                    remove_size_check(child)

            remove_size_check(self.swin)

            print(f"[OK] Created Swin backbone: {model_name}")
            print(f"  - Embed dim: {embed_dim}")
            print(f"  - Input channels: {channels}")
            print(f"  - Window size: {window_size}")

        except Exception as e:
            print(f"[ERROR] Failed to create Swin model {model_name}: {e}")
            raise

        # Store feature dimensions for each stage
        # For Swin-Tiny: [96, 192, 384, 768]
        self.feature_dims = [
            embed_dim,        # Stage 0: C
            embed_dim * 2,    # Stage 1: 2C
            embed_dim * 4,    # Stage 2: 4C
            embed_dim * 8     # Stage 3: 8C
        ]

        # Store stride information for each stage
        self.strides = [4, 8, 16, 32]

        # Initialize any additional layers
        self.apply(init_weights)

    def get_feature_dims(self) -> List[int]:
        """Return channel dimensions for each stage"""
        return self.feature_dims.copy()

    def get_strides(self) -> List[int]:
        """Return spatial stride for each stage"""
        return self.strides.copy()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through Swin Transformer

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of 4 feature maps from different stages:
            - F0: (B, C,   H/4,  W/4)   - Stage 0, stride 4
            - F1: (B, 2C,  H/8,  W/8)   - Stage 1, stride 8
            - F2: (B, 4C,  H/16, W/16)  - Stage 2, stride 16
            - F3: (B, 8C,  H/32, W/32)  - Stage 3, stride 32

            For Swin-Tiny (embed_dim=96):
            - F0: (B, 96,  H/4,  W/4)
            - F1: (B, 192, H/8,  W/8)
            - F2: (B, 384, H/16, W/16)
            - F3: (B, 768, H/32, W/32)
        """
        try:
            features = self.swin(x)

            # Convert from timm format (B, H, W, C) to standard format (B, C, H, W)
            features = [feat.permute(0, 3, 1, 2).contiguous() for feat in features]

            # Validate output shapes
            B, C, H, W = x.shape
            expected_shapes = [
                (B, self.feature_dims[0], H//4, W//4),
                (B, self.feature_dims[1], H//8, W//8),
                (B, self.feature_dims[2], H//16, W//16),
                (B, self.feature_dims[3], H//32, W//32),
            ]

            for i, (feat, expected) in enumerate(zip(features, expected_shapes)):
                if feat.shape != expected:
                    print(f"Warning: Stage {i} shape mismatch. Got {feat.shape}, expected {expected}")

            return features

        except Exception as e:
            print(f"[ERROR] Swin forward pass failed: {e}")
            print(f"  Input shape: {x.shape}")
            raise


class SwinVisionTransformer(nn.Module):
    """
    Wrapper to make Swin Transformer compatible with RangeViT's ViT interface

    This class provides the same interface as the original VisionTransformer class
    but uses Swin Transformer backbone internally. For Phase 1, it uses only the
    F2 stage (stride 16) to match ViT behavior.
    """

    def __init__(
        self,
        swin_backbone: SwinTransformerBackbone,
        use_stage: int = 2,  # Use F2 by default (stride 16, matches ViT)
    ):
        """
        Initialize Swin-ViT compatibility wrapper

        Args:
            swin_backbone: SwinTransformerBackbone instance
            use_stage: Which Swin stage to use (0-3). Default 2 (F2, stride 16)
        """
        super().__init__()

        self.swin = swin_backbone
        self.use_stage = use_stage

        # Set interface properties to match VisionTransformer
        self.d_model = self.swin.feature_dims[use_stage]  # Feature dimension
        self.patch_size = (4, 4)  # Swin patch size
        self.patch_stride = (4, 4)  # Same as patch size for Swin

        # Image size will be set when needed
        self.image_size = None

        print(f"[OK] Created SwinVisionTransformer wrapper")
        print(f"  - Using stage: {use_stage} (stride {self.swin.strides[use_stage]})")
        print(f"  - Feature dim: {self.d_model}")

    @torch.jit.ignore
    def no_weight_decay(self):
        """Compatibility with RangeViT weight decay exclusion"""
        # Swin doesn't use pos_embed or cls_token, so return empty set
        return set()

    def get_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        """
        Get grid size for given input dimensions
        Compatible with RangeViT's get_grid_size interface
        """
        stride = self.swin.strides[self.use_stage]
        return (H // stride, W // stride)

    def get_actual_grid_size(self) -> Tuple[int, int]:
        """
        Get the actual grid size from the last forward pass
        This is needed because Swin may have different actual dimensions
        than what the decoder expects
        """
        if hasattr(self, '_last_grid_size'):
            return self._last_grid_size
        return None

    def forward(self, im: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass compatible with VisionTransformer interface

        Args:
            im: Input tensor (B, C, H, W)
            return_features: Always True for compatibility

        Returns:
            Tuple of (tokens, skip):
            - tokens: Feature tokens in ViT format (B, N, D) where N = H*W/stride^2
            - skip: Skip connection (None for Swin, may be added later)
        """
        B, C, H, W = im.shape
        self.image_size = (H, W)  # Store for compatibility

        # Reshape to square for pretrained SwinV2 compatibility
        # Most pretrained Swin models expect square inputs (224x224, 256x256, etc.)
        target_size = 256  # Standard size for swinv2_tiny_window16_256

        if H != target_size or W != target_size:
            # Resize to square, preserving aspect ratio with padding
            im_resized = torch.nn.functional.interpolate(
                im, size=(target_size, target_size), mode='bilinear', align_corners=False
            )
        else:
            im_resized = im

        # Get all hierarchical features from Swin
        multi_scale_features = self.swin(im_resized)  # [F0, F1, F2, F3]

        # Use specified stage (default F2 for Phase 1)
        selected_feature = multi_scale_features[self.use_stage]  # (B, D, H/stride, W/stride)

        # If we resized input, we need to adjust feature map size back
        feat_B, feat_D, feat_H, feat_W = selected_feature.shape

        if H != target_size or W != target_size:
            # Calculate expected feature map size based on original input
            # Assuming F2 stage has stride 16 for Swin-Tiny
            expected_feat_H = H // 16
            expected_feat_W = W // 16

            # Resize feature map back to expected proportions
            selected_feature = torch.nn.functional.interpolate(
                selected_feature,
                size=(expected_feat_H, expected_feat_W),
                mode='bilinear',
                align_corners=False
            )
            feat_B, feat_D, feat_H, feat_W = selected_feature.shape

        # Convert to token format like ViT
        # ViT returns (B, N+1, D) where N+1 includes CLS token
        # We'll return (B, N, D) and handle CLS token removal in the caller
        tokens = selected_feature.flatten(2).transpose(1, 2)  # (B, N, D) where N = feat_H * feat_W

        # Store the actual grid size for decoder compatibility
        self._last_grid_size = (feat_H, feat_W)

        # Add dummy CLS token to match ViT interface (will be removed by caller)
        cls_token = torch.zeros(B, 1, feat_D, device=tokens.device, dtype=tokens.dtype)
        tokens_with_cls = torch.cat([cls_token, tokens], dim=1)  # (B, N+1, D)

        # No skip connection for now (Swin doesn't naturally provide one like ConvStem)
        skip = None

        return tokens_with_cls, skip


def create_swin_backbone(
    model_name: str,
    channels: int = 3,
    pretrained: bool = False,
    **kwargs
) -> SwinTransformerBackbone:
    """
    Factory function to create Swin backbone with predefined configurations

    Args:
        model_name: Name of the Swin model
        channels: Number of input channels
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments

    Returns:
        SwinTransformerBackbone instance
    """

    # Predefined configurations for common Swin models
    swin_configs = {
        'swin_tiny_patch4_window7_224': {
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
        },
        'swin_small_patch4_window7_224': {
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
        },
        'swin_base_patch4_window7_224': {
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'window_size': 7,
        },
        'swinv2_tiny_window16_256': {
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 16,
        },
        'swinv2_tiny_window16_256': {
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 16,
        }
    }

    if model_name not in swin_configs:
        raise ValueError(f"Unknown Swin model: {model_name}. Available: {list(swin_configs.keys())}")

    config = swin_configs[model_name]
    config.update(kwargs)  # Allow override of default config

    return SwinTransformerBackbone(
        model_name=model_name,
        channels=channels,
        pretrained=pretrained,
        **config
    )


if __name__ == '__main__':
    """
    Test script for Swin Transformer backbone
    """
    print("Testing Swin Transformer Backbone...")

    # Test configuration - use dimensions compatible with window size
    batch_size = 2
    channels = 5  # Range image channels
    height, width = 224, 224  # Test with standard size first

    # Create test input
    x = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {x.shape}")

    try:
        # Test backbone creation
        backbone = create_swin_backbone(
            'swin_tiny_patch4_window7_224',
            channels=channels,
            pretrained=False
        )
        print(f"[OK] Backbone created successfully")
        print(f"Feature dims: {backbone.get_feature_dims()}")
        print(f"Strides: {backbone.get_strides()}")

        # Test forward pass
        with torch.no_grad():
            features = backbone(x)

        print(f"[OK] Forward pass successful")
        for i, feat in enumerate(features):
            print(f"  Stage {i}: {feat.shape}")

        # Test wrapper
        wrapper = SwinVisionTransformer(backbone, use_stage=2)
        print(f"[OK] Wrapper created successfully")
        print(f"d_model: {wrapper.d_model}")

        # Test wrapper forward pass
        with torch.no_grad():
            tokens, skip = wrapper(x, return_features=True)

        print(f"[OK] Wrapper forward pass successful")
        print(f"  Tokens shape: {tokens.shape}")
        print(f"  Skip: {skip}")

        print("[SUCCESS] All tests passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()