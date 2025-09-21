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
from typing import List

from .model_utils import init_weights


class MockSwinBackbone(nn.Module):
    """
    Mock Swin Transformer backbone for testing multi-scale decoder

    This generates hierarchical features that match the expected Swin output format:
    - F0: (B, 96,  H/4,  W/4)   - Stage 0, stride 4
    - F1: (B, 192, H/8,  W/8)   - Stage 1, stride 8
    - F2: (B, 384, H/16, W/16)  - Stage 2, stride 16
    - F3: (B, 768, H/32, W/32)  - Stage 3, stride 32

    Used for testing and development when timm Swin models have compatibility issues.
    """

    def __init__(
        self,
        channels: int = 5,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
    ):
        """
        Initialize Mock Swin Backbone

        Args:
            channels: Number of input channels
            embed_dim: Base embedding dimension (C)
            depths: Number of layers in each stage (not used in mock)
            num_heads: Number of attention heads in each stage (not used in mock)
        """
        super().__init__()

        self.channels = channels
        self.embed_dim = embed_dim

        # Feature dimensions for each stage
        self.feature_dims = [
            embed_dim,        # Stage 0: C = 96
            embed_dim * 2,    # Stage 1: 2C = 192
            embed_dim * 4,    # Stage 2: 4C = 384
            embed_dim * 8     # Stage 3: 8C = 768
        ]

        # Stride information for each stage
        self.strides = [4, 8, 16, 32]

        # Create simple convolutional stages that mimic Swin hierarchical structure
        self.stage0 = nn.Sequential(
            nn.Conv2d(channels, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, 4, stride=4),  # Downsample to H/4, W/4
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim * 2, 2, stride=2),  # Downsample to H/8, W/8
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim * 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, embed_dim * 4, 2, stride=2),  # Downsample to H/16, W/16
            nn.BatchNorm2d(embed_dim * 4),
            nn.ReLU(inplace=True)
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim * 4, 3, padding=1),
            nn.BatchNorm2d(embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 4, embed_dim * 8, 2, stride=2),  # Downsample to H/32, W/32
            nn.BatchNorm2d(embed_dim * 8),
            nn.ReLU(inplace=True)
        )

        # Initialize weights
        self.apply(init_weights)

        print(f"[OK] Created Mock Swin backbone")
        print(f"  - Input channels: {channels}")
        print(f"  - Embed dim: {embed_dim}")
        print(f"  - Feature dims: {self.feature_dims}")

    def get_feature_dims(self) -> List[int]:
        """Return channel dimensions for each stage"""
        return self.feature_dims.copy()

    def get_strides(self) -> List[int]:
        """Return spatial stride for each stage"""
        return self.strides.copy()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through Mock Swin Transformer

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of 4 feature maps from different stages
        """
        # Stage 0: Input -> H/4, W/4
        f0 = self.stage0(x)

        # Stage 1: F0 -> H/8, W/8
        f1 = self.stage1(f0)

        # Stage 2: F1 -> H/16, W/16
        f2 = self.stage2(f1)

        # Stage 3: F2 -> H/32, W/32
        f3 = self.stage3(f2)

        features = [f0, f1, f2, f3]

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
                print(f"Warning: Mock Stage {i} shape mismatch. Got {feat.shape}, expected {expected}")

        return features


class MockSwinVisionTransformer(nn.Module):
    """
    Mock Swin-ViT compatibility wrapper

    Provides the same interface as SwinVisionTransformer but uses MockSwinBackbone
    """

    def __init__(
        self,
        channels: int = 5,
        embed_dim: int = 96,
        use_stage: int = 2,
        multi_scale: bool = False,
        native_input: bool = False,
    ):
        """
        Initialize Mock Swin-ViT wrapper

        Args:
            channels: Number of input channels
            embed_dim: Base embedding dimension
            use_stage: Which stage to use for single-scale mode
            multi_scale: Whether to return all stages
            native_input: Whether to use native input (no resizing)
        """
        super().__init__()

        self.mock_swin = MockSwinBackbone(
            channels=channels,
            embed_dim=embed_dim
        )

        self.use_stage = use_stage
        self.multi_scale = multi_scale
        self.native_input = native_input

        # Set interface properties to match SwinVisionTransformer
        if multi_scale:
            self.d_model = 256  # Unified decoder channel dimension
        else:
            self.d_model = self.mock_swin.feature_dims[use_stage]

        self.patch_size = (4, 4)
        self.patch_stride = (4, 4)
        self.image_size = None

        print(f"[OK] Created Mock SwinVisionTransformer")
        if multi_scale:
            print(f"  - Mode: Multi-scale (all stages)")
            print(f"  - Feature dims: {self.mock_swin.feature_dims}")
        else:
            print(f"  - Mode: Single-scale (stage {use_stage})")
            print(f"  - Feature dim: {self.d_model}")

    @torch.jit.ignore
    def no_weight_decay(self):
        """Compatibility with RangeViT weight decay exclusion"""
        return set()

    def get_grid_size(self, H: int, W: int):
        """Get grid size for given input dimensions"""
        stride = self.mock_swin.strides[self.use_stage]
        return (H // stride, W // stride)

    def get_actual_grid_size(self):
        """Get actual grid size from last forward pass"""
        if hasattr(self, '_last_grid_size'):
            return self._last_grid_size
        return None

    def forward(self, im: torch.Tensor, return_features: bool = False):
        """
        Forward pass compatible with SwinVisionTransformer interface

        Args:
            im: Input tensor (B, C, H, W)
            return_features: Always True for compatibility

        Returns:
            For single-scale mode: (tokens_with_cls, skip)
            For multi-scale mode: (multi_scale_features, skip)
        """
        B, C, H, W = im.shape
        self.image_size = (H, W)

        # Get all hierarchical features from mock Swin
        multi_scale_features = self.mock_swin(im)

        if self.multi_scale:
            # Multi-scale mode - return all feature maps
            return multi_scale_features, None
        else:
            # Single-scale mode - return selected stage in ViT format
            selected_feature = multi_scale_features[self.use_stage]
            feat_B, feat_D, feat_H, feat_W = selected_feature.shape

            # Convert to token format like ViT
            tokens = selected_feature.flatten(2).transpose(1, 2)  # (B, N, D)

            # Store grid size
            self._last_grid_size = (feat_H, feat_W)

            # Add dummy CLS token
            cls_token = torch.zeros(feat_B, 1, feat_D, device=tokens.device, dtype=tokens.dtype)
            tokens_with_cls = torch.cat([cls_token, tokens], dim=1)  # (B, N+1, D)

            return tokens_with_cls, None


def create_mock_swin_backbone(
    model_name: str,
    channels: int = 3,
    pretrained: bool = False,
    **kwargs
) -> MockSwinBackbone:
    """
    Factory function to create Mock Swin backbone

    Args:
        model_name: Model name (ignored for mock)
        channels: Number of input channels
        pretrained: Pretrained flag (ignored for mock)
        **kwargs: Additional arguments

    Returns:
        MockSwinBackbone instance
    """
    print(f"[INFO] Creating Mock Swin backbone (model_name={model_name} ignored)")

    return MockSwinBackbone(
        channels=channels,
        embed_dim=96,  # Swin-Tiny default
        **kwargs
    )


if __name__ == '__main__':
    """
    Test Mock Swin Backbone
    """
    print("Testing Mock Swin Backbone...")

    # Test configuration
    batch_size = 2
    channels = 5
    height, width = 32, 384  # Range image dimensions

    # Create test input
    x = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {x.shape}")

    try:
        # Test backbone
        backbone = create_mock_swin_backbone('mock_swin_tiny', channels=channels)

        with torch.no_grad():
            features = backbone(x)

        print(f"[OK] Mock backbone forward pass successful")
        for i, feat in enumerate(features):
            print(f"  Stage {i}: {feat.shape}")

        # Test wrapper in both modes
        print("\nTesting Mock wrapper - single scale:")
        wrapper_single = MockSwinVisionTransformer(
            channels=channels,
            multi_scale=False,
            use_stage=2
        )

        with torch.no_grad():
            tokens, skip = wrapper_single(x)
        print(f"  Tokens: {tokens.shape}")

        print("\nTesting Mock wrapper - multi scale:")
        wrapper_multi = MockSwinVisionTransformer(
            channels=channels,
            multi_scale=True
        )

        with torch.no_grad():
            multi_features, skip = wrapper_multi(x)
        print(f"  Multi-scale features: {len(multi_features)} stages")

        print("\n[SUCCESS] All Mock Swin tests passed!")

    except Exception as e:
        print(f"[FAIL] Mock Swin test failed: {e}")
        import traceback
        traceback.print_exc()