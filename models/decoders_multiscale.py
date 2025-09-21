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
from einops import rearrange
from typing import List, Tuple

from .model_utils import get_grid_size_2d, init_weights


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) for capturing multi-scale context information.

    Takes the highest-level feature map (F3 from Swin) and applies adaptive pooling
    at multiple scales to capture global context at different granularities.

    Reference: Pyramid Scene Parsing Network (PSPNet)
    """

    def __init__(
        self,
        in_channels: int = 768,         # F3 channels for Swin-Tiny
        pool_scales: List[int] = [1, 2, 3, 6],  # Pooling scales
        out_channels: int = 256,        # Final output channels
        reduce_channels: int = 192      # Intermediate channels after pooling
    ):
        """
        Initialize Pyramid Pooling Module

        Args:
            in_channels: Input channel dimension (typically F3 channels)
            pool_scales: List of pooling scales for adaptive pooling
            out_channels: Final output channel dimension
            reduce_channels: Intermediate channel dimension after each pooling
        """
        super().__init__()

        self.in_channels = in_channels
        self.pool_scales = pool_scales
        self.out_channels = out_channels
        self.reduce_channels = reduce_channels

        # Pooling branches: one for each scale
        self.pool_branches = nn.ModuleList()
        for scale in pool_scales:
            branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),                    # Adaptive pooling to scale×scale
                nn.Conv2d(in_channels, reduce_channels, 1),    # 1×1 conv to reduce channels
                nn.BatchNorm2d(reduce_channels),
                nn.ReLU(inplace=True)
            )
            self.pool_branches.append(branch)

        # Process original features (no pooling)
        self.original_branch = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, 1),
            nn.BatchNorm2d(reduce_channels),
            nn.ReLU(inplace=True)
        )

        # Final fusion: concatenated features → output channels
        # Total input channels: original + pooled = reduce_channels * (1 + len(pool_scales))
        total_channels = reduce_channels * (1 + len(pool_scales))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Pyramid Pooling Module

        Args:
            x: Input feature map (B, in_channels, H, W)

        Returns:
            Enhanced feature map (B, out_channels, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # Process original features
        original_features = self.original_branch(x)  # (B, reduce_channels, H, W)

        # Process pooled features
        pooled_features = []
        for branch in self.pool_branches:
            # Apply pooling and 1×1 conv
            pooled = branch(x)  # (B, reduce_channels, scale, scale)

            # Upsample back to original size
            upsampled = F.interpolate(
                pooled,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )  # (B, reduce_channels, H, W)

            pooled_features.append(upsampled)

        # Concatenate all features
        all_features = [original_features] + pooled_features
        concatenated = torch.cat(all_features, dim=1)  # (B, total_channels, H, W)

        # Final fusion
        output = self.fusion_conv(concatenated)  # (B, out_channels, H, W)

        return output


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module for FPN-style top-down feature fusion.

    Combines high-level semantic features with low-level spatial features
    through lateral connections and top-down paths.
    """

    def __init__(self, high_channels: int, low_channels: int, out_channels: int = 256):
        """
        Initialize Feature Fusion Module

        Args:
            high_channels: Channels from higher-level (lower resolution) features
            low_channels: Channels from lower-level (higher resolution) features
            out_channels: Output channel dimension
        """
        super().__init__()

        # Lateral connection: reduce low-level feature channels
        self.lateral_conv = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Top-down connection: process high-level features
        self.top_down_conv = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Post-fusion refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.apply(init_weights)

    def forward(self, high_feat: torch.Tensor, low_feat: torch.Tensor) -> torch.Tensor:
        """
        Fuse high-level and low-level features

        Args:
            high_feat: High-level features (B, high_channels, H_high, W_high)
            low_feat: Low-level features (B, low_channels, H_low, W_low)

        Returns:
            Fused features at low_feat resolution (B, out_channels, H_low, W_low)
        """
        # Get target size from low-level features
        _, _, H_target, W_target = low_feat.shape

        # Process high-level features and upsample
        high_processed = self.top_down_conv(high_feat)
        high_upsampled = F.interpolate(
            high_processed,
            size=(H_target, W_target),
            mode='bilinear',
            align_corners=False
        )

        # Process low-level features
        low_processed = self.lateral_conv(low_feat)

        # Element-wise addition
        fused = high_upsampled + low_processed

        # Refinement
        output = self.refine_conv(fused)

        return output


class DecoderMultiScaleFPN(nn.Module):
    """
    Multi-Scale Feature Pyramid Network (FPN) Decoder for Swin Transformer features.

    Processes hierarchical features from Swin Transformer backbone:
    - F0: (B, 96,  H/4,  W/4)   - Stage 0, finest spatial resolution
    - F1: (B, 192, H/8,  W/8)   - Stage 1
    - F2: (B, 384, H/16, W/16)  - Stage 2
    - F3: (B, 768, H/32, W/32)  - Stage 3, coarsest resolution, highest semantics

    Uses PPM on F3 for global context, then FPN for multi-scale fusion.
    """

    def __init__(
        self,
        n_cls: int,
        swin_channels: List[int] = [96, 192, 384, 768],  # Swin-Tiny channel progression
        pyramid_channels: int = 256,                     # Unified FPN channel dimension
        use_ppm: bool = True,                           # Enable Pyramid Pooling Module
        ppm_scales: List[int] = [1, 2, 3, 6],          # PPM pooling scales
        patch_size: Tuple[int, int] = (4, 4),          # For compatibility (not used in multi-scale)
        patch_stride: Tuple[int, int] = None            # For compatibility (not used in multi-scale)
    ):
        """
        Initialize Multi-Scale FPN Decoder

        Args:
            n_cls: Number of output classes
            swin_channels: Channel dimensions for each Swin stage [C0, C1, C2, C3]
            pyramid_channels: Unified channel dimension in FPN
            use_ppm: Whether to use Pyramid Pooling Module on F3
            ppm_scales: Pooling scales for PPM
            patch_size: Patch size (for compatibility with existing interface)
            patch_stride: Patch stride (for compatibility)
        """
        super().__init__()

        self.n_cls = n_cls
        self.swin_channels = swin_channels
        self.pyramid_channels = pyramid_channels
        self.use_ppm = use_ppm

        # Store for compatibility with existing decoder interface
        self.patch_size = patch_size
        self.patch_stride = patch_stride or patch_size

        # Pyramid Pooling Module for F3 (highest semantic level)
        if use_ppm:
            self.ppm = PyramidPoolingModule(
                in_channels=swin_channels[3],  # F3 channels (768 for Swin-Tiny)
                pool_scales=ppm_scales,
                out_channels=pyramid_channels,
                reduce_channels=pyramid_channels // 4 * 3  # 192 for 256 output
            )
        else:
            # Simple 1×1 conv if no PPM
            self.ppm = nn.Sequential(
                nn.Conv2d(swin_channels[3], pyramid_channels, 1),
                nn.BatchNorm2d(pyramid_channels),
                nn.ReLU(inplace=True)
            )

        # FPN fusion modules: top-down pathway
        # F3→F2 fusion
        self.fusion_3_2 = FeatureFusionModule(
            high_channels=pyramid_channels,   # F3 after PPM
            low_channels=swin_channels[2],    # F2 channels (384)
            out_channels=pyramid_channels
        )

        # F2→F1 fusion
        self.fusion_2_1 = FeatureFusionModule(
            high_channels=pyramid_channels,   # F2 after fusion
            low_channels=swin_channels[1],    # F1 channels (192)
            out_channels=pyramid_channels
        )

        # F1→F0 fusion
        self.fusion_1_0 = FeatureFusionModule(
            high_channels=pyramid_channels,   # F1 after fusion
            low_channels=swin_channels[0],    # F0 channels (96)
            out_channels=pyramid_channels
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(pyramid_channels, pyramid_channels, 3, padding=1),
            nn.BatchNorm2d(pyramid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(pyramid_channels, n_cls, 1)
        )

        # Initialize weights
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Compatibility with RangeViT weight decay exclusion"""
        return set()

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        im_size: Tuple[int, int],
        skip: torch.Tensor = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through Multi-Scale FPN Decoder

        Args:
            multi_scale_features: List of [F0, F1, F2, F3] from Swin backbone
            im_size: Original image size (H, W) - for final upsampling
            skip: Skip connections (unused in this version, for compatibility)
            return_features: If True, return features before classification

        Returns:
            Segmentation logits (B, n_cls, H, W) at original image resolution
        """
        # Validate inputs
        assert len(multi_scale_features) == 4, f"Expected 4 feature maps, got {len(multi_scale_features)}"

        F0, F1, F2, F3 = multi_scale_features
        H_orig, W_orig = im_size

        # Validate feature dimensions
        expected_channels = self.swin_channels
        for i, (feat, expected_ch) in enumerate(zip(multi_scale_features, expected_channels)):
            B, C, H, W = feat.shape
            assert C == expected_ch, f"F{i}: expected {expected_ch} channels, got {C}"

        # Stage 1: Process F3 with PPM for global context
        P3 = self.ppm(F3)  # (B, pyramid_channels, H/32, W/32)

        # Stage 2: Top-down FPN fusion
        # F3→F2 fusion
        P2 = self.fusion_3_2(P3, F2)  # (B, pyramid_channels, H/16, W/16)

        # F2→F1 fusion
        P1 = self.fusion_2_1(P2, F1)  # (B, pyramid_channels, H/8, W/8)

        # F1→F0 fusion
        P0 = self.fusion_1_0(P1, F0)  # (B, pyramid_channels, H/4, W/4)

        # Return features if requested (for visualization/analysis)
        if return_features:
            return P0

        # Stage 3: Classification
        logits = self.classifier(P0)  # (B, n_cls, H/4, W/4)

        # Stage 4: Upsample to original resolution
        # From H/4×W/4 to H×W (4× upsampling)
        final_logits = F.interpolate(
            logits,
            size=(H_orig, W_orig),
            mode='bilinear',
            align_corners=False
        )  # (B, n_cls, H, W)

        return final_logits


def create_multiscale_decoder(encoder, decoder_cfg):
    """
    Factory function to create multi-scale decoder

    Args:
        encoder: Swin encoder (for compatibility)
        decoder_cfg: Decoder configuration dictionary

    Returns:
        DecoderMultiScaleFPN instance
    """
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop('name')

    if name == 'multi_scale_fpn':
        # Extract Swin-specific parameters
        if hasattr(encoder, 'swin'):
            # Multi-scale Swin encoder
            swin_channels = encoder.swin.get_feature_dims()
        else:
            # Default Swin-Tiny channels
            swin_channels = [96, 192, 384, 768]

        decoder = DecoderMultiScaleFPN(
            swin_channels=swin_channels,
            **decoder_cfg
        )

        return decoder
    else:
        raise ValueError(f'Unknown multi-scale decoder: {name}')


if __name__ == '__main__':
    """
    Test multi-scale decoder components
    """
    print("Testing Multi-Scale Decoder Components...")

    # Test PPM
    print("\n1. Testing PyramidPoolingModule...")
    ppm = PyramidPoolingModule(in_channels=768, out_channels=256)
    F3_test = torch.randn(2, 768, 1, 12)  # Typical F3 for range images (32×384 → 1×12 at stride 32)
    ppm_output = ppm(F3_test)
    print(f"   PPM Input: {F3_test.shape} → Output: {ppm_output.shape}")
    assert ppm_output.shape == (2, 256, 1, 12), f"PPM output shape mismatch: {ppm_output.shape}"

    # Test Feature Fusion
    print("\n2. Testing FeatureFusionModule...")
    fusion = FeatureFusionModule(high_channels=256, low_channels=384, out_channels=256)
    high_feat = torch.randn(2, 256, 1, 12)  # F3 processed
    low_feat = torch.randn(2, 384, 2, 24)   # F2 raw
    fusion_output = fusion(high_feat, low_feat)
    print(f"   Fusion High: {high_feat.shape} + Low: {low_feat.shape} → Output: {fusion_output.shape}")
    assert fusion_output.shape == (2, 256, 2, 24), f"Fusion output shape mismatch: {fusion_output.shape}"

    # Test Multi-Scale Decoder
    print("\n3. Testing DecoderMultiScaleFPN...")
    decoder = DecoderMultiScaleFPN(n_cls=17, pyramid_channels=256)

    # Create mock Swin features for range image (32×384)
    F0 = torch.randn(2, 96, 8, 96)    # H/4=8, W/4=96
    F1 = torch.randn(2, 192, 4, 48)   # H/8=4, W/8=48
    F2 = torch.randn(2, 384, 2, 24)   # H/16=2, W/16=24
    F3 = torch.randn(2, 768, 1, 12)   # H/32=1, W/32=12

    multi_scale_features = [F0, F1, F2, F3]

    # Test forward pass
    output = decoder(multi_scale_features, im_size=(32, 384))
    print(f"   Decoder Input: {[f.shape for f in multi_scale_features]}")
    print(f"   Decoder Output: {output.shape}")
    assert output.shape == (2, 17, 32, 384), f"Decoder output shape mismatch: {output.shape}"

    # Test feature return
    features = decoder(multi_scale_features, im_size=(32, 384), return_features=True)
    print(f"   Feature Output: {features.shape}")
    assert features.shape == (2, 256, 8, 96), f"Feature output shape mismatch: {features.shape}"

    print("\n✅ All tests passed! Multi-scale decoder is working correctly.")

    # Memory usage estimation
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n📊 Decoder Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")