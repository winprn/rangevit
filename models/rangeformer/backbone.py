# RangeFormer Backbone Implementation
# Reference: Kong et al. 2023 - RangeFormer: Toward Fast and Accurate 3D Object Detection
# Reuses components from RangeViT codebase where possible

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Import reusable components from RangeViT
from ..blocks import Attention, FeedForward


class REM(nn.Module):
    """
    Range Embedding Module: 6 -> 64 -> 128 -> 128 (1x1 convs) with BN+GELU

    This is RangeFormer-specific and replaces RangeViT's ConvStem.
    Converts 6-channel range image to 128-channel feature map.
    """
    def __init__(self):
        super().__init__()
        self.rem = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: (B, 6, H, W) range image

        Returns:
            out: (B, 128, H, W) embedded features
        """
        # Assert input shape
        assert x.dim() == 4, f"REM: Expected 4D input (B, C, H, W), got {x.dim()}D input with shape {x.shape}"
        assert x.shape[1] == 6, f"REM: Expected 6 input channels [x,y,z,depth,intensity,existence], got {x.shape[1]} channels"

        B, C, H, W = x.shape
        out = self.rem(x)  # (B, 128, H, W)

        # Assert output shape
        assert out.shape == (B, 128, H, W), f"REM: Expected output shape ({B}, 128, {H}, {W}), got {out.shape}"

        return out


class PatchEmbedOverlap(nn.Module):
    """
    3x3 overlapping patch embedding for RangeFormer.

    Different from RangeViT's PatchEmbedding which uses larger patches (16x16).
    RangeFormer uses small overlapping patches for hierarchical processing.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: (B, in_ch, H, W) input features

        Returns:
            out: (B, out_ch, H//stride, W//stride) output features
        """
        # Assert input shape
        assert x.dim() == 4, f"PatchEmbedOverlap: Expected 4D input (B, C, H, W), got {x.dim()}D with shape {x.shape}"
        assert x.shape[1] == self.in_ch, f"PatchEmbedOverlap: Expected {self.in_ch} input channels, got {x.shape[1]}"

        B, C, H, W = x.shape
        x = self.proj(x)
        x = self.norm(x)
        out = self.act(x)

        # Assert output shape
        expected_H = (H + 2 * 1 - 3) // self.stride + 1  # With padding=1, kernel=3
        expected_W = (W + 2 * 1 - 3) // self.stride + 1
        assert out.shape == (B, self.out_ch, expected_H, expected_W), \
            f"PatchEmbedOverlap: Expected output shape ({B}, {self.out_ch}, {expected_H}, {expected_W}), got {out.shape}"

        return out


class TransformerBlock2D(nn.Module):
    """
    Transformer block for 2D range image processing.

    Key differences from RangeViT's Block:
    - Operates on 2D spatial features (B, C, H, W) instead of sequences (B, N, C)
    - Adds 3x3 conv branch in FFN for local spatial information
    - No CLS token handling

    Reuses Attention and FeedForward from RangeViT's blocks.py
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)

        # Reuse RangeViT's Attention module
        self.attn = Attention(dim, num_heads, dropout=0.0)

        self.norm2 = nn.LayerNorm(dim)

        # Reuse RangeViT's FeedForward module
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, hidden_dim, dropout=0.0)

        # RangeFormer-specific: convolution branch in FFN to inject local spatial info
        self.conv_branch = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input features where C = self.dim

        Returns:
            out: (B, C, H, W) output features (same shape as input)
        """
        # Assert input shape
        assert x.dim() == 4, f"TransformerBlock2D: Expected 4D input (B, C, H, W), got {x.dim()}D with shape {x.shape}"
        B, C, H, W = x.shape
        assert C == self.dim, f"TransformerBlock2D: Expected {self.dim} channels, got {C}"
        assert C % self.num_heads == 0, f"TransformerBlock2D: dim ({C}) must be divisible by num_heads ({self.num_heads})"

        N = H * W

        # Flatten spatial dims and transpose to (B, N, C) for attention
        x_flat = x.view(B, C, N).permute(0, 2, 1).contiguous()  # (B, N, C)
        assert x_flat.shape == (B, N, C), f"TransformerBlock2D: Flattening error, expected ({B}, {N}, {C}), got {x_flat.shape}"

        # Attention block
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm)  # (B, N, C)
        assert attn_out.shape == (B, N, C), f"TransformerBlock2D: Attention output shape mismatch, expected ({B}, {N}, {C}), got {attn_out.shape}"
        x2 = x_flat + attn_out

        # FFN block
        x2_norm = self.norm2(x2)
        ffn = self.mlp(x2_norm)  # (B, N, C)
        assert ffn.shape == (B, N, C), f"TransformerBlock2D: FFN output shape mismatch, expected ({B}, {N}, {C}), got {ffn.shape}"

        # Conv branch: operate on original spatial map
        x_spatial = x  # (B, C, H, W)
        conv_out = self.conv_branch(x_spatial)  # (B, C, H, W)
        assert conv_out.shape == (B, C, H, W), f"TransformerBlock2D: Conv branch output shape mismatch, expected ({B}, {C}, {H}, {W}), got {conv_out.shape}"
        conv_out_flat = conv_out.view(B, C, N).permute(0, 2, 1).contiguous()

        # Residual add
        x_out = x2 + ffn + conv_out_flat

        # Reshape back to (B, C, H, W)
        x_out = x_out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        assert x_out.shape == x.shape, f"TransformerBlock2D: Output shape {x_out.shape} != Input shape {x.shape}"

        return x_out


class RangeFormerBackbone(nn.Module):
    """
    Hierarchical backbone for RangeFormer.

    Architecture:
    - REM: 6 -> 128 channels
    - Stage 1: 128 channels, stride 1 (H, W)
    - Stage 2: 128 channels, stride 2 (H/2, W/2)
    - Stage 3: 320 channels, stride 2 (H/4, W/4)
    - Stage 4: 512 channels, stride 2 (H/8, W/8)

    Key differences from RangeViT:
    - Multi-scale hierarchical structure vs single-scale ViT
    - Progressive downsampling vs fixed patch size
    - Stage-wise channel expansion vs uniform d_model
    """
    def __init__(self,
                 H: int,
                 W: int,
                 num_classes: int,
                 depths: List[int] = [2, 2, 6, 2],
                 stage_channels: List[int] = [128, 128, 320, 512],
                 heads: List[int] = [3, 4, 6, 3]):
        super().__init__()
        assert len(depths) == 4, f"RangeFormerBackbone: Expected 4 depth values, got {len(depths)}"
        assert len(stage_channels) == 4, f"RangeFormerBackbone: Expected 4 stage_channels values, got {len(stage_channels)}"
        assert len(heads) == 4, f"RangeFormerBackbone: Expected 4 heads values, got {len(heads)}"

        # Verify channels are divisible by heads
        # for i, (ch, h) in enumerate(zip(stage_channels, heads)):
        #     assert ch % h == 0, f"RangeFormerBackbone: Stage {i+1} channels ({ch}) must be divisible by heads ({h})"

        self.H = H
        self.W = W
        self.stage_channels = stage_channels
        self.depths = depths

        # REM: 6 -> 128 channels
        self.rem = REM()

        # Patch embedding layers for each stage
        self.patch1 = PatchEmbedOverlap(128, stage_channels[0], stride=1)
        self.patch2 = PatchEmbedOverlap(stage_channels[0], stage_channels[1], stride=2)
        self.patch3 = PatchEmbedOverlap(stage_channels[1], stage_channels[2], stride=2)
        self.patch4 = PatchEmbedOverlap(stage_channels[2], stage_channels[3], stride=2)

        # Transformer stacks for each stage
        self.stage1_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[0], num_heads=heads[0])
            for _ in range(depths[0])
        ])
        self.stage2_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[1], num_heads=heads[1])
            for _ in range(depths[1])
        ])
        self.stage3_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[2], num_heads=heads[2])
            for _ in range(depths[2])
        ])
        self.stage4_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[3], num_heads=heads[3])
            for _ in range(depths[3])
        ])

    def forward(self, x):
        """
        Args:
            x: (B, 6, H, W) range image

        Returns:
            list of stage features: [F1, F2, F3, F4] with shapes:
                F1: (B, C1, H, W)
                F2: (B, C2, H/2, W/2)
                F3: (B, C3, H/4, W/4)
                F4: (B, C4, H/8, W/8)
        """
        # Assert input shape
        assert x.dim() == 4, f"RangeFormerBackbone: Expected 4D input (B, C, H, W), got {x.dim()}D with shape {x.shape}"
        assert x.shape[1] == 6, f"RangeFormerBackbone: Expected 6 input channels, got {x.shape[1]}"
        B, _, H, W = x.shape

        # Range Embedding Module
        x = self.rem(x)  # (B, 128, H, W)
        assert x.shape == (B, 128, H, W), f"RangeFormerBackbone: REM output shape mismatch, expected ({B}, 128, {H}, {W}), got {x.shape}"

        # Stage 1: stride 1, maintain spatial resolution
        x1 = self.patch1(x)
        assert x1.shape[1] == self.stage_channels[0], f"RangeFormerBackbone: Stage1 patch embed channel mismatch"
        assert x1.shape[2] == H and x1.shape[3] == W, f"RangeFormerBackbone: Stage1 should maintain spatial size, expected ({H}, {W}), got ({x1.shape[2]}, {x1.shape[3]})"

        for i, blk in enumerate(self.stage1_blocks):
            x1_before = x1.shape
            x1 = blk(x1)
            assert x1.shape == x1_before, f"RangeFormerBackbone: Stage1 block {i} changed shape from {x1_before} to {x1.shape}"

        # Stage 2: stride 2, downsample
        x2 = self.patch2(x1)
        assert x2.shape[1] == self.stage_channels[1], f"RangeFormerBackbone: Stage2 channel mismatch, expected {self.stage_channels[1]}, got {x2.shape[1]}"
        expected_H2 = H // 2
        expected_W2 = W // 2
        assert x2.shape[2] == expected_H2 and x2.shape[3] == expected_W2, \
            f"RangeFormerBackbone: Stage2 spatial size mismatch, expected ({expected_H2}, {expected_W2}), got ({x2.shape[2]}, {x2.shape[3]})"

        for i, blk in enumerate(self.stage2_blocks):
            x2_before = x2.shape
            x2 = blk(x2)
            assert x2.shape == x2_before, f"RangeFormerBackbone: Stage2 block {i} changed shape from {x2_before} to {x2.shape}"

        # Stage 3: stride 2, downsample
        x3 = self.patch3(x2)
        assert x3.shape[1] == self.stage_channels[2], f"RangeFormerBackbone: Stage3 channel mismatch, expected {self.stage_channels[2]}, got {x3.shape[1]}"
        expected_H3 = H // 4
        expected_W3 = W // 4
        assert x3.shape[2] == expected_H3 and x3.shape[3] == expected_W3, \
            f"RangeFormerBackbone: Stage3 spatial size mismatch, expected ({expected_H3}, {expected_W3}), got ({x3.shape[2]}, {x3.shape[3]})"

        for i, blk in enumerate(self.stage3_blocks):
            x3_before = x3.shape
            x3 = blk(x3)
            assert x3.shape == x3_before, f"RangeFormerBackbone: Stage3 block {i} changed shape from {x3_before} to {x3.shape}"

        # Stage 4: stride 2, downsample
        x4 = self.patch4(x3)
        assert x4.shape[1] == self.stage_channels[3], f"RangeFormerBackbone: Stage4 channel mismatch, expected {self.stage_channels[3]}, got {x4.shape[1]}"
        expected_H4 = H // 8
        expected_W4 = W // 8
        assert x4.shape[2] == expected_H4 and x4.shape[3] == expected_W4, \
            f"RangeFormerBackbone: Stage4 spatial size mismatch, expected ({expected_H4}, {expected_W4}), got ({x4.shape[2]}, {x4.shape[3]})"

        for i, blk in enumerate(self.stage4_blocks):
            x4_before = x4.shape
            x4 = blk(x4)
            assert x4.shape == x4_before, f"RangeFormerBackbone: Stage4 block {i} changed shape from {x4_before} to {x4.shape}"

        return [x1, x2, x3, x4]
