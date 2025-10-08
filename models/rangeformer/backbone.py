# backbone.py
# RangeFormer Backbone Implementation (fixed)
# Reference: Kong et al., 2023 - RangeFormer: Toward Fast and Accurate 3D Object Detection
# Paper link: https://arxiv.org/pdf/2303.05367.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Import reusable components from RangeViT
from ..blocks import Attention, FeedForward


class REM(nn.Module):
    """
    Range Embedding Module (REM)
    Input:  (B, 6, H, W)
    Output: (B, 128, H, W)
    Layers: 6 -> 64 -> 128 -> 128 (1x1 convs) + BN + GELU
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
        assert x.shape[1] == 6, f"REM: expected 6 channels, got {x.shape[1]}"
        return self.rem(x)


class PatchEmbedOverlap(nn.Module):
    """
    Overlapping patch embedding (3x3 conv) for hierarchical processing.
    Args:
        in_ch  - input channels
        out_ch - output channels
        stride - stride for spatial downsampling
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class TransformerBlock2D(nn.Module):
    """
    Transformer block for 2D range view.
    Operates on (B, C, H, W) feature maps.
    Includes:
    - Multi-head self-attention (from RangeViT)
    - FeedForward MLP (from RangeViT)
    - Depthwise 3x3 conv branch (adds local spatial inductive bias)
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout=0.0)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, hidden_dim, dropout=0.0)

        # ✅ FIXED: depthwise conv (groups=dim) instead of full conv
        self.conv_branch = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dims to (B, N, C)
        x_flat = x.view(B, C, N).permute(0, 2, 1).contiguous()

        # Attention
        x1 = self.norm1(x_flat)
        attn_out, _ = self.attn(x1)
        x2 = x_flat + attn_out

        # FeedForward
        x2_norm = self.norm2(x2)
        ffn_out = self.mlp(x2_norm)

        # Conv branch: local context (depthwise)
        conv_out = self.conv_branch(x)
        conv_out_flat = conv_out.view(B, C, N).permute(0, 2, 1).contiguous()

        # Combine branches
        out = x2 + ffn_out + conv_out_flat

        # Reshape back to (B, C, H, W)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return out


class RangeFormerBackbone(nn.Module):
    """
    Hierarchical backbone for RangeFormer.
    Architecture:
        Stage 1: 128 channels, stride 1, depth 2
        Stage 2: 128 channels, stride 2, depth 2
        Stage 3: 320 channels, stride 2, depth 6
        Stage 4: 512 channels, stride 2, depth 2
    """
    def __init__(self,
                 H: int,
                 W: int,
                 num_classes: int,
                 depths: List[int] = [2, 2, 6, 2],
                 stage_channels: List[int] = [128, 128, 320, 512],
                 heads: List[int] = [3, 4, 6, 3]):
        super().__init__()

        self.H, self.W = H, W
        self.stage_channels = stage_channels
        self.depths = depths

        # Stage 0: Range Embedding
        self.rem = REM()

        # Stage 1–4 patch embeddings
        self.patch1 = PatchEmbedOverlap(128, stage_channels[0], stride=1)
        self.patch2 = PatchEmbedOverlap(stage_channels[0], stage_channels[1], stride=2)
        self.patch3 = PatchEmbedOverlap(stage_channels[1], stage_channels[2], stride=2)
        self.patch4 = PatchEmbedOverlap(stage_channels[2], stage_channels[3], stride=2)

        # Transformer blocks for each stage
        self.stage1_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[0], num_heads=heads[0]) for _ in range(depths[0])
        ])
        self.stage2_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[1], num_heads=heads[1]) for _ in range(depths[1])
        ])
        self.stage3_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[2], num_heads=heads[2]) for _ in range(depths[2])
        ])
        self.stage4_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[3], num_heads=heads[3]) for _ in range(depths[3])
        ])

    def forward(self, x):
        assert x.shape[1] == 6, f"Expected 6 input channels, got {x.shape[1]}"

        # REM: (B,6,H,W) → (B,128,H,W)
        x = self.rem(x)

        # Stage 1
        x1 = self.patch1(x)
        for blk in self.stage1_blocks:
            x1 = blk(x1)

        # Stage 2
        x2 = self.patch2(x1)
        for blk in self.stage2_blocks:
            x2 = blk(x2)

        # Stage 3
        x3 = self.patch3(x2)
        for blk in self.stage3_blocks:
            x3 = blk(x3)

        # Stage 4
        x4 = self.patch4(x3)
        for blk in self.stage4_blocks:
            x4 = blk(x4)

        # Output multi-scale features
        return [x1, x2, x3, x4]


# Quick sanity test
if __name__ == "__main__":
    model = RangeFormerBackbone(H=64, W=512, num_classes=19)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Params: {total_params:.2f}M")  # should be close to ~24M
