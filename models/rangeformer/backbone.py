# RangeFormer Backbone Implementation
# Reference: Kong et al., 2023 - RangeFormer: Toward Fast and Accurate 3D Object Detection

from typing import List

import torch
import torch.nn as nn

from ..blocks import FeedForward


class REM(nn.Module):
    """
    Range Embedding Module (REM)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 6, f"REM expects 6 channels, got {x.shape[1]}"
        return self.rem(x)


class PatchEmbedOverlap(nn.Module):
    """
    Overlapping patch embedding (3x3 conv).
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class TransformerBlock2D(nn.Module):
    """
    Transformer block with spatial-reduction attention and depthwise conv branch.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, sr_ratio: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio,
                                groups=dim, bias=False)
            self.sr_norm = nn.LayerNorm(dim)
        else:
            self.sr = None

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, hidden_dim, dropout=0.0)

        self.conv_branch = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        x_flat = x.view(B, C, N).permute(0, 2, 1).contiguous()
        q_input = self.norm_q(x_flat)
        q = self.q(q_input).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_sr = self.sr(x).view(B, C, -1).permute(0, 2, 1).contiguous()
            x_sr = self.sr_norm(x_sr)
        else:
            x_sr = q_input

        kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj(attn_out)

        x2 = x_flat + attn_out
        x2_norm = self.norm2(x2)
        ffn_out = self.mlp(x2_norm)

        conv_out = self.conv_branch(x).view(B, C, N).permute(0, 2, 1).contiguous()

        out = x2 + ffn_out + conv_out
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return out


class RangeFormerBackbone(nn.Module):
    def __init__(self,
                 H: int,
                 W: int,
                 num_classes: int,
                 depths: List[int] = [2, 2, 6, 2],
                 stage_channels: List[int] = [128, 128, 320, 512],
                 heads: List[int] = [3, 4, 6, 3],
                 mlp_ratio: float = 4.0,
                 sr_ratios: List[int] = [8, 4, 2, 1]):
        super().__init__()

        assert len(stage_channels) == 4
        assert len(heads) == 4
        assert len(depths) == 4
        assert len(sr_ratios) == 4

        self.H, self.W = H, W

        self.rem = REM()
        self.patch1 = PatchEmbedOverlap(128, stage_channels[0], stride=1)
        self.patch2 = PatchEmbedOverlap(stage_channels[0], stage_channels[1], stride=2)
        self.patch3 = PatchEmbedOverlap(stage_channels[1], stage_channels[2], stride=2)
        self.patch4 = PatchEmbedOverlap(stage_channels[2], stage_channels[3], stride=2)

        self.stage1_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[0], num_heads=heads[0], mlp_ratio=mlp_ratio, sr_ratio=sr_ratios[0])
            for _ in range(depths[0])
        ])
        self.stage2_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[1], num_heads=heads[1], mlp_ratio=mlp_ratio, sr_ratio=sr_ratios[1])
            for _ in range(depths[1])
        ])
        self.stage3_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[2], num_heads=heads[2], mlp_ratio=mlp_ratio, sr_ratio=sr_ratios[2])
            for _ in range(depths[2])
        ])
        self.stage4_blocks = nn.ModuleList([
            TransformerBlock2D(stage_channels[3], num_heads=heads[3], mlp_ratio=mlp_ratio, sr_ratio=sr_ratios[3])
            for _ in range(depths[3])
        ])

    def forward(self, x: torch.Tensor):
        assert x.shape[1] == 6, f"Expected 6 input channels, got {x.shape[1]}"

        x = self.rem(x)

        x1 = self.patch1(x)
        for blk in self.stage1_blocks:
            x1 = blk(x1)

        x2 = self.patch2(x1)
        for blk in self.stage2_blocks:
            x2 = blk(x2)

        x3 = self.patch3(x2)
        for blk in self.stage3_blocks:
            x3 = blk(x3)

        x4 = self.patch4(x3)
        for blk in self.stage4_blocks:
            x4 = blk(x4)

        return [x1, x2, x3, x4]


if __name__ == "__main__":
    model = RangeFormerBackbone(H=64, W=512, num_classes=19)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Params: {total_params:.2f}M")
