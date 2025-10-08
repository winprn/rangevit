# RangeFormer: Main Model Implementation
# Reference: Kong et al. 2023 - RangeFormer: Toward Fast and Accurate 3D Object Detection
# Built on top of RangeViT codebase, reusing components where applicable

import torch
import torch.nn as nn
from typing import List

from .backbone import RangeFormerBackbone
from .decoder import SegmentationHead


class RangeFormer(nn.Module):
    """
    RangeFormer model for range image semantic segmentation.

    Architecture:
    - Input: 6-channel range image [x, y, z, depth, intensity, existence]
    - Backbone: REM + 4-stage hierarchical transformer encoder
    - Decoder: Multi-scale feature fusion with auxiliary heads
    - Output: Per-pixel semantic predictions

    Key differences from RangeViT:
    - Hierarchical multi-scale vs single-scale ViT
    - Overlapping patches (3x3) vs large patches (16x16)
    - Multi-scale feature fusion decoder vs single-scale decoder
    - Auxiliary supervision at each stage
    """
    def __init__(self,
                 H: int,
                 W: int,
                 num_classes: int,
                 depths: List[int] = [2, 2, 6, 2],
                 stage_channels: List[int] = [128, 256, 384, 512],
                 heads: List[int] = [4, 4, 8, 16],
                 decoder_unify_ch: int = 512,
                 mlp_ratio: float = 4.0,
                 sr_ratios: List[int] = [8, 4, 2, 1]):
        """
        Args:
            H: height of range image (vertical resolution)
            W: width of range image (horizontal resolution)
            num_classes: number of semantic classes
            depths: number of transformer blocks per stage [stage1, stage2, stage3, stage4]
            stage_channels: channel dimensions per stage [C1, C2, C3, C4]
            heads: number of attention heads per stage
            decoder_unify_ch: unified channel dimension in decoder (default 256)
        """
        super().__init__()

        # Assert valid inputs
        assert H > 0 and W > 0, f"RangeFormer: H and W must be positive, got H={H}, W={W}"
        assert H % 8 == 0, f"RangeFormer: H must be divisible by 8 (for 3 downsampling stages), got H={H}"
        assert W % 8 == 0, f"RangeFormer: W must be divisible by 8 (for 3 downsampling stages), got W={W}"
        assert num_classes > 0, f"RangeFormer: num_classes must be positive, got {num_classes}"
        assert len(depths) == 4, f"RangeFormer: depths must have 4 values, got {len(depths)}"
        assert len(stage_channels) == 4, f"RangeFormer: stage_channels must have 4 values, got {len(stage_channels)}"
        assert len(heads) == 4, f"RangeFormer: heads must have 4 values, got {len(heads)}"
        assert decoder_unify_ch > 0, f"RangeFormer: decoder_unify_ch must be positive, got {decoder_unify_ch}"
        assert len(sr_ratios) == 4, f"RangeFormer: sr_ratios must have 4 values, got {len(sr_ratios)}"

        self.H = H
        self.W = W
        self.num_classes = num_classes
        self.depths = depths
        self.stage_channels = stage_channels
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.sr_ratios = sr_ratios

        # Backbone: REM + hierarchical transformer encoder
        self.backbone = RangeFormerBackbone(
            H=H,
            W=W,
            num_classes=num_classes,
            depths=depths,
            stage_channels=stage_channels,
            heads=heads,
            mlp_ratio=mlp_ratio,
            sr_ratios=sr_ratios
        )

        # Decoder: multi-scale feature fusion head
        self.head = SegmentationHead(
            stage_channels=stage_channels,
            out_ch_unify=decoder_unify_ch,
            num_classes=num_classes,
            H=H,
            W=W
        )

    def forward(self, rv: torch.Tensor):
        """
        Forward pass through RangeFormer.

        Args:
            rv: (B, 6, H, W) range image tensor
                Channels: [x, y, z, depth, intensity, existence]

        Returns:
            logits_main: (B, num_classes, H, W) - main semantic predictions
            aux_logits: list of (B, num_classes, H, W) - auxiliary predictions
        """
        # Assert input shape
        assert rv.dim() == 4, f"RangeFormer: Expected 4D input (B, 6, H, W), got {rv.dim()}D with shape {rv.shape}"
        assert rv.shape[1] == 6, f"RangeFormer: Expected 6 input channels [x,y,z,depth,intensity,existence], got {rv.shape[1]}"
        B, _, H, W = rv.shape

        # Extract multi-scale features
        features = self.backbone(rv)  # [F1, F2, F3, F4]

        # Assert backbone output
        assert isinstance(features, list), f"RangeFormer: Backbone should return list, got {type(features)}"
        assert len(features) == 4, f"RangeFormer: Expected 4 feature maps from backbone, got {len(features)}"

        # Decode to semantic predictions
        logits, auxs = self.head(features)

        # Assert final output shapes
        assert logits.shape == (B, self.num_classes, H, W), f"RangeFormer: Main output shape mismatch, expected ({B}, {self.num_classes}, {H}, {W}), got {logits.shape}"
        assert isinstance(auxs, list), f"RangeFormer: Auxiliary outputs should be list, got {type(auxs)}"
        assert len(auxs) == 4, f"RangeFormer: Expected 4 auxiliary outputs, got {len(auxs)}"

        for i, aux in enumerate(auxs):
            assert aux.shape[0] == B, f"RangeFormer: Aux {i} batch size mismatch"
            assert aux.shape[2] == H and aux.shape[3] == W, \
                f"RangeFormer: Aux {i} spatial size mismatch, expected ({H}, {W}), got ({aux.shape[2]}, {aux.shape[3]})"

        return logits, auxs

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_component(self):
        """Count parameters by component for analysis."""
        stats = {}
        stats['total'] = self.count_parameters()
        stats['backbone'] = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        stats['decoder'] = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        stats['rem'] = sum(p.numel() for p in self.backbone.rem.parameters() if p.requires_grad)
        return stats


def create_rangeformer(config: dict):
    """
    Factory function to create RangeFormer model from config.

    Args:
        config: dictionary with model configuration
            Required keys:
            - H: int, height of range image
            - W: int, width of range image
            - num_classes: int, number of semantic classes

            Optional keys:
            - depths: list of ints, default [2, 2, 6, 2]
            - stage_channels: list of ints, default [128, 128, 320, 512]
            - heads: list of ints, default [3, 4, 6, 3]
            - decoder_unify_ch: int, default 256
            - mlp_ratio: float, default 4.0
            - sr_ratios: list of ints, default [8, 4, 2, 1]

    Returns:
        RangeFormer model instance
    """
    # Required parameters
    H = config['H']
    W = config['W']
    num_classes = config['num_classes']

    # Optional parameters with defaults
    depths = config.get('depths', [2, 2, 6, 2])
    stage_channels = config.get('stage_channels', [128, 256, 384, 512])
    heads = config.get('heads', [4, 4, 8, 16])
    decoder_unify_ch = config.get('decoder_unify_ch', 512)
    mlp_ratio = config.get('mlp_ratio', 4.0)
    sr_ratios = config.get('sr_ratios', [8, 4, 2, 1])

    model = RangeFormer(
        H=H,
        W=W,
        num_classes=num_classes,
        depths=depths,
        stage_channels=stage_channels,
        heads=heads,
        decoder_unify_ch=decoder_unify_ch,
        mlp_ratio=mlp_ratio,
        sr_ratios=sr_ratios
    )

    return model


if __name__ == '__main__':
    # Quick sanity test
    H, W = 64, 1024
    num_classes = 19

    model = RangeFormer(H=H, W=W, num_classes=num_classes)
    model.eval()

    # Test forward pass
    x = torch.randn(2, 6, H, W)
    with torch.no_grad():
        logits, auxs = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Main output shape: {logits.shape}")
    print(f"Number of auxiliary outputs: {len(auxs)}")
    for i, aux in enumerate(auxs):
        print(f"  Aux {i+1} shape: {aux.shape}")

    stats = model.count_parameters_by_component()
    print("\nParameter counts:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
