# RangeFormer Decoder Implementation
# Reference: Kong et al. 2023 - RangeFormer paper

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SegmentationHead(nn.Module):
    """
    Multi-scale decoder head for RangeFormer semantic segmentation.

    Architecture:
    - Channel unification: map each Fi (with dim di) -> 256 via 1x1 conv
    - Spatial unification: upsample Fi (for i>1) to HxW using bilinear interp
    - Concatenate four 256-feature maps -> MLP (conv1x1 + GELU + conv1x1) to classes
    - Auxiliary heads: 1x1 conv per Fi to classes (for auxiliary losses)

    Key differences from RangeViT's decoders:
    - Multi-scale feature fusion vs single-scale upsampling
    - Auxiliary supervision at each stage
    - Channel unification before concatenation
    """
    def __init__(self,
                 stage_channels: List[int],
                 out_ch_unify: int = 256,
                 num_classes: int = 19,
                 H: int = 64,
                 W: int = 2048):
        """
        Args:
            stage_channels: list of channel dims for each stage [C1, C2, C3, C4]
            out_ch_unify: unified channel dimension (default 256)
            num_classes: number of semantic classes
            H, W: output spatial dimensions
        """
        super().__init__()

        # Assert valid inputs
        assert len(stage_channels) == 4, f"SegmentationHead: Expected 4 stage channels, got {len(stage_channels)}"
        assert out_ch_unify > 0, f"SegmentationHead: out_ch_unify must be positive, got {out_ch_unify}"
        assert num_classes > 0, f"SegmentationHead: num_classes must be positive, got {num_classes}"
        assert H > 0 and W > 0, f"SegmentationHead: H and W must be positive, got H={H}, W={W}"

        self.stage_channels = stage_channels
        self.out_ch_unify = out_ch_unify
        self.num_classes = num_classes
        self.H = H
        self.W = W

        # Channel unification layers: map each stage to out_ch_unify
        self.unify_layers = nn.ModuleList([
            nn.Conv2d(c, out_ch_unify, kernel_size=1) for c in stage_channels
        ])

        # Auxiliary heads for each stage (for auxiliary losses during training)
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch_unify, out_ch_unify // 2, 1),
                nn.GELU(),
                nn.Conv2d(out_ch_unify // 2, num_classes, 1)
            ) for _ in stage_channels
        ])

        # Main MLP head for final prediction
        self.main_mlp = nn.Sequential(
            nn.Conv2d(out_ch_unify * 4, out_ch_unify, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch_unify),
            nn.GELU(),
            nn.Conv2d(out_ch_unify, num_classes, kernel_size=1)
        )

        self.H = H
        self.W = W

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features: [F1, F2, F3, F4] as output by backbone
                F1: (B, C1, H, W)
                F2: (B, C2, H/2, W/2)
                F3: (B, C3, H/4, W/4)
                F4: (B, C4, H/8, W/8)

        Returns:
            logits_main: (B, num_classes, H, W) - main predictions
            aux_logits: list of (B, num_classes, H, W) - auxiliary predictions from each stage
        """
        # Assert input
        assert isinstance(features, list), f"SegmentationHead: Expected list of features, got {type(features)}"
        assert len(features) == 4, f"SegmentationHead: Expected 4 feature maps, got {len(features)}"

        # Verify all features have same batch size
        B = features[0].shape[0]
        for i, f in enumerate(features):
            assert f.dim() == 4, f"SegmentationHead: Feature {i} should be 4D (B,C,H,W), got {f.dim()}D with shape {f.shape}"
            assert f.shape[0] == B, f"SegmentationHead: Feature {i} batch size mismatch, expected {B}, got {f.shape[0]}"

        ups = []
        auxs = []

        for i, f in enumerate(features):
            # Unify channels to out_ch_unify
            f_unify = self.unify_layers[i](f)  # (B, out_ch_unify, Hi, Wi)
            assert f_unify.shape[0] == B, f"SegmentationHead: Stage {i} unify batch mismatch"
            assert f_unify.shape[1] == self.unify_layers[i].out_channels, \
                f"SegmentationHead: Stage {i} unify channel mismatch, expected {self.unify_layers[i].out_channels}, got {f_unify.shape[1]}"

            # Upsample to (H, W)
            f_up = F.interpolate(f_unify, size=(self.H, self.W), mode='bilinear', align_corners=False)
            assert f_up.shape == (B, f_unify.shape[1], self.H, self.W), \
                f"SegmentationHead: Stage {i} upsample shape mismatch, expected ({B}, {f_unify.shape[1]}, {self.H}, {self.W}), got {f_up.shape}"
            ups.append(f_up)

            # Auxiliary prediction
            aux = self.aux_heads[i](f_unify)
            assert aux.shape[1] == self.aux_heads[i][-1].out_channels, \
                f"SegmentationHead: Stage {i} aux head output channel mismatch"

            aux_up = F.interpolate(aux, size=(self.H, self.W), mode='bilinear', align_corners=False)
            assert aux_up.shape == (B, aux.shape[1], self.H, self.W), \
                f"SegmentationHead: Stage {i} aux upsample shape mismatch"
            auxs.append(aux_up)

        # Concatenate all unified features
        cat = torch.cat(ups, dim=1)  # (B, out_ch_unify*4, H, W)
        expected_concat_ch = ups[0].shape[1] * 4
        assert cat.shape == (B, expected_concat_ch, self.H, self.W), \
            f"SegmentationHead: Concatenation shape mismatch, expected ({B}, {expected_concat_ch}, {self.H}, {self.W}), got {cat.shape}"

        # Main prediction
        logits = self.main_mlp(cat)  # (B, num_classes, H, W)
        assert logits.shape == (B, self.main_mlp[-1].out_channels, self.H, self.W), \
            f"SegmentationHead: Main output shape mismatch, expected ({B}, {self.main_mlp[-1].out_channels}, {self.H}, {self.W}), got {logits.shape}"

        # Verify all aux outputs
        for i, aux in enumerate(auxs):
            assert aux.shape == (B, aux.shape[1], self.H, self.W), \
                f"SegmentationHead: Aux {i} final shape mismatch"

        return logits, auxs
