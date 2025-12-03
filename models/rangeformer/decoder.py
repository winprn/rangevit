# decoder.py
# RangeFormer Decoder / Segmentation Head (corrected to follow the paper)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SegmentationHead(nn.Module):
    """
    Multi-scale decoder head for RangeFormer semantic segmentation.

    Follows Sec. 3.2 / 6.2 of the paper:
      - Channel unification: Fi (Ci, Hi, Wi) -> 256 channels via 1x1 conv ("MLP")
      - Spatial unification: bilinear upsample to HxW (same size as Stage-1 feature)
      - Main head: concat four unified maps [H1..H4] along channel dim -> 2-layer MLP
      - Aux heads: one extra MLP (1x1 conv) per Hi for auxiliary supervision
    """

    def __init__(
        self,
        stage_channels: List[int],
        out_ch_unify: int = 256,
        num_classes: int = 19,
    ):
        super().__init__()

        self.stage_channels = list(stage_channels)
        self.out_ch_unify = out_ch_unify
        self.num_classes = num_classes

        # Channel unification layers: Conv1x1 -> BN -> GELU
        # (implements the "Linear" = MLP used to map Fi to unified dim)
        self.unify_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_ch_unify, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch_unify),
                nn.GELU()
            ) for in_c in self.stage_channels
        ])

        # Auxiliary heads: one extra MLP layer per Hi
        # Paper: "we add an extra MLP layer for each Hi as the auxiliary head."
        # Implemented as a single 1x1 Conv (per-pixel linear classifier).
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(out_ch_unify, num_classes, kernel_size=1, bias=True)
            for _ in self.stage_channels
        ])

        # Main head: concat H1..H4 (4 * out_ch_unify channels) ->
        # 1x1 conv -> GELU -> 1x1 conv -> num_classes
        # Paper: "another two MLP layers" for the main head.
        self.main_mlp = nn.Sequential(
            nn.Conv2d(out_ch_unify * 4, out_ch_unify, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch_unify),
            nn.GELU(),
            nn.Conv2d(out_ch_unify, num_classes, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            features: list [F1, F2, F3, F4] where
                F1: (B, C1, H,   W)
                F2: (B, C2, H/2, W/2)
                F3: (B, C3, H/4, W/4)
                F4: (B, C4, H/8, W/8)

        Returns:
            logits_main: (B, num_classes, H, W)
            aux_logits: list of 4 tensors each (B, num_classes, H, W)
        """
        # Target spatial size: resolution of F1 (the finest stage)
        B, _, H, W = features[0].shape

        unified_up = []   # list of Hi: (B, 256, H, W)
        auxs = []         # list of aux logits: (B, num_classes, H, W)

        for i, f in enumerate(features):
            # 1) Channel unification: Fi -> unified channels (Ci -> 256)
            f_un = self.unify_layers[i](f)  # (B, out_ch_unify, Hi, Wi)

            # 2) Spatial unification: bilinear upsample to (H, W)
            f_up = F.interpolate(
                f_un, size=(H, W), mode='bilinear', align_corners=False
            )  # (B, out_ch_unify, H, W)
            unified_up.append(f_up)

            # 3) Aux head on Hi (apply on smaller feature map f_un, then upsample)
            # Optimization: Apply conv on smaller map to save VRAM and FLOPs
            aux_logits_small = self.aux_heads[i](f_un)  # (B, num_classes, Hi, Wi)
            aux_logits = F.interpolate(
                aux_logits_small, size=(H, W), mode='bilinear', align_corners=False
            ) # (B, num_classes, H, W)
            auxs.append(aux_logits)

        # Concatenate unified features along channel dimension:
        # [H1, H2, H3, H4] -> (B, 4 * out_ch_unify, H, W)
        cat = torch.cat(unified_up, dim=1)

        # Main head: 2-layer MLP on concatenated feature map
        logits_main = self.main_mlp(cat)  # (B, num_classes, H, W)

        return logits_main, auxs


if __name__ == "__main__":
    # Quick sanity test: instantiate head with common RangeFormer sizes
    stage_channels = [128, 128, 320, 512]
    H, W = 64, 512  # typical range image resolution (H x W)

    head = SegmentationHead(
        stage_channels=stage_channels,
        out_ch_unify=256,
        num_classes=19,
    )

    test_feats = [
        torch.randn(2, stage_channels[0], H,     W),
        torch.randn(2, stage_channels[1], H // 2, W // 2),
        torch.randn(2, stage_channels[2], H // 4, W // 4),
        torch.randn(2, stage_channels[3], H // 8, W // 8),
    ]

    logits, auxs = head(test_feats)
    print("Main logits shape:", logits.shape)              # (2, 19, 64, 512)
    print("Aux logits shapes:", [a.shape for a in auxs])   # each (2, 19, 64, 512)
    total_params = sum(p.numel() for p in head.parameters()) / 1e6
    print(f"SegmentationHead params: {total_params:.3f} M")
