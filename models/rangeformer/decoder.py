# decoder.py
# Fixed RangeFormer Decoder / Segmentation Head
# Reference: Kong et al., 2023 - RangeFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SegmentationHead(nn.Module):
    """
    Multi-scale decoder head for RangeFormer semantic segmentation.

    - Channel unification: map each Fi (with dim di) -> out_ch_unify via 1x1 conv + BN + GELU
    - Spatial unification: upsample Fi (for i>1) to HxW using bilinear interp
    - Concatenate four unified maps -> main MLP head (1x1 conv + BN + GELU + 1x1 conv)
    - Auxiliary heads: small conv block per unified feature -> classes (upsampled)
    """

    def __init__(self,
                 stage_channels: List[int],
                 out_ch_unify: int = 256,
                 num_classes: int = 19,
                 H: int = 64,
                 W: int = 2048):
        super().__init__()

        assert len(stage_channels) == 4, "SegmentationHead expects 4 stage channel sizes"
        assert out_ch_unify > 0 and num_classes > 0 and H > 0 and W > 0

        self.stage_channels = list(stage_channels)
        self.out_ch_unify = out_ch_unify
        self.num_classes = num_classes
        self.H = H
        self.W = W

        # Channel unification layers: Conv1x1 -> BN -> GELU
        self.unify_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_ch_unify, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch_unify),
                nn.GELU()
            ) for in_c in self.stage_channels
        ])

        # Auxiliary heads: small classifier on each unified feature
        # (Conv1x1 -> BN -> GELU -> Conv1x1 -> logits)
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch_unify, out_ch_unify // 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch_unify // 2),
                nn.GELU(),
                nn.Conv2d(out_ch_unify // 2, num_classes, kernel_size=1, bias=True)
            ) for _ in self.stage_channels
        ])

        # Main MLP head: concat 4 unified features -> conv(1x1) -> BN -> GELU -> conv(1x1) -> num_classes
        self.main_mlp = nn.Sequential(
            nn.Conv2d(out_ch_unify * 4, out_ch_unify, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch_unify),
            nn.GELU(),
            nn.Conv2d(out_ch_unify, num_classes, kernel_size=1, bias=True)
        )

        # Weight init
        self._init_weights()

    def _init_weights(self):
        # Kaiming init for convs (except final classifier convs which are small)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # For small final classifier convs it's still fine to use kaiming
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
                F1: (B, C1, H, W)
                F2: (B, C2, H/2, W/2)
                F3: (B, C3, H/4, W/4)
                F4: (B, C4, H/8, W/8)

        Returns:
            logits_main: (B, num_classes, H, W)
            aux_logits: list of 4 tensors each (B, num_classes, H, W)
        """
        assert isinstance(features, list) and len(features) == 4, "SegmentationHead.forward expects list of 4 feature maps"
        B = features[0].shape[0]

        unified = []
        auxs = []

        # unify channels and upsample; compute aux predictions from unified features
        for i, f in enumerate(features):
            assert f.dim() == 4, f"Feature {i} must be 4D"
            # unify channels
            f_un = self.unify_layers[i](f)  # (B, out_ch_unify, Hi, Wi)
            # upsample to (H, W)
            f_up = F.interpolate(f_un, size=(self.H, self.W), mode='bilinear', align_corners=False)
            unified.append(f_up)

            # auxiliary logits (from pre-upsample unified or from f_un directly)
            aux = self.aux_heads[i](f_un)  # (B, num_classes, Hi, Wi)
            aux_up = F.interpolate(aux, size=(self.H, self.W), mode='bilinear', align_corners=False)
            auxs.append(aux_up)

        # Concatenate unified features along channel
        cat = torch.cat(unified, dim=1)  # (B, out_ch_unify*4, H, W)
        assert cat.shape[1] == self.out_ch_unify * 4, "Concatenated channel mismatch"

        # Main logits
        logits_main = self.main_mlp(cat)  # (B, num_classes, H, W)
        assert logits_main.shape[1] == self.num_classes

        return logits_main, auxs


if __name__ == "__main__":
    # Quick sanity test: instantiate head with common RangeFormer sizes
    stage_channels = [128, 128, 320, 512]
    H, W = 64, 512  # typical range image resolution (HxW)
    head = SegmentationHead(stage_channels=stage_channels, out_ch_unify=256, num_classes=19, H=H, W=W)

    test_feats = [
        torch.randn(2, stage_channels[0], H, W),
        torch.randn(2, stage_channels[1], H // 2, W // 2),
        torch.randn(2, stage_channels[2], H // 4, W // 4),
        torch.randn(2, stage_channels[3], H // 8, W // 8),
    ]

    logits, auxs = head(test_feats)
    print("Main logits shape:", logits.shape)
    print("Aux logits shapes:", [a.shape for a in auxs])
    total_params = sum(p.numel() for p in head.parameters()) / 1e6
    print(f"SegmentationHead params: {total_params:.3f} M")
