import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_utils import init_weights


class TinyViMFPNDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        n_cls,
        out_channels=256,
        head_channels=128,
        dropout_ratio=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_channels = head_channels
        self.n_cls = n_cls

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1) for ch in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

        self.head_convs = nn.ModuleList([
            nn.Conv2d(out_channels, head_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])
        self.fuse_conv = nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(head_channels, n_cls, kernel_size=1)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size, skip=None, return_features=False):
        if not isinstance(skip, (list, tuple)):
            raise ValueError('TinyViMFPNDecoder expects a list of feature maps in skip.')

        feats = skip
        if len(feats) != len(self.in_channels):
            raise ValueError('Number of feature maps does not match in_channels.')

        # Build laterals
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode='bilinear', align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up

        # FPN outputs
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Head fusion at the highest resolution
        head_feats = []
        target_size = outs[0].shape[-2:]
        for i, out in enumerate(outs):
            h = self.head_convs[i](out)
            if i > 0:
                h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)
            head_feats.append(h)

        fused = sum(head_feats)
        fused = self.fuse_conv(fused)
        fused = self.dropout(fused)

        logits = self.cls_seg(fused)
        if im_size is not None:
            logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=False)

        if return_features:
            return fused
        return logits
