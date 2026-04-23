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
        # Use horizontal kernels for deeper stages (Stage3/Stage4) to emphasize azimuth context.
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 5) if i >= 2 else 3,
                padding=(0, 2) if i >= 2 else 1,
            )
            for i, _ in enumerate(in_channels)
        ])

        self.head_convs = nn.ModuleList([
            nn.Conv2d(
                out_channels,
                head_channels,
                kernel_size=(1, 5) if i >= 2 else 3,
                padding=(0, 2) if i >= 2 else 1,
            )
            for i, _ in enumerate(in_channels)
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

        if return_features:
            if im_size is not None and fused.shape[-2:] != im_size:
                fused = F.interpolate(fused, size=im_size, mode='bilinear', align_corners=False)
            return fused

        logits = self.cls_seg(fused)
        if im_size is not None and logits.shape[-2:] != im_size:
            logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=False)
        return logits


class TinyViMFPNGatedDecoder(nn.Module):
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
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 5) if i >= 2 else 3,
                padding=(0, 2) if i >= 2 else 1,
            )
            for i, _ in enumerate(in_channels)
        ])
        self.head_convs = nn.ModuleList([
            nn.Conv2d(
                out_channels,
                head_channels,
                kernel_size=(1, 5) if i >= 2 else 3,
                padding=(0, 2) if i >= 2 else 1,
            )
            for i, _ in enumerate(in_channels)
        ])
        self.gate = nn.Sequential(
            nn.Conv2d(len(in_channels) * head_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.GELU(),
            nn.Conv2d(head_channels, len(in_channels), kernel_size=1, bias=True),
        )
        self.fuse_conv = nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(head_channels, n_cls, kernel_size=1)

        self.apply(init_weights)
        self._init_gate()

    def _init_gate(self):
        # Start from uniform scale fusion. This keeps the gated decoder close to
        # the plain FPN sum at initialization and avoids early saturated gates.
        final_gate = self.gate[-1]
        nn.init.zeros_(final_gate.weight)
        nn.init.zeros_(final_gate.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size, skip=None, return_features=False):
        if not isinstance(skip, (list, tuple)):
            raise ValueError('TinyViMFPNGatedDecoder expects a list of feature maps in skip.')

        feats = skip
        if len(feats) != len(self.in_channels):
            raise ValueError('Number of feature maps does not match in_channels.')

        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]

        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode='bilinear', align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        head_feats = []
        target_size = outs[0].shape[-2:]
        for i, out in enumerate(outs):
            h = self.head_convs[i](out)
            if i > 0:
                h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)
            head_feats.append(h)

        cat = torch.cat(head_feats, dim=1)
        with torch.cuda.amp.autocast(enabled=False):
            gate_logits = self.gate(cat.float()).clamp(-30.0, 30.0)
        gates = torch.softmax(gate_logits, dim=1).to(cat.dtype)

        fused = torch.zeros_like(head_feats[0])
        for i, feat in enumerate(head_feats):
            fused = fused + gates[:, i:i + 1] * feat

        fused = self.fuse_conv(fused)
        fused = self.dropout(fused)

        if return_features:
            if im_size is not None and fused.shape[-2:] != im_size:
                fused = F.interpolate(fused, size=im_size, mode='bilinear', align_corners=False)
            return fused

        logits = self.cls_seg(fused)
        if im_size is not None and logits.shape[-2:] != im_size:
            logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=False)
        return logits


class DetailRefineBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.block(x)


class TinyViMFPNGatedDetailDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        n_cls,
        out_channels=256,
        head_channels=128,
        detail_channels=64,
        dropout_ratio=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_cls = n_cls
        self.out_channels = out_channels
        self.head_channels = head_channels
        self.detail_channels = detail_channels

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1) for ch in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 5) if i >= 2 else 3,
                padding=(0, 2) if i >= 2 else 1,
            )
            for i, _ in enumerate(in_channels)
        ])
        self.head_convs = nn.ModuleList([
            nn.Conv2d(
                out_channels,
                head_channels,
                kernel_size=(1, 5) if i >= 2 else 3,
                padding=(0, 2) if i >= 2 else 1,
            )
            for i, _ in enumerate(in_channels)
        ])
        self.gate = nn.Sequential(
            nn.Conv2d(len(in_channels) * head_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.GELU(),
            nn.Conv2d(head_channels, len(in_channels), kernel_size=1, bias=True),
        )
        self.detail_proj0 = nn.Sequential(
            nn.Conv2d(in_channels[0], detail_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.GELU(),
        )
        self.detail_proj1 = nn.Sequential(
            nn.Conv2d(in_channels[1], detail_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.GELU(),
        )
        self.detail_refine = DetailRefineBlock(detail_channels)
        self.detail_to_main = nn.Conv2d(detail_channels, head_channels, kernel_size=1, bias=False)
        self.detail_scale = nn.Parameter(torch.zeros(1))

        self.fuse_conv = nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(head_channels, n_cls, kernel_size=1)

        self.apply(init_weights)
        self._init_gate()

    def _init_gate(self):
        # Start from uniform scale fusion. The detail branch is also scaled from
        # zero, so this decoder initially behaves like the stable gated variant.
        final_gate = self.gate[-1]
        nn.init.zeros_(final_gate.weight)
        nn.init.zeros_(final_gate.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size, skip=None, return_features=False):
        if not isinstance(skip, (list, tuple)):
            raise ValueError('TinyViMFPNGatedDetailDecoder expects a list of feature maps in skip.')

        feats = skip
        if len(feats) != len(self.in_channels):
            raise ValueError('Number of feature maps does not match in_channels.')

        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]

        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode='bilinear', align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        head_feats = []
        target_size = outs[0].shape[-2:]
        for i, out in enumerate(outs):
            h = self.head_convs[i](out)
            if i > 0:
                h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)
            head_feats.append(h)

        cat = torch.cat(head_feats, dim=1)
        with torch.cuda.amp.autocast(enabled=False):
            gate_logits = self.gate(cat.float()).clamp(-30.0, 30.0)
        gates = torch.softmax(gate_logits, dim=1).to(cat.dtype)

        fused_main = torch.zeros_like(head_feats[0])
        for i, feat in enumerate(head_feats):
            fused_main = fused_main + gates[:, i:i + 1] * feat

        d0 = self.detail_proj0(feats[0])
        d1 = self.detail_proj1(feats[1])
        if d1.shape[-2:] != d0.shape[-2:]:
            d1 = F.interpolate(d1, size=d0.shape[-2:], mode='bilinear', align_corners=False)

        detail = d0 + d1
        detail = detail + self.detail_refine(detail)
        detail = self.detail_to_main(detail)

        fused = fused_main + self.detail_scale.to(detail.dtype) * detail
        fused = self.fuse_conv(fused)
        fused = self.dropout(fused)

        if return_features:
            if im_size is not None and fused.shape[-2:] != im_size:
                fused = F.interpolate(fused, size=im_size, mode='bilinear', align_corners=False)
            return fused

        logits = self.cls_seg(fused)
        if im_size is not None and logits.shape[-2:] != im_size:
            logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=False)
        return logits
