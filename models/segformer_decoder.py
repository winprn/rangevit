import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import init_weights


class _ReduceBlock(nn.Module):
    """1x1 reduction followed by depthwise 3x3 to mimic SegFormer MLP."""
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.act1 = nn.ReLU(inplace=True)
        self.dw = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1,
                            groups=embed_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.proj(x)))
        x = self.act2(self.bn2(self.dw(x)))
        return x


class SegFormerDecoder(nn.Module):
    """
    Lightweight SegFormer-style head:
      - Reduce each feature to embed_dim with 1x1 + depthwise 3x3
      - Upsample all to the highest resolution
      - Concatenate and fuse, then predict logits
    """
    def __init__(
        self,
        in_channels,
        n_cls,
        embed_dim=128,
        head_channels=128,
        dropout_ratio=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.head_channels = head_channels
        self.n_cls = n_cls

        self.reduce_layers = nn.ModuleList([
            _ReduceBlock(ch, embed_dim) for ch in in_channels
        ])
        fusion_in = embed_dim * len(in_channels)
        self.fuse_conv = nn.Conv2d(fusion_in, head_channels, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(head_channels)
        self.fuse_act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(head_channels, n_cls, kernel_size=1)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size, skip=None, return_features=False):
        if not isinstance(skip, (list, tuple)):
            raise ValueError('SegFormerDecoder expects a list of feature maps in skip.')
        feats = skip
        if len(feats) != len(self.in_channels):
            raise ValueError('Number of feature maps does not match in_channels.')

        target_size = feats[0].shape[-2:]
        reduced = []
        for feat, reduce in zip(feats, self.reduce_layers):
            y = reduce(feat)
            if y.shape[-2:] != target_size:
                y = F.interpolate(y, size=target_size, mode='bilinear', align_corners=False)
            reduced.append(y)

        fused = torch.cat(reduced, dim=1)
        fused = self.fuse_act(self.fuse_bn(self.fuse_conv(fused)))
        fused = self.dropout(fused)

        if return_features:
            if im_size is not None and fused.shape[-2:] != im_size:
                fused = F.interpolate(fused, size=im_size, mode='bilinear', align_corners=False)
            return fused

        logits = self.cls_seg(fused)
        if im_size is not None and logits.shape[-2:] != im_size:
            logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=False)
        return logits
