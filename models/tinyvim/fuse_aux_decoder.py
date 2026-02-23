import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_utils import init_weights


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyViMFuseAuxDecoder(nn.Module):
    """
    TinyViM-only multi-stage fusion decoder with optional auxiliary heads.
    Expects a list of stage feature maps in `skip`.
    """
    supports_aux = True

    def __init__(
        self,
        in_channels,
        n_cls,
        proj_channels=128,
        mid_channels=256,
        out_channels=128,
        use_aux=True,
        use_preproj=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_cls = n_cls
        self.use_aux = bool(use_aux)
        self.use_preproj = bool(use_preproj)

        if self.use_preproj:
            self.proj_convs = nn.ModuleList([
                nn.Conv2d(ch, proj_channels, kernel_size=1) for ch in in_channels
            ])
            fused_in_channels = proj_channels * len(in_channels)
            aux_head_in_channels = [proj_channels] * max(0, len(in_channels) - 1)
        else:
            self.proj_convs = None
            fused_in_channels = sum(in_channels)
            aux_head_in_channels = list(in_channels[1:])
        self.conv_1 = ConvBNAct(fused_in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv_2 = ConvBNAct(mid_channels, out_channels, kernel_size=3, padding=1)
        self.semantic_output = nn.Conv2d(out_channels, n_cls, kernel_size=1)

        if self.use_aux:
            # Aux heads supervise deeper levels after spatial alignment.
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(ch, n_cls, kernel_size=1) for ch in aux_head_in_channels
            ])
        else:
            self.aux_heads = nn.ModuleList([])

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size, skip=None, return_features=False, return_aux=False):
        if not isinstance(skip, (list, tuple)):
            raise ValueError('TinyViMFuseAuxDecoder expects a list of feature maps in skip.')

        feats = list(skip)
        if len(feats) != len(self.in_channels):
            raise ValueError('Number of feature maps does not match in_channels.')

        target_size = feats[0].shape[-2:]
        aligned_feats = []
        if self.use_preproj:
            for feat, proj in zip(feats, self.proj_convs):
                feat = proj(feat)
                if feat.shape[-2:] != target_size:
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                aligned_feats.append(feat)
        else:
            for feat in feats:
                if feat.shape[-2:] != target_size:
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                aligned_feats.append(feat)

        fused = torch.cat(aligned_feats, dim=1)
        fused = self.conv_1(fused)
        fused = self.conv_2(fused)

        if return_features:
            if im_size is not None and fused.shape[-2:] != im_size:
                fused = F.interpolate(fused, size=im_size, mode='bilinear', align_corners=False)
            return fused

        logits = self.semantic_output(fused)
        if im_size is not None and logits.shape[-2:] != im_size:
            logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=False)

        if not (self.use_aux and return_aux):
            return logits

        aux_logits = []
        # Use all but the first (highest-resolution) aligned feature for aux supervision.
        for feat, head in zip(aligned_feats[1:], self.aux_heads):
            a = head(feat)
            if im_size is not None and a.shape[-2:] != im_size:
                a = F.interpolate(a, size=im_size, mode='bilinear', align_corners=False)
            aux_logits.append(a)

        return logits, aux_logits
