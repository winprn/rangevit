# models/create_rangeswin.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rangeswin import RangeSwinUPerNet  # the swin+upernet module you already made
from .rangevit_kpconv import KPClassifier, RangeViT_KPConv  # for KPConv integration (optional)


def create_rangeswin(model_cfg, use_kpconv=False):
    in_ch = model_cfg.get('in_channels', 5)
    n_cls = model_cfg.get('n_cls', 17)
    swin_name = model_cfg.get('swin_name', 'swin_tiny_patch4_window7_224')
    out_ch = model_cfg.get('out_channels', 256)
    pretrained_path = model_cfg.get('pretrained_path', None)

    backbone = RangeSwinUPerNet(
        in_channels=in_ch,
        n_cls=n_cls,
        swin_name=swin_name,
        pretrained_path=pretrained_path,
        out_channels=out_ch
    )

    if use_kpconv:
        kpclassifier = KPClassifier(
            in_channels=out_ch,
            out_channels=out_ch,
            num_classes=n_cls
        )
        model = RangeSwin_KPConv(backbone, kpclassifier, n_cls=n_cls)
    else:
        model = RangeSwin_noKPConv(backbone, out_channels=out_ch, in_channels=in_ch, n_cls=n_cls)

    return model


class RangeSwin_KPConv(nn.Module):
    def __init__(self, backbone: RangeSwinUPerNet, kpclassifier: KPClassifier, n_cls: int):
        super().__init__()
        self.backbone = backbone   # produces [B, D_h, H, W]
        self.kpclassifier = kpclassifier
        self.n_cls = n_cls

        # Add swin_encoder attribute for compatibility with train.py validation check
        self.swin_encoder = backbone

        # metadata to keep parity with RangeViT_KPConv
        self.patch_size = (4, 4)     # Swin patch embed default
        self.patch_stride = (4, 4)

    def forward_2d_features(self, im):
        """
        Produce 2D dense features [B, D_h, H, W] from Swin+UPerNet.
        """
        feats = self.backbone(im, return_features=True)
        return feats

    def counter_model_parameters(self):
        stats = {}
        stats['total_num_parameters'] = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Swin backbone
        if hasattr(self, "backbone"):
            stats['backbone_num_parameters'] = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        else:
            stats['backbone_num_parameters'] = 0

        # Decoder head
        if hasattr(self, "decode_head"):
            stats['decode_head_num_parameters'] = sum(p.numel() for p in self.decode_head.parameters() if p.requires_grad)
        else:
            stats['decode_head_num_parameters'] = 0

        # KPConv classifier
        if hasattr(self, "kpclassifier"):
            stats['kpclassifier_num_parameters'] = sum(p.numel() for p in self.kpclassifier.parameters() if p.requires_grad)
        else:
            stats['kpclassifier_num_parameters'] = 0

        return stats


    def forward(self, im, px, py, pxyz, pknn, num_points):
        """
        im: [B, in_ch, H, W]
        px, py, pxyz, pknn, num_points: same as RangeViT_KPConv
        returns: per-point logits [N_points, n_cls]
        """
        feats = self.forward_2d_features(im)   # [B, D_h, H, W]
        masks3d = self.kpclassifier(feats, px, py, pxyz, pknn, num_points)
        return masks3d

class RangeSwin_noKPConv(nn.Module):
    """
    Wrapper having the same runtime behaviour as RangeViT_noKPConv:
    - pads input according to patch / window size if needed,
    - runs Swin+UPerNet (backbone_decoder),
    - upsamples to (H,W) and unpad to original size,
    - returns [B, out_ch, H_ori, W_ori].
    """
    def __init__(self, backbone_decoder, out_channels, in_channels=5, n_cls=17):
        super().__init__()
        self.backbone_decoder = backbone_decoder
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.n_cls = n_cls

        # Add swin_encoder attribute for compatibility with train.py validation check
        self.swin_encoder = backbone_decoder

        # metadata so other code expecting RangeViT fields doesn't break
        # patch_size: Swin patch size (timm's default is 4 for many variants)
        # patch_stride: we set equal to patch_size (common convention)
        # d_model: we set to the out_channels of the UPer head for compatibility
        self.patch_size = (4, 4)   # default; you can change if using other swin variants
        self.patch_stride = (4, 4)
        self.d_model = out_channels

    def counter_model_parameters(self):
        stats = {}
        stats['total_num_parameters'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        stats['decoder_num_parameters'] = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        stats['stem_num_parameters'] = sum(p.numel() for p in self.encoder.patch_embed.parameters() if p.requires_grad)
        stats['encoder_num_parameters'] = (
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) - stats['stem_num_parameters']
        )
        return stats

    def forward(self, im):
        """
        im: [B, in_channels, H_ori, W_ori]
        returns: feats [B, out_channels, H_ori, W_ori]
        """
        H_ori, W_ori = im.size(2), im.size(3)

        # --- padding so Swin windows and patching are safe
        # Use the swin model's internal window size if present
        window_size = getattr(self.backbone_decoder.backbone, 'window_size', None)
        # window_size may be int or tuple
        if isinstance(window_size, int):
            ws_h = ws_w = window_size
        elif isinstance(window_size, (tuple, list)):
            ws_h, ws_w = window_size[0], window_size[1]
        else:
            # fallback - ensure divisibility by 32 (Swin downsamples by 32 in total)
            ws_h, ws_w = 32, 32

        # pad to nearest multiple of window size
        H_pad = ((H_ori + ws_h - 1) // ws_h) * ws_h
        W_pad = ((W_ori + ws_w - 1) // ws_w) * ws_w
        pad_h = H_pad - H_ori
        pad_w = W_pad - W_ori

        # pad right and bottom (left, top = 0)
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # ensure backbone's patch_embed.img_size matches padded size
        if hasattr(self.backbone_decoder.backbone, 'patch_embed'):
            self.backbone_decoder.backbone.patch_embed.img_size = (H_pad, W_pad)

        # forward through the combined swin + upernet module
        feats = self.backbone_decoder(im_padded, return_features=True)  # [B, out_ch, H_pad, W_pad]

        # unpad to original size
        if pad_h != 0 or pad_w != 0:
            feats = feats[:, :, :H_ori, :W_ori]

        return feats
