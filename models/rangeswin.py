# models/rangeswin.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .model_utils import adapt_input_conv, padding, unpadding, init_weights

# A compact UPer-like head (sufficient for fusing multi-scale Swin features).
class SimpleUPerHead(nn.Module):
    def __init__(self, in_channels, channels=256, num_classes=256, pool_scales=(1,2,3,6),
                 norm_layer=nn.BatchNorm2d, dropout=0.1):
        """
        in_channels: list of channels from backbone (C1..C4)
        channels: internal FPN channel
        num_classes: final output channels (we'll set to D_h)
        pool_scales: scales for pyramid pooling
        dropout: dropout rate for regularization
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.pool_scales = pool_scales
        self.dropout = dropout

        # lateral convs to unify channels
        self.lateral_convs = nn.ModuleList()
        for c in in_channels:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(c, channels, kernel_size=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True)
            ))

        # FPN convs after fusion (add light dropout for regularization)
        self.fpn_convs = nn.ModuleList()
        for _ in in_channels:
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout * 0.5) if dropout > 0 else nn.Identity()  # lighter dropout in FPN
            ))

        # Simple PPM on the last stage
        self.ppm_convs = nn.ModuleList()
        last_c = in_channels[-1]
        for p in pool_scales:
            self.ppm_convs.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(p, p)),
                nn.Conv2d(last_c, channels, kernel_size=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True)
            ))
        # final conv to combine PPM features
        self.ppm_last = nn.Sequential(
            nn.Conv2d(last_c + len(pool_scales) * channels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True)
        )

        # final fusion conv (after upsampling to highest resolution)
        self.conv_seg = nn.Sequential(
            nn.Conv2d(channels * len(in_channels), channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),  # Add dropout
            nn.Conv2d(channels, num_classes, kernel_size=1)
        )

        self.apply(init_weights)

    def forward(self, feats):
        """
        feats: list of 4 feature maps from backbone [C1, C2, C3, C4]
            shapes: [B, C1, H/4, W/4], [B, C2, H/8, W/8], ... [B, C4, H/32, W/32]
        returns:
            out: [B, num_classes, H/4, W/4]  (we will upsample later in RangeSwin to original H)
        """
        # build ppm on last feature
        last_feat = feats[-1]
        ppm_outs = [last_feat]
        for ppm in self.ppm_convs:
            x = ppm(last_feat)
            x = F.interpolate(x, size=last_feat.shape[2:], mode='bilinear', align_corners=False)
            ppm_outs.append(x)
        ppm_cat = torch.cat(ppm_outs, dim=1)  # [B, last_c + len(pool)*channels, h4, w4]
        ppm_feat = self.ppm_last(ppm_cat)     # [B, channels, h4, w4]

        # lateral conv + fpn top-down
        laterals = [l_conv(f) for l_conv, f in zip(self.lateral_convs, feats)]
        # replace last lateral with ppm fused features projected to channels
        laterals[-1] = ppm_feat

        # top-down fusion
        for i in range(len(laterals)-2, -1, -1):
            size = laterals[i].shape[2:]
            laterals[i] = laterals[i] + F.interpolate(laterals[i+1], size=size, mode='bilinear', align_corners=False)

        # apply fpn convs
        fpn_outs = [fpn(l) for fpn, l in zip(self.fpn_convs, laterals)]

        # upsample all to highest resolution (feats[0] resolution)
        upsampled = [F.interpolate(f, size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=False) for f in fpn_outs]
        cat = torch.cat(upsampled, dim=1)

        out = self.conv_seg(cat)  # [B, num_classes, H/4, W/4]
        return out


class RangeSwinUPerNet(nn.Module):
    def __init__(self,
                 in_channels=5,
                 n_cls=17,
                 swin_name='swinv2_tiny_window16_256',
                 pretrained_path=None,
                 out_channels=256,    # D_h
                 swin_features_out_indices=(0,1,2,3),
                 swin_features_only=True,
                 swin_pretrained=True):
        super().__init__()

        self.in_channels = in_channels
        self.n_cls = n_cls
        self.out_channels = out_channels
        # Image size will be set dynamically based on input or from timm model config
        self.range_image_size = None


        # Create timm swin backbone (features_only gives list of feature maps)
        # Note: img_size will be set to input size dynamically during forward pass
        self.backbone = timm.create_model(
            swin_name,
            pretrained=False,   # we will optionally load custom checkpoint manually
            features_only=swin_features_only,
            in_chans=in_channels,   # timm supports in_chans param
            out_indices=swin_features_out_indices
        )


        # timm returns feature channels in backbone.feature_info.channels
        # feat_channels = self.backbone.feature_info.channels  # list like [96,192,384,768]


        ####################
        feat_info = getattr(self.backbone, 'feature_info', None)
        if feat_info is None:
            raise RuntimeError("Backbone has no `feature_info`. Make sure features_only=True when creating the timm model.")
        # feature_info.channels can be either an attribute (list) or a callable (method) depending on timm version
        channels_attr = getattr(feat_info, 'channels', None)
        if channels_attr is None:
            # fallback: some timm versions have feature_info.info with entries containing 'num_chs'
            if hasattr(feat_info, 'info'):
                feat_channels = [x['num_chs'] for x in feat_info.info]  # last resort
            else:
                raise RuntimeError("Cannot read feature channels from backbone.feature_info")
        else:
            feat_channels = channels_attr() if callable(channels_attr) else channels_attr

        # final check
        if not isinstance(feat_channels, (list, tuple)):
            raise RuntimeError(f"Unexpected feature_info.channels type: {type(feat_channels)}. Got value: {feat_channels}")

        ####################


        # UPer head that produces feature maps with out_channels (D_h)
        # Note: For non-KPConv mode, we need a classification head to map D_h -> n_cls
        # Use smaller decoder channels for better parameter efficiency (similar to RangeViT)
        decoder_channels = min(256, out_channels * 2)  # Use 256 for D_h=128, scale up for larger D_h

        self.decode_head = SimpleUPerHead(
            in_channels=feat_channels,
            channels=decoder_channels,
            num_classes=out_channels,  # Output D_h features (for KPConv compatibility)
            dropout=0.1  # Regularization
        )

        # Add classification head for direct semantic segmentation (non-KPConv mode)
        # This converts D_h features to n_classes logits
        self.classifier = nn.Conv2d(out_channels, n_cls, kernel_size=1)
        # Initialize classifier with smaller weights for better convergence
        nn.init.normal_(self.classifier.weight, std=0.01)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

        # Optional auxiliary classifier on C3 (1/16 resolution) for deep supervision
        # This helps train the backbone better, especially early layers
        self.aux_classifier = None
        if len(feat_channels) >= 3:  # Only if we have C3 features
            self.aux_classifier = nn.Sequential(
                nn.Conv2d(feat_channels[2], out_channels // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channels // 2, n_cls, kernel_size=1)
            )
            # Initialize auxiliary classifier
            for m in self.aux_classifier.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # optionally load pretrained checkpoint and adapt first conv
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path, in_channels)

    def _load_pretrained(self, pretrained_path, in_chans):
        # Supports checkpoints that are dict with 'model' or raw timm format.
        ckpt = torch.load(pretrained_path, map_location='cpu')
        # find state_dict inside ckpt
        if isinstance(ckpt, dict) and ('model' in ckpt):
            sd = ckpt['model']
        elif isinstance(ckpt, dict) and any(k.startswith('backbone.patch_embed') or k.startswith('patch_embed') for k in ckpt.keys()):
            # some checkpoints have top-level keys like patch_embed.proj.weight
            sd = ckpt
        else:
            sd = ckpt

        # adapt first conv weights if necessary
        # timm Swin uses 'patch_embed.proj.weight' as first conv key
        # find conv key and adapt if in_chans != 3
        first_conv_key = None
        for k in list(sd.keys()):
            if 'patch_embed.proj.weight' in k:
                first_conv_key = k
                break
            # if timm prefix includes 'backbone.' try that too
            if 'backbone.patch_embed.proj.weight' in k:
                first_conv_key = k
                break

        if first_conv_key is not None:
            # adapt conv weight format
            conv_w = sd[first_conv_key]
            if conv_w.shape[1] != in_chans:
                # adapt_input_conv supports repeating / scaling
                new_w = adapt_input_conv(in_chans, conv_w)
                sd[first_conv_key] = new_w

        # Now load to backbone state dict
        # timm model keys usually have no 'backbone.' prefix, so try to map keys
        model_sd = self.backbone.state_dict()
        # Keep only keys that exist in backbone to avoid decode_head mismatch
        sd_to_load = {}
        for k, v in sd.items():
            # try multiple prefix variants
            candidate_keys = [k, k.replace('backbone.', ''), k.replace('encoder.', ''), k.replace('module.', '')]
            loaded = False
            for cand in candidate_keys:
                if cand in model_sd and model_sd[cand].shape == v.shape:
                    sd_to_load[cand] = v
                    loaded = True
                    break
            # skip otherwise

        msg = self.backbone.load_state_dict(sd_to_load, strict=False)
        print('Swin preload:', msg)

    def forward(self, im, return_features=False, return_aux=False):
        """
        Forward pass through Swin backbone and UPerNet decoder.

        Args:
            im: Input tensor [B, in_channels, H, W]
            return_features: If True, return D_h features; else return class logits
            return_aux: If True and aux_classifier exists, return auxiliary logits for training

        Returns:
            If return_features=True: [B, out_channels, H, W] features (for KPConv)
            If return_features=False and return_aux=False: [B, n_cls, H, W] class logits
            If return_features=False and return_aux=True: tuple of (main_logits, aux_logits)
        """
        B, C, H, W = im.shape

        # Set image size for patch embedding (done dynamically to handle variable input sizes)
        if hasattr(self.backbone, "patch_embed"):
            self.backbone.patch_embed.img_size = (H, W)

        # Forward through Swin backbone - returns list of multi-scale feature maps
        # Note: timm Swin with features_only=True outputs NHWC format (not NCHW!)
        # Shapes: [B, H/4, W/4, C1], [B, H/8, W/8, C2], [B, H/16, W/16, C3], [B, H/32, W/32, C4]
        feats = self.backbone(im)

        # Convert all features from NHWC to NCHW format
        # timm Swin with features_only=True outputs features in NHWC format
        proc_feats = []
        for f in feats:
            if f.ndim == 4:
                # Convert from NHWC [B, H, W, C] to NCHW [B, C, H, W]
                f_nchw = f.permute(0, 3, 1, 2).contiguous()
                proc_feats.append(f_nchw)
            else:
                proc_feats.append(f)

        # Decode with UPerNet-like head -> [B, D_h, H/4, W/4]
        seg_feat = self.decode_head(proc_feats)

        # Upsample to original input size
        seg_feat_up = F.interpolate(
            seg_feat, size=(H, W), mode="bilinear", align_corners=False
        )

        # Return features for KPConv mode, or class logits for direct segmentation
        if return_features:
            return seg_feat_up  # [B, out_channels, H, W]
        else:
            # Apply classification head to get class logits
            logits = self.classifier(seg_feat_up)  # [B, n_cls, H, W]

            # Optionally return auxiliary logits for deep supervision during training
            if return_aux and self.aux_classifier is not None and self.training:
                # Apply auxiliary classifier on C3 features (1/16 resolution)
                aux_logits = self.aux_classifier(proc_feats[2])  # [B, n_cls, H/16, W/16]
                aux_logits = F.interpolate(aux_logits, size=(H, W), mode="bilinear", align_corners=False)
                return logits, aux_logits
            else:
                return logits