# models/rangeswin.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .model_utils import adapt_input_conv, padding, unpadding, init_weights

# A compact UPer-like head (sufficient for fusing multi-scale Swin features).
class SimpleUPerHead(nn.Module):
    def __init__(self, in_channels, channels=256, num_classes=256, pool_scales=(1,2,3,6), norm_layer=nn.BatchNorm2d):
        """
        in_channels: list of channels from backbone (C1..C4)
        channels: internal FPN channel
        num_classes: final output channels (we'll set to D_h)
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.pool_scales = pool_scales

        # lateral convs to unify channels
        self.lateral_convs = nn.ModuleList()
        for c in in_channels:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(c, channels, kernel_size=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True)
            ))

        # FPN convs after fusion
        self.fpn_convs = nn.ModuleList()
        for _ in in_channels:
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True)
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
        self.dataset = "Semantic-kitti" # TODO: change to depend on dataset
        if self.dataset == "Semantic-kitti":
            self.range_image_size = (64, 384)
        else:
            self.range_image_size = (32, 384)


        # Create timm swin backbone (features_only gives list of feature maps)
        self.backbone = timm.create_model(
            swin_name,
            pretrained=False,   # we will optionally load custom checkpoint manually
            features_only=swin_features_only,
            in_chans=in_channels,   # timm supports in_chans param
            img_size=self.range_image_size,
            out_indices=swin_features_out_indices
        )
        if hasattr(self.backbone, 'patch_embed'):
            self.backbone.patch_embed.img_size = self.range_image_size


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


        # UPer head that produces out_channels (D_h)
        self.decode_head = SimpleUPerHead(in_channels=feat_channels, channels=max(64, out_channels), num_classes=out_channels)

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

    def forward(self, im, return_features=False):
        """
        im: [B, in_channels, H, W]
        returns: [B, out_channels, H, W] (upsampled to original padded H,W)
        """
        #### VERSION 1 ####
        
        # B, C, H, W = im.shape
        # im_padded = padding(im, (4, 4))  # just ensure compatibility, Patch sizes differ; but padding accepts patch_size param - using (4,4) safe
        # H_pad, W_pad = im_padded.shape[2], im_padded.shape[3]

        # feats = self.backbone(im_padded)   # list of 4 feature maps
        # # decode head -> produces [B, out_channels, H/4, W/4] (depends on swin config)
        # seg_feat = self.decode_head(feats)

        # # upsample to the padded resolution
        # seg_feat_up = F.interpolate(seg_feat, size=(H_pad, W_pad), mode='bilinear', align_corners=False)
        # seg_feat_up = unpadding(seg_feat_up, (H, W))  # bring back to original
        # if return_features:
        #     return seg_feat_up  # [B, out_channels, H, W]
        # else:
        #     # for compatibility with RangeViT_noKPConv / KPConv flows, return final features
        #     return seg_feat_up

        #### VERSION 2 ####
        """
        im: [B, in_channels, H, W]
        returns: [B, out_channels, H, W]
        """
        B, C, H, W = im.shape

        # make sure patch_embed expects the right size
        if hasattr(self.backbone, "patch_embed"):
            self.backbone.patch_embed.img_size = (H, W)

        # forward through Swin backbone
        feats = self.backbone(im)  # list of feature maps

        # ensure format is [B, C, H, W]
        proc_feats = []
        for f in feats:
            if f.ndim == 4 and f.shape[1] < f.shape[-1]:  # looks like NHWC
                f = f.permute(0, 3, 1, 2).contiguous()
            proc_feats.append(f)

        # decode with UPerNet-like head
        seg_feat = self.decode_head(proc_feats)  # [B, out_channels, h', w']

        # upsample back to original H, W
        seg_feat_up = F.interpolate(
            seg_feat, size=(H, W), mode="bilinear", align_corners=False
        )

        if return_features:
            return seg_feat_up  # [B, out_channels, H, W]
        else:
            return seg_feat_up