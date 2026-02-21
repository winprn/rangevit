import torch
import torch.nn as nn
from .tinyvim.tvimblock import Conv2d_BN
from .tinyvim.tinyvim import TinyViM, TinyViM_depth, TinyViM_width, Embedding

class TinyViMAdapter(nn.Module):
    def __init__(self,
                 backbone_name='tinyvim_small',
                 image_size=(64, 2048),
                 patch_size=None,
                 patch_stride=None,
                 channels=3, # RangeViT uses 'channels' kwarg
                 in_channels=None, # For compatibility if passed explicitly
                 pretrained_path=None,
                 load_pretrained_stem=False,  # If True, keep 3-channel stem for weight loading
                 d_model=None, # RangeViT passes d_model, but we might ignore or verify
                 use_fpn_decoder=False,
                 **kwargs):
        # Resolve in_channels
        if in_channels is None:
            in_channels = channels
        
        super().__init__()
        
        # Parse backbone name to get capacity
        backbone_name = backbone_name.lower()
        suffix = backbone_name.split('_')[-1]
        if 'small' in backbone_name or suffix == 's':
            capacity = 'S'
            embed_dims = TinyViM_width['S']
            layers = TinyViM_depth['S']
        elif 'base' in backbone_name or suffix == 'b':
            capacity = 'B'
            embed_dims = TinyViM_width['B']
            layers = TinyViM_depth['B']
        elif 'large' in backbone_name or suffix == 'l':
            capacity = 'L'
            embed_dims = TinyViM_width['L']
            layers = TinyViM_depth['L']
        else:
            raise ValueError(f"Unknown backbone capacity in {backbone_name}")

        # Default: keep height and width in stem; stage-wise downsampling happens later.
        stem_stride = kwargs.pop('stem_stride', (1, 1))
        down_stride = kwargs.pop('down_stride', (1, 2))
        height_downsample_stage = kwargs.pop('height_downsample_stage', None)
        self.patch_size = tuple(stem_stride) if isinstance(stem_stride, (list, tuple)) else (stem_stride, stem_stride)
        self.patch_stride = self.patch_size
        self.embed_dims = embed_dims
        self.use_fpn_decoder = use_fpn_decoder
        # Downsample every stage along width only: /2, /4, /8, /16 (width).
        downsamples = kwargs.pop('downsamples', [True, True, True, True])

        # Initialize TinyViM
        self.model = TinyViM(
            layers=layers,
            embed_dims=embed_dims,
            downsamples=downsamples,
            vit_num=1,
            num_classes=0, # No classification head
            fork_feat=False, # We handle feature extraction manually or change this
            stem_stride=stem_stride,
            down_stride=down_stride,
            height_downsample_stage=height_downsample_stage,
        )
        
        self.d_model = embed_dims[-1]

        # Handle input channels adaptation
        # Only reconstruct if NOT loading pretrained weights (which adapts weights in rangevit.py)
        if in_channels != 3 and not load_pretrained_stem:
            first_conv_bn = self.model.patch_embed[0]
            old_conv = first_conv_bn.c

            # Reconstruct the first Conv2d_BN block
            # args: a, b, ks, stride, pad, dilation, groups, bn_weight_init, resolution
            new_conv_bn = Conv2d_BN(
                a=in_channels,
                b=old_conv.out_channels,
                ks=old_conv.kernel_size,
                stride=old_conv.stride,
                pad=old_conv.padding, # note: Conv2d_BN uses 'pad'
                groups=old_conv.groups,
                # dilation is not stored as attr in standard conv if 1? It is.
                dilation=old_conv.dilation,
                # bn_weight_init: logic in Conv2d_BN init uses constant 1 usually.
                # We assume 1.
                bn_weight_init=1
            )

            # Replace in Sequential
            self.model.patch_embed[0] = new_conv_bn

        self.in_channels = in_channels
        
    def forward(self, x, return_features=True):
        # x: [B, C, H, W]
        
        # Forward through Stem
        # TinyViM.forward calls patch_embed then forward_tokens
        # We need intermediate output for 'skip'
        
        # patch_embed is nn.Sequential
        # 0: Conv2d_BN (stride 2)
        # 1: GELU
        # 2: Conv2d_BN (stride 2)
        # 3: GELU
        
        # RangeViT expects spatial features for FPN.
            
        # Let's look at TinyViM stem execution
        x_stem = x
        for i, layer in enumerate(self.model.patch_embed):
            x_stem = layer(x_stem)
            
        # x_stem is now [B, embed_dims[0], H/stride_h, W/stride_w]
        # Keep stem output as the first spatial feature.
        
        # Forward tokens through stages
        tokens = x_stem
        stage_features = []
        # TinyViM.forward_tokens iterates self.network modules
        # Copied from tinyvim.py
        for idx, block in enumerate(self.model.network):
            tokens = block(tokens)
            if not isinstance(block, Embedding):
                stage_features.append(tokens)
        
        # tokens is now [B, embed_dims[-1], H/32, W/32] usually (stride 32 total)

        if return_features:
            # Return spatial features only (no CLS tokens).
            if self.use_fpn_decoder:
                return stage_features, None
            return tokens, None

        return tokens
