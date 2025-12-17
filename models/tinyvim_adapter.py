import torch
import torch.nn as nn
from .tinyvim.tvimblock import Conv2d_BN
from .tinyvim.tinyvim import TinyViM, TinyViM_depth, TinyViM_width

class TinyViMAdapter(nn.Module):
    def __init__(self,
                 backbone_name='tinyvim_small',
                 image_size=(64, 2048),
                 patch_size=None,
                 patch_stride=None,
                 channels=3, # RangeViT uses 'channels' kwarg
                 in_channels=None, # For compatibility if passed explicitly
                 pretrained_path=None,
                 d_model=None, # RangeViT passes d_model, but we might ignore or verify
                 **kwargs):
        # Resolve in_channels
        if in_channels is None:
            in_channels = channels
        
        super().__init__()
        
        # Parse backbone name to get capacity
        if 'small' in backbone_name or 's' in backbone_name.split('_')[-1]:
            capacity = 'S'
            embed_dims = TinyViM_width['S']
            layers = TinyViM_depth['S']
        elif 'base' in backbone_name or 'b' in backbone_name.split('_')[-1]:
            capacity = 'B'
            embed_dims = TinyViM_width['B']
            layers = TinyViM_depth['B']
        elif 'large' in backbone_name or 'l' in backbone_name.split('_')[-1]:
            capacity = 'L'
            embed_dims = TinyViM_width['L']
            layers = TinyViM_depth['L']
        else:
            raise ValueError(f"Unknown backbone capacity in {backbone_name}")
            
        self.patch_size = (4, 4) # TinyViM Stem is 4x downsample (2x conv -> gelu -> 2x conv -> gelu). 
        # Actually in tinyvim.py: stem = Conv2d(.., stride=2) -> GELU -> Conv2d(.., stride=2) -> GELU. Total stride is 4.
        self.patch_stride = (4, 4) 
        
        # Initialize TinyViM
        self.model = TinyViM(
            layers=layers,
            embed_dims=embed_dims,
            downsamples=[True, True, True, True],
            vit_num=1,
            num_classes=0, # No classification head
            fork_feat=False # We handle feature extraction manually or change this
        )
        
        self.d_model = embed_dims[-1]
        
        # Handle input channels adaptation
        if in_channels != 3:
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
        
        # RangeViT expects skip from STEM.
        # Usually stem output is the "skip". 
        # In RangeViT/models/rangevit.py: x, skip = self.encoder(im, return_features=True)
        # In ConvStem (rangevit/models/stems.py): returns x (flattened), x_base (skip)
        # x_base is output of "conv_block" before "proj_block".
            
        # Let's look at TinyViM stem execution
        x_stem = x
        for i, layer in enumerate(self.model.patch_embed):
            x_stem = layer(x_stem)
            
        # x_stem is now [B, embed_dims[0], H/4, W/4]
        skip = x_stem
        
        # Forward tokens through stages
        tokens = x_stem
        
        # TinyViM.forward_tokens iterates self.network modules
        # Copied from tinyvim.py
        for idx, block in enumerate(self.model.network):
            tokens = block(tokens)
            # We don't need 'outs' logic from original forward_tokens unless fork_feat is True
        
        # tokens is now [B, embed_dims[-1], H/32, W/32] usually (stride 32 total)
        
        if return_features:
            # RangeViT Decoder expects:
            # x: [B, N, D] (flattened tokens)
            # BUT: 
            # In rangevit.py: 
            #   x, skip = self.encoder(im, return_features=True) # x.shape = [16, 577, 384]
            #   x = x[:, num_extra_tokens:] (removes CLS)
            #   feats = self.decoder(x, (H, W), skip)
            
            # The standard ViT returns [B, N+1, D] (CLS + tokens).
            # TinyViM does NOT have CLS token.
            # So we should return [B, N, D] or [B, 1+N, D] with dummy CLS?
            # RangeViT explicitly effectively ignores tokens it calls "extra_tokens".
            # num_extra_tokens = 1.
            # If we return [B, N, D], RangeViT will slice [:, 1:], losing the first real token!
            # So we MUST prepend a dummy CLS token.
            
            B, C, H_out, W_out = tokens.shape
            # Flatten
            tokens_flat = tokens.flatten(2).transpose(1, 2) # [B, N, C]
            
            # Add dummy CLS token
            # RangeViT encoder.cls_token is [1, 1, D]
            dummy_cls = torch.zeros((B, 1, C), device=tokens.device, dtype=tokens.dtype)
            
            # Concatenate
            out_tokens = torch.cat((dummy_cls, tokens_flat), dim=1)
            
            return out_tokens, skip
            
        return tokens

