# Copyright 2023 - Valeo Comfort and Driving Assistance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import timm
from timm.models.layers import trunc_normal_

from .blocks import Block
from .model_utils import adapt_input_conv, padding, unpadding, resize_pos_embed, init_weights
from .stems import PatchEmbedding, ConvStem
from .decoders import DecoderLinear, DecoderUpConv
from .decoders_multiscale import DecoderMultiScaleFPN
from .rangevit_kpconv import RangeViT_KPConv, KPClassifier
from .swin_transformer_fixed import create_swin_backbone, SwinVisionTransformer


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        channels=3,
        ls_init_values=None,
        patch_stride=None,
        conv_stem='none',
        stem_base_channels=32,
        stem_hidden_dim=None,
    ):
        super().__init__()

        self.conv_stem = conv_stem

        if self.conv_stem == 'none':
            self.patch_embed = PatchEmbedding(
                image_size,
                patch_size,
                patch_stride,
                d_model,
                channels,)
        else:   # in this case self.conv_stem = 'ConvStem'
            assert patch_stride == patch_size # patch_size = patch_stride if a convolutional stem is used

            self.patch_embed = ConvStem(
                in_channels=channels,
                base_channels=stem_base_channels,
                img_size=image_size,
                patch_stride=patch_stride,
                embed_dim=d_model,
                flatten=True,
                hidden_dim=stem_hidden_dim)

        self.patch_size = patch_size
        self.PS_H, self.PS_W = patch_size
        self.patch_stride = patch_stride
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        self.image_size = image_size

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model))

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
                [Block(d_model, n_heads, d_ff, dropout, dpr[i], init_values=ls_init_values) for i in range(n_layers)]
            )

        self.norm = nn.LayerNorm(d_model)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_grid_size(self, H, W):
        return self.patch_embed.get_grid_size(H, W)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        x, skip = self.patch_embed(im) # x.shape = [16, 576, 384]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # x.shape = [16, 577, 384]

        pos_embed = self.pos_embed
        num_extra_tokens = 1

        if x.shape[1] != pos_embed.shape[1]:
            grid_H, grid_W = self.get_grid_size(H, W)
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (grid_H, grid_W),
                num_extra_tokens,
            )

        x = x + pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, skip  # x.shape = [16, 577, 384]


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone_name = model_cfg.pop('backbone')

    # Check if this is a Swin model
    if backbone_name.startswith('swin'):
        # Extract multi-scale configuration
        use_all_stages = model_cfg.get('use_all_stages', False)  # Phase 2A: multi-scale mode
        native_input = model_cfg.get('native_input', False)     # Phase 2C: native input mode
        use_stage = model_cfg.get('use_stage', 2)               # Default F2 for single-scale

        # Create Swin-ViT wrapper (creates backbone internally with auto-fallback)
        model = SwinVisionTransformer(
            model_name=backbone_name,
            channels=model_cfg['channels'],
            use_stage=use_stage,
            multi_scale=use_all_stages,
            native_input=native_input
        )

        # Set required attributes for compatibility
        model.image_size = model_cfg['image_size']
        model.n_cls = model_cfg['n_cls']

        return model

    else:
        # Original ViT path
        mlp_expansion_ratio = 4
        model_cfg['d_ff'] = mlp_expansion_ratio * model_cfg['d_model']

        new_patch_size = model_cfg.pop('new_patch_size')
        new_patch_stride = model_cfg.pop('new_patch_stride')

        if (new_patch_size is not None):
            if new_patch_stride is None:
                new_patch_stride = new_patch_size
            model_cfg['patch_size'] = new_patch_size
            model_cfg['patch_stride'] = new_patch_stride

        model = VisionTransformer(**model_cfg)

        return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop('name')

    if name == 'multi_scale_fpn':
        # Phase 2A: Multi-scale decoder for Swin Transformer
        # Extract Swin-specific parameters
        if hasattr(encoder, 'swin_backbone') and hasattr(encoder, 'multi_scale') and encoder.multi_scale:
            # Multi-scale Swin encoder
            if hasattr(encoder.swin_backbone, 'get_feature_dims'):
                # Real Swin backbone
                swin_channels = encoder.swin_backbone.get_feature_dims()
            elif hasattr(encoder.swin_backbone, 'feature_dims'):
                # Real MockSwinBackbone case
                swin_channels = encoder.swin_backbone.feature_dims
            elif hasattr(encoder.swin_backbone, 'mock_swin'):
                # MockSwinVisionTransformer case
                swin_channels = encoder.swin_backbone.mock_swin.feature_dims
            else:
                # Default fallback
                swin_channels = [96, 192, 384, 768]
        else:
            # Default Swin-Tiny channels for backward compatibility
            swin_channels = [96, 192, 384, 768]

        decoder = DecoderMultiScaleFPN(
            swin_channels=swin_channels,
            **decoder_cfg
        )
        # Set d_decoder for KPConv compatibility
        # Multi-scale FPN outputs at pyramid_channels dimension
        decoder.d_decoder = decoder_cfg.get('pyramid_channels', 256)
    else:
        # Original decoders (linear, up_conv) for ViT and single-scale Swin
        decoder_cfg['d_encoder'] = encoder.d_model
        decoder_cfg['patch_size'] = encoder.patch_size

        if name == 'linear':
            decoder_cfg['patch_stride'] = encoder.patch_stride
            decoder = DecoderLinear(**decoder_cfg)
        elif name == 'up_conv':
            decoder_cfg['patch_stride'] = encoder.patch_stride
            decoder = DecoderUpConv(**decoder_cfg)
        else:
            raise ValueError(f'Unknown decoder: {name}')

    return decoder


def create_rangevit(model_cfg, use_kpconv=False):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop('decoder')
    decoder_cfg['n_cls'] = model_cfg['n_cls']

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)

    if use_kpconv:
        # Get d_decoder from decoder object (for multi-scale) or config (for traditional decoders)
        d_decoder = getattr(decoder, 'd_decoder', decoder_cfg.get('d_decoder'))
        if d_decoder is None:
            raise ValueError("d_decoder not found in decoder config or decoder object for KPConv")

        kpclassifier = KPClassifier(
            in_channels=d_decoder,
            out_channels=d_decoder,
            num_classes=model_cfg['n_cls'])
        model = RangeViT_KPConv(encoder, decoder, kpclassifier, n_cls=model_cfg['n_cls'])
    else:
        model = RangeViT_noKPConv(encoder, decoder, n_cls=model_cfg['n_cls'])

    return model


class RangeViT_noKPConv(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.patch_stride = encoder.patch_stride
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay('encoder.', self.encoder).union(
            append_prefix_no_weight_decay('decoder.', self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x, skip = self.encoder(im, return_features=True)

        # Check if this is a Swin encoder and if it's in multi-scale mode
        is_swin = hasattr(self.encoder, 'swin_backbone')
        is_multi_scale = hasattr(self.encoder, 'multi_scale') and self.encoder.multi_scale

        if is_swin and is_multi_scale:
            # Phase 2A: Multi-scale Swin mode
            # x is now a list of [F0, F1, F2, F3] feature maps
            multi_scale_features = x

            # Pass directly to multi-scale decoder
            # The multi-scale decoder handles upsampling to original resolution internally
            feats = self.decoder(multi_scale_features, (H_ori, W_ori), skip)

        elif is_swin:
            # Phase 1: Single-scale Swin mode (backward compatibility)
            # For Swin: remove dummy CLS token (first token)
            x = x[:, 1:]  # Remove dummy CLS token added for compatibility

            # For Swin, we need to handle the decoder differently
            # The decoder expects tokens that can be reshaped to a 2D grid
            # We need to pass dimensions that match the actual feature map size

            # Get actual grid size from Swin backbone
            actual_grid = self.encoder.get_actual_grid_size()
            if actual_grid is not None:
                feat_H, feat_W = actual_grid
                # For the decoder, we need to find dimensions that work with its patch calculations
                # The decoder calculates: GS_H = get_grid_size_1d(H, PS_H, H_stride)
                # We want: GS_H = feat_H, so we need to find H such that this works

                # Get decoder's patch info
                if hasattr(self.decoder, 'patch_size') and hasattr(self.decoder, 'patch_stride'):
                    dec_patch_size = self.decoder.patch_size
                    dec_patch_stride = self.decoder.patch_stride

                    if isinstance(dec_patch_size, int):
                        PS_H = PS_W = dec_patch_size
                    else:
                        PS_H, PS_W = dec_patch_size

                    if dec_patch_stride is not None:
                        if isinstance(dec_patch_stride, int):
                            H_stride = W_stride = dec_patch_stride
                        else:
                            H_stride, W_stride = dec_patch_stride
                    else:
                        H_stride, W_stride = PS_H, PS_W

                    # Calculate input dimensions that would produce the desired grid size
                    # From get_grid_size_1d: grid_size = (length - patch_size) // stride + 1
                    # Solving for length: length = (grid_size - 1) * stride + patch_size
                    decoder_H = (feat_H - 1) * H_stride + PS_H
                    decoder_W = (feat_W - 1) * W_stride + PS_W
                else:
                    # Fallback: use stride-based calculation
                    stride = self.encoder.swin_backbone.strides[self.encoder.use_stage]
                    decoder_H, decoder_W = feat_H * stride, feat_W * stride
            else:
                decoder_H, decoder_W = H, W

            feats = self.decoder(x, (decoder_H, decoder_W), skip)
            feats = F.interpolate(feats, size=(H, W), mode='bilinear')

        else:
            # Original ViT mode
            # For ViT: remove real CLS token
            num_extra_tokens = 1
            x = x[:, num_extra_tokens:]
            decoder_H, decoder_W = H, W

            feats = self.decoder(x, (decoder_H, decoder_W), skip)
            feats = F.interpolate(feats, size=(H, W), mode='bilinear')

        # Final unpadding (only needed for non-multi-scale decoders)
        if not (is_swin and is_multi_scale):
            feats = unpadding(feats, (H_ori, W_ori))

        return feats


class RangeViT(nn.Module):
    def __init__(
        self,
        in_channels=5,
        n_cls=17,
        backbone='vit_small_patch16_384',
        image_size=(32, 384),
        pretrained_path=None,
        new_patch_size=None,
        new_patch_stride=None,
        reuse_pos_emb=False,
        reuse_patch_emb=False,
        conv_stem='none',
        stem_base_channels=32,
        stem_hidden_dim=None,
        skip_filters=0,
        decoder='up_conv',
        up_conv_d_decoder=64,
        up_conv_scale_factor=(2, 8),
        use_kpconv=False,
        use_all_stages=False,  # Multi-scale mode for Swin transformers
        native_input=False,    # Native input mode for Swin transformers
        ):
        super(RangeViT, self).__init__()

        self.n_cls = n_cls

        if backbone == 'vit_small_patch16_384':
            n_heads = 6
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 384
        elif backbone == 'vit_base_patch16_384':
            n_heads = 12
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 768
        elif backbone == 'vit_large_patch16_384':
            n_heads = 16
            n_layers = 24
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 1024
        elif backbone == 'swin_tiny_patch4_window7_224':
            # Swin-Tiny configuration
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            n_heads = 12  # Use F2 stage heads for compatibility
            n_layers = sum(depths)  # Total layers
            window_size = 7
            patch_size = 4
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 384  # Use F2 stage output dimension
        elif backbone == 'swinv2_tiny_window16_256':
            # SwinV2-Tiny configuration
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            n_heads = 12  # Use F2 stage heads for compatibility
            n_layers = sum(depths)  # Total layers
            window_size = 16
            patch_size = 4
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 384  # Use F2 stage output dimension
        else:
            raise NameError('Not known ViT/Swin backbone.')

        # Decoder config
        print(f'Decoder: {decoder}')
        if decoder == 'linear':
            decoder_cfg = {'n_cls': n_cls, 'name': 'linear'}
        elif decoder == 'up_conv':
            decoder_cfg = {
                'n_cls': n_cls, 'name': 'up_conv',
                'd_decoder': up_conv_d_decoder, # hidden dim of the decoder
                'scale_factor': up_conv_scale_factor, # scaling factor in the PixelShuffle layer
                'skip_filters': skip_filters,} # channel dim of the skip connection (between the convolutional stem and the up_conv decoder)
        elif decoder == 'multi_scale_fpn':
            decoder_cfg = {
                'n_cls': n_cls,
                'name': 'multi_scale_fpn',
                'pyramid_channels': 256,  # Default unified FPN channel dimension
                'use_ppm': True,          # Enable Pyramid Pooling Module
                'ppm_scales': [1, 2, 3, 6]  # PPM pooling scales
            }
        else:
            raise NameError('Not known decoder.')

        # ViT encoder and stem config
        net_kwargs = {
            'backbone': backbone,
            'd_model': d_model, # dim of features
            'decoder': decoder_cfg,
            'drop_path_rate': drop_path_rate,
            'dropout': dropout,
            'channels': in_channels, # nb of channels for the 3D point projections
            'image_size': image_size,
            'n_cls': n_cls,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'patch_size': patch_size, # old patch size for the ViT encoder
            'new_patch_size': new_patch_size, # new patch size for the ViT encoder
            'new_patch_stride': new_patch_stride, # new patch stride for the ViT encoder
            'conv_stem': conv_stem,
            'stem_base_channels': stem_base_channels,
            'stem_hidden_dim': stem_hidden_dim,
            # Multi-scale configuration for Swin transformers
            'use_all_stages': use_all_stages,
            'native_input': native_input,
        }


        # Create RangeViT model
        self.rangevit = create_rangevit(net_kwargs, use_kpconv)

        old_state_dict = self.rangevit.state_dict()

        # Loading pre-trained weights in the encoder
        if pretrained_path is not None:
            print(f'Loading pretrained parameters from {pretrained_path}')

            # Check if this is a Swin model
            is_swin_model = backbone.startswith('swin')

            if pretrained_path == 'timmImageNet21k':
                if is_swin_model:
                    # For Swin models, load through timm directly in the backbone
                    print('Note: Swin models will use timm pretrained weights during backbone creation')
                    pretrained_state_dict = {}  # Empty dict, weights loaded in backbone
                else:
                    # Original ViT path
                    vit_imagenet = timm.create_model(backbone, pretrained=True)
                    pretrained_state_dict = vit_imagenet.state_dict()
                    all_keys = list(pretrained_state_dict.keys())
                    for key in all_keys:
                        pretrained_state_dict['encoder.'+key] = pretrained_state_dict.pop(key)
            else:
                # Load from file
                pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
                if 'model' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['model']

                if is_swin_model:
                    # Handle Swin weight loading
                    print('Loading Swin Transformer weights...')

                    # For Swin models, we need to prefix keys properly
                    # The structure is: rangevit.encoder.swin.{swin_keys}
                    all_keys = list(pretrained_state_dict.keys())
                    for key in all_keys:
                        # Add proper prefix for Swin weights
                        new_key = f'rangevit.encoder.swin.{key}'
                        pretrained_state_dict[new_key] = pretrained_state_dict.pop(key)

                    # Remove ViT-specific keys that don't exist in Swin
                    keys_to_remove = []
                    for key in list(pretrained_state_dict.keys()):
                        # Remove keys that are ViT-specific
                        if any(skip_key in key for skip_key in ['pos_embed', 'cls_token', 'head']):
                            keys_to_remove.append(key)

                    for key in keys_to_remove:
                        del pretrained_state_dict[key]
                        print(f'Removed ViT-specific key: {key}')

                else:
                    # Original ViT handling
                    if 'pos_embed' in pretrained_state_dict.keys():
                        all_keys = list(pretrained_state_dict.keys())
                        for key in all_keys:
                            pretrained_state_dict['encoder.'+key] = pretrained_state_dict.pop(key)

            # Handle positional embeddings (ViT only)
            if not is_swin_model:
                # ViT positional embedding handling
                if reuse_pos_emb:
                    # Resize the existing position embeddings to the desired size
                    print('Reusing positional embeddings.')
                    gs_new_h = int((image_size[0] - new_patch_size[0]) // new_patch_stride[0] + 1)
                    gs_new_w = int((image_size[1] - new_patch_size[1]) // new_patch_stride[1] + 1)
                    num_extra_tokens = 1
                    resized_pos_emb = resize_pos_embed(pretrained_state_dict['encoder.pos_embed'],
                                                       grid_old_shape=None,
                                                       grid_new_shape=(gs_new_h, gs_new_w),
                                                       num_extra_tokens=num_extra_tokens)
                    pretrained_state_dict['encoder.pos_embed'] = resized_pos_emb
                else:
                    if 'encoder.pos_embed' in pretrained_state_dict:
                        del pretrained_state_dict['encoder.pos_embed'] # remove positional embeddings
            else:
                # Swin doesn't use absolute positional embeddings
                print('Swin Transformer uses relative positional bias, skipping pos_embed handling.')

            # Handle patch embeddings (ViT only)
            if not is_swin_model:
                # ViT patch embedding handling
                if reuse_patch_emb:
                    assert conv_stem=='none' # no patch embedding if a convolutional stem is used
                    print('Reusing patch embeddings.')

                    assert old_state_dict['encoder.patch_embed.proj.bias'].shape == pretrained_state_dict['encoder.patch_embed.proj.bias'].shape
                    old_state_dict['encoder.patch_embed.proj.bias'] = pretrained_state_dict['encoder.patch_embed.proj.bias']

                    _, _, gs_new_h, gs_new_w = old_state_dict['encoder.patch_embed.proj.weight'].shape
                    reshaped_weight = adapt_input_conv(in_channels, pretrained_state_dict['encoder.patch_embed.proj.weight'])
                    reshaped_weight = F.interpolate(reshaped_weight, size=(gs_new_h, gs_new_w), mode='bilinear')
                    pretrained_state_dict['encoder.patch_embed.proj.weight'] = reshaped_weight
                else:
                    if 'encoder.patch_embed.proj.weight' in pretrained_state_dict:
                        del pretrained_state_dict['encoder.patch_embed.proj.weight'] # remove patch embedding layers
                    if 'encoder.patch_embed.proj.bias' in pretrained_state_dict:
                        del pretrained_state_dict['encoder.patch_embed.proj.bias'] # remove patch embedding layers
            else:
                # For Swin, handle input channel adaptation at the patch embedding level
                print('Swin Transformer patch embedding will be adapted for multi-channel input.')
                # Note: Channel adaptation for Swin happens in the timm model creation

            # Delete the pre-trained weights of the decoder
            decoder_keys = []
            for key in pretrained_state_dict.keys():
                if 'decoder' in key:
                    decoder_keys.append(key)
            for decoder_key in decoder_keys:
                del pretrained_state_dict[decoder_key]

            msg = self.rangevit.load_state_dict(pretrained_state_dict, strict=False)
            print(f'{msg}')

    def counter_model_parameters(self):
        stats = {}
        stats['total_num_parameters'] = count_parameters(self.rangevit)
        stats['decoder_num_parameters'] = count_parameters(self.rangevit.decoder)

        # Handle both ViT and Swin models
        if hasattr(self.rangevit.encoder, 'patch_embed'):
            # Original ViT model
            stats['stem_num_parameters'] = count_parameters(self.rangevit.encoder.patch_embed)
        elif hasattr(self.rangevit.encoder, 'swin'):
            # Swin model - count patch embedding in the backbone
            stats['stem_num_parameters'] = count_parameters(self.rangevit.encoder.swin.swin.patch_embed) if hasattr(self.rangevit.encoder.swin.swin, 'patch_embed') else 0
        else:
            stats['stem_num_parameters'] = 0

        stats['encoder_num_parameters'] = count_parameters(self.rangevit.encoder) - stats['stem_num_parameters']
        return stats

    def forward(self, *args):
        return self.rangevit(*args)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = RangeViT(in_channels=5,
                     n_cls=17,
                     backbone='vit_small_patch16_384',
                     decoder='linear',
                     image_size=(32, 384),
                     pretrained_path='/root/checkpoint.pth',
                     reuse_pos_emb=True)

    predictions = model(x)
