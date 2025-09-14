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
from .rangevit_kpconv import RangeViT_KPConv, KPClassifier
from .swin_transformer_v2 import SwinTransformerV2, create_swin_v2


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
        self.PS_H, self.PS_W = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
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
        x, skip = self.patch_embed(im) # x.shape = [B, num_patches, d_model]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

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

        return x, skip


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    model_cfg.pop('backbone')
    mlp_expansion_ratio = 4
    model_cfg['d_ff'] = mlp_expansion_ratio * model_cfg['d_model']

    new_patch_size = model_cfg.pop('new_patch_size', None)
    new_patch_stride = model_cfg.pop('new_patch_stride', None)

    if new_patch_size is not None:
        if new_patch_stride is None:
            new_patch_stride = new_patch_size
        model_cfg['patch_size'] = new_patch_size
        model_cfg['patch_stride'] = new_patch_stride

    # Remove Swin-specific parameters
    model_cfg.pop('window_size', None)
    model_cfg.pop('mlp_ratio', None)
    model_cfg.pop('depths', None)
    model_cfg.pop('num_heads', None)

    model = VisionTransformer(**model_cfg)
    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop('name')
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

    # Choose encoder architecture
    backbone = model_cfg.get('backbone', 'vit_small_patch16_384')
    if backbone.startswith('swin'):
        encoder = create_swin_v2(model_cfg)
    else:
        encoder = create_vit(model_cfg)
    
    decoder = create_decoder(encoder, decoder_cfg)

    if use_kpconv:
        kpclassifier = KPClassifier(
            in_channels=decoder_cfg['d_decoder'],
            out_channels=decoder_cfg['d_decoder'],
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

        # Remove CLS tokens for decoding (only for ViT, not Swin)  
        if hasattr(self.encoder, 'cls_token') and not hasattr(self.encoder, 'layers'):
            num_extra_tokens = 1
            x = x[:, num_extra_tokens:]

        feats = self.decoder(x, (H, W), skip)
        feats = F.interpolate(feats, size=(H, W), mode='bilinear', align_corners=False)
        feats = unpadding(feats, (H_ori, W_ori))
        return feats


class RangeViT(nn.Module):
    def __init__(
            self,
            in_channels=3,
            n_cls=1000,
            backbone='vit_small_patch16_384',
            decoder='linear',
            image_size=(224, 224),
            pretrained_path=None,
            reuse_pos_emb=False,
            reuse_patch_emb=False,
            new_patch_size=None,
            new_patch_stride=None,
            conv_stem='none',
            stem_base_channels=32,
            stem_hidden_dim=None,
            up_conv_d_decoder=256,
            up_conv_scale_factor=(2, 2),
            skip_filters=0,
            use_kpconv=False,
            d_model=384,
            n_layers=12,
            n_heads=6,
            dropout=0.0,
            drop_path_rate=0.1,
            window_size=7,
            mlp_ratio=4.0,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
    ):
        super().__init__()

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
        elif backbone == 'swin_small_patch4_window7_224':
            n_heads = 3
            n_layers = sum(depths)
            patch_size = 4
            dropout = 0.0
            drop_path_rate = 0.3
            d_model = 96
        elif backbone == 'swin_base_patch4_window7_224':
            n_heads = 4
            n_layers = sum(depths)
            patch_size = 4
            dropout = 0.0
            drop_path_rate = 0.5
            d_model = 128
        elif backbone == 'swin_large_patch4_window7_224':
            n_heads = 6
            n_layers = sum(depths)
            patch_size = 4
            dropout = 0.0
            drop_path_rate = 0.5
            d_model = 192
        else:
            raise NameError('Not known ViT backbone.')

        # Decoder config
        if decoder == 'linear':
            decoder_cfg = {'n_cls': n_cls, 'name': 'linear'}
        elif decoder == 'up_conv':
            decoder_cfg = {
                'n_cls': n_cls, 'name': 'up_conv',
                'd_decoder': up_conv_d_decoder,
                'scale_factor': up_conv_scale_factor,
                'skip_filters': skip_filters,}

        # ViT encoder and stem config
        net_kwargs = {
            'backbone': backbone,
            'd_model': d_model,
            'decoder': decoder_cfg,
            'drop_path_rate': drop_path_rate,
            'dropout': dropout,
            'channels': in_channels,
            'image_size': image_size,
            'n_cls': n_cls,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'patch_size': patch_size,
            'new_patch_size': new_patch_size,
            'new_patch_stride': new_patch_stride,
            'conv_stem': conv_stem,
            'stem_base_channels': stem_base_channels,
            'stem_hidden_dim': stem_hidden_dim,
            'window_size': window_size,
            'mlp_ratio': mlp_ratio,
            'depths': depths,
            'num_heads': num_heads,
        }

        # Create RangeViT model
        self.rangevit = create_rangevit(net_kwargs, use_kpconv)

        old_state_dict = self.rangevit.state_dict()

        # Loading pre-trained weights in the ViT encoder
        if pretrained_path is not None:
            print(f'Loading pretrained parameters from {pretrained_path}')
            if pretrained_path == 'timmImageNet21k':
                vit_imagenet = timm.create_model(backbone, pretrained=True)
                pretrained_state_dict = vit_imagenet.state_dict()
                all_keys = list(pretrained_state_dict.keys())
                for key in all_keys:
                    pretrained_state_dict['encoder.'+key] = pretrained_state_dict.pop(key)
            else:
                pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
                all_keys = list(pretrained_state_dict.get('state_dict', pretrained_state_dict).keys())
                for key in all_keys:
                    if key.startswith('backbone.'):
                        new_key = key.replace('backbone.', '')
                        pretrained_state_dict['state_dict'][new_key] = pretrained_state_dict['state_dict'].pop(key)
                if 'model' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['model']
                elif 'pos_embed' in pretrained_state_dict.get('state_dict', pretrained_state_dict).keys():
                    all_keys = list(pretrained_state_dict['state_dict'].keys())
                    for key in all_keys:
                        pretrained_state_dict['encoder.'+key] = pretrained_state_dict['state_dict'].pop(key)

            # Handle positional embeddings
            if reuse_pos_emb and not backbone.startswith('swin'):
                gs_new_h = int((image_size[0] - new_patch_size[0]) // new_patch_stride[0] + 1)
                gs_new_w = int((image_size[1] - new_patch_size[1]) // new_patch_stride[1] + 1)
                num_extra_tokens = 1
                resized_pos_emb = resize_pos_embed(
                    pretrained_state_dict['encoder.pos_embed'],
                    grid_old_shape=None,
                    grid_new_shape=(gs_new_h, gs_new_w),
                    num_extra_tokens=num_extra_tokens)
                pretrained_state_dict['encoder.pos_embed'] = resized_pos_emb
            elif backbone.startswith('swin'):
                # Skip pos_embed for Swin, as it uses relative position bias
                if 'encoder.pos_embed' in pretrained_state_dict:
                    del pretrained_state_dict['encoder.pos_embed']

            # Reuse pre-trained patch embeddings
            if reuse_patch_emb:
                assert conv_stem == 'none'
                print('Reusing patch embeddings.')
                assert old_state_dict['encoder.patch_embed.proj.bias'].shape == pretrained_state_dict['encoder.patch_embed.proj.bias'].shape
                old_state_dict['encoder.patch_embed.proj.bias'] = pretrained_state_dict['encoder.patch_embed.proj.bias']
                _, _, gs_new_h, gs_new_w = old_state_dict['encoder.patch_embed.proj.weight'].shape
                reshaped_weight = adapt_input_conv(in_channels, pretrained_state_dict['encoder.patch_embed.proj.weight'])
                reshaped_weight = F.interpolate(reshaped_weight, size=(gs_new_h, gs_new_w), mode='bilinear')
                pretrained_state_dict['encoder.patch_embed.proj.weight'] = reshaped_weight
            elif conv_stem == 'ConvStem' and 'encoder.patch_embed.proj.weight' in pretrained_state_dict:
                # Adapt ConvStem's first conv layer from pretrained patch_embed
                conv1_weight = adapt_input_conv(in_channels, pretrained_state_dict['encoder.patch_embed.proj.weight'])
                pretrained_state_dict['encoder.patch_embed.conv_block.0.conv1.weight'] = conv1_weight
            else:
                if 'encoder.patch_embed.projection.weight' in pretrained_state_dict:
                    del pretrained_state_dict['encoder.patch_embed.projection.weight']
                if 'encoder.patch_embed.projection.bias' in pretrained_state_dict:
                    del pretrained_state_dict['encoder.patch_embed.projection.bias']

            # Delete pretrained weights of decoder and kpclassifier
            pretrained_dict = pretrained_state_dict.get('state_dict', pretrained_state_dict)
            decoder_keys = [key for key in pretrained_dict.keys() if 'decoder' in key or 'kpclassifier' in key]
            for key in decoder_keys:
                pretrained_dict.pop(key, None)

            msg = self.rangevit.load_state_dict(pretrained_state_dict, strict=False)
            print(f'{msg}')

    def counter_model_parameters(self):
        stats = {}
        stats['total_num_parameters'] = count_parameters(self.rangevit)
        stats['decoder_num_parameters'] = count_parameters(self.rangevit.decoder)
        stats['stem_num_parameters'] = count_parameters(self.rangevit.encoder.patch_embed)
        stats['encoder_num_parameters'] = count_parameters(self.rangevit.encoder) - stats['stem_num_parameters']
        return stats

    def forward(self, *args):
        return self.rangevit(*args)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = RangeViT(
        in_channels=5,
        n_cls=17,
        backbone='swin_small_patch4_window7_224',
        decoder='up_conv',
        image_size=(32, 384),
        pretrained_path='/path/to/swinv2_small_window8_256.pth',
        reuse_pos_emb=False,
        reuse_patch_emb=False,
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
        conv_stem='ConvStem',
        stem_base_channels=32,
        stem_hidden_dim=256,
        up_conv_d_decoder=256,
        up_conv_scale_factor=(2, 8),
        skip_filters=256,
        use_kpconv=True,
        d_model=96,
        n_layers=24,  # Sum of depths for Swin
        n_heads=3,
        dropout=0.0,
        drop_path_rate=0.3,
        window_size=7,
        mlp_ratio=4.0,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
    )
    x = torch.randn(1, 5, 32, 384)
    px = torch.randn(1000)
    py = torch.randn(1000)
    pxyz = torch.randn(1000, 3)
    pknn = torch.randint(0, 1000, (1000, 16))
    num_points = torch.tensor([1000])
    predictions = model(x, px, py, pxyz, pknn, num_points)
    # print(predictions.shape)
    # print(model.counter_model_parameters())