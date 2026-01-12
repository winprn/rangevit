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

from .model_utils import adapt_input_conv, padding, unpadding, resize_pos_embed
from .decoders import DecoderLinear, DecoderUpConv
from .rangevit_kpconv import RangeViT_KPConv, KPClassifier
from .encoders.factory import create_encoder
from .encoders.tinyvim import TinyViMEncoder
from .tinyvim.fpn_decoder import TinyViMFPNDecoder


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
    elif name == 'fpn':
        if not isinstance(encoder, TinyViMEncoder):
            raise ValueError('FPN decoder is only supported for TinyViM backbones.')
        decoder = TinyViMFPNDecoder(
            in_channels=encoder.embed_dims,
            n_cls=decoder_cfg['n_cls'],
            out_channels=decoder_cfg.get('fpn_out_channels', 256),
            head_channels=decoder_cfg.get('fpn_head_channels', 128),
            dropout_ratio=decoder_cfg.get('fpn_dropout', 0.1),
        )
    else:
        raise ValueError(f'Unknown decoder: {name}')
    return decoder


def create_range_model(model_cfg, use_kpconv=False):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop('decoder')
    decoder_cfg['n_cls'] = model_cfg['n_cls']

    encoder = create_encoder(model_cfg, decoder_name=decoder_cfg.get('name'))
    
    decoder = create_decoder(encoder, decoder_cfg)

    if use_kpconv:
        kpclassifier = KPClassifier(
            in_channels=decoder_cfg['d_decoder'] ,
            out_channels=decoder_cfg['d_decoder'],
            num_classes=model_cfg['n_cls'])
        model = RangeViT_KPConv(encoder, decoder, kpclassifier, n_cls=model_cfg['n_cls'])
    else:
        model = RangeSegNoKPConv(encoder, decoder, n_cls=model_cfg['n_cls'])

    return model


def create_rangevit(model_cfg, use_kpconv=False):
    return create_range_model(model_cfg, use_kpconv=use_kpconv)


class RangeSegNoKPConv(nn.Module):
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

        x, skip = self.encoder(im, return_features=True) # x.shape = [16, 577, 384]

        # remove CLS tokens for decoding
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:] # x.shape = [16, 576, 384]

        feats = self.decoder(x, (H, W), skip) # feats.shape = [16, 17, 24, 24]
        feats = F.interpolate(feats, size=(H, W), mode='bilinear')
        feats = unpadding(feats, (H_ori, W_ori)) # feats.shape = [16, 17, 384, 384]

        return feats


class RangeSeg(nn.Module):
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
        pretrained_channel_adaptation='repeat',
        conv_stem='none',
        stem_base_channels=32,
        stem_hidden_dim=None,
        skip_filters=0,
        decoder='up_conv',
        up_conv_d_decoder=64,
        up_conv_scale_factor=(2, 8),
        fpn_out_channels=256,
        fpn_head_channels=128,
        fpn_dropout=0.1,
        use_kpconv=False,
        ):
        super(RangeSeg, self).__init__()

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
        elif backbone == 'swin_small_patch4_window7_224':
            n_heads = 3
            n_layers = 12
            patch_size = 4
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 96
        elif backbone == 'swin_base_patch4_window7_224':
            n_heads = 4
            n_layers = 12
            patch_size = 4
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 128
        elif backbone == 'swin_large_patch4_window7_224':
            n_heads = 6
            n_layers = 12
            patch_size = 4
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 192
        elif backbone == 'tinyvim_small':
            n_heads = 1
            n_layers = 0 
            patch_size = 4
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 224
        elif backbone == 'tinyvim_base':
            n_heads = 1
            n_layers = 0
            patch_size = 4
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 384
        elif backbone == 'tinyvim_large':
            n_heads = 1
            n_layers = 0
            patch_size = 4
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 512
        else:
            raise NameError('Not known ViT backbone.')

        # Decoder config
        if decoder == 'linear':
            decoder_cfg = {'n_cls': n_cls, 'name': 'linear'}
        elif decoder == 'up_conv':
            decoder_cfg = {
                'n_cls': n_cls, 'name': 'up_conv',
                'd_decoder': up_conv_d_decoder, # hidden dim of the decoder
                'scale_factor': up_conv_scale_factor, # scaling factor in the PixelShuffle layer
                'skip_filters': skip_filters,} # channel dim of the skip connection (between the convolutional stem and the up_conv decoder)
        elif decoder == 'fpn':
            decoder_cfg = {
                'n_cls': n_cls, 'name': 'fpn',
                'fpn_out_channels': fpn_out_channels,
                'fpn_head_channels': fpn_head_channels,
                'fpn_dropout': fpn_dropout,
                # Needed for KPConv head channel size; FPN returns head_channels when return_features=True.
                'd_decoder': fpn_head_channels,
            }

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
        }


        # Create RangeSeg model
        self.rangevit = create_range_model(net_kwargs, use_kpconv)

        old_state_dict = self.rangevit.state_dict()

        # Loading pre-trained weights in the ViT encoder
        if pretrained_path is not None:
            print(f'Loading pretrained parameters from {pretrained_path}')
            if pretrained_path == 'timmImageNet21k':
                vit_imagenet = timm.create_model(backbone, pretrained=True) #.cuda()
                pretrained_state_dict = vit_imagenet.state_dict() # nb keys: 152
                all_keys = list(pretrained_state_dict.keys())
                for key in all_keys:
                    pretrained_state_dict['encoder.'+key] = pretrained_state_dict.pop(key)
            else:
                pretrained_state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                if 'state_dict' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['state_dict']
                
                all_keys = list(pretrained_state_dict.keys())
                # all_keys = list(pretrained_state_dict['state_dict'].keys())
                for key in all_keys:
                    if key.startswith('backbone.'):
                        new_key = key.replace('backbone.', '')
                        pretrained_state_dict[new_key] = pretrained_state_dict.pop(key)
                if 'model' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['model']
                elif 'pos_embed' in pretrained_state_dict.keys():
                    all_keys = list(pretrained_state_dict.keys())
                    for key in all_keys:
                        pretrained_state_dict['encoder.'+key] = pretrained_state_dict.pop(key)
                elif backbone.startswith('tinyvim'):
                    all_keys = list(pretrained_state_dict.keys())
                    for key in all_keys:
                        # TinyViMAdapter has .model attribute
                        pretrained_state_dict['encoder.model.'+key] = pretrained_state_dict.pop(key)

                    # Adapt first conv layer for 5-channel input (TinyViM specific)
                    if in_channels != 3:
                        first_conv_key = 'encoder.model.patch_embed.0.c.weight'

                        if first_conv_key in pretrained_state_dict:
                            original_weight = pretrained_state_dict[first_conv_key]

                            # Check if already adapted (e.g., loading a finetuned checkpoint)
                            if original_weight.shape[1] == in_channels:
                                print(f'First conv already has {in_channels} channels, skipping adaptation.')
                            elif original_weight.shape[1] == 3:
                                print(f'Adapting TinyViM first conv: 3 → {in_channels} channels '
                                      f'(method: {pretrained_channel_adaptation})')
                                print(f'  Original shape: {original_weight.shape}')

                                adapted_weight = adapt_input_conv(
                                    in_channels,
                                    original_weight,
                                    method=pretrained_channel_adaptation
                                )
                                pretrained_state_dict[first_conv_key] = adapted_weight
                                print(f'  Adapted shape: {adapted_weight.shape}')
                            else:
                                raise ValueError(f'Unexpected input channels in checkpoint: {original_weight.shape[1]}')
                        else:
                            print(f'WARNING: {first_conv_key} not found in checkpoint. '
                                  f'First conv will be randomly initialized.')

            # Reuse pre-trained positional embeddings
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
            # else:
            #     del pretrained_state_dict['encoder.pos_embed'] # remove positional embeddings

            # Reuse pre-trained patch embeddings
            # Only if not TinyViM (which handles its own stem/embeds differently)
            if reuse_patch_emb and not backbone.startswith('tinyvim'):
                assert conv_stem=='none' # no patch embedding if a convolutional stem is used
                print('Reusing patch embeddings.')

                assert old_state_dict['encoder.patch_embed.proj.bias'].shape == pretrained_state_dict['encoder.patch_embed.proj.bias'].shape
                old_state_dict['encoder.patch_embed.proj.bias'] = pretrained_state_dict['encoder.patch_embed.proj.bias']

                _, _, gs_new_h, gs_new_w = old_state_dict['encoder.patch_embed.proj.weight'].shape
                reshaped_weight = adapt_input_conv(in_channels, pretrained_state_dict['encoder.patch_embed.proj.weight'])
                reshaped_weight = F.interpolate(reshaped_weight, size=(gs_new_h, gs_new_w), mode='bilinear')
                pretrained_state_dict['encoder.patch_embed.proj.weight'] = reshaped_weight
            # else:
            #     del pretrained_state_dict['encoder.patch_embed.projection.weight'] # remove patch embedding layers
            #     del pretrained_state_dict['encoder.patch_embed.projection.bias'] # remove patch embedding layers

            # Delete the pre-trained weights of the decoder
            decoder_keys = []
            for key in pretrained_state_dict.keys():
                if 'decoder' in key:
                    decoder_keys.append(key)
            # for decoder_key in decoder_keys:
            #     del pretrained_state_dict[decoder_key]

            msg = self.rangevit.load_state_dict(pretrained_state_dict, strict=False)
            print(f'{msg}')

    def counter_model_parameters(self):
        stats = {}
        stats['total_num_parameters'] = count_parameters(self.rangevit)
        stats['decoder_num_parameters'] = count_parameters(self.rangevit.decoder)
        # TinyViMAdapter does not expose patch_embed; fall back to its internal stem if present
        stem_params = 0
        encoder = self.rangevit.encoder
        if hasattr(encoder, 'patch_embed'):
            stem_params = count_parameters(encoder.patch_embed)
        elif hasattr(encoder, 'model') and hasattr(encoder.model, 'patch_embed'):
            stem_params = count_parameters(encoder.model.patch_embed)
        stats['stem_num_parameters'] = stem_params
        stats['encoder_num_parameters'] = count_parameters(encoder) - stem_params
        return stats

    def forward(self, *args):
        return self.rangevit(*args)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = RangeSeg(in_channels=5,
                     n_cls=17,
                     backbone='vit_small_patch16_384',
                     decoder='linear',
                     image_size=(32, 384),
                     pretrained_path='/root/checkpoint.pth',
                     reuse_pos_emb=True)

    predictions = model(x)
