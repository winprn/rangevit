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
from timm.models.layers import DropPath, trunc_normal_

from .blocks import Block
from .model_utils import adapt_input_conv, padding, unpadding, resize_pos_embed, init_weights
from .stems import PatchEmbedding, ConvStem
from .decoders import DecoderLinear, DecoderUpConv
from .rangevit_kpconv import RangeViT_KPConv, KPClassifier
from .swin_transformer_v2 import SwinTransformerV2, create_swin_v2


def project_to_bev(points, bev_size=None, **kwargs):
    """
    Placeholder for projecting 3D point clouds to a BEV raster.
    This is intentionally minimal; BEV projection is now handled in the data loader.
    See dataset.preprocess.bev_projection.BEVProjection for the actual implementation.
    """
    raise NotImplementedError(
        "BEV projection should be done in the data loader. "
        "See dataset/preprocess/bev_projection.py for implementation. "
        "Enable BEV by setting use_bev=True in your config file."
    )


class BEVEncoder(nn.Module):
    """
    Lightweight BEV encoder that downsamples a rasterized BEV tensor and projects it
    to the ViT token dimension. GroupNorm is used to stay consistent with transformer
    normalization choices.
    """
    def __init__(self, in_channels, embed_dim, base_channels=64, num_layers=3, dropout=0.0):
        super().__init__()
        layers = []
        in_ch = in_channels
        out_ch = base_channels
        for i in range(num_layers):
            stride = 2 if i < num_layers - 1 else 1
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1))
            layers.append(nn.GroupNorm(num_groups=8, num_channels=out_ch))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = out_ch
            out_ch = min(out_ch * 2, embed_dim)
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1)

    def forward(self, bev_image):
        """
        Args:
            bev_image: [B, C_b, H_b, W_b]
        Returns:
            torch.Tensor: [B, embed_dim, H_b, W_b]
        """
        input_hw = bev_image.shape[-2:]
        x = self.body(bev_image)
        # Restore the original BEV grid resolution if downsampled.
        if x.shape[-2:] != input_hw:
            x = F.interpolate(x, size=input_hw, mode='bilinear', align_corners=False)
        return self.proj(x)


class CrossModalAdapter(nn.Module):
    """
    BALViT-style adapter that lets range-view tokens attend to BEV tokens.
    Queries come from RV tokens, keys/values from BEV tokens.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        hidden_dim = int(dim * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, rv_tokens, bev_tokens):
        """
        Args:
            rv_tokens: [B, N_r, C] range-view tokens (queries)
            bev_tokens: [B, N_b, C] BEV tokens (keys/values)
        Returns:
            torch.Tensor: [B, N_r, C] enhanced RV tokens
        """
        q = self.norm_q(rv_tokens)
        kv = self.norm_kv(bev_tokens)
        attn_out, _ = self.attn(q, kv, kv)
        rv_tokens = rv_tokens + self.drop_path(attn_out)
        rv_tokens = rv_tokens + self.drop_path(self.mlp(self.norm_mlp(rv_tokens)))
        return rv_tokens


class BEVDecoder(nn.Module):
    """
    Minimal BEV decoder that maps BEV features back to class logits on the BEV grid.
    """
    def __init__(self, in_channels, n_cls, hidden_channels=128, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_channels, n_cls, kernel_size=1)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, bev_feat):
        logits = self.body(bev_feat)
        return self.head(logits)


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
        bev_channels=None,
        bev_base_channels=64,
        bev_num_layers=3,
        adapter_indices=None,
        adapter_mlp_ratio=4.0,
        freeze_vit=False,
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
        self.adapter_indices = sorted(adapter_indices) if adapter_indices else []

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

        # Optional BEV branch
        self.bev_encoder = None
        if bev_channels is not None:
            self.bev_encoder = BEVEncoder(
                in_channels=bev_channels,
                embed_dim=d_model,
                base_channels=bev_base_channels,
                num_layers=bev_num_layers,
                dropout=dropout,
            )

        self.cross_modal_adapters = nn.ModuleDict()
        if self.adapter_indices:
            for idx in self.adapter_indices:
                # Align drop path with the matching encoder block index.
                adapter_drop_path = dpr[idx] if idx < len(dpr) else 0.0
                self.cross_modal_adapters[str(idx)] = CrossModalAdapter(
                    dim=d_model,
                    num_heads=n_heads,
                    mlp_ratio=adapter_mlp_ratio,
                    dropout=dropout,
                    drop_path=adapter_drop_path,
                )

        if freeze_vit:
            self._freeze_vit_backbone()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _freeze_vit_backbone(self):
        """
        Freeze most of the ViT encoder while keeping norms trainable.
        """
        for name, param in self.named_parameters():
            if name.startswith('bev_encoder') or name.startswith('cross_modal_adapters'):
                continue  # keep BEV/adapter trainable
            if 'norm' in name:
                continue  # allow LayerNorm fine-tuning
            param.requires_grad = False

    def get_grid_size(self, H, W):
        return self.patch_embed.get_grid_size(H, W)

    def _encode_bev_to_tokens(self, bev_image):
        bev_feat = self.bev_encoder(bev_image)
        bev_tokens = bev_feat.flatten(2).transpose(1, 2)
        return bev_feat, bev_tokens

    def forward(self, im, bev_image=None, return_features=False):
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

        bev_tokens = None
        bev_feat = None
        if bev_image is not None and self.bev_encoder is not None:
            bev_feat, bev_tokens = self._encode_bev_to_tokens(bev_image)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if bev_tokens is not None and str(i) in self.cross_modal_adapters:
                # Do not mix the CLS token in cross-modal attention.
                cls_token, rv_tokens = x[:, :1], x[:, 1:]
                rv_tokens = self.cross_modal_adapters[str(i)](rv_tokens, bev_tokens)
                x = torch.cat([cls_token, rv_tokens], dim=1)

        x = self.norm(x)

        if bev_feat is not None:
            return x, skip, {'bev_feat': bev_feat, 'bev_tokens': bev_tokens}
        return x, skip  # x.shape = [16, 577, 384]


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    model_cfg.pop('backbone')
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
    use_bev_decoder = model_cfg.pop('use_bev_decoder', False)
    bev_decoder_hidden = model_cfg.pop('bev_decoder_hidden', 128)
    use_bev_fusion = model_cfg.pop('use_bev_fusion', False)

    # Choose encoder architecture
    backbone = model_cfg.get('backbone', 'vit_small_patch16_384')
    if backbone.startswith('swin'):
        # Swin backbone currently does not support the BALViT adapters/BEV path.
        if model_cfg.get('bev_channels') is not None:
            raise ValueError('BALViT BEV/adapters are only supported with ViT backbones, not Swin.')
        encoder = create_swin_v2(model_cfg)
    else:
        encoder = create_vit(model_cfg)
    
    decoder = create_decoder(encoder, decoder_cfg)
    bev_decoder = None
    if use_bev_decoder:
        bev_decoder = BEVDecoder(
            in_channels=encoder.d_model,
            n_cls=model_cfg['n_cls'],
            hidden_channels=bev_decoder_hidden)

    if use_kpconv:
        kpclassifier = KPClassifier(
            in_channels=decoder_cfg['d_decoder'] ,
            out_channels=decoder_cfg['d_decoder'],
            num_classes=model_cfg['n_cls'])
        model = RangeViT_KPConv(encoder, decoder, kpclassifier, n_cls=model_cfg['n_cls'])
    else:
        model = RangeViT_noKPConv(
            encoder,
            decoder,
            n_cls=model_cfg['n_cls'],
            bev_decoder=bev_decoder,
            use_bev_fusion=use_bev_fusion)

    return model


class RangeViT_noKPConv(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        bev_decoder=None,
        use_bev_fusion=False,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.patch_stride = encoder.patch_stride
        self.encoder = encoder
        self.decoder = decoder
        self.bev_decoder = bev_decoder
        self.use_bev_fusion = use_bev_fusion

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay('encoder.', self.encoder).union(
            append_prefix_no_weight_decay('decoder.', self.decoder)
        )
        if self.bev_decoder is not None:
            nwd_params = nwd_params.union(
                append_prefix_no_weight_decay('bev_decoder.', self.bev_decoder)
            )
        return nwd_params

    def forward(self, im, bev_image=None):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        encoder_outputs = self.encoder(im, bev_image=bev_image, return_features=True) # x.shape = [16, 577, 384]
        if len(encoder_outputs) == 2:
            x, skip = encoder_outputs
            bev_ctx = {}
        else:
            x, skip, bev_ctx = encoder_outputs

        # remove CLS tokens for decoding
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:] # x.shape = [16, 576, 384]

        rv_logits = self.decoder(x, (H, W), skip) # feats.shape = [16, 17, 24, 24]
        rv_logits = F.interpolate(rv_logits, size=(H, W), mode='bilinear')
        rv_logits = unpadding(rv_logits, (H_ori, W_ori)) # feats.shape = [16, 17, 384, 384]

        bev_logits = None
        if self.bev_decoder is not None and bev_ctx.get('bev_feat') is not None:
            bev_logits = self.bev_decoder(bev_ctx['bev_feat'])
            bev_logits = F.interpolate(
                bev_logits, size=rv_logits.shape[-2:], mode='bilinear', align_corners=False)

        if self.use_bev_fusion and bev_logits is not None:
            return (rv_logits + bev_logits) / 2.0

        return rv_logits


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
        bev_channels=None,
        bev_base_channels=64,
        bev_num_layers=3,
        adapter_indices=None,
        adapter_mlp_ratio=4.0,
        freeze_vit=False,
        use_bev_decoder=False,
        bev_decoder_hidden=128,
        use_bev_fusion=False,
        ):
        super(RangeViT, self).__init__()

        self.n_cls = n_cls

        if backbone == 'vit_tiny_patch16_384':
            n_heads = 6
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 192
        elif backbone == 'vit_small_patch16_384':
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
            'bev_channels': bev_channels,
            'bev_base_channels': bev_base_channels,
            'bev_num_layers': bev_num_layers,
            'adapter_indices': adapter_indices,
            'adapter_mlp_ratio': adapter_mlp_ratio,
            'freeze_vit': freeze_vit,
            'use_bev_decoder': use_bev_decoder,
            'bev_decoder_hidden': bev_decoder_hidden,
            'use_bev_fusion': use_bev_fusion,
        }


        # Create RangeViT model
        self.rangevit = create_rangevit(net_kwargs, use_kpconv)

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
                pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
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
            if reuse_patch_emb:
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
        stats['stem_num_parameters'] = count_parameters(self.rangevit.encoder.patch_embed)
        stats['encoder_num_parameters'] = count_parameters(self.rangevit.encoder) - stats['stem_num_parameters']
        return stats

    def forward(self, *args, **kwargs):
        return self.rangevit(*args, **kwargs)

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
