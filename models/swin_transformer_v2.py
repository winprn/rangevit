# Copyright 2023 - Valeo Comfort and Driving Assistance
# Swin Transformer V2 implementation adapted for RangeViT
# Based on Microsoft Swin Transformer V2 design

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .model_utils import adapt_input_conv, padding, unpadding, init_weights
from .stems import PatchEmbedding, ConvStem


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention with relative position bias and cosine attention."""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative coordinates table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        # Resize relative_coords_table if pretrained window size differs
        if pretrained_window_size[0] > 0 and (pretrained_window_size[0] != self.window_size[0] or pretrained_window_size[1] != self.window_size[1]):
            relative_coords_table = F.interpolate(
                relative_coords_table.permute(0, 3, 1, 2),
                size=self.window_size,
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(torch.tensor(self.scale), max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # Handle case where window_size might be a list/tuple
        window_size_val = self.window_size[0] if isinstance(self.window_size, (list, tuple)) else self.window_size
        if min(self.input_resolution) <= window_size_val:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        window_size_check = self.window_size[0] if isinstance(self.window_size, (list, tuple)) else self.window_size
        assert 0 <= self.shift_size < window_size_check, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        
        # Handle dynamic input sizes during inference - simplified
        if L != H * W:
            # Use simple square-like factorization
            import math
            sqrt_L = int(math.sqrt(L))
            
            # Find the closest factors
            for h in range(sqrt_L, 0, -1):
                if L % h == 0:
                    H, W = h, L // h
                    break
            else:
                # Fallback: use 1 x L
                H, W = 1, L

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        window_size_val = self.window_size[0] if isinstance(self.window_size, (list, tuple)) else self.window_size
        
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (window_size_val - W % window_size_val) % window_size_val
        pad_b = (window_size_val - H % window_size_val) % window_size_val
        shifted_x = F.pad(shifted_x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = shifted_x.shape
        
        x_windows = window_partition(shifted_x, window_size_val)
        x_windows = x_windows.view(-1, window_size_val * window_size_val, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1, window_size_val, window_size_val, C)
        shifted_x = window_reverse(attn_windows, window_size_val, Hp, Wp)
        
        # unpad if necessary
        if pad_r > 0 or pad_b > 0:
            shifted_x = shifted_x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        
        # Handle dynamic input sizes during inference - simplified
        if L != H * W:
            # Use simple square-like factorization
            import math
            sqrt_L = int(math.sqrt(L))
            
            # Find the closest factors
            for h in range(sqrt_L, 0, -1):
                if L % h == 0:
                    H, W = h, L // h
                    break
            else:
                # Fallback: use 1 x L
                H, W = 1, L
        
        # Ensure dimensions are even for patch merging
        if H % 2 != 0 or W % 2 != 0:
            # Pad to make dimensions even
            pad_h = H % 2
            pad_w = W % 2
            x = x.view(B, H, W, C)
            if pad_h or pad_w:
                x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # pad right and bottom
                H = H + pad_h
                W = W + pad_w
                x = x.view(B, -1, C)
                L = H * W

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else (window_size[0] if isinstance(window_size, (list, tuple)) else window_size) // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerV2(nn.Module):
    """Swin Transformer V2 adapted for RangeViT architecture."""

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
        drop_path_rate=0.1,
        channels=3,
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        patch_stride=None,
        conv_stem='none',
        stem_base_channels=32,
        stem_hidden_dim=None,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
    ):
        super().__init__()
        
        self.conv_stem = conv_stem
        self.num_layers = len(depths)
        self.embed_dim = d_model
        # Keep d_model consistent with decoder expectations
        self.d_model = d_model  # Keep original dimension for decoder compatibility
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride or patch_size
        self.n_cls = n_cls
        self.image_size = image_size
        
        # Patch embedding
        if self.conv_stem == 'none':
            self.patch_embed = PatchEmbedding(
                image_size, patch_size, self.patch_stride, d_model, channels)
        else:
            assert patch_stride == patch_size
            self.patch_embed = ConvStem(
                in_channels=channels,
                base_channels=stem_base_channels,
                img_size=image_size,
                patch_stride=patch_stride,
                embed_dim=d_model,
                flatten=True,
                hidden_dim=stem_hidden_dim)

        self.pos_drop = nn.Dropout(p=dropout)

        # Build layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer_depth = depths[i_layer] if i_layer < len(depths) else 2
            layer_heads = num_heads[i_layer] if i_layer < len(num_heads) else n_heads
            
            # Calculate resolution for this layer
            if i_layer == 0:
                layer_resolution = (
                    image_size[0] // self.patch_stride[0],
                    image_size[1] // self.patch_stride[1]
                )
                layer_dim = d_model
            else:
                prev_resolution = self.layers[i_layer-1].input_resolution
                layer_resolution = (prev_resolution[0] // 2, prev_resolution[1] // 2)
                layer_dim = d_model * (2 ** i_layer)

            layer = BasicLayer(
                dim=layer_dim,
                input_resolution=layer_resolution,
                depth=layer_depth,
                num_heads=layer_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=dropout,
                attn_drop=dropout,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        # Use final layer dimension for norm
        final_dim = d_model * (2 ** (self.num_layers - 1))
        self.norm = nn.LayerNorm(final_dim)
        
        # Add projection layer to match decoder expectations
        self.final_proj = nn.Linear(final_dim, d_model)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def get_grid_size(self, H, W):
        return self.patch_embed.get_grid_size(H, W)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        x, skip = self.patch_embed(im)
        x = self.pos_drop(x)
        
        # Extract features from different layers for skip connections
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        
        x = self.norm(x)
        x = self.final_proj(x)  # Project to decoder expected dimension
        
        # Ensure output format matches decoder expectations
        # The decoder expects (B, L, C) where L = grid_H * grid_W
        # After Swin layers, we need to ensure proper spatial arrangement
        B, L, C = x.shape
        
        # Calculate expected grid size based on original image and patch stride
        expected_H = H // self.patch_stride[0]
        expected_W = W // self.patch_stride[1]
        expected_L = expected_H * expected_W
        
        # If the current tensor length doesn't match expected, we need to reshape properly
        if L != expected_L:
            # This can happen due to patch merging in Swin layers
            # We'll interpolate to match the expected output size
            import math
            current_H = int(math.sqrt(L))
            current_W = L // current_H
            
            # Reshape to spatial format, interpolate, then flatten back
            x_spatial = x.view(B, current_H, current_W, C).permute(0, 3, 1, 2)  # B, C, H, W
            x_spatial = F.interpolate(x_spatial, size=(expected_H, expected_W), mode='bilinear', align_corners=False)
            x = x_spatial.permute(0, 2, 3, 1).view(B, expected_L, C)  # B, L, C
        
        if return_features:
            return x, skip, features
        return x, skip


def create_swin_v2(model_cfg):
    """Create Swin Transformer V2 model."""
    model_cfg = model_cfg.copy()
    model_cfg.pop('backbone')
    
    new_patch_size = model_cfg.pop('new_patch_size', None)
    new_patch_stride = model_cfg.pop('new_patch_stride', None)

    if new_patch_size is not None:
        if new_patch_stride is None:
            new_patch_stride = new_patch_size
        model_cfg['patch_size'] = new_patch_size
        model_cfg['patch_stride'] = new_patch_stride
    
    # Calculate d_ff based on mlp_ratio and d_model
    mlp_ratio = model_cfg.get('mlp_ratio', 4.0)
    d_model = model_cfg.get('d_model', 96)
    model_cfg['d_ff'] = int(mlp_ratio * d_model)
    
    # Set appropriate depths and heads based on d_model (if not already set)
    if 'depths' not in model_cfg or 'num_heads' not in model_cfg:
        if d_model == 96:
            model_cfg['depths'] = [2, 2, 18, 2]  # Swin V2 Small
            model_cfg['num_heads'] = [3, 6, 12, 24]
        elif d_model == 128:
            model_cfg['depths'] = [2, 2, 18, 2]
            model_cfg['num_heads'] = [4, 8, 16, 32]
        elif d_model == 192:
            model_cfg['depths'] = [2, 2, 18, 2]
            model_cfg['num_heads'] = [6, 12, 24, 48]
        elif d_model == 384:
            model_cfg['depths'] = [2, 2, 6, 2]
            model_cfg['num_heads'] = [3, 6, 12, 24]
        elif d_model == 768:
            model_cfg['depths'] = [2, 2, 18, 2]
            model_cfg['num_heads'] = [4, 8, 16, 32]
        else:
            raise ValueError(f"Unsupported d_model for Swin Transformer V2: {d_model}")

    model = SwinTransformerV2(**model_cfg)
    return model