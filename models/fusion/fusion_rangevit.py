# Copyright 2024 - Fusion Extension for RangeViT
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

"""
FusionRangeViT: Multi-view fusion model combining ViT range branch with MinkUNet voxel branch.

Architecture:
    - Range Branch: ConvStem → ViT Encoder → DecoderUpConv (existing RangeViT)
    - Voxel Branch: MinkUNet sparse 3D U-Net
    - Fusion: 3-point fusion with point representation as hub
    - Output: Per-point classification using z1 + z2 (last 2 fused features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchsparse import PointTensor, SparseTensor

from ..rangevit import VisionTransformer, create_vit, create_decoder
from ..stems import ConvStem
from ..decoders import DecoderUpConv
from ..model_utils import padding, unpadding, resize_pos_embed, adapt_input_conv

from .minkunet_voxel import MinkUNetVoxelEncoder
from .fusion_modules import FusionMLP, PointTransform
from .representation_utils import (
    initial_voxelize,
    voxel_to_point,
    point_to_voxel,
    range_to_point,
    range_to_point_from_tokens,
    point_to_range_fast,
)

__all__ = ['FusionRangeViT']


class FusionRangeViT(nn.Module):
    """
    Fusion model combining ViT-based range branch with MinkUNet voxel branch.

    Fusion Strategy:
        - Point representation as communication hub
        - 3 fusion points: after-stem, after-encoder, after-decoder
        - Simple concatenation + MLP at each fusion point
        - Final classifier uses z1 + z2 (last 2 fused features)

    Args:
        # Range branch config
        range_in_channels: Input channels for range image (default: 5)
        n_cls: Number of classes
        vit_backbone: ViT backbone type ('vit_small_patch16_384', 'vit_base_patch16_384', etc.)
        image_size: Range image size (H, W)
        range_pretrained_path: Path to pretrained range encoder weights
        patch_size: Patch size for ViT
        patch_stride: Patch stride for ViT
        conv_stem: Stem type ('ConvStem' or 'none')
        stem_base_channels: Base channels for ConvStem
        stem_hidden_dim: Hidden dimension for ConvStem
        decoder: Decoder type ('up_conv' or 'linear')
        decoder_d_decoder: Decoder hidden dimension
        skip_filters: Skip connection channels from stem to decoder

        # Voxel branch config
        voxel_in_channels: Input channels for voxel (default: 4 for x,y,z,intensity)
        voxel_num_layer: Number of blocks per MinkUNet stage
        voxel_block_type: 'Bottleneck' or 'ResBlock'
        voxel_cr: Channel ratio for MinkUNet
        voxel_planes: Base channel dimensions for MinkUNet
        voxel_pres: Point resolution
        voxel_vres: Voxel resolution
        voxel_pretrained_path: Path to pretrained voxel encoder weights

        # Fusion config
        fusion_hidden_ratio: Hidden layer ratio for FusionMLP

        # Training config
        if_dist: Whether to use distributed training (SyncBatchNorm)
        dropout_p: Dropout probability
    """

    def __init__(
        self,
        # Range branch config
        range_in_channels: int = 5,
        n_cls: int = 20,
        vit_backbone: str = 'vit_small_patch16_384',
        image_size: tuple = (64, 2048),
        range_pretrained_path: str = None,
        patch_size: tuple = (2, 8),
        patch_stride: tuple = (2, 8),
        conv_stem: str = 'ConvStem',
        stem_base_channels: int = 32,
        stem_hidden_dim: int = 64,
        decoder: str = 'up_conv',
        decoder_d_decoder: int = 64,
        skip_filters: int = 64,

        # Voxel branch config
        voxel_in_channels: int = 4,
        voxel_num_layer: list = None,
        voxel_block_type: str = 'Bottleneck',
        voxel_cr: float = 1.0,
        voxel_planes: list = None,
        voxel_pres: float = 0.05,
        voxel_vres: float = 0.05,
        voxel_pretrained_path: str = None,

        # Fusion config
        fusion_hidden_ratio: float = 2.0,

        # Training config
        if_dist: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        if voxel_num_layer is None:
            voxel_num_layer = [2, 3, 4, 6, 2, 2, 2, 2]
        if voxel_planes is None:
            voxel_planes = [32, 32, 64, 128, 256, 256, 128, 96, 96]

        self.n_cls = n_cls
        self.if_dist = if_dist
        self.voxel_pres = voxel_pres
        self.voxel_vres = voxel_vres
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Get backbone dimensions
        self.d_model = self._get_vit_dim(vit_backbone)

        # === Build Range Branch ===
        self.range_encoder = self._build_range_encoder(
            range_in_channels, vit_backbone, image_size,
            patch_size, patch_stride, conv_stem,
            stem_base_channels, stem_hidden_dim
        )

        self.range_decoder = self._build_range_decoder(
            decoder, decoder_d_decoder, skip_filters
        )

        # Store range branch dimensions
        self.range_stem_channels = stem_hidden_dim  # Skip connection dim
        self.range_encoder_channels = self.d_model  # ViT output dim
        self.range_decoder_channels = decoder_d_decoder  # Decoder output dim

        # Load range pretrained weights if provided
        if range_pretrained_path is not None:
            self._load_range_checkpoint(range_pretrained_path, vit_backbone, range_in_channels)

        # === Build Voxel Branch ===
        self.voxel_branch = MinkUNetVoxelEncoder(
            in_feature_dim=voxel_in_channels,
            num_layer=voxel_num_layer,
            block_type=voxel_block_type,
            cr=voxel_cr,
            planes=voxel_planes,
            pres=voxel_pres,
            vres=voxel_vres,
            if_dist=if_dist,
            dropout_p=dropout_p,
        )

        # Store voxel branch dimensions
        self.voxel_stem_channels = self.voxel_branch.stem_out_channels
        self.voxel_bottleneck_channels = self.voxel_branch.bottleneck_out_channels
        self.voxel_final_channels = self.voxel_branch.final_out_channels

        # Load voxel pretrained weights if provided
        if voxel_pretrained_path is not None:
            self._load_voxel_checkpoint(voxel_pretrained_path)

        # === Build Point Transforms ===
        # Transform point features to match dimensions at each fusion stage
        self.point_transforms = nn.ModuleList([
            # Fusion 1: raw point features (4) → stem dim
            PointTransform(voxel_in_channels, self.voxel_stem_channels, if_dist=if_dist),
            # Fusion 2: stem dim → bottleneck dim
            PointTransform(self.voxel_stem_channels, self.voxel_bottleneck_channels, if_dist=if_dist),
            # Fusion 3: bottleneck dim → final dim
            PointTransform(self.voxel_bottleneck_channels, self.voxel_final_channels, if_dist=if_dist),
        ])

        # === Build Fusion MLPs ===
        # Fusion 1: After stem
        self.fusion_modules = nn.ModuleList([
            FusionMLP(
                in_channels_range=self.range_stem_channels,
                in_channels_voxel=self.voxel_stem_channels,
                in_channels_point=self.voxel_stem_channels,
                out_channels=self.voxel_stem_channels,
                hidden_ratio=fusion_hidden_ratio,
                if_dist=if_dist,
            ),
            # Fusion 2: After encoder/bottleneck
            FusionMLP(
                in_channels_range=self.range_encoder_channels,
                in_channels_voxel=self.voxel_bottleneck_channels,
                in_channels_point=self.voxel_bottleneck_channels,
                out_channels=self.voxel_bottleneck_channels,
                hidden_ratio=fusion_hidden_ratio,
                if_dist=if_dist,
            ),
            # Fusion 3: After decoder
            FusionMLP(
                in_channels_range=self.range_decoder_channels,
                in_channels_voxel=self.voxel_final_channels,
                in_channels_point=self.voxel_final_channels,
                out_channels=self.voxel_final_channels,
                hidden_ratio=fusion_hidden_ratio,
                if_dist=if_dist,
            ),
        ])

        # === Final Classifier ===
        # Uses z1 + z2 (bottleneck + final fused features)
        classifier_in = self.voxel_bottleneck_channels + self.voxel_final_channels
        self.classifier = nn.Linear(classifier_in, n_cls)

        self.dropout = nn.Dropout(dropout_p, inplace=True)

    def _get_vit_dim(self, backbone: str) -> int:
        """Get d_model dimension for ViT backbone."""
        dims = {
            'vit_small_patch16_384': 384,
            'vit_base_patch16_384': 768,
            'vit_large_patch16_384': 1024,
        }
        return dims.get(backbone, 384)

    def _build_range_encoder(
        self,
        in_channels,
        backbone,
        image_size,
        patch_size,
        patch_stride,
        conv_stem,
        stem_base_channels,
        stem_hidden_dim,
    ):
        """Build range branch encoder (ViT with optional ConvStem)."""
        # Get backbone config
        if backbone == 'vit_small_patch16_384':
            n_heads, n_layers, d_model = 6, 12, 384
        elif backbone == 'vit_base_patch16_384':
            n_heads, n_layers, d_model = 12, 12, 768
        elif backbone == 'vit_large_patch16_384':
            n_heads, n_layers, d_model = 16, 24, 1024
        else:
            n_heads, n_layers, d_model = 6, 12, 384

        dropout = 0.0
        drop_path_rate = 0.1

        encoder = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=4 * d_model,
            n_heads=n_heads,
            n_cls=self.n_cls,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            channels=in_channels,
            patch_stride=patch_stride,
            conv_stem=conv_stem,
            stem_base_channels=stem_base_channels,
            stem_hidden_dim=stem_hidden_dim,
        )

        return encoder

    def _build_range_decoder(self, decoder_type, d_decoder, skip_filters):
        """Build range branch decoder."""
        if decoder_type == 'up_conv':
            decoder = DecoderUpConv(
                n_cls=self.n_cls,
                patch_size=self.patch_size,
                d_encoder=self.d_model,
                d_decoder=d_decoder,
                scale_factor=self.patch_stride,
                patch_stride=self.patch_stride,
                skip_filters=skip_filters,
            )
        else:
            raise ValueError(f"Fusion model only supports 'up_conv' decoder, got {decoder_type}")

        return decoder

    def _load_range_checkpoint(self, path, backbone, in_channels):
        """Load pretrained range encoder weights."""
        print(f'Loading range branch pretrained parameters from {path}')

        if path == 'timmImageNet21k':
            vit_imagenet = timm.create_model(backbone, pretrained=True)
            pretrained_state_dict = vit_imagenet.state_dict()
            all_keys = list(pretrained_state_dict.keys())
            for key in all_keys:
                pretrained_state_dict['range_encoder.' + key] = pretrained_state_dict.pop(key)
        else:
            pretrained_state_dict = torch.load(path, map_location='cpu')
            if 'state_dict' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['state_dict']
            elif 'model' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['model']

            # Handle different key prefixes
            all_keys = list(pretrained_state_dict.keys())
            new_state_dict = {}
            for key in all_keys:
                if key.startswith('rangevit.encoder.'):
                    new_key = key.replace('rangevit.encoder.', 'range_encoder.')
                    new_state_dict[new_key] = pretrained_state_dict[key]
                elif key.startswith('encoder.'):
                    new_key = key.replace('encoder.', 'range_encoder.')
                    new_state_dict[new_key] = pretrained_state_dict[key]
                elif key.startswith('rangevit.decoder.'):
                    new_key = key.replace('rangevit.decoder.', 'range_decoder.')
                    new_state_dict[new_key] = pretrained_state_dict[key]
                elif key.startswith('decoder.'):
                    new_key = key.replace('decoder.', 'range_decoder.')
                    new_state_dict[new_key] = pretrained_state_dict[key]
            pretrained_state_dict = new_state_dict

        msg = self.load_state_dict(pretrained_state_dict, strict=False)
        print(f'Range checkpoint loading: {msg}')

    def _load_voxel_checkpoint(self, path):
        """Load pretrained voxel encoder weights."""
        print(f'Loading voxel branch pretrained parameters from {path}')

        state_dict = torch.load(path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Filter to only voxel branch keys
        voxel_state = {}
        for k, v in state_dict.items():
            if k.startswith('voxel_branch.'):
                voxel_state[k.replace('voxel_branch.', '')] = v
            elif not any(k.startswith(p) for p in ['range_', 'fusion_', 'point_', 'classifier']):
                # Assume it's a standalone MinkUNet checkpoint
                voxel_state[k] = v

        msg = self.voxel_branch.load_state_dict(voxel_state, strict=False)
        print(f'Voxel checkpoint loading: {msg}')

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return parameters that should not have weight decay."""
        nwd = {'range_encoder.cls_token', 'range_encoder.pos_embed'}
        return nwd

    def forward(
        self,
        range_image: torch.Tensor,
        point_features: torch.Tensor,
        point_coords: torch.Tensor,
        batch_indices: torch.Tensor,
        range_pxpy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with multi-view fusion.

        Args:
            range_image: [B, 5, H, W] range image (5 channels for range branch)
            point_features: [N, 4] raw point features (x, y, z, intensity)
            point_coords: [N, 3] point coordinates
            batch_indices: [N] batch index per point
            range_pxpy: [N, 2] projection coords (px, py normalized to [-1,1])

        Returns:
            logits: [N, n_cls] per-point class predictions
        """
        B, _, H_ori, W_ori = range_image.shape

        # Padding for ViT
        range_image_padded = padding(range_image, self.patch_size)
        H, W = range_image_padded.size(2), range_image_padded.size(3)

        # Initialize PointTensor for voxel operations
        coords_with_batch = torch.cat([
            point_coords,
            batch_indices.float().unsqueeze(1)
        ], dim=1)

        z = PointTensor(
            point_features,
            coords_with_batch,
            idx_query={},
            weights={},
        )
        z.additional_features = {'idx_query': {}, 'counts': {}}

        # ========== FUSION 1: After Stem ==========
        # Range stem
        range_stem_out, skip = self.range_encoder.patch_embed(range_image_padded)
        # skip: [B, D_h, H, W] (ConvStem hidden dim output before proj)

        # DEBUG: Print input point info
        print(f"[DEBUG] Input points: {point_coords.shape[0]} points, coords range: min={point_coords.min(dim=0).values.tolist()}, max={point_coords.max(dim=0).values.tolist()}")
        print(f"[DEBUG] voxel_pres={self.voxel_pres}, voxel_vres={self.voxel_vres}")
        import sys; sys.stdout.flush()

        # Voxel stem
        x0 = initial_voxelize(z, self.voxel_pres, self.voxel_vres)
        print(f"[DEBUG] After voxelize: coords={x0.C.shape}, feats={x0.F.shape}, stride={x0.s}")
        print(f"[DEBUG] After voxelize coords range: min={x0.C.min(dim=0).values.tolist()}, max={x0.C.max(dim=0).values.tolist()}")
        import sys; sys.stdout.flush()

        x0 = self.voxel_branch.stem(x0)
        print(f"[DEBUG] After stem: coords={x0.C.shape}, feats={x0.F.shape}, stride={x0.s}")
        import sys; sys.stdout.flush()

        # Fusion at point level
        z0_voxel = voxel_to_point(x0, z)
        z0_range = range_to_point(skip, range_pxpy, batch_indices, B)
        z0_point = self.point_transforms[0](point_features)
        z0 = self.fusion_modules[0](z0_range, z0_voxel.F, z0_point)

        # Update PointTensor with fused features
        z_fused = PointTensor(z0, z.C, idx_query=z.idx_query, weights=z.weights)
        z_fused.additional_features = z.additional_features

        # Update voxel branch with fused features
        x0_fused = point_to_voxel(x0, z_fused)

        # ========== FUSION 2: After Encoder ==========
        # Range encoder (full ViT forward)
        range_enc_out, _ = self.range_encoder(range_image_padded)
        # range_enc_out: [B, N_tokens+1, d_model]
        range_enc = range_enc_out[:, 1:]  # Remove CLS token: [B, N_tokens, d_model]

        # Voxel encoder stages
        x1 = self.voxel_branch.stage1(x0_fused)

        # DEBUG: Print tensor info before stage2
        print(f"[DEBUG] x0_fused: coords={x0_fused.C.shape}, feats={x0_fused.F.shape}, stride={x0_fused.s}")
        print(f"[DEBUG] x0_fused coords range: min={x0_fused.C.min(dim=0).values.tolist()}, max={x0_fused.C.max(dim=0).values.tolist()}")
        print(f"[DEBUG] x1: coords={x1.C.shape}, feats={x1.F.shape}, stride={x1.s}")
        print(f"[DEBUG] x1 coords range: min={x1.C.min(dim=0).values.tolist()}, max={x1.C.max(dim=0).values.tolist()}")
        print(f"[DEBUG] x1 num voxels: {x1.C.shape[0]}")
        import sys; sys.stdout.flush()

        x2 = self.voxel_branch.stage2(x1)
        x3 = self.voxel_branch.stage3(x2)
        x4 = self.voxel_branch.stage4(x3)  # Bottleneck

        # Fusion at point level
        z1_voxel = voxel_to_point(x4, z_fused)
        z1_range = range_to_point_from_tokens(
            range_enc, range_pxpy, batch_indices, B, H, W, self.patch_stride
        )
        z1_point = self.point_transforms[1](z0)
        z1 = self.fusion_modules[1](z1_range, z1_voxel.F, z1_point)

        # Update PointTensor
        z_fused = PointTensor(z1, z_fused.C, idx_query=z_fused.idx_query, weights=z_fused.weights)
        z_fused.additional_features = z.additional_features

        # Update voxel branch
        x4_fused = point_to_voxel(x4, z_fused)

        # ========== FUSION 3: After Decoder ==========
        # Range decoder
        GS_H, GS_W = H // self.patch_stride[0], W // self.patch_stride[1]
        range_dec = self.range_decoder(range_enc, (H, W), skip, return_features=True)
        # range_dec: [B, d_decoder, H, W] (full resolution features)

        # Interpolate decoder output to original size
        range_dec = F.interpolate(range_dec, size=(H, W), mode='bilinear', align_corners=False)
        range_dec = unpadding(range_dec, (H_ori, W_ori))

        # Voxel decoder
        x4_drop = SparseTensor(self.dropout(x4_fused.F), x4_fused.C, x4_fused.s)
        x4_drop._caches = x4_fused._caches

        y1 = self.voxel_branch.up1_deconv(x4_drop)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.voxel_branch.up1_blocks(y1)

        y2 = self.voxel_branch.up2_deconv(y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.voxel_branch.up2_blocks(y2)

        y2_drop = SparseTensor(self.dropout(y2.F), y2.C, y2.s)
        y2_drop._caches = y2._caches

        y3 = self.voxel_branch.up3_deconv(y2_drop)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.voxel_branch.up3_blocks(y3)

        y4 = self.voxel_branch.up4_deconv(y3)
        y4 = torchsparse.cat([y4, x0_fused])
        y4 = self.voxel_branch.up4_blocks(y4)

        # Fusion at point level
        z2_voxel = voxel_to_point(y4, z_fused)
        z2_range = range_to_point(range_dec, range_pxpy, batch_indices, B)
        z2_point = self.point_transforms[2](z1)
        z2 = self.fusion_modules[2](z2_range, z2_voxel.F, z2_point)

        # ========== FINAL: Classification ==========
        # Use last 2 fused point features (z1, z2)
        out = self.classifier(torch.cat([z1, z2], dim=1))

        return out  # [N, n_cls]

    def get_checkpoint_state(self) -> dict:
        """Get state dict organized by component for checkpoint saving."""
        return {
            'range_encoder': self.range_encoder.state_dict(),
            'range_decoder': self.range_decoder.state_dict(),
            'voxel_branch': self.voxel_branch.state_dict(),
            'fusion_modules': self.fusion_modules.state_dict(),
            'point_transforms': self.point_transforms.state_dict(),
            'classifier': self.classifier.state_dict(),
        }

    def load_checkpoint_state(self, checkpoint: dict, strict: bool = False):
        """Load state dict from checkpoint organized by component."""
        if 'range_encoder' in checkpoint:
            self.range_encoder.load_state_dict(checkpoint['range_encoder'], strict=strict)
        if 'range_decoder' in checkpoint:
            self.range_decoder.load_state_dict(checkpoint['range_decoder'], strict=strict)
        if 'voxel_branch' in checkpoint:
            self.voxel_branch.load_state_dict(checkpoint['voxel_branch'], strict=strict)
        if 'fusion_modules' in checkpoint:
            self.fusion_modules.load_state_dict(checkpoint['fusion_modules'], strict=strict)
        if 'point_transforms' in checkpoint:
            self.point_transforms.load_state_dict(checkpoint['point_transforms'], strict=strict)
        if 'classifier' in checkpoint:
            self.classifier.load_state_dict(checkpoint['classifier'], strict=strict)

    def counter_model_parameters(self):
        """Count trainable parameters in each model component."""
        stats = {}
        stats['total_num_parameters'] = count_parameters(self)
        stats['range_encoder_num_parameters'] = count_parameters(self.range_encoder)
        stats['range_decoder_num_parameters'] = count_parameters(self.range_decoder)
        stats['voxel_branch_num_parameters'] = count_parameters(self.voxel_branch)
        stats['fusion_modules_num_parameters'] = count_parameters(self.fusion_modules)
        stats['point_transforms_num_parameters'] = count_parameters(self.point_transforms)
        stats['classifier_num_parameters'] = count_parameters(self.classifier)
        return stats


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Import torchsparse at module level for cat operation
try:
    import torchsparse
except ImportError:
    raise ImportError("torchsparse is required for FusionRangeViT. Install with: pip install torchsparse")
