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
PointFusionRangeViT: Efficient fusion model combining ViT range branch with Point MLP + Cross-Attention.

Replaces MinkUNet voxel branch with lightweight point processing for improved speed.

Architecture:
    - Range Branch: ConvStem -> ViT Encoder -> DecoderUpConv (existing RangeViT)
    - Point Branch: PointMLPEncoder (lightweight MLPs)
    - Fusion: Cross-attention at encoder and decoder stages
    - Output: Per-point classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ..rangevit import VisionTransformer
from ..decoders import DecoderUpConv
from ..model_utils import padding, unpadding

from .point_encoder import PointMLPEncoder
from .fusion_modules import PointToRangeCrossAttention

__all__ = ['PointFusionRangeViT']


class PointFusionRangeViT(nn.Module):
    """
    Fusion model combining ViT-based range branch with Point MLP + Cross-Attention.

    Fusion Strategy:
        - Point MLP encodes raw point features (xyz, intensity, cluster_offset)
        - Cross-attention at encoder output: points query encoder features
        - Cross-attention at decoder output: points query decoder features
        - Final classifier on fused point features

    Args:
        # Range branch config
        range_in_channels: Input channels for range image (default: 5)
        n_cls: Number of classes
        vit_backbone: ViT backbone type
        image_size: Range image size (H, W)
        range_pretrained_path: Path to pretrained range encoder weights
        patch_size: Patch size for ViT
        patch_stride: Patch stride for ViT
        conv_stem: Stem type ('ConvStem' or 'none')
        stem_base_channels: Base channels for ConvStem
        stem_hidden_dim: Hidden dimension for ConvStem
        decoder: Decoder type ('up_conv')
        decoder_d_decoder: Decoder hidden dimension
        skip_filters: Skip connection channels

        # Point branch config
        point_in_channels: Input channels for points (default: 7)
        point_hidden_dim: Hidden dimension for point encoder

        # Fusion config
        cross_attn_window: Window size for cross-attention
        cross_attn_heads: Number of attention heads

        # Training config
        if_dist: Whether to use distributed training
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

        # Point branch config
        point_in_channels: int = 7,
        point_hidden_dim: int = 256,

        # Fusion config
        cross_attn_window: int = 3,
        cross_attn_heads: int = 4,

        # Training config
        if_dist: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        self.n_cls = n_cls
        self.if_dist = if_dist
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

        # Store dimensions
        self.range_encoder_channels = self.d_model
        self.range_decoder_channels = decoder_d_decoder

        # Load range pretrained weights if provided
        if range_pretrained_path is not None:
            self._load_range_checkpoint(range_pretrained_path, vit_backbone, range_in_channels)

        # === Build Point Branch ===
        self.point_encoder = PointMLPEncoder(
            in_channels=point_in_channels,
            hidden_dim=point_hidden_dim,
            if_dist=if_dist,
        )

        # Project point features to match ViT dimensions for cross-attention
        BatchNorm = nn.SyncBatchNorm if if_dist else nn.BatchNorm1d
        self.point_proj_encoder = nn.Sequential(
            nn.Linear(point_hidden_dim, self.d_model),
            BatchNorm(self.d_model),
            nn.ReLU(inplace=False),
        )
        self.point_proj_decoder = nn.Sequential(
            nn.Linear(self.d_model, decoder_d_decoder),
            BatchNorm(decoder_d_decoder),
            nn.ReLU(inplace=False),
        )

        # === Build Cross-Attention Fusion ===
        self.cross_attn_encoder = PointToRangeCrossAttention(
            dim=self.d_model,
            window_size=cross_attn_window,
            num_heads=cross_attn_heads,
            if_dist=if_dist,
        )
        self.cross_attn_decoder = PointToRangeCrossAttention(
            dim=decoder_d_decoder,
            window_size=cross_attn_window,
            num_heads=cross_attn_heads,
            if_dist=if_dist,
        )

        # === Final Classifier ===
        # Uses concatenation of encoder and decoder fused features
        classifier_in = self.d_model + decoder_d_decoder
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, classifier_in // 2),
            BatchNorm(classifier_in // 2),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_p),
            nn.Linear(classifier_in // 2, n_cls),
        )

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
        if backbone == 'vit_small_patch16_384':
            n_heads, n_layers, d_model = 6, 12, 384
        elif backbone == 'vit_base_patch16_384':
            n_heads, n_layers, d_model = 12, 12, 384
        elif backbone == 'vit_large_patch16_384':
            n_heads, n_layers, d_model = 16, 24, 1024
        else:
            n_heads, n_layers, d_model = 6, 12, 384

        encoder = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=4 * d_model,
            n_heads=n_heads,
            n_cls=self.n_cls,
            dropout=0.0,
            drop_path_rate=0.1,
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
            raise ValueError(f"Only 'up_conv' decoder supported, got {decoder_type}")

        return decoder

    def _load_range_checkpoint(self, path, backbone, in_channels):
        """Load pretrained range encoder weights."""
        print(f'Loading range branch pretrained parameters from {path}')

        if path == 'timmImageNet21k':
            vit_imagenet = timm.create_model(backbone, pretrained=True)
            state_dict = vit_imagenet.state_dict()
            for key in list(state_dict.keys()):
                state_dict['range_encoder.' + key] = state_dict.pop(key)
        else:
            state_dict = torch.load(path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            # Handle key prefixes
            new_state_dict = {}
            for key in list(state_dict.keys()):
                if key.startswith('rangevit.encoder.'):
                    new_key = key.replace('rangevit.encoder.', 'range_encoder.')
                    new_state_dict[new_key] = state_dict[key]
                elif key.startswith('encoder.'):
                    new_key = key.replace('encoder.', 'range_encoder.')
                    new_state_dict[new_key] = state_dict[key]
                elif key.startswith('rangevit.decoder.'):
                    new_key = key.replace('rangevit.decoder.', 'range_decoder.')
                    new_state_dict[new_key] = state_dict[key]
                elif key.startswith('decoder.'):
                    new_key = key.replace('decoder.', 'range_decoder.')
                    new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        msg = self.load_state_dict(state_dict, strict=False)
        print(f'Range checkpoint loading: {msg}')

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return parameters that should not have weight decay."""
        return {'range_encoder.cls_token', 'range_encoder.pos_embed'}

    def forward(
        self,
        range_image: torch.Tensor,
        point_features: torch.Tensor,
        cluster_offset: torch.Tensor,
        batch_indices: torch.Tensor,
        range_pxpy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with Point-Range cross-attention fusion.

        Args:
            range_image: [B, 5, H, W] range image
            point_features: [N, 4] raw point features (x, y, z, intensity)
            cluster_offset: [N, 3] cluster center offset
            batch_indices: [N] batch index per point
            range_pxpy: [N, 2] projection coords (px, py normalized to [-1,1])

        Returns:
            logits: [N, n_cls] per-point class predictions
        """
        # Handle empty point cloud edge case
        if point_features.shape[0] == 0:
            return torch.empty(0, self.n_cls, device=range_image.device)

        B, _, H_ori, W_ori = range_image.shape

        # Padding for ViT
        range_image_padded = padding(range_image, self.patch_size)
        H, W = range_image_padded.size(2), range_image_padded.size(3)

        # ========== RANGE BRANCH ==========
        # Range stem + encoder (returns encoder output and skip connection)
        range_enc_out, skip = self.range_encoder(range_image_padded)
        range_enc = range_enc_out[:, 1:]  # Remove CLS token: [B, N_tokens, d_model]

        # Range decoder
        range_dec = self.range_decoder(range_enc, (H, W), skip, return_features=True)
        range_dec = F.interpolate(range_dec, size=(H, W), mode='bilinear', align_corners=False)
        range_dec = unpadding(range_dec, (H_ori, W_ori))  # [B, D_dec, H_ori, W_ori]

        # ========== POINT BRANCH ==========
        # Combine point features with cluster offset: [N, 7]
        point_input = torch.cat([point_features, cluster_offset], dim=1)
        point_feats = self.point_encoder(point_input)  # [N, hidden_dim]

        # ========== FUSION 1: After Encoder ==========
        # Reshape encoder output to spatial: [B, H', W', d_model]
        H_enc = H // self.patch_stride[0]
        W_enc = W // self.patch_stride[1]
        range_enc_spatial = range_enc.view(B, H_enc, W_enc, self.d_model)

        # Project points to encoder dimension
        point_feats_enc = self.point_proj_encoder(point_feats)  # [N, d_model]

        # Convert normalized pxpy to pixel indices for encoder resolution
        proj_y_enc, proj_x_enc = self._pxpy_to_indices(range_pxpy, H_enc, W_enc)

        # Cross-attention (process each batch separately)
        fused_enc_list = []
        for b in range(B):
            mask_b = batch_indices == b
            if mask_b.sum() == 0:
                continue
            point_b = point_feats_enc[mask_b]  # [N_b, d_model]
            vit_b = range_enc_spatial[b]  # [H', W', d_model]
            proj_y_b = proj_y_enc[mask_b]
            proj_x_b = proj_x_enc[mask_b]
            fused_b = self.cross_attn_encoder(point_b, vit_b, proj_y_b, proj_x_b)
            fused_enc_list.append(fused_b)

        fused_enc = torch.cat(fused_enc_list, dim=0)  # [N, d_model]

        # ========== FUSION 2: After Decoder ==========
        # Reshape decoder output: [B, H, W, D_dec]
        range_dec_spatial = range_dec.permute(0, 2, 3, 1)  # [B, H, W, D_dec]

        # Project fused encoder features to decoder dimension
        point_feats_dec = self.point_proj_decoder(fused_enc)  # [N, D_dec]

        # Convert pxpy to pixel indices for decoder resolution
        proj_y_dec, proj_x_dec = self._pxpy_to_indices(range_pxpy, H_ori, W_ori)

        # Cross-attention (process each batch separately)
        fused_dec_list = []
        for b in range(B):
            mask_b = batch_indices == b
            if mask_b.sum() == 0:
                continue
            point_b = point_feats_dec[mask_b]  # [N_b, D_dec]
            vit_b = range_dec_spatial[b]  # [H, W, D_dec]
            proj_y_b = proj_y_dec[mask_b]
            proj_x_b = proj_x_dec[mask_b]
            fused_b = self.cross_attn_decoder(point_b, vit_b, proj_y_b, proj_x_b)
            fused_dec_list.append(fused_b)

        fused_dec = torch.cat(fused_dec_list, dim=0)  # [N, D_dec]

        # ========== CLASSIFICATION ==========
        # Concatenate encoder and decoder fused features
        out = torch.cat([fused_enc, fused_dec], dim=1)  # [N, d_model + D_dec]
        out = self.classifier(out)  # [N, n_cls]

        return out

    def _pxpy_to_indices(
        self,
        range_pxpy: torch.Tensor,
        H: int,
        W: int,
    ) -> tuple:
        """
        Convert normalized pxpy [-1, 1] to pixel indices [0, H-1], [0, W-1].

        Args:
            range_pxpy: [N, 2] normalized coordinates (px, py)
            H, W: Target resolution

        Returns:
            proj_y: [N] row indices
            proj_x: [N] col indices
        """
        # pxpy is (px, py) normalized to [-1, 1]
        # Convert to [0, 1] then to pixel indices
        px = range_pxpy[:, 0]  # [N]
        py = range_pxpy[:, 1]  # [N]

        # [-1, 1] -> [0, 1]
        px_norm = (px + 1) / 2
        py_norm = (py + 1) / 2

        # [0, 1] -> [0, W-1] and [0, H-1]
        proj_x = (px_norm * (W - 1)).long().clamp(0, W - 1)
        proj_y = (py_norm * (H - 1)).long().clamp(0, H - 1)

        return proj_y, proj_x

    def count_parameters(self) -> dict:
        """Count trainable parameters in each model component."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            'total': count(self),
            'range_encoder': count(self.range_encoder),
            'range_decoder': count(self.range_decoder),
            'point_encoder': count(self.point_encoder),
            'point_projections': count(self.point_proj_encoder) + count(self.point_proj_decoder),
            'cross_attention': count(self.cross_attn_encoder) + count(self.cross_attn_decoder),
            'classifier': count(self.classifier),
        }

    def counter_model_parameters(self) -> dict:
        """Alias for count_parameters to match other models."""
        return self.count_parameters()
