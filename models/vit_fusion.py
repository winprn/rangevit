# Copyright 2026 - RangeViT-Fusion
# Vision Transformer with bidirectional point-pixel fusion

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .blocks import Block
from .model_utils import resize_pos_embed, init_weights
from .stems import PatchEmbedding, ConvStem
from .fusion_modules import (
    EfficientTransformationPipeline,
    PointFusionLayer,
    PixelFusionLayer,
    AuxiliaryHead,
)


class VisionTransformerFusion(nn.Module):
    """
    Vision Transformer with bidirectional point-pixel fusion at specified blocks.

    This model extends the standard Vision Transformer by adding fusion operations
    at configurable block indices. At each fusion point:
    1. Pixel features are mapped to points using projection coordinates
    2. Point features are fused with the mapped pixel features
    3. Point features are aggregated back to pixel space
    4. Pixel features are fused with the aggregated point features
    5. Auxiliary heads provide intermediate supervision

    Args:
        image_size: Input image size (H, W)
        patch_size: Patch size for tokenization
        n_layers: Number of transformer blocks
        d_model: Transformer hidden dimension
        d_ff: Feed-forward hidden dimension
        n_heads: Number of attention heads
        n_cls: Number of output classes
        dropout: Dropout rate
        drop_path_rate: Stochastic depth rate
        channels: Number of input channels
        ls_init_values: LayerScale initialization values
        patch_stride: Stride for patch embedding (defaults to patch_size)
        fusion_blocks: List of block indices (1-indexed) for fusion
        conv_stem: Type of convolutional stem ('none' for standard patch embedding)
        stem_base_channels: Base channels for ConvStem
        stem_hidden_dim: Hidden dimension for ConvStem
    """

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
        channels=5,
        ls_init_values=None,
        patch_stride=None,
        fusion_blocks=[4, 8, 12],
        conv_stem='none',
        stem_base_channels=32,
        stem_hidden_dim=None,
    ):
        super().__init__()

        # Store config
        self.conv_stem = conv_stem
        self.fusion_blocks = fusion_blocks
        self.d_model = d_model
        self.n_cls = n_cls
        self.n_layers = n_layers

        if patch_stride is None:
            patch_stride = patch_size

        # Patch embedding / ConvStem
        if self.conv_stem == 'none':
            self.patch_embed = PatchEmbedding(
                image_size, patch_size, patch_stride, d_model, channels)
        else:
            self.patch_embed = ConvStem(
                in_channels=channels,
                base_channels=stem_base_channels,
                img_size=image_size,
                patch_stride=patch_stride,
                embed_dim=d_model,
                flatten=True,
                hidden_dim=stem_hidden_dim)

        self.patch_size = patch_size
        self.PS_H, self.PS_W = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        self.patch_stride = patch_stride

        # Grid size
        if isinstance(patch_stride, (list, tuple)):
            self.grid_h = image_size[0] // patch_stride[0]
            self.grid_w = image_size[1] // patch_stride[1]
        else:
            self.grid_h = image_size[0] // patch_stride
            self.grid_w = image_size[1] // patch_stride

        # Transformation pipeline for fusion
        self.etp = EfficientTransformationPipeline(self.grid_h, self.grid_w)

        # CLS token and pos embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, dpr[i], init_values=ls_init_values)
            for i in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Fusion layers (one per fusion point)
        self.point_fusion_layers = nn.ModuleList([
            PointFusionLayer(d_model) for _ in fusion_blocks
        ])
        self.pixel_fusion_layers = nn.ModuleList([
            PixelFusionLayer(d_model) for _ in fusion_blocks
        ])

        # Auxiliary heads for pixel supervision
        self.aux_heads = nn.ModuleList([
            AuxiliaryHead(d_model, n_cls) for _ in fusion_blocks
        ])

        # Initialize weights
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_grid_size(self, H, W):
        return self.patch_embed.get_grid_size(H, W)

    def _reshape_tokens_to_2d(self, tokens, grid_h, grid_w):
        """
        Reshape tokens from sequence to 2D spatial format.

        Args:
            tokens: (B, N, D) token sequence
            grid_h: Height of the grid
            grid_w: Width of the grid

        Returns:
            pixel_feats: (B, D, H, W) spatial feature map
        """
        B, N, D = tokens.shape
        return tokens.transpose(1, 2).reshape(B, D, grid_h, grid_w)

    def _reshape_2d_to_tokens(self, pixel_feats):
        """
        Reshape 2D spatial features back to token sequence.

        Args:
            pixel_feats: (B, D, H, W) spatial feature map

        Returns:
            tokens: (B, N, D) token sequence
        """
        B, D, H, W = pixel_feats.shape
        return pixel_feats.flatten(2).transpose(1, 2)

    def _convert_coords_to_patch_space(self, coords):
        """
        Convert pixel-space coordinates to patch-space coordinates.

        Args:
            coords: (N, 3) coordinates [batch_idx, y, x] in pixel space

        Returns:
            patch_coords: (N, 3) coordinates [batch_idx, y, x] in patch space
        """
        patch_coords = coords.clone()
        if isinstance(self.patch_stride, (list, tuple)):
            patch_coords[:, 1] = coords[:, 1] // self.patch_stride[0]
            patch_coords[:, 2] = coords[:, 2] // self.patch_stride[1]
        else:
            patch_coords[:, 1] = coords[:, 1] // self.patch_stride
            patch_coords[:, 2] = coords[:, 2] // self.patch_stride
        return patch_coords

    def forward(self, im, point_feats=None, coords=None):
        """
        Forward pass with optional bidirectional point-pixel fusion.

        Args:
            im: (B, C, H, W) input range image
            point_feats: (N_total, D) point features (optional)
            coords: (N_total, 3) point coordinates [batch_idx, y, x] in pixel space

        Returns:
            pixel_feats: (B, D, grid_H, grid_W) final pixel features
            point_feats: (N_total, D) updated point features (or None if not provided)
            aux_outputs: List of (B, n_cls, grid_H, grid_W) auxiliary logits
            skip: (B, D_h, H, W) skip features from ConvStem (or None if no ConvStem)
        """
        B, _, H, W = im.shape

        # Compute actual grid size from input dimensions (handles variable-size inputs)
        if isinstance(self.patch_stride, (list, tuple)):
            actual_grid_h = H // self.patch_stride[0]
            actual_grid_w = W // self.patch_stride[1]
        else:
            actual_grid_h = H // self.patch_stride
            actual_grid_w = W // self.patch_stride

        # Patch embedding
        x, skip = self.patch_embed(im)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Positional embedding
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

        # Convert coords to patch space
        patch_coords = None
        if coords is not None:
            patch_coords = self._convert_coords_to_patch_space(coords)

        # Create dynamic ETP for actual grid size (needed for fusion operations)
        dynamic_etp = EfficientTransformationPipeline(actual_grid_h, actual_grid_w)

        aux_outputs = []
        fusion_idx = 0

        # Process transformer blocks
        for block_idx, blk in enumerate(self.blocks):
            x = blk(x)

            # Check if fusion point (1-indexed in config)
            block_num = block_idx + 1
            if block_num in self.fusion_blocks:
                # Remove CLS for spatial ops
                cls_token = x[:, :1]
                tokens = x[:, 1:]

                # Reshape to 2D using actual grid size
                pixel_feats = self._reshape_tokens_to_2d(tokens, actual_grid_h, actual_grid_w)

                if point_feats is not None and patch_coords is not None:
                    # Pixel -> Point: map pixel features to point locations
                    mapped_pixel = dynamic_etp.pixel2point(pixel_feats, patch_coords)

                    # Point fusion: combine mapped pixel features with point features
                    point_feats = self.point_fusion_layers[fusion_idx](mapped_pixel, point_feats)

                    # Point -> Cluster: aggregate point features into voxels
                    voxel_coords, cluster_feats = dynamic_etp.point2cluster(point_feats, patch_coords)

                    # Cluster -> Pixel: convert sparse voxel features to dense grid
                    pixel_from_points = dynamic_etp.cluster2pixel(cluster_feats, voxel_coords, B)

                    # Pixel fusion: combine aggregated point features with pixel features
                    pixel_feats = self.pixel_fusion_layers[fusion_idx](pixel_from_points, pixel_feats)

                # Auxiliary output for intermediate supervision
                aux_logits = self.aux_heads[fusion_idx](pixel_feats)
                aux_outputs.append(aux_logits)

                # Reshape back to token sequence
                tokens = self._reshape_2d_to_tokens(pixel_feats)
                x = torch.cat([cls_token, tokens], dim=1)

                fusion_idx += 1

        # Final layer normalization
        x = self.norm(x)

        # Remove CLS and reshape to 2D spatial format using actual grid size
        tokens = x[:, 1:]
        pixel_feats = self._reshape_tokens_to_2d(tokens, actual_grid_h, actual_grid_w)

        return pixel_feats, point_feats, aux_outputs, skip
