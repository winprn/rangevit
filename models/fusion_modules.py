# Copyright 2026 - RangeViT-Fusion
# Bidirectional point-pixel fusion components

import torch
import torch.nn as nn
from typing import Tuple, Optional

# Try to import torch_scatter for efficient scatter operations
try:
    from torch_scatter import scatter_max
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False


class EfficientTransformationPipeline:
    """
    Handles bidirectional mappings between pixel (2D) and point (3D) features.

    This is the core of the fusion mechanism, enabling feature exchange between
    the range image domain and the point cloud domain.

    Args:
        ny: Grid height (number of rows in range image)
        nx: Grid width (number of columns in range image)
    """

    def __init__(self, ny: int, nx: int):
        self.ny = ny
        self.nx = nx

    def pixel2point(
        self,
        pixel_feats: torch.Tensor,
        coords: torch.Tensor,
        stride: int = 1
    ) -> torch.Tensor:
        """
        Map pixel features to points using projection coordinates.

        Args:
            pixel_feats: (B, D, H, W) pixel feature tensor
            coords: (N, 3) coordinates [batch_idx, y, x] for each point
            stride: Stride factor for coordinate scaling (for multi-scale features)

        Returns:
            point_feats: (N, D) features gathered for each point
        """
        B, D, H, W = pixel_feats.shape
        N = coords.shape[0]

        if N == 0:
            return torch.zeros(0, D, device=pixel_feats.device, dtype=pixel_feats.dtype)

        # Extract coordinates and apply stride
        batch_idx = coords[:, 0].long()
        y = (coords[:, 1] / stride).long().clamp(0, H - 1)
        x = (coords[:, 2] / stride).long().clamp(0, W - 1)

        # Gather features from pixel locations
        # pixel_feats: (B, D, H, W) -> need to index [batch_idx, :, y, x]
        point_feats = pixel_feats[batch_idx, :, y, x]  # (N, D)

        return point_feats

    def point2cluster(
        self,
        point_feats: torch.Tensor,
        coords: torch.Tensor,
        stride: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate point features into voxel clusters using max pooling.

        Args:
            point_feats: (N, D) point feature tensor
            coords: (N, 3) coordinates [batch_idx, y, x] for each point
            stride: Stride factor for voxel grid resolution

        Returns:
            cluster_coords: (M, 3) unique voxel coordinates [batch_idx, y, x]
            cluster_feats: (M, D) aggregated features per voxel
        """
        N, D = point_feats.shape
        device = point_feats.device
        dtype = point_feats.dtype

        if N == 0:
            return (
                torch.zeros(0, 3, device=device, dtype=coords.dtype),
                torch.zeros(0, D, device=device, dtype=dtype)
            )

        # Scale coordinates by stride
        batch_idx = coords[:, 0].long()
        y = (coords[:, 1] / stride).long()
        x = (coords[:, 2] / stride).long()

        # Compute scaled grid dimensions
        H = (self.ny + stride - 1) // stride
        W = (self.nx + stride - 1) // stride

        # Clamp to valid range
        y = y.clamp(0, H - 1)
        x = x.clamp(0, W - 1)

        # Compute linear indices for voxels: batch_idx * H * W + y * W + x
        # Find max batch index to compute proper offsets
        max_batch = batch_idx.max().item() + 1 if N > 0 else 1
        voxel_idx = batch_idx * (H * W) + y * W + x

        # Get unique voxels and inverse mapping
        unique_voxels, inverse_idx = torch.unique(voxel_idx, return_inverse=True)
        M = unique_voxels.shape[0]

        if HAS_TORCH_SCATTER:
            # Use torch_scatter for efficient max pooling
            cluster_feats, _ = scatter_max(point_feats, inverse_idx, dim=0, dim_size=M)
        else:
            # Fallback: loop-based implementation
            cluster_feats = torch.zeros(M, D, device=device, dtype=dtype)
            # Initialize with very negative values for max pooling
            cluster_feats.fill_(float('-inf'))

            for i in range(N):
                idx = inverse_idx[i]
                cluster_feats[idx] = torch.max(cluster_feats[idx], point_feats[i])

            # Replace -inf with 0 for voxels that had no points (shouldn't happen but safety)
            cluster_feats = torch.where(
                cluster_feats == float('-inf'),
                torch.zeros_like(cluster_feats),
                cluster_feats
            )

        # Recover voxel coordinates from linear indices
        cluster_batch = unique_voxels // (H * W)
        remainder = unique_voxels % (H * W)
        cluster_y = remainder // W
        cluster_x = remainder % W

        cluster_coords = torch.stack([cluster_batch, cluster_y, cluster_x], dim=1)

        return cluster_coords, cluster_feats

    def cluster2pixel(
        self,
        cluster_feats: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
        stride: int = 1
    ) -> torch.Tensor:
        """
        Convert sparse voxel features to dense pixel grid.

        Args:
            cluster_feats: (M, D) voxel feature tensor
            coords: (M, 3) voxel coordinates [batch_idx, y, x]
            batch_size: Number of samples in batch
            stride: Stride factor for output grid resolution

        Returns:
            pixel_feats: (B, D, H, W) dense pixel feature tensor
        """
        M, D = cluster_feats.shape
        device = cluster_feats.device
        dtype = cluster_feats.dtype

        # Compute output grid dimensions
        H = (self.ny + stride - 1) // stride
        W = (self.nx + stride - 1) // stride

        # Initialize output tensor with zeros
        pixel_feats = torch.zeros(batch_size, D, H, W, device=device, dtype=dtype)

        if M == 0:
            return pixel_feats

        # Extract coordinates
        batch_idx = coords[:, 0].long()
        y = coords[:, 1].long().clamp(0, H - 1)
        x = coords[:, 2].long().clamp(0, W - 1)

        # Scatter features to pixel locations
        # Use index_put_ for assignment
        pixel_feats[batch_idx, :, y, x] = cluster_feats

        return pixel_feats


class PointFusionLayer(nn.Module):
    """
    Fuses mapped pixel features with point features.

    Architecture: Concat -> Linear(2D, D) -> BN1d -> ReLU

    Args:
        d_model: Feature dimension
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.d_model = d_model

        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        mapped_pixel_feats: torch.Tensor,
        point_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            mapped_pixel_feats: (N, D) pixel features mapped to points
            point_feats: (N, D) original point features

        Returns:
            fused_feats: (N, D) fused point features
        """
        # Concatenate along feature dimension
        concat_feats = torch.cat([mapped_pixel_feats, point_feats], dim=1)  # (N, 2D)

        # Apply fusion
        fused_feats = self.fusion(concat_feats)  # (N, D)

        return fused_feats


class PixelFusionLayer(nn.Module):
    """
    Fuses mapped point features with pixel features.

    Architecture: Concat -> Conv2d(2D, D, 1x1) -> BN2d -> Hardswish

    Args:
        d_model: Feature dimension
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.d_model = d_model

        self.fusion = nn.Sequential(
            nn.Conv2d(2 * d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.Hardswish(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        pixel_from_points: torch.Tensor,
        pixel_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pixel_from_points: (B, D, H, W) pixel features derived from points
            pixel_feats: (B, D, H, W) original pixel features

        Returns:
            fused_feats: (B, D, H, W) fused pixel features
        """
        # Concatenate along channel dimension
        concat_feats = torch.cat([pixel_from_points, pixel_feats], dim=1)  # (B, 2D, H, W)

        # Apply fusion
        fused_feats = self.fusion(concat_feats)  # (B, D, H, W)

        return fused_feats


class AuxiliaryHead(nn.Module):
    """
    Lightweight head for pixel supervision during training.

    Architecture: Conv2d(D, n_classes, 1x1)

    Args:
        d_model: Input feature dimension
        n_classes: Number of output classes
    """

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()

        self.d_model = d_model
        self.n_classes = n_classes

        self.head = nn.Conv2d(d_model, n_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.head.weight, mode='fan_out', nonlinearity='relu')
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, pixel_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_feats: (B, D, H, W) pixel features

        Returns:
            logits: (B, n_classes, H, W) class logits
        """
        return self.head(pixel_feats)
