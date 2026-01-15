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
Representation conversion utilities for multi-view fusion.

Functions for converting between:
- Voxel representation (SparseTensor)
- Point representation (PointTensor)
- Range image representation (dense 2D tensor)

Adapted from OpenPCSeg RPVNet: pcseg/model/segmentor/fusion/rpvnet/utils.py
"""

import torch
import torch.nn.functional as F
import torchsparse
import torchsparse.nn.functional as spF
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets

__all__ = [
    'initial_voxelize',
    'voxel_to_point',
    'point_to_voxel',
    'range_to_point',
    'point_to_range',
]


def initial_voxelize(z: PointTensor, init_res: float, after_res: float) -> SparseTensor:
    """
    Convert PointTensor to SparseTensor via voxelization.

    Args:
        z: PointTensor with features [N, C] and coordinates [N, 4] (x, y, z, batch_idx)
        init_res: Initial resolution (typically point resolution)
        after_res: Target voxel resolution

    Returns:
        SparseTensor with voxelized features
    """
    # Scale spatial coordinates from meters to voxel units
    # Input coordinates are in meters, voxel resolution is after_res (e.g., 0.05m)
    # Formula: coord_meters / voxel_size_meters = coord_voxels
    scaled_coords = z.C[:, :3] / after_res

    # Offset spatial coordinates to be non-negative
    # This prevents issues with torchsparse's handling of negative coordinates
    # during strided convolutions
    coord_min = scaled_coords.min(dim=0).values
    scaled_coords = scaled_coords - coord_min  # Now all >= 0

    # Scale batch index to survive stride operations
    # torchsparse applies stride to all coordinates including batch dimension
    # With 4 stride-2 stages, we need scale factor of 2^4 = 16
    # This ensures batch indices remain distinguishable after all downsampling
    batch_scale = 16
    scaled_batch = z.C[:, -1:] * batch_scale

    new_float_coord = torch.cat(
        [scaled_coords, scaled_batch], 1,
    )

    pc_hash = spF.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = spF.sphashquery(pc_hash, sparse_hash)
    counts = spF.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = spF.spvoxelize(
        torch.floor(new_float_coord),
        idx_query,
        counts,
    )
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = spF.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor._caches.cmaps.setdefault(new_tensor.stride, new_tensor.coords)

    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


def point_to_voxel(x: SparseTensor, z: PointTensor) -> SparseTensor:
    """
    Aggregate point features to voxels (P2V).

    Uses averaging to aggregate features from points that fall into the same voxel.

    Args:
        x: SparseTensor (voxel representation) to update
        z: PointTensor with features to aggregate

    Returns:
        SparseTensor with aggregated point features
    """
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
                x.s) is None:
        pc_hash = spF.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = spF.sphash(x.C)
        idx_query = spF.sphashquery(pc_hash, sparse_hash)
        counts = spF.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = spF.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    # Only set the coordinate map for current stride, don't copy entire cache
    # Copying stale caches causes CUDA errors in downstream convolutions
    new_tensor._caches.cmaps.setdefault(new_tensor.stride, new_tensor.coords)

    return new_tensor


def voxel_to_point(x: SparseTensor, z: PointTensor, nearest: bool = False) -> PointTensor:
    """
    Interpolate voxel features to points (V2P).

    Uses trilinear interpolation by default, or nearest neighbor if specified.

    Args:
        x: SparseTensor with voxel features
        z: PointTensor with point coordinates to interpolate to
        nearest: If True, use nearest neighbor instead of trilinear interpolation

    Returns:
        PointTensor with interpolated features at point locations
    """
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = spF.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = spF.sphash(x.C.to(z.F.device))
        idx_query = spF.sphashquery(old_hash, pc_hash)
        weights = spF.calc_ti_weights(z.C, idx_query,
                                      scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = spF.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = spF.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor


def range_to_point(
    feature_map: torch.Tensor,
    pxpy: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Sample range image features at point locations (R2P).

    Uses bilinear interpolation to sample features from the 2D range image
    at the projected point locations.

    Args:
        feature_map: [B, C, H, W] range image features
        pxpy: [N, 2] projection coordinates (px, py) normalized to [-1, 1]
        batch_indices: [N] batch index per point
        batch_size: Number of samples in batch
        mode: Interpolation mode ('bilinear' or 'nearest')

    Returns:
        [N, C] point features sampled from range image
    """
    N = pxpy.shape[0]
    C = feature_map.shape[1]

    # Output tensor for all points
    output = torch.zeros(N, C, device=feature_map.device, dtype=feature_map.dtype)

    # Process each batch sample
    for b in range(batch_size):
        # Get points belonging to this batch
        mask = batch_indices == b
        if not mask.any():
            continue

        # Get projection coordinates for this batch's points
        # pxpy is [N, 2] with (px, py) in [-1, 1]
        coords = pxpy[mask]  # [Nb, 2]

        # Reshape for grid_sample: [1, 1, Nb, 2]
        grid = coords.unsqueeze(0).unsqueeze(0)  # [1, 1, Nb, 2]

        # Sample from feature map
        # Input: [1, C, H, W], Grid: [1, 1, Nb, 2]
        # Output: [1, C, 1, Nb]
        sampled = F.grid_sample(
            feature_map[b:b+1],  # [1, C, H, W]
            grid,
            mode=mode,
            padding_mode='border',
            align_corners=False
        )

        # Reshape: [1, C, 1, Nb] -> [Nb, C]
        sampled = sampled.squeeze(0).squeeze(1).transpose(0, 1)  # [Nb, C]

        output[mask] = sampled

    return output


def range_to_point_from_tokens(
    tokens: torch.Tensor,
    pxpy: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    H: int,
    W: int,
    patch_stride: tuple = (2, 8)
) -> torch.Tensor:
    """
    Sample features from ViT token representation at point locations.

    Converts ViT tokens back to spatial feature map, then samples at point locations.
    Used for Fusion 2 (after encoder) where features are in token form.

    Args:
        tokens: [B, N_tokens, d_model] ViT encoder output (without CLS token)
        pxpy: [N, 2] projection coordinates (px, py) normalized to [-1, 1]
        batch_indices: [N] batch index per point
        batch_size: Number of samples in batch
        H, W: Original image height and width
        patch_stride: Patch stride used in ViT

    Returns:
        [N, d_model] point features sampled from token representation
    """
    B, N_tokens, d_model = tokens.shape

    # Compute spatial dimensions of token grid
    GH = H // patch_stride[0]
    GW = W // patch_stride[1]

    # Reshape tokens to spatial: [B, N_tokens, d_model] -> [B, d_model, GH, GW]
    feature_map = tokens.transpose(1, 2).reshape(B, d_model, GH, GW)

    # Use standard range_to_point with the spatial feature map
    return range_to_point(feature_map, pxpy, batch_indices, batch_size, mode='bilinear')


def point_to_range(
    point_features: torch.Tensor,
    pxpy: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    H: int,
    W: int,
    mode: str = 'scatter_mean'
) -> torch.Tensor:
    """
    Scatter point features to range image (P2R).

    Aggregates point features into a 2D range image representation.

    Args:
        point_features: [N, C] point features
        pxpy: [N, 2] projection coordinates (px, py) normalized to [-1, 1]
        batch_indices: [N] batch index per point
        batch_size: Number of samples in batch
        H, W: Output range image dimensions
        mode: Aggregation mode ('scatter_mean' supported)

    Returns:
        [B, C, H, W] range image with scattered point features
    """
    N, C = point_features.shape
    device = point_features.device
    dtype = point_features.dtype

    # Output tensor
    output = torch.zeros(batch_size, C, H, W, device=device, dtype=dtype)
    count = torch.zeros(batch_size, 1, H, W, device=device, dtype=dtype)

    # Convert normalized coordinates [-1, 1] to pixel indices [0, H-1] and [0, W-1]
    px_normalized = pxpy[:, 0]  # [-1, 1]
    py_normalized = pxpy[:, 1]  # [-1, 1]

    # Convert to pixel coordinates
    px_pixel = ((px_normalized + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
    py_pixel = ((py_normalized + 1) / 2 * (H - 1)).long().clamp(0, H - 1)

    # Scatter features
    for i in range(N):
        b = batch_indices[i].long()
        x = px_pixel[i]
        y = py_pixel[i]

        output[b, :, y, x] += point_features[i]
        count[b, 0, y, x] += 1

    # Average where count > 0
    mask = count > 0
    output = output / (count + (count == 0).float())  # Avoid division by zero

    return output


def point_to_range_fast(
    point_features: torch.Tensor,
    pxpy: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Scatter point features to range image (P2R) - vectorized version.

    Uses scatter_add for faster execution compared to the loop-based version.

    Args:
        point_features: [N, C] point features
        pxpy: [N, 2] projection coordinates (px, py) normalized to [-1, 1]
        batch_indices: [N] batch index per point (long tensor)
        batch_size: Number of samples in batch
        H, W: Output range image dimensions

    Returns:
        [B, C, H, W] range image with scattered point features
    """
    N, C = point_features.shape
    device = point_features.device
    dtype = point_features.dtype

    # Convert normalized coordinates [-1, 1] to pixel indices
    px_normalized = pxpy[:, 0]
    py_normalized = pxpy[:, 1]

    px_pixel = ((px_normalized + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
    py_pixel = ((py_normalized + 1) / 2 * (H - 1)).long().clamp(0, H - 1)

    # Compute linear indices: batch * H * W + y * W + x
    batch_idx = batch_indices.long()
    linear_idx = batch_idx * H * W + py_pixel * W + px_pixel  # [N]

    # Expand linear_idx for scatter: [N] -> [N, C]
    linear_idx_expanded = linear_idx.unsqueeze(1).expand(-1, C)  # [N, C]

    # Initialize output and count tensors as flat
    output_flat = torch.zeros(batch_size * H * W, C, device=device, dtype=dtype)
    count_flat = torch.zeros(batch_size * H * W, 1, device=device, dtype=dtype)

    # Scatter add
    output_flat.scatter_add_(0, linear_idx_expanded, point_features)
    count_flat.scatter_add_(0, linear_idx.unsqueeze(1), torch.ones(N, 1, device=device, dtype=dtype))

    # Reshape
    output = output_flat.view(batch_size, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
    count = count_flat.view(batch_size, H, W, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]

    # Average where count > 0
    output = output / (count + (count == 0).float())

    return output
