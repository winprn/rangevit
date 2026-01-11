"""
Voxelization module for converting point clouds to voxel grids and projecting to range images.
Implements custom PyTorch-based voxelization without heavy dependencies.
"""

import numpy as np
import torch
from typing import Tuple, Optional


class VoxelGrid:
    """
    Voxelizes point clouds and projects voxel features to range images.

    Phase 1: Non-learnable features (mean intensity + density)
    Phase 2: Will support learnable encoding
    """

    def __init__(self, voxel_size: float = 0.05, grid_bounds: Optional[Tuple] = None):
        """
        Initialize voxel grid.

        Args:
            voxel_size: Size of each voxel cube (default 0.05m = 5cm)
            grid_bounds: Optional fixed bounds (min_coords, max_coords), or None for automatic
        """
        self.voxel_size = voxel_size
        self.grid_bounds = grid_bounds

    def compute_grid_bounds(self, points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute tight grid bounds from point cloud with padding.

        Args:
            points_xyz: Point coordinates [N, 3]

        Returns:
            min_bound: Minimum coordinates [3]
            max_bound: Maximum coordinates [3]
        """
        pmin = np.floor(points_xyz.min(axis=0) / self.voxel_size) * self.voxel_size
        pmax = np.ceil(points_xyz.max(axis=0) / self.voxel_size) * self.voxel_size

        # Add padding (one voxel on each side)
        min_bound = pmin - self.voxel_size
        max_bound = pmax + self.voxel_size

        return min_bound, max_bound

    def voxelize(self, pointcloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert point cloud to voxel grid with aggregated features.

        Args:
            pointcloud: Point cloud [N, 4] where columns are [x, y, z, intensity]

        Returns:
            voxel_coords: Voxel center coordinates [V, 3]
            voxel_features: Aggregated features [V, 2] where channels are [mean_intensity, normalized_density]
        """
        if pointcloud.shape[0] == 0:
            # Empty point cloud
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

        points_xyz = pointcloud[:, :3]  # [N, 3]
        intensity = pointcloud[:, 3]    # [N]

        # Compute grid bounds
        if self.grid_bounds is not None:
            min_bound, max_bound = self.grid_bounds
        else:
            min_bound, max_bound = self.compute_grid_bounds(points_xyz)

        # Compute voxel indices for each point
        voxel_indices = np.floor((points_xyz - min_bound) / self.voxel_size).astype(np.int32)

        # Create unique voxel keys using hash-based indexing
        # Hash function: z * grid_y * grid_x + y * grid_x + x
        grid_size = np.ceil((max_bound - min_bound) / self.voxel_size).astype(np.int32)
        voxel_keys = (voxel_indices[:, 2] * grid_size[1] * grid_size[0] +
                      voxel_indices[:, 1] * grid_size[0] +
                      voxel_indices[:, 0])

        # Get unique voxels and inverse indices
        unique_keys, inverse_indices = np.unique(voxel_keys, return_inverse=True)
        num_voxels = len(unique_keys)

        # Aggregate features per voxel
        voxel_features = np.zeros((num_voxels, 2), dtype=np.float32)
        voxel_point_counts = np.zeros(num_voxels, dtype=np.int32)

        # Use numpy bincount for efficient aggregation
        # Channel 0: Mean intensity
        intensity_sum = np.bincount(inverse_indices, weights=intensity, minlength=num_voxels)
        point_counts = np.bincount(inverse_indices, minlength=num_voxels)

        # Avoid division by zero (though this shouldn't happen with unique voxels)
        valid_mask = point_counts > 0
        voxel_features[valid_mask, 0] = intensity_sum[valid_mask] / point_counts[valid_mask]

        # Channel 1: Normalized density (count / max_count)
        max_count = point_counts.max()
        if max_count > 0:
            voxel_features[:, 1] = point_counts.astype(np.float32) / max_count

        # Compute voxel center coordinates
        # Reconstruct voxel indices from keys
        voxel_coords_indices = np.zeros((num_voxels, 3), dtype=np.int32)
        voxel_coords_indices[:, 0] = unique_keys % grid_size[0]
        voxel_coords_indices[:, 1] = (unique_keys // grid_size[0]) % grid_size[1]
        voxel_coords_indices[:, 2] = unique_keys // (grid_size[0] * grid_size[1])

        # Convert indices to world coordinates (voxel centers)
        voxel_coords = (voxel_coords_indices.astype(np.float32) + 0.5) * self.voxel_size + min_bound

        return voxel_coords, voxel_features

    def project_to_range(self,
                         voxel_coords: np.ndarray,
                         voxel_features: np.ndarray,
                         projection) -> np.ndarray:
        """
        Project voxel features to range image using spherical projection.

        Args:
            voxel_coords: Voxel center coordinates [V, 3]
            voxel_features: Voxel features [V, D]
            projection: RangeProjection instance with FOV and resolution settings

        Returns:
            range_voxel_features: Projected features [H, W, D]
        """
        if voxel_coords.shape[0] == 0:
            # No voxels, return zeros
            feature_dim = voxel_features.shape[1] if voxel_features.ndim > 1 else 1
            return np.zeros((projection.proj_h, projection.proj_w, feature_dim), dtype=np.float32)

        # Compute depth (range) from voxel centers
        depth = np.linalg.norm(voxel_coords, axis=1)  # [V]

        # Avoid division by zero
        valid_mask = depth > 1e-6
        if not np.any(valid_mask):
            feature_dim = voxel_features.shape[1] if voxel_features.ndim > 1 else 1
            return np.zeros((projection.proj_h, projection.proj_w, feature_dim), dtype=np.float32)

        # Apply valid mask
        voxel_coords = voxel_coords[valid_mask]
        voxel_features = voxel_features[valid_mask]
        depth = depth[valid_mask]

        # Compute spherical coordinates
        # Yaw: azimuth angle in horizontal plane
        yaw = -np.arctan2(voxel_coords[:, 1], voxel_coords[:, 0])  # [V]

        # Pitch: elevation angle
        pitch = np.arcsin(np.clip(voxel_coords[:, 2] / depth, -1.0, 1.0))  # [V]

        # Project to range image pixel coordinates
        # Following the same convention as RangeProjection
        fov_left = getattr(projection, 'fov_left', -np.pi)
        fov_right = getattr(projection, 'fov_right', np.pi)
        fov_up = getattr(projection, 'fov_up', 0.0523599)  # 3 degrees in radians
        fov_down = getattr(projection, 'fov_down', -0.436332)  # -25 degrees in radians

        fov_h = fov_right - fov_left  # Horizontal FOV range
        fov_v = fov_up - fov_down     # Vertical FOV range

        # Normalize to [0, 1]
        proj_x = (yaw - fov_left) / fov_h
        proj_y = 1.0 - (pitch - fov_down) / fov_v

        # Scale to pixel coordinates
        px_float = proj_x * projection.proj_w
        py_float = proj_y * projection.proj_h

        # Clip and convert to integer indices
        px = np.floor(np.clip(px_float, 0, projection.proj_w - 1)).astype(np.int32)
        py = np.floor(np.clip(py_float, 0, projection.proj_h - 1)).astype(np.int32)

        # Depth-weighted averaging for multiple voxels per pixel
        feature_dim = voxel_features.shape[1] if voxel_features.ndim > 1 else 1
        range_voxel_features = np.zeros((projection.proj_h, projection.proj_w, feature_dim), dtype=np.float32)
        weight_sum = np.zeros((projection.proj_h, projection.proj_w), dtype=np.float32)

        # Compute depth weights (closer voxels weighted higher)
        depth_weights = 1.0 / (depth + 1e-6)  # [V]

        # Aggregate features with depth weighting
        for i in range(len(voxel_coords)):
            px_i, py_i = px[i], py[i]
            weight = depth_weights[i]
            range_voxel_features[py_i, px_i] += weight * voxel_features[i]
            weight_sum[py_i, px_i] += weight

        # Normalize by weight sum (avoid division by zero)
        valid_pixels = weight_sum > 0
        range_voxel_features[valid_pixels] /= weight_sum[valid_pixels, np.newaxis]

        # Empty pixels remain zero-filled (semantically means "no voxel info")

        return range_voxel_features


def test_voxelization():
    """
    Unit test for voxelization module.
    """
    print("Testing VoxelGrid class...")

    # Create synthetic point cloud
    np.random.seed(42)
    num_points = 1000
    points_xyz = np.random.randn(num_points, 3) * 5.0  # Random points in 3D
    intensity = np.random.rand(num_points)
    pointcloud = np.column_stack([points_xyz, intensity])

    # Initialize voxelizer
    voxelizer = VoxelGrid(voxel_size=0.05)

    # Test voxelization
    voxel_coords, voxel_features = voxelizer.voxelize(pointcloud)

    print(f"Input: {num_points} points")
    print(f"Output: {len(voxel_coords)} voxels")
    print(f"Voxel coords shape: {voxel_coords.shape}")
    print(f"Voxel features shape: {voxel_features.shape}")
    print(f"Feature ranges - Intensity: [{voxel_features[:, 0].min():.3f}, {voxel_features[:, 0].max():.3f}]")
    print(f"Feature ranges - Density: [{voxel_features[:, 1].min():.3f}, {voxel_features[:, 1].max():.3f}]")

    # Test projection (mock projection object)
    class MockProjection:
        def __init__(self):
            self.proj_h = 64
            self.proj_w = 2048
            self.fov_left = -np.pi
            self.fov_right = np.pi
            self.fov_up = np.deg2rad(3.0)
            self.fov_down = np.deg2rad(-25.0)

    mock_proj = MockProjection()
    range_features = voxelizer.project_to_range(voxel_coords, voxel_features, mock_proj)

    print(f"\nProjected range features shape: {range_features.shape}")
    non_zero_pixels = (range_features != 0).any(axis=2).sum()
    print(f"Non-zero pixels: {non_zero_pixels} / {64 * 2048} ({100 * non_zero_pixels / (64 * 2048):.2f}%)")

    print("\nVoxelization test passed!")


if __name__ == "__main__":
    test_voxelization()
