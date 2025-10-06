# Test script for RangeFormer model
# Demonstrates usage of the refactored modular implementation
# This replaces the monolithic test_model.py

import math
import numpy as np
import torch
import torch.nn as nn

# Import RangeFormer components from modular structure
from models.rangeformer import RangeFormer, create_rangeformer

# Import RangeViT's projection utilities (reusing battle-tested code)
from dataset.preprocess.projection import RangeProjection

# Import RangeAug functions (RangeFormer-specific augmentations)
from dataset.preprocess.range_aug import range_mix, range_union, range_paste, range_shift


# ---------------------------
# Helper functions for inference
# ---------------------------
def rv_preds_to_point_labels(pred_map: np.ndarray, index_map: np.ndarray,
                             num_points: int, default_label: int = 0):
    """
    Map 2D predictions back to original points.

    Args:
        pred_map: (H, W) predicted class ids
        index_map: (H, W) mapping pixel -> original point index or -1
        num_points: total number of points
        default_label: label for unmapped points

    Returns:
        point_labels: (num_points,) per-point labels
    """
    point_labels = np.ones((num_points,), dtype=np.int32) * default_label
    H, W = index_map.shape
    for r in range(H):
        for c in range(W):
            idx = index_map[r, c]
            if idx >= 0:
                point_labels[idx] = int(pred_map[r, c])
    return point_labels


def split_into_views(points: np.ndarray, Z: int):
    """
    Split scan into Z views based on azimuth angle.
    Used for STR (Scan-to-Range with multiple views) strategy.

    Args:
        points: (N, >=4) point cloud
        Z: number of views

    Returns:
        list of point arrays, one per view
    """
    azim = np.arctan2(points[:, 1], points[:, 0])  # [-pi, pi]
    # Map to [0, 2pi)
    azim_pos = azim.copy()
    azim_pos[azim_pos < 0] += 2 * math.pi
    bins = np.linspace(0, 2 * math.pi, num=Z + 1)
    views = []
    for i in range(Z):
        mask = (azim_pos >= bins[i]) & (azim_pos < bins[i + 1])
        views.append(points[mask])
    return views


def range_post_inference(model: nn.Module, rasterizer: RangeProjection,
                         scan: np.ndarray, num_sub: int, device='cpu'):
    """
    RangePost inference strategy from RangeFormer paper.
    Split scan into subclouds, rasterize each, predict, then merge.

    Args:
        model: RangeFormer model
        rasterizer: RangeProjection instance
        scan: (N, >=4) point cloud
        num_sub: number of subclouds to split into
        device: torch device

    Returns:
        final_pred: (N,) per-point predictions
    """
    # Step 1: Split into subclouds
    subclouds = []
    indices = []
    for i in range(num_sub):
        sub = scan[i::num_sub, :]
        subclouds.append(sub)
        idxs = np.arange(i, scan.shape[0], num_sub)
        indices.append(idxs)

    # Step 2: Rasterize each subcloud
    rvs = []
    index_maps = []
    for sub in subclouds:
        proj_pc, proj_range, proj_idx, proj_mask = rasterizer.doProjection(sub)
        # Convert to 6-channel format [x, y, z, depth, intensity, existence]
        H, W = proj_range.shape
        rv = np.zeros((6, H, W), dtype=np.float32)
        rv[0] = proj_pc[:, :, 0]  # x
        rv[1] = proj_pc[:, :, 1]  # y
        rv[2] = proj_pc[:, :, 2]  # z
        rv[3] = proj_range         # depth
        rv[4] = proj_pc[:, :, 3] if proj_pc.shape[2] > 3 else 0  # intensity
        rv[5] = proj_mask          # existence
        rvs.append(rv)
        index_maps.append(proj_idx)

    # Step 3: Batch predict
    batch_rv = torch.from_numpy(np.stack(rvs, axis=0)).float().to(device)
    with torch.no_grad():
        logits_batch, _ = model(batch_rv)
        preds_batch = logits_batch.argmax(dim=1).cpu().numpy()  # (num_sub, H, W)

    # Step 4: Map back to points
    final_pred = np.zeros(scan.shape[0], dtype=np.int32)
    for j in range(len(preds_batch)):
        pred_j = preds_batch[j]
        idxs = indices[j]
        ind_map = index_maps[j]
        for r in range(ind_map.shape[0]):
            for c in range(ind_map.shape[1]):
                idx = ind_map[r, c]
                if idx >= 0:
                    global_idx = idxs[idx]
                    final_pred[global_idx] = int(pred_j[r, c])

    return final_pred


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("RangeFormer Model Test")
    print("Using refactored modular implementation")
    print("Reusing RangeViT components where applicable")
    print("=" * 60)

    # Model configuration
    H, W = 64, 1024  # Can be 64x512, 64x1024, or 64x2048
    num_classes = 19  # e.g., SemanticKITTI

    # Create model using factory function
    config = {
        'H': H,
        'W': W,
        'num_classes': num_classes,
        'depths': [2, 2, 6, 2],  # Can adjust for different model sizes
        'stage_channels': [128, 128, 320, 512],
        'heads': [3, 4, 6, 3],
        'decoder_unify_ch': 256
    }

    print("\n1. Creating RangeFormer model...")
    model = create_rangeformer(config)
    model.eval()

    # Print model statistics
    stats = model.count_parameters_by_component()
    print(f"\nModel statistics:")
    print(f"  Total parameters: {stats['total']:,}")
    print(f"  Backbone parameters: {stats['backbone']:,}")
    print(f"  Decoder parameters: {stats['decoder']:,}")
    print(f"  REM parameters: {stats['rem']:,}")

    # Generate fake point cloud
    print("\n2. Generating fake point cloud...")
    N = 120000  # Typical SemanticKITTI size
    rng = np.random.RandomState(42)
    thetas = rng.rand(N) * 2 * math.pi - math.pi
    phis = (rng.rand(N) - 0.5) * math.radians(28)
    ranges = rng.rand(N) * 80 + 1.0
    x = ranges * np.cos(phis) * np.cos(thetas)
    y = ranges * np.cos(phis) * np.sin(thetas)
    z = ranges * np.sin(phis)
    intensity = rng.rand(N).astype(np.float32)
    points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
    print(f"  Point cloud shape: {points.shape}")

    # Create rasterizer using RangeViT's projection (battle-tested)
    print("\n3. Creating range projection (using RangeViT's implementation)...")
    rasterizer = RangeProjection(
        fov_up=3.0,
        fov_down=-25.0,
        proj_w=W,
        proj_h=H
    )

    # Rasterize to range image
    print("\n4. Projecting point cloud to range image...")
    proj_pc, proj_range, proj_idx, proj_mask = rasterizer.doProjection(points)

    # Convert to 6-channel format
    rv_np = np.zeros((6, H, W), dtype=np.float32)
    rv_np[0] = proj_pc[:, :, 0]  # x
    rv_np[1] = proj_pc[:, :, 1]  # y
    rv_np[2] = proj_pc[:, :, 2]  # z
    rv_np[3] = proj_range         # depth
    rv_np[4] = proj_pc[:, :, 3]   # intensity
    rv_np[5] = proj_mask          # existence
    print(f"  Range image shape: {rv_np.shape}")
    print(f"  Valid pixels: {int(rv_np[5].sum())} / {H*W}")

    # Run inference
    print("\n5. Running inference...")
    rv_t = torch.from_numpy(rv_np[None, ...]).float()  # (1, 6, H, W)
    with torch.no_grad():
        logits, auxs = model(rv_t)
        preds2d = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

    print(f"  Predictions shape: {preds2d.shape}")
    print(f"  Main output shape: {logits.shape}")
    print(f"  Number of auxiliary outputs: {len(auxs)}")

    # Map back to points
    print("\n6. Mapping predictions back to points...")
    per_point_labels = rv_preds_to_point_labels(preds2d, proj_idx, N, default_label=0)
    print(f"  Per-point labels shape: {per_point_labels.shape}")
    print(f"  Unique labels: {np.unique(per_point_labels)}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    # Optional: Test RangePost inference
    print("\n7. Testing RangePost inference strategy...")
    final_pred = range_post_inference(model, rasterizer, points, num_sub=4, device='cpu')
    print(f"  Final predictions shape: {final_pred.shape}")
    print(f"  Unique labels: {np.unique(final_pred)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
