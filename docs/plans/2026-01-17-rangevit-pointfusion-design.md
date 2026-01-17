# RangeViT-PointFusion Design

## Overview

Replace MinkUNet voxel branch with lightweight Point MLP + Cross-Attention fusion for improved speed while maintaining accuracy gains from multi-view fusion.

**Goal:** Efficiently fuse range (ViT) and point features to improve accuracy over range-only, faster than current MinkUNet approach.

**Key decisions:**
- Lightweight MLPs on raw points (replacing MinkUNet's sparse 3D convs)
- Dual fusion at encoder and decoder stages
- Cross-attention mechanism (Point → Range direction)
- Point features: xyz + intensity + cluster center offset (7 channels)

---

## Architecture Overview

```
Input: LiDAR Point Cloud (N points)
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐  ┌──────────────┐
│ Range   │  │ Point MLP    │
│ Project │  │ Encoder      │
└────┬────┘  └──────┬───────┘
     │              │
     ▼              │ (point features: 7ch → hidden dim)
┌─────────┐         │
│ ConvStem│         │
└────┬────┘         │
     │              │
     ▼              │
┌─────────┐         │
│ ViT     │◄────────┤ Cross-Attention Fusion 1 (encoder)
│ Encoder │         │
└────┬────┘         │
     │              │
     ▼              │
┌─────────┐         │
│ Decoder │◄────────┘ Cross-Attention Fusion 2 (decoder)
└────┬────┘
     │
     ▼
  Per-point predictions
```

**Design principles:**
- ViT remains the primary encoder (preserves pre-training benefits)
- Point MLP encoder is lightweight (~50K params vs MinkUNet's ~5M)
- Cross-attention lets points selectively query relevant ViT features
- Dual fusion at encoder output and decoder output

---

## Component 1: Point MLP Encoder

**Purpose:** Encode raw point features into a learned representation for fusion.

**Input features (7 channels):**
- `xyz` (3): 3D coordinates
- `intensity` (1): LiDAR reflectance
- `cluster_offset` (3): relative position to voxel center `(x - cx, y - cy, z - cz)`

**Architecture:**

```python
class PointMLPEncoder(nn.Module):
    """
    Encode raw point features into learned representations.

    Input: (N, 7) - N points, 7 features
    Output: (N, hidden_dim) - N points, hidden_dim features
    """
    def __init__(self, in_channels=7, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, point_features):
        # point_features: (N, 7)
        return self.mlp(point_features)  # (N, hidden_dim)
```

**Design notes:**
- Shared MLPs (same weights for all points) - efficient, parallelizable
- BatchNorm for training stability
- `hidden_dim` matches ViT's feature dimension for easier fusion
- No pooling/aggregation - keep per-point features for cross-attention

**Cluster center computation (preprocessing):**
- Voxelize points with resolution ~10cm (configurable)
- Compute mean position per voxel
- `cluster_offset = point_xyz - voxel_center_xyz`

**Estimated parameters:** ~50K

---

## Component 2: Cross-Attention Fusion Module

**Purpose:** Let each point query relevant features from ViT's range representation.

**Inputs:**
- Point features: `(N, D)` from Point MLP Encoder
- ViT features: `(H, W, D)` range image feature map
- Point-to-pixel indices: `(N,)` mapping each point to its range pixel

**Architecture:**

```python
class PointToRangeCrossAttention(nn.Module):
    """
    Cross-attention where points query local ViT features.

    Each point attends to a 3x3 neighborhood in the range image.
    """
    def __init__(self, dim, window_size=3):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.scale = dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, point_feats, vit_feats, proj_y, proj_x):
        """
        Args:
            point_feats: (N, D) point features
            vit_feats: (H, W, D) ViT feature map
            proj_y: (N,) row indices for each point
            proj_x: (N,) col indices for each point

        Returns:
            fused_feats: (N, D) fused point features
        """
        N, D = point_feats.shape
        H, W, _ = vit_feats.shape

        # Gather 3x3 neighborhood for each point
        neighbor_feats = self.gather_neighbors(vit_feats, proj_y, proj_x)  # (N, 9, D)

        # Cross-attention
        Q = self.q_proj(point_feats).unsqueeze(1)  # (N, 1, D)
        K = self.k_proj(neighbor_feats)             # (N, 9, D)
        V = self.v_proj(neighbor_feats)             # (N, 9, D)

        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)  # (N, 1, 9)
        out = (attn @ V).squeeze(1)  # (N, D)

        # Residual connection
        fused_feats = point_feats + self.out_proj(out)

        return fused_feats

    def gather_neighbors(self, vit_feats, proj_y, proj_x):
        """Gather 3x3 window around each point's projected pixel."""
        H, W, D = vit_feats.shape
        N = proj_y.shape[0]

        # Generate 3x3 offsets
        offsets = torch.tensor([[-1,-1], [-1,0], [-1,1],
                                [0,-1],  [0,0],  [0,1],
                                [1,-1],  [1,0],  [1,1]], device=vit_feats.device)

        # Compute neighbor coordinates (N, 9)
        ny = (proj_y.unsqueeze(1) + offsets[:, 0]).clamp(0, H-1)
        nx = (proj_x.unsqueeze(1) + offsets[:, 1]).clamp(0, W-1)

        # Gather features (N, 9, D)
        neighbor_feats = vit_feats[ny, nx]

        return neighbor_feats
```

**Key design choices:**
- **Local attention window (3x3):** Each point attends to 9 nearby pixels, not entire image. Efficient and spatially relevant.
- **Residual connection:** Preserves original point features, attention adds context.
- **Single head:** Start simple. Can add multi-head later if needed.

**Estimated parameters:** ~260K per module (x2 = ~520K total)

---

## Component 3: Integration Flow

**Complete forward pass:**

```python
class RangeViTPointFusion(nn.Module):
    def __init__(self, vit_config, hidden_dim=256, num_classes=20):
        super().__init__()

        # Existing RangeViT components
        self.stem = ConvStem(...)
        self.encoder = ViTEncoder(...)
        self.decoder = Decoder(...)

        # New PointFusion components
        self.point_encoder = PointMLPEncoder(in_channels=7, hidden_dim=hidden_dim)
        self.fusion_encoder = PointToRangeCrossAttention(dim=hidden_dim)
        self.fusion_decoder = PointToRangeCrossAttention(dim=hidden_dim)

        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, range_image, point_features, proj_y, proj_x):
        """
        Args:
            range_image: (B, C, H, W) projected range image
            point_features: (N, 7) raw point features [xyz, intensity, cluster_offset]
            proj_y: (N,) row indices
            proj_x: (N,) col indices

        Returns:
            predictions: (N, num_classes) per-point logits
        """
        # Range branch (existing)
        x = self.stem(range_image)           # (B, D, H', W')
        encoder_feats = self.encoder(x)       # (B, D, H', W')
        decoder_feats = self.decoder(encoder_feats)  # (B, D, H, W)

        # Point branch (new)
        point_feats = self.point_encoder(point_features)  # (N, D)

        # Fusion 1: after encoder
        enc_feats_hwc = encoder_feats.permute(0, 2, 3, 1).squeeze(0)  # (H', W', D)
        point_feats = self.fusion_encoder(point_feats, enc_feats_hwc,
                                          proj_y // scale, proj_x // scale)

        # Fusion 2: after decoder
        dec_feats_hwc = decoder_feats.permute(0, 2, 3, 1).squeeze(0)  # (H, W, D)
        point_feats = self.fusion_decoder(point_feats, dec_feats_hwc,
                                          proj_y, proj_x)

        # Classification
        predictions = self.classifier(point_feats)  # (N, num_classes)

        return predictions
```

**Fusion flow:**
1. Point MLP encodes raw points → `point_feats`
2. After ViT encoder: `point_feats` attends to `encoder_feats` → `point_feats_v1`
3. After decoder: `point_feats_v1` attends to `decoder_feats` → `point_feats_v2`
4. Classifier predicts on `point_feats_v2`

---

## Data Flow & Required Changes

**Data loader changes:**

```python
# Current loader provides:
- range_image: (H, W, 5)        # projected range image
- point_cloud: (N, 4)           # xyz + intensity
- labels: (N,)                  # per-point labels
- proj_y, proj_x: (N,)          # point → pixel mapping

# Additional fields needed:
- cluster_offset: (N, 3)        # point_xyz - cluster_center
```

**Preprocessing (computed per sample):**

```python
def compute_cluster_features(points_xyz, voxel_size=0.1):
    """Compute cluster center offset for each point."""
    # Voxelize
    voxel_coords = (points_xyz / voxel_size).floor().long()

    # Compute center per voxel using scatter
    unique_voxels, inverse_idx = torch.unique(voxel_coords, dim=0, return_inverse=True)
    cluster_centers = scatter_mean(points_xyz, inverse_idx, dim=0)

    # Map back to points
    point_cluster_center = cluster_centers[inverse_idx]
    cluster_offset = points_xyz - point_cluster_center

    return cluster_offset
```

**Files to modify/create:**

| File | Change |
|------|--------|
| `dataset/range_view_loader.py` | Add cluster offset computation |
| `models/fusion/point_encoder.py` | **New:** PointMLPEncoder class |
| `models/fusion/cross_attention.py` | **New:** PointToRangeCrossAttention class |
| `models/fusion/pointfusion_rangevit.py` | **New:** Main model integrating components |
| `config_pointfusion_kitti.yaml` | **New:** Config for PointFusion architecture |

---

## Expected Performance

**Parameter comparison:**

| Component | Current (MinkUNet) | Proposed (PointFusion) |
|-----------|-------------------|------------------------|
| Voxel/Point branch | ~5M params | ~50K params |
| Fusion modules | ~200K params | ~520K params (attention) |
| Total added to ViT | ~5.2M | ~570K |

**Speed comparison (estimated):**

| Operation | MinkUNet | PointFusion |
|-----------|----------|-------------|
| Sparse 3D convs | ~50ms | N/A |
| Point MLPs | N/A | ~5ms |
| Cross-attention (local 3x3) | N/A | ~10ms |
| **Total branch overhead** | ~50-80ms | ~15-20ms |

**Expected improvements:**
- ~3-4x faster than MinkUNet branch
- No torchsparse dependency (pure PyTorch)
- Easier to debug and modify
- Smaller memory footprint
- 10x fewer parameters in fusion branch

**Accuracy expectations:**
- Cross-attention is expressive - should capture point↔range relationships
- Dual fusion provides multi-scale context
- If accuracy insufficient, can deepen Point MLP or add multi-head attention

---

## Future Extensions (if needed)

1. **Multi-head attention:** Add multiple attention heads for richer fusion
2. **Deeper Point MLP:** Add more layers if point features need more capacity
3. **Bidirectional attention:** Add Range→Point direction for ViT to benefit from 3D context
4. **Larger attention window:** Expand from 3x3 to 5x5 if more context helps
