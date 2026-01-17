# RangeViT-PointFusion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace MinkUNet voxel branch with lightweight Point MLP + Cross-Attention fusion for improved speed while maintaining accuracy.

**Architecture:** ViT encodes range images (unchanged). New Point MLP encoder processes raw points (xyz, intensity, cluster offset). Cross-attention modules let points query ViT features at encoder and decoder stages. Final classifier operates on fused point features.

**Tech Stack:** PyTorch, torch_scatter (for cluster center computation), existing RangeViT components

---

## Task 1: Create PointMLPEncoder Module

**Files:**
- Create: `models/fusion/point_encoder.py`

**Step 1: Create the point encoder module file**

```python
# models/fusion/point_encoder.py
"""
Point MLP Encoder for PointFusion architecture.

Encodes raw point features (xyz, intensity, cluster offset) into learned representations.
"""

import torch
import torch.nn as nn

__all__ = ['PointMLPEncoder']


class PointMLPEncoder(nn.Module):
    """
    Encode raw point features into learned representations.

    Input features (7 channels):
        - xyz (3): 3D coordinates
        - intensity (1): LiDAR reflectance
        - cluster_offset (3): relative position to voxel center

    Args:
        in_channels: Number of input channels (default: 7)
        hidden_dim: Output feature dimension (default: 256)
        if_dist: Whether to use SyncBatchNorm for distributed training
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 256,
        if_dist: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        BatchNorm = nn.SyncBatchNorm if if_dist else nn.BatchNorm1d

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            BatchNorm(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            BatchNorm(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, hidden_dim),
            BatchNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """
        Encode point features.

        Args:
            point_features: [N, in_channels] raw point features

        Returns:
            [N, hidden_dim] encoded point features
        """
        return self.mlp(point_features)
```

**Step 2: Verify file was created correctly**

Run: `python -c "from models.fusion.point_encoder import PointMLPEncoder; print('Import successful')"`
Expected: `Import successful`

**Step 3: Commit**

```bash
git add models/fusion/point_encoder.py
git commit -m "feat(fusion): add PointMLPEncoder module

Lightweight MLP encoder for raw point features (xyz, intensity, cluster offset).
Part of PointFusion architecture replacing MinkUNet."
```

---

## Task 2: Create PointToRangeCrossAttention Module

**Files:**
- Modify: `models/fusion/fusion_modules.py` (add new class at end)

**Step 1: Add cross-attention module to fusion_modules.py**

Add the following class at the end of `models/fusion/fusion_modules.py`:

```python
class PointToRangeCrossAttention(nn.Module):
    """
    Cross-attention where points query local ViT features.

    Each point attends to a local neighborhood (e.g., 3x3) in the range image
    around its projected pixel location.

    Args:
        dim: Feature dimension for Q, K, V projections
        window_size: Size of local attention window (default: 3 for 3x3)
        num_heads: Number of attention heads (default: 1)
        if_dist: Whether to use SyncBatchNorm for distributed training
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 3,
        num_heads: int = 1,
        if_dist: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        BatchNorm = nn.SyncBatchNorm if if_dist else nn.BatchNorm1d
        self.norm = BatchNorm(dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)

    def forward(
        self,
        point_feats: torch.Tensor,
        vit_feats: torch.Tensor,
        proj_y: torch.Tensor,
        proj_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attention from points to range image features.

        Args:
            point_feats: [N, D] point features (queries)
            vit_feats: [H, W, D] ViT feature map (keys/values)
            proj_y: [N] row indices for each point (0 to H-1)
            proj_x: [N] col indices for each point (0 to W-1)

        Returns:
            fused_feats: [N, D] fused point features
        """
        N, D = point_feats.shape
        H, W, _ = vit_feats.shape

        # Gather local neighborhood for each point
        neighbor_feats = self._gather_neighbors(vit_feats, proj_y, proj_x)  # [N, K, D]
        K = neighbor_feats.shape[1]  # window_size^2

        # Multi-head attention
        Q = self.q_proj(point_feats).view(N, 1, self.num_heads, self.head_dim)  # [N, 1, H, D/H]
        K_feat = self.k_proj(neighbor_feats).view(N, K, self.num_heads, self.head_dim)  # [N, K, H, D/H]
        V = self.v_proj(neighbor_feats).view(N, K, self.num_heads, self.head_dim)  # [N, K, H, D/H]

        # Transpose for attention: [N, H, 1, D/H] and [N, H, K, D/H]
        Q = Q.transpose(1, 2)  # [N, H, 1, D/H]
        K_feat = K_feat.transpose(1, 2)  # [N, H, K, D/H]
        V = V.transpose(1, 2)  # [N, H, K, D/H]

        # Attention scores
        attn = torch.matmul(Q, K_feat.transpose(-2, -1)) * self.scale  # [N, H, 1, K]
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, V)  # [N, H, 1, D/H]
        out = out.transpose(1, 2).reshape(N, D)  # [N, D]

        # Output projection + residual
        out = self.out_proj(out)
        out = point_feats + out  # Residual connection
        out = self.norm(out)

        return out

    def _gather_neighbors(
        self,
        vit_feats: torch.Tensor,
        proj_y: torch.Tensor,
        proj_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather local window around each point's projected pixel.

        Args:
            vit_feats: [H, W, D] feature map
            proj_y: [N] row indices
            proj_x: [N] col indices

        Returns:
            neighbor_feats: [N, window_size^2, D]
        """
        H, W, D = vit_feats.shape
        N = proj_y.shape[0]
        device = vit_feats.device

        # Generate window offsets
        half = self.window_size // 2
        offsets_y = torch.arange(-half, half + 1, device=device)
        offsets_x = torch.arange(-half, half + 1, device=device)
        grid_y, grid_x = torch.meshgrid(offsets_y, offsets_x, indexing='ij')
        offsets = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)  # [K, 2]
        K = offsets.shape[0]

        # Compute neighbor coordinates [N, K]
        ny = proj_y.unsqueeze(1) + offsets[:, 0].unsqueeze(0)  # [N, K]
        nx = proj_x.unsqueeze(1) + offsets[:, 1].unsqueeze(0)  # [N, K]

        # Clamp to valid range
        ny = ny.clamp(0, H - 1)
        nx = nx.clamp(0, W - 1)

        # Gather features [N, K, D]
        neighbor_feats = vit_feats[ny, nx]

        return neighbor_feats
```

**Step 2: Update __all__ in fusion_modules.py**

Change line 26 from:
```python
__all__ = ['FusionMLP', 'PointTransform']
```
to:
```python
__all__ = ['FusionMLP', 'PointTransform', 'PointToRangeCrossAttention']
```

**Step 3: Verify import works**

Run: `python -c "from models.fusion.fusion_modules import PointToRangeCrossAttention; print('Import successful')"`
Expected: `Import successful`

**Step 4: Commit**

```bash
git add models/fusion/fusion_modules.py
git commit -m "feat(fusion): add PointToRangeCrossAttention module

Cross-attention where points query local neighborhood in range features.
Supports multi-head attention and configurable window size."
```

---

## Task 3: Add Cluster Offset Computation to Data Loader

**Files:**
- Modify: `dataset/range_view_loader.py`

**Step 1: Add compute_cluster_offset function**

Add this function after the imports (around line 22, before class RangeViewLoader):

```python
def compute_cluster_offset(points_xyz: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """
    Compute cluster center offset for each point.

    Voxelizes points and computes the offset from each point to its voxel center.

    Args:
        points_xyz: [N, 3] point coordinates
        voxel_size: Voxel size for clustering (default: 0.1m = 10cm)

    Returns:
        cluster_offset: [N, 3] offset from each point to its cluster center
    """
    # Voxelize: compute voxel indices for each point
    voxel_coords = np.floor(points_xyz / voxel_size).astype(np.int32)

    # Get unique voxels and inverse mapping
    unique_voxels, inverse_idx = np.unique(voxel_coords, axis=0, return_inverse=True)

    # Compute mean position per voxel
    num_voxels = len(unique_voxels)
    voxel_sums = np.zeros((num_voxels, 3), dtype=np.float64)
    voxel_counts = np.zeros(num_voxels, dtype=np.int32)

    np.add.at(voxel_sums, inverse_idx, points_xyz)
    np.add.at(voxel_counts, inverse_idx, 1)

    voxel_centers = voxel_sums / voxel_counts[:, np.newaxis]

    # Map cluster centers back to points
    point_cluster_center = voxel_centers[inverse_idx]

    # Compute offset
    cluster_offset = points_xyz - point_cluster_center

    return cluster_offset.astype(np.float32)
```

**Step 2: Add import for numpy at top if not present**

Verify `import numpy as np` exists at top of file (it does at line 15).

**Step 3: Modify get_item_for_fusion to include cluster_offset**

In the `get_item_for_fusion` method (starting at line 189), modify the output dictionary to include cluster_offset.

Find this section (around lines 265-278):

```python
        # Prepare point data
        # Point features: x, y, z, intensity
        if self.is_train:
            # After cropping, we only have valid points
            # Get intensity from the projection for cropped points
            point_features = np.zeros((points_xyz.shape[0], 4), dtype=np.float32)
            point_features[:, :3] = points_xyz
            # Note: intensity may not be preserved after cropping, using zeros
            # A more robust solution would track the original point indices
            point_features_tensor = torch.from_numpy(point_features).float()
            point_coords_tensor = torch.from_numpy(points_xyz).float()
            point_labels_tensor = torch.from_numpy(sem_label_mapped).long()
        else:
            point_features_tensor = torch.from_numpy(pointcloud).float()  # [N, 4]
            point_coords_tensor = torch.from_numpy(points_xyz).float()  # [N, 3]
            point_labels_tensor = torch.from_numpy(sem_label_mapped).long()  # [N]
```

Replace with:

```python
        # Prepare point data
        # Point features: x, y, z, intensity
        if self.is_train:
            # After cropping, we only have valid points
            point_features = np.zeros((points_xyz.shape[0], 4), dtype=np.float32)
            point_features[:, :3] = points_xyz
            # Note: intensity may not be preserved after cropping, using zeros
            point_features_tensor = torch.from_numpy(point_features).float()
            point_coords_tensor = torch.from_numpy(points_xyz).float()
            point_labels_tensor = torch.from_numpy(sem_label_mapped).long()
        else:
            point_features_tensor = torch.from_numpy(pointcloud).float()  # [N, 4]
            point_coords_tensor = torch.from_numpy(points_xyz).float()  # [N, 3]
            point_labels_tensor = torch.from_numpy(sem_label_mapped).long()  # [N]

        # Compute cluster offset for PointFusion
        cluster_offset = compute_cluster_offset(points_xyz, voxel_size=0.1)
        cluster_offset_tensor = torch.from_numpy(cluster_offset).float()  # [N, 3]
```

**Step 4: Add cluster_offset to output dictionary**

Find the output dictionary (around lines 286-297):

```python
        output = {
            'range_image': proj_tensor[:5],  # [5, H, W]
            'range_label': proj_tensor[5],   # [H, W]
            'range_mask': proj_tensor[6],    # [H, W]
            'point_features': point_features_tensor,  # [N, 4]
            'point_coords': point_coords_tensor,      # [N, 3]
            'point_labels': point_labels_tensor,      # [N]
            'range_pxpy': range_pxpy_tensor,          # [N, 2]
            'num_points': point_features_tensor.shape[0],
            'index': index,
        }
```

Replace with:

```python
        output = {
            'range_image': proj_tensor[:5],  # [5, H, W]
            'range_label': proj_tensor[5],   # [H, W]
            'range_mask': proj_tensor[6],    # [H, W]
            'point_features': point_features_tensor,  # [N, 4]
            'point_coords': point_coords_tensor,      # [N, 3]
            'point_labels': point_labels_tensor,      # [N]
            'range_pxpy': range_pxpy_tensor,          # [N, 2]
            'cluster_offset': cluster_offset_tensor,  # [N, 3]
            'num_points': point_features_tensor.shape[0],
            'index': index,
        }
```

**Step 5: Update custom_collate_fusion_fn to include cluster_offset**

Find the collate function (around lines 438-480) and add cluster_offset handling.

Add after `range_pxpy_list = []` (around line 457):
```python
    cluster_offset_list = []
```

Add inside the for loop after `range_pxpy_list.append(d['range_pxpy'])`:
```python
        cluster_offset_list.append(d['cluster_offset'])
```

Add after `output['range_pxpy'] = ...` (around line 475):
```python
    output['cluster_offset'] = torch.cat(cluster_offset_list, dim=0)  # [N_total, 3]
```

**Step 6: Verify data loader works**

Run: `python -c "from dataset.range_view_loader import compute_cluster_offset; import numpy as np; xyz = np.random.randn(100, 3); off = compute_cluster_offset(xyz); print(f'Shape: {off.shape}')"`
Expected: `Shape: (100, 3)`

**Step 7: Commit**

```bash
git add dataset/range_view_loader.py
git commit -m "feat(data): add cluster offset computation for PointFusion

Computes relative position of each point to its voxel center.
Used as additional input feature for PointMLPEncoder."
```

---

## Task 4: Create PointFusionRangeViT Model

**Files:**
- Create: `models/fusion/pointfusion_rangevit.py`

**Step 1: Create the main model file**

```python
# models/fusion/pointfusion_rangevit.py
"""
PointFusionRangeViT: Efficient fusion model combining ViT range branch with Point MLP + Cross-Attention.

Replaces MinkUNet voxel branch with lightweight point processing for improved speed.

Architecture:
    - Range Branch: ConvStem → ViT Encoder → DecoderUpConv (existing RangeViT)
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
            nn.ReLU(inplace=True),
        )
        self.point_proj_decoder = nn.Sequential(
            nn.Linear(self.d_model, decoder_d_decoder),
            BatchNorm(decoder_d_decoder),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
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
            n_heads, n_layers, d_model = 12, 12, 768
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
        B, _, H_ori, W_ori = range_image.shape

        # Padding for ViT
        range_image_padded = padding(range_image, self.patch_size)
        H, W = range_image_padded.size(2), range_image_padded.size(3)

        # ========== RANGE BRANCH ==========
        # Range stem + encoder
        range_enc_out, _ = self.range_encoder(range_image_padded)
        range_enc = range_enc_out[:, 1:]  # Remove CLS token: [B, N_tokens, d_model]

        # Get skip connection from stem
        _, skip = self.range_encoder.patch_embed(range_image_padded)

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
        proj_y_enc, proj_x_enc = self._pxpy_to_indices(
            range_pxpy, batch_indices, H_enc, W_enc, B
        )

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
        proj_y_dec, proj_x_dec = self._pxpy_to_indices(
            range_pxpy, batch_indices, H_ori, W_ori, B
        )

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
        batch_indices: torch.Tensor,
        H: int,
        W: int,
        B: int,
    ) -> tuple:
        """
        Convert normalized pxpy [-1, 1] to pixel indices [0, H-1], [0, W-1].

        Args:
            range_pxpy: [N, 2] normalized coordinates (px, py)
            batch_indices: [N] batch indices
            H, W: Target resolution
            B: Batch size

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
```

**Step 2: Verify import works**

Run: `python -c "from models.fusion.pointfusion_rangevit import PointFusionRangeViT; print('Import successful')"`
Expected: `Import successful`

**Step 3: Commit**

```bash
git add models/fusion/pointfusion_rangevit.py
git commit -m "feat(fusion): add PointFusionRangeViT model

Main fusion model combining ViT range branch with Point MLP + Cross-Attention.
Replaces MinkUNet for faster inference while maintaining accuracy."
```

---

## Task 5: Update fusion __init__.py

**Files:**
- Modify: `models/fusion/__init__.py`

**Step 1: Read current __init__.py**

Check contents of `models/fusion/__init__.py`.

**Step 2: Add exports for new modules**

Update the file to include new exports:

```python
from .fusion_rangevit import FusionRangeViT
from .fusion_modules import FusionMLP, PointTransform, PointToRangeCrossAttention
from .point_encoder import PointMLPEncoder
from .pointfusion_rangevit import PointFusionRangeViT

__all__ = [
    'FusionRangeViT',
    'FusionMLP',
    'PointTransform',
    'PointToRangeCrossAttention',
    'PointMLPEncoder',
    'PointFusionRangeViT',
]
```

**Step 3: Verify imports work**

Run: `python -c "from models.fusion import PointFusionRangeViT, PointMLPEncoder, PointToRangeCrossAttention; print('All imports successful')"`
Expected: `All imports successful`

**Step 4: Commit**

```bash
git add models/fusion/__init__.py
git commit -m "feat(fusion): export PointFusion modules from package"
```

---

## Task 6: Create Configuration File

**Files:**
- Create: `config_pointfusion_kitti.yaml`

**Step 1: Create the config file**

```yaml
# config_pointfusion_kitti.yaml
# Configuration for PointFusionRangeViT on SemanticKITTI

# General config
num_workers: 4
id: "exp_pointfusion_kitti"

# MLflow config
mlflow:
  enable: true
  tracking_uri: "http://140.245.117.232:5000"
  experiment_name: "rangevit_pointfusion"
  run_name: null
  nested: false
  log_checkpoints: false
  log_code_snapshot: false

# Data config
dataset: "SemanticKitti"
n_classes: 20  # 19 + 1(ignored)
use_trainval: false

# Train config
has_label: true
val_frequency: 1
n_epochs: 60
warmup_epochs: 10
batch_size: 4
batch_size_val: 1
lr: 0.0004
train_result_frequency: 100

# Enable PointFusion model
use_pointfusion: true
use_fusion_voxel: true  # Use fusion data loader (provides point data)

# Voxel features disabled
voxel_features:
  enable: false

# Point branch config (for PointFusion)
point_branch:
  in_channels: 7  # xyz (3) + intensity (1) + cluster_offset (3)
  hidden_dim: 256
  cluster_voxel_size: 0.1  # 10cm voxels for cluster computation

# Cross-attention config
cross_attention:
  window_size: 3  # 3x3 local attention
  num_heads: 4

# Model config (range branch)
vit_backbone: "vit_base_patch16_384"
in_channels: 5  # Range branch: range, x, y, z, intensity
patch_size: [2, 8]
patch_stride: [2, 8]
image_size: [64, 768]
window_size: [64, 768]
window_stride: [64, 256]
original_image_size: [64, 2048]

# Stem
conv_stem: "ConvStem"
stem_base_channels: 32
D_h: 128

# Decoder
decoder: "up_conv"
skip_filters: 128

# 3D refiner disabled
use_kpconv: false

# Checkpoint model
checkpoint: null
pretrained_model: null

# Pretrained paths
range_pretrained_model: null  # Path to pretrained RangeViT checkpoint

# Loading pre-trained patch and positional embeddings
reuse_pos_emb: false
reuse_patch_emb: false

# Dropout
dropout_p: 0.3

# Data augmentation config
augmentation:
  # flip
  p_flipx: 0.
  p_flipy: 0.5

  # translation
  p_transx: 0.5
  trans_xmin: -5
  trans_xmax: 5
  p_transy: 0.5
  trans_ymin: -3
  trans_ymax: 3
  p_transz: 0.5
  trans_zmin: -1
  trans_zmax: 0.

  # rotation
  p_rot_roll: 0.5
  rot_rollmin: -5
  rot_rollmax: 5
  p_rot_pitch: 0.5
  rot_pitchmin: -5
  rot_pitchmax: 5
  p_rot_yaw: 0.5
  rot_yawmin: 5
  rot_yawmax: -5

sensor:
  name: "HDL64"
  type: "spherical"
  scan_proj: true
  proj_h: 64
  proj_w: 2048
  fov_up: 3.
  fov_down: -25.
  fov_left: -180
  fov_right: 180
  img_mean:
    - 12.12  # range
    - 10.88  # x
    - 0.23   # y
    - -1.04  # z
    - 0.21   # intensity
  img_stds:
    - 12.32  # range
    - 11.47  # x
    - 6.91   # y
    - 0.86   # z
    - 0.16   # intensity
```

**Step 2: Verify config loads**

Run: `python -c "import yaml; cfg = yaml.safe_load(open('config_pointfusion_kitti.yaml')); print(f'Config loaded: {cfg[\"id\"]}')"`
Expected: `Config loaded: exp_pointfusion_kitti`

**Step 3: Commit**

```bash
git add config_pointfusion_kitti.yaml
git commit -m "feat(config): add PointFusion configuration for SemanticKITTI"
```

---

## Task 7: Update main.py to Support PointFusion Model

**Files:**
- Modify: `main.py`

**Step 1: Add import for PointFusionRangeViT**

Find the imports section and add:

```python
from models.fusion.pointfusion_rangevit import PointFusionRangeViT
```

**Step 2: Add model building logic**

Find the model building section (search for `use_fusion_voxel` or model instantiation).
Add a branch for `use_pointfusion`:

```python
    # Check for PointFusion model
    if config.get('use_pointfusion', False):
        point_config = config.get('point_branch', {})
        cross_attn_config = config.get('cross_attention', {})

        model = PointFusionRangeViT(
            range_in_channels=config['in_channels'],
            n_cls=config['n_classes'],
            vit_backbone=config['vit_backbone'],
            image_size=config['original_image_size'],
            range_pretrained_path=config.get('range_pretrained_model'),
            patch_size=config['patch_size'],
            patch_stride=config['patch_stride'],
            conv_stem=config['conv_stem'],
            stem_base_channels=config['stem_base_channels'],
            stem_hidden_dim=config['D_h'],
            decoder=config['decoder'],
            decoder_d_decoder=config['D_h'],
            skip_filters=config['skip_filters'],
            point_in_channels=point_config.get('in_channels', 7),
            point_hidden_dim=point_config.get('hidden_dim', 256),
            cross_attn_window=cross_attn_config.get('window_size', 3),
            cross_attn_heads=cross_attn_config.get('num_heads', 4),
            if_dist=distributed,
            dropout_p=config.get('dropout_p', 0.3),
        )
```

**Step 3: Verify main.py syntax is valid**

Run: `python -m py_compile main.py && echo "Syntax OK"`
Expected: `Syntax OK`

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat(main): add support for PointFusionRangeViT model"
```

---

## Task 8: Update train.py for PointFusion Forward Pass

**Files:**
- Modify: `train.py`

**Step 1: Find the forward pass for fusion model**

Look for the section handling `use_fusion_voxel` data and model forward.

**Step 2: Add PointFusion forward pass handling**

The forward pass needs to pass `cluster_offset` to the model. Find the fusion forward pass and add:

```python
        # PointFusion model forward
        if self.config.get('use_pointfusion', False):
            outputs = self.model(
                range_image=batch['range_image'].cuda(),
                point_features=batch['point_features'].cuda(),
                cluster_offset=batch['cluster_offset'].cuda(),
                batch_indices=batch['batch_indices'].cuda(),
                range_pxpy=batch['range_pxpy'].cuda(),
            )
```

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat(train): add PointFusion forward pass handling"
```

---

## Task 9: Integration Test

**Files:**
- Create: `tests/test_pointfusion.py`

**Step 1: Create test file**

```python
# tests/test_pointfusion.py
"""Integration tests for PointFusion model."""

import torch
import pytest


def test_point_mlp_encoder():
    """Test PointMLPEncoder forward pass."""
    from models.fusion.point_encoder import PointMLPEncoder

    encoder = PointMLPEncoder(in_channels=7, hidden_dim=256, if_dist=False)
    x = torch.randn(1000, 7)  # 1000 points, 7 features
    out = encoder(x)

    assert out.shape == (1000, 256), f"Expected (1000, 256), got {out.shape}"
    print("PointMLPEncoder test passed!")


def test_cross_attention():
    """Test PointToRangeCrossAttention forward pass."""
    from models.fusion.fusion_modules import PointToRangeCrossAttention

    attn = PointToRangeCrossAttention(dim=256, window_size=3, num_heads=4, if_dist=False)

    point_feats = torch.randn(100, 256)  # 100 points
    vit_feats = torch.randn(64, 768, 256)  # Range image features
    proj_y = torch.randint(0, 64, (100,))
    proj_x = torch.randint(0, 768, (100,))

    out = attn(point_feats, vit_feats, proj_y, proj_x)

    assert out.shape == (100, 256), f"Expected (100, 256), got {out.shape}"
    print("CrossAttention test passed!")


def test_pointfusion_model():
    """Test PointFusionRangeViT forward pass."""
    from models.fusion.pointfusion_rangevit import PointFusionRangeViT

    model = PointFusionRangeViT(
        range_in_channels=5,
        n_cls=20,
        vit_backbone='vit_small_patch16_384',
        image_size=(64, 768),
        patch_size=(2, 8),
        patch_stride=(2, 8),
        point_in_channels=7,
        point_hidden_dim=256,
        cross_attn_window=3,
        cross_attn_heads=4,
        if_dist=False,
    )

    # Simulate batch
    B = 2
    H, W = 64, 768
    N_per_batch = 5000

    range_image = torch.randn(B, 5, H, W)
    point_features = torch.randn(B * N_per_batch, 4)
    cluster_offset = torch.randn(B * N_per_batch, 3)
    batch_indices = torch.cat([
        torch.zeros(N_per_batch, dtype=torch.long),
        torch.ones(N_per_batch, dtype=torch.long),
    ])
    range_pxpy = torch.rand(B * N_per_batch, 2) * 2 - 1  # [-1, 1]

    with torch.no_grad():
        out = model(range_image, point_features, cluster_offset, batch_indices, range_pxpy)

    assert out.shape == (B * N_per_batch, 20), f"Expected ({B * N_per_batch}, 20), got {out.shape}"
    print("PointFusionRangeViT test passed!")

    # Print parameter counts
    params = model.count_parameters()
    print("\nParameter counts:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")


if __name__ == '__main__':
    test_point_mlp_encoder()
    test_cross_attention()
    test_pointfusion_model()
    print("\nAll tests passed!")
```

**Step 2: Run integration tests**

Run: `python tests/test_pointfusion.py`
Expected: `All tests passed!`

**Step 3: Commit**

```bash
git add tests/test_pointfusion.py
git commit -m "test: add integration tests for PointFusion model"
```

---

## Summary

After completing all tasks, you will have:

1. **PointMLPEncoder** (`models/fusion/point_encoder.py`) - Lightweight MLP for point features
2. **PointToRangeCrossAttention** (`models/fusion/fusion_modules.py`) - Cross-attention module
3. **Updated data loader** (`dataset/range_view_loader.py`) - Cluster offset computation
4. **PointFusionRangeViT** (`models/fusion/pointfusion_rangevit.py`) - Main fusion model
5. **Configuration** (`config_pointfusion_kitti.yaml`) - Training config
6. **Updated main.py and train.py** - Model building and forward pass
7. **Integration tests** (`tests/test_pointfusion.py`) - Verification

**To train:**
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63545 \
    --use_env main.py 'config_pointfusion_kitti.yaml' \
    --data_root '<path_to_semantic_kitti_dataset>/dataset/sequences/' \
    --save_path '<path_to_log>'
```
