# RangeViT-Fusion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement bidirectional point-pixel fusion in RangeViT at ViT blocks 4, 8, and 12, replacing KPConv with continuous 3D information flow.

**Architecture:** Vision Transformer backbone with parallel point branch. At fusion points, reshape tokens to 2D grid, exchange features bidirectionally with point branch via pixel2point/point2cluster/cluster2pixel mappings, then flatten back. Final prediction via MLP head on concatenated pixel+point features.

**Tech Stack:** PyTorch, torch_scatter (for point aggregation), timm (ViT weights), existing RangeViT codebase patterns.

---

## Prerequisites

Before starting, ensure `torch_scatter` is installed:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## Task 1: Create Test Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_fusion_modules.py`

**Step 1: Create tests directory structure**

```bash
mkdir -p tests
```

**Step 2: Create tests/__init__.py**

```python
# tests/__init__.py
```

**Step 3: Create initial test file with imports**

```python
# tests/test_fusion_modules.py
import pytest
import torch
import torch.nn as nn

# Test device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_placeholder():
    """Placeholder test to verify test infrastructure works."""
    assert True
```

**Step 4: Run test to verify infrastructure**

Run: `python -m pytest tests/test_fusion_modules.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/
git commit -m "test: add test infrastructure for fusion modules"
```

---

## Task 2: Implement FeaturesEncoder

**Files:**
- Create: `models/features_encoder.py`
- Modify: `tests/test_fusion_modules.py`

**Step 1: Write the failing test**

Add to `tests/test_fusion_modules.py`:

```python
def test_features_encoder_output_shape():
    """Test FeaturesEncoder produces correct output shape."""
    from models.features_encoder import FeaturesEncoder

    batch_size = 2
    n_points = 1000
    in_channels = 5  # xyz + intensity + range
    d_model = 384

    encoder = FeaturesEncoder(in_channels=in_channels, d_model=d_model).to(DEVICE)

    # Input: raw point features (B*N, C)
    point_attrs = torch.randn(batch_size * n_points, in_channels, device=DEVICE)

    output = encoder(point_attrs)

    assert output.shape == (batch_size * n_points, d_model)


def test_features_encoder_gradient_flow():
    """Test gradients flow through FeaturesEncoder."""
    from models.features_encoder import FeaturesEncoder

    encoder = FeaturesEncoder(in_channels=5, d_model=384).to(DEVICE)
    point_attrs = torch.randn(100, 5, device=DEVICE, requires_grad=True)

    output = encoder(point_attrs)
    loss = output.sum()
    loss.backward()

    assert point_attrs.grad is not None
    assert not torch.isnan(point_attrs.grad).any()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion_modules.py::test_features_encoder_output_shape -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

Create `models/features_encoder.py`:

```python
# Copyright 2026 - RangeViT-Fusion
# Adapted from HARP-NeXt FeaturesEncoder

import torch
import torch.nn as nn


class FeaturesEncoder(nn.Module):
    """
    Encodes raw point attributes (xyz, intensity, range) into feature vectors.

    Architecture: Linear(in, 64) -> BN -> ReLU -> Linear(64, 128) -> BN -> ReLU -> Linear(128, d_model) -> BN -> ReLU

    Args:
        in_channels: Number of input channels per point (default: 5 for xyz + intensity + range)
        d_model: Output feature dimension (should match ViT d_model)
    """

    def __init__(self, in_channels: int = 5, d_model: int = 384):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_model),
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

    def forward(self, point_attrs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_attrs: (N, in_channels) raw point attributes

        Returns:
            point_feats: (N, d_model) encoded point features
        """
        return self.mlp(point_attrs)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fusion_modules.py::test_features_encoder_output_shape tests/test_fusion_modules.py::test_features_encoder_gradient_flow -v`
Expected: PASS

**Step 5: Commit**

```bash
git add models/features_encoder.py tests/test_fusion_modules.py
git commit -m "feat: add FeaturesEncoder for point feature initialization"
```

---

## Task 3: Implement EfficientTransformationPipeline

**Files:**
- Create: `models/fusion_modules.py`
- Modify: `tests/test_fusion_modules.py`

**Step 1: Write the failing tests**

Add to `tests/test_fusion_modules.py`:

```python
def test_pixel2point_mapping():
    """Test pixel2point extracts correct pixel features for each point."""
    from models.fusion_modules import EfficientTransformationPipeline

    batch_size = 2
    n_points = 100
    d_model = 384
    H, W = 16, 48  # Patch grid size

    etp = EfficientTransformationPipeline(H, W)

    # Pixel features: (B, D, H, W)
    pixel_feats = torch.randn(batch_size, d_model, H, W, device=DEVICE)

    # Point coordinates: (N, 3) -> [batch_idx, y, x]
    coords = torch.zeros(batch_size * n_points, 3, dtype=torch.long, device=DEVICE)
    coords[:, 0] = torch.arange(batch_size).repeat_interleave(n_points)  # batch indices
    coords[:, 1] = torch.randint(0, H, (batch_size * n_points,))  # y coords
    coords[:, 2] = torch.randint(0, W, (batch_size * n_points,))  # x coords

    point_feats = etp.pixel2point(pixel_feats, coords)

    assert point_feats.shape == (batch_size * n_points, d_model)


def test_point2cluster_aggregation():
    """Test point2cluster aggregates points to voxels correctly."""
    from models.fusion_modules import EfficientTransformationPipeline

    H, W = 16, 48
    etp = EfficientTransformationPipeline(H, W)

    n_points = 100
    d_model = 384

    # Point features: (N, D)
    point_feats = torch.randn(n_points, d_model, device=DEVICE)

    # Point coordinates: (N, 3) -> [batch_idx, y, x]
    coords = torch.zeros(n_points, 3, dtype=torch.long, device=DEVICE)
    coords[:, 0] = 0  # all same batch
    coords[:, 1] = torch.randint(0, H, (n_points,))
    coords[:, 2] = torch.randint(0, W, (n_points,))

    voxel_coords, cluster_feats = etp.point2cluster(point_feats, coords)

    # Should have fewer or equal clusters than points
    assert cluster_feats.shape[0] <= n_points
    assert cluster_feats.shape[1] == d_model
    assert voxel_coords.shape[1] == 3


def test_cluster2pixel_dense_conversion():
    """Test cluster2pixel converts sparse clusters to dense pixel grid."""
    from models.fusion_modules import EfficientTransformationPipeline

    batch_size = 2
    H, W = 16, 48
    d_model = 384

    etp = EfficientTransformationPipeline(H, W)

    # Create some cluster features
    n_clusters = 50
    cluster_feats = torch.randn(n_clusters, d_model, device=DEVICE)

    # Cluster coordinates: (N, 3) -> [batch_idx, y, x]
    coords = torch.zeros(n_clusters, 3, dtype=torch.long, device=DEVICE)
    coords[:n_clusters//2, 0] = 0
    coords[n_clusters//2:, 0] = 1
    coords[:, 1] = torch.randint(0, H, (n_clusters,))
    coords[:, 2] = torch.randint(0, W, (n_clusters,))

    pixel_feats = etp.cluster2pixel(cluster_feats, coords, batch_size)

    assert pixel_feats.shape == (batch_size, d_model, H, W)


def test_roundtrip_pixel_point_pixel():
    """Test pixel -> point -> pixel roundtrip preserves information structure."""
    from models.fusion_modules import EfficientTransformationPipeline

    batch_size = 1
    H, W = 8, 16
    d_model = 64
    n_points = 50

    etp = EfficientTransformationPipeline(H, W)

    # Create pixel features
    pixel_feats = torch.randn(batch_size, d_model, H, W, device=DEVICE)

    # Create point coords (one point per pixel for simplicity)
    coords = torch.zeros(H * W, 3, dtype=torch.long, device=DEVICE)
    coords[:, 0] = 0
    coords[:, 1] = torch.arange(H).repeat_interleave(W)
    coords[:, 2] = torch.arange(W).repeat(H)

    # pixel -> point
    point_feats = etp.pixel2point(pixel_feats, coords)

    # point -> cluster -> pixel
    voxel_coords, cluster_feats = etp.point2cluster(point_feats, coords)
    pixel_feats_reconstructed = etp.cluster2pixel(cluster_feats, voxel_coords, batch_size)

    # Should recover original (with one point per pixel, max pooling = identity)
    assert torch.allclose(pixel_feats, pixel_feats_reconstructed, atol=1e-5)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fusion_modules.py::test_pixel2point_mapping -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Create `models/fusion_modules.py`:

```python
# Copyright 2026 - RangeViT-Fusion
# Adapted from HARP-NeXt EfficientTransformationPipeline

import torch
import torch.nn as nn

try:
    import torch_scatter
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False
    print("Warning: torch_scatter not installed. Using fallback implementation.")


class EfficientTransformationPipeline:
    """
    Handles bidirectional mappings between pixel (2D) and point (3D) features.

    - pixel2point: Look up pixel features for each point
    - point2cluster: Aggregate point features into voxel clusters (max pooling)
    - cluster2pixel: Convert sparse voxel features to dense 2D grid

    Args:
        ny: Height of pixel grid (number of patches in y)
        nx: Width of pixel grid (number of patches in x)
    """

    def __init__(self, ny: int, nx: int):
        self.ny = ny
        self.nx = nx

    def pixel2point(self, pixel_feats: torch.Tensor, coords: torch.Tensor, stride: int = 1) -> torch.Tensor:
        """
        Map pixel features to individual points.

        Args:
            pixel_feats: (B, D, H, W) dense pixel feature grid
            coords: (N, 3) point coordinates [batch_idx, y, x]
            stride: coordinate stride (for multi-scale)

        Returns:
            point_feats: (N, D) feature for each point
        """
        batch_indices = coords[:, 0]
        y_indices = coords[:, 1] // stride
        x_indices = coords[:, 2] // stride

        # Clamp to valid range
        y_indices = y_indices.clamp(0, pixel_feats.shape[2] - 1)
        x_indices = x_indices.clamp(0, pixel_feats.shape[3] - 1)

        # Index: (N, D)
        point_feats = pixel_feats[batch_indices, :, y_indices, x_indices]

        return point_feats.contiguous()

    def point2cluster(self, point_feats: torch.Tensor, coords: torch.Tensor, stride: int = 1) -> tuple:
        """
        Aggregate point features into voxel clusters using max pooling.

        Args:
            point_feats: (N, D) per-point features
            coords: (N, 3) point coordinates [batch_idx, y, x]
            stride: coordinate stride

        Returns:
            voxel_coords: (M, 3) unique voxel coordinates
            cluster_feats: (M, D) aggregated features per voxel
        """
        # Apply stride to coordinates
        strided_coords = coords.clone()
        strided_coords[:, 1] = coords[:, 1] // stride
        strided_coords[:, 2] = coords[:, 2] // stride

        # Find unique voxels
        voxel_coords, inverse_map = torch.unique(strided_coords, return_inverse=True, dim=0, sorted=True)

        if HAS_TORCH_SCATTER:
            # Use torch_scatter for efficient aggregation
            cluster_feats = torch_scatter.scatter_max(point_feats, inverse_map, dim=0)[0]
        else:
            # Fallback: slower but works without torch_scatter
            n_voxels = voxel_coords.shape[0]
            d_model = point_feats.shape[1]
            cluster_feats = torch.full((n_voxels, d_model), float('-inf'),
                                       device=point_feats.device, dtype=point_feats.dtype)

            for i in range(point_feats.shape[0]):
                voxel_idx = inverse_map[i]
                cluster_feats[voxel_idx] = torch.max(cluster_feats[voxel_idx], point_feats[i])

        return voxel_coords, cluster_feats

    def cluster2pixel(self, cluster_feats: torch.Tensor, coords: torch.Tensor,
                      batch_size: int, stride: int = 1) -> torch.Tensor:
        """
        Convert sparse voxel features to dense 2D pixel grid.

        Args:
            cluster_feats: (M, D) features per voxel
            coords: (M, 3) voxel coordinates [batch_idx, y, x]
            batch_size: number of samples in batch
            stride: coordinate stride

        Returns:
            pixel_feats: (B, D, H, W) dense pixel feature grid
        """
        ny = self.ny // stride
        nx = self.nx // stride
        d_model = cluster_feats.shape[1]

        # Initialize output tensor
        pixel_feats = torch.zeros(batch_size, d_model, ny, nx,
                                  device=cluster_feats.device, dtype=cluster_feats.dtype)

        # Scatter cluster features into pixel grid
        batch_indices = coords[:, 0].long()
        y_indices = coords[:, 1].long().clamp(0, ny - 1)
        x_indices = coords[:, 2].long().clamp(0, nx - 1)

        # Use index assignment (works for non-overlapping voxels)
        pixel_feats[batch_indices, :, y_indices, x_indices] = cluster_feats

        return pixel_feats


class PointFusionLayer(nn.Module):
    """
    Fuses mapped pixel features with point features.

    Architecture: Linear(2D, D) -> BN -> ReLU
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, mapped_pixel_feats: torch.Tensor, point_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mapped_pixel_feats: (N, D) pixel features mapped to points
            point_feats: (N, D) current point features

        Returns:
            fused_feats: (N, D) updated point features
        """
        fused = torch.cat([mapped_pixel_feats, point_feats], dim=1)
        return self.fusion(fused)


class PixelFusionLayer(nn.Module):
    """
    Fuses mapped point features with pixel features.

    Architecture: Conv2d(2D, D, 1x1) -> BN -> Hardswish
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.Hardswish(inplace=True),
        )

    def forward(self, pixel_from_points: torch.Tensor, pixel_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_from_points: (B, D, H, W) point features aggregated to pixels
            pixel_feats: (B, D, H, W) current pixel features

        Returns:
            fused_feats: (B, D, H, W) updated pixel features
        """
        fused = torch.cat([pixel_from_points, pixel_feats], dim=1)
        return self.fusion(fused)


class AuxiliaryHead(nn.Module):
    """
    Lightweight auxiliary head for pixel-level supervision during training.

    Architecture: Conv2d(D, n_classes, 1x1)
    """

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.head = nn.Conv2d(d_model, n_classes, kernel_size=1)

    def forward(self, pixel_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_feats: (B, D, H, W) pixel features

        Returns:
            logits: (B, n_classes, H, W) class logits
        """
        return self.head(pixel_feats)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fusion_modules.py -v -k "pixel2point or point2cluster or cluster2pixel or roundtrip"`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add models/fusion_modules.py tests/test_fusion_modules.py
git commit -m "feat: add EfficientTransformationPipeline and fusion layers"
```

---

## Task 4: Implement FusionHead (Prediction Head)

**Files:**
- Create: `models/fusion_head.py`
- Modify: `tests/test_fusion_modules.py`

**Step 1: Write the failing tests**

Add to `tests/test_fusion_modules.py`:

```python
def test_fusion_head_output_shape():
    """Test FusionHead produces correct per-point logits."""
    from models.fusion_head import FusionHead

    batch_size = 2
    n_points = 1000
    d_model = 384
    n_classes = 17

    head = FusionHead(d_model=d_model, n_classes=n_classes).to(DEVICE)

    # Inputs
    mapped_pixel_feats = torch.randn(batch_size * n_points, d_model, device=DEVICE)
    point_feats = torch.randn(batch_size * n_points, d_model, device=DEVICE)

    logits = head(mapped_pixel_feats, point_feats)

    assert logits.shape == (batch_size * n_points, n_classes)


def test_fusion_head_gradient_flow():
    """Test gradients flow through FusionHead."""
    from models.fusion_head import FusionHead

    head = FusionHead(d_model=384, n_classes=17).to(DEVICE)

    mapped_pixel_feats = torch.randn(100, 384, device=DEVICE, requires_grad=True)
    point_feats = torch.randn(100, 384, device=DEVICE, requires_grad=True)

    logits = head(mapped_pixel_feats, point_feats)
    loss = logits.sum()
    loss.backward()

    assert mapped_pixel_feats.grad is not None
    assert point_feats.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion_modules.py::test_fusion_head_output_shape -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Create `models/fusion_head.py`:

```python
# Copyright 2026 - RangeViT-Fusion
# HARP-NeXt style prediction head

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """
    Prediction head that combines pixel and point features for per-point classification.

    Architecture:
        concat(mapped_pixel, point) -> Linear(2D, D) -> BN -> ReLU
                                    -> Linear(D, D//2) -> BN -> ReLU
                                    -> Linear(D//2, n_classes)

    Args:
        d_model: Feature dimension
        n_classes: Number of semantic classes
    """

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, n_classes),
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

    def forward(self, mapped_pixel_feats: torch.Tensor, point_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mapped_pixel_feats: (N, D) pixel features mapped to points
            point_feats: (N, D) final point features

        Returns:
            logits: (N, n_classes) per-point class logits
        """
        combined = torch.cat([mapped_pixel_feats, point_feats], dim=1)
        return self.mlp(combined)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fusion_modules.py -v -k "fusion_head"`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add models/fusion_head.py tests/test_fusion_modules.py
git commit -m "feat: add FusionHead for per-point prediction"
```

---

## Task 5: Implement VisionTransformerFusion (Modified ViT with Fusion)

**Files:**
- Create: `models/vit_fusion.py`
- Modify: `tests/test_fusion_modules.py`

**Step 1: Write the failing tests**

Add to `tests/test_fusion_modules.py`:

```python
def test_vit_fusion_forward_shape():
    """Test VisionTransformerFusion produces correct output shapes."""
    from models.vit_fusion import VisionTransformerFusion

    batch_size = 2
    n_points = 1000
    d_model = 384
    H, W = 32, 384
    in_channels = 5

    # Patch size determines grid size
    patch_size = (2, 8)
    grid_h = H // patch_size[0]  # 16
    grid_w = W // patch_size[1]  # 48

    model = VisionTransformerFusion(
        image_size=(H, W),
        patch_size=patch_size,
        n_layers=12,
        d_model=d_model,
        d_ff=d_model * 4,
        n_heads=6,
        n_cls=17,
        channels=in_channels,
        fusion_blocks=[4, 8, 12],
    ).to(DEVICE)

    # Inputs
    images = torch.randn(batch_size, in_channels, H, W, device=DEVICE)
    point_feats = torch.randn(batch_size * n_points, d_model, device=DEVICE)

    # Point coords in pixel space (before patching)
    coords = torch.zeros(batch_size * n_points, 3, dtype=torch.long, device=DEVICE)
    coords[:, 0] = torch.arange(batch_size, device=DEVICE).repeat_interleave(n_points)
    coords[:, 1] = torch.randint(0, H, (batch_size * n_points,), device=DEVICE)
    coords[:, 2] = torch.randint(0, W, (batch_size * n_points,), device=DEVICE)

    pixel_feats, updated_point_feats, aux_outputs = model(images, point_feats, coords)

    # Check output shapes
    assert pixel_feats.shape == (batch_size, d_model, grid_h, grid_w)
    assert updated_point_feats.shape == (batch_size * n_points, d_model)
    assert len(aux_outputs) == 3  # One per fusion block


def test_vit_fusion_without_points():
    """Test VisionTransformerFusion works without point features (inference mode)."""
    from models.vit_fusion import VisionTransformerFusion

    batch_size = 2
    d_model = 384
    H, W = 32, 384
    patch_size = (2, 8)

    model = VisionTransformerFusion(
        image_size=(H, W),
        patch_size=patch_size,
        n_layers=12,
        d_model=d_model,
        d_ff=d_model * 4,
        n_heads=6,
        n_cls=17,
        channels=5,
        fusion_blocks=[4, 8, 12],
    ).to(DEVICE)

    images = torch.randn(batch_size, 5, H, W, device=DEVICE)

    # Forward without point features
    pixel_feats, _, _ = model(images, point_feats=None, coords=None)

    grid_h = H // patch_size[0]
    grid_w = W // patch_size[1]
    assert pixel_feats.shape == (batch_size, d_model, grid_h, grid_w)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion_modules.py::test_vit_fusion_forward_shape -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Create `models/vit_fusion.py`:

```python
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

    At each fusion block:
    1. Reshape tokens to 2D grid
    2. Exchange features with point branch (pixel2point, point2cluster, cluster2pixel)
    3. Flatten back to tokens and continue

    Args:
        image_size: (H, W) input image size
        patch_size: (PH, PW) patch size
        n_layers: Number of transformer blocks
        d_model: Feature dimension
        d_ff: Feedforward dimension
        n_heads: Number of attention heads
        n_cls: Number of classes (for auxiliary heads)
        dropout: Dropout rate
        drop_path_rate: Stochastic depth rate
        channels: Input channels
        fusion_blocks: List of block indices after which to perform fusion (1-indexed)
        conv_stem: 'none' or 'ConvStem'
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

        self.conv_stem = conv_stem
        self.fusion_blocks = fusion_blocks
        self.d_model = d_model
        self.n_cls = n_cls

        # Patch embedding / ConvStem
        if patch_stride is None:
            patch_stride = patch_size

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
        self.PS_H, self.PS_W = patch_size
        self.patch_stride = patch_stride
        self.n_layers = n_layers

        # Grid size for fusion operations
        if isinstance(patch_stride, (list, tuple)):
            self.grid_h = image_size[0] // patch_stride[0]
            self.grid_w = image_size[1] // patch_stride[1]
        else:
            self.grid_h = image_size[0] // patch_stride
            self.grid_w = image_size[1] // patch_stride

        # Transformation pipeline for fusion
        self.etp = EfficientTransformationPipeline(self.grid_h, self.grid_w)

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model))

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

        # Auxiliary heads for pixel supervision (training only)
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

    def _reshape_tokens_to_2d(self, tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Reshape tokens (B, N, D) to spatial grid (B, D, H, W)."""
        B, N, D = tokens.shape
        return tokens.transpose(1, 2).reshape(B, D, grid_h, grid_w)

    def _reshape_2d_to_tokens(self, pixel_feats: torch.Tensor) -> torch.Tensor:
        """Reshape spatial grid (B, D, H, W) to tokens (B, N, D)."""
        B, D, H, W = pixel_feats.shape
        return pixel_feats.flatten(2).transpose(1, 2)

    def _convert_coords_to_patch_space(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert pixel-space coordinates to patch-space coordinates."""
        patch_coords = coords.clone()
        if isinstance(self.patch_stride, (list, tuple)):
            patch_coords[:, 1] = coords[:, 1] // self.patch_stride[0]
            patch_coords[:, 2] = coords[:, 2] // self.patch_stride[1]
        else:
            patch_coords[:, 1] = coords[:, 1] // self.patch_stride
            patch_coords[:, 2] = coords[:, 2] // self.patch_stride
        return patch_coords

    def forward(self, im: torch.Tensor, point_feats: torch.Tensor = None,
                coords: torch.Tensor = None) -> tuple:
        """
        Forward pass with optional point fusion.

        Args:
            im: (B, C, H, W) input range image
            point_feats: (N_total, D) point features (optional)
            coords: (N_total, 3) point coordinates [batch_idx, y, x] in pixel space (optional)

        Returns:
            pixel_feats: (B, D, grid_H, grid_W) final pixel features
            point_feats: (N_total, D) updated point features (or None if not provided)
            aux_outputs: List of (B, n_cls, grid_H, grid_W) auxiliary pixel logits
        """
        B, _, H, W = im.shape

        # Patch embedding
        x, skip = self.patch_embed(im)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
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

        # Convert coords to patch space if provided
        patch_coords = None
        if coords is not None:
            patch_coords = self._convert_coords_to_patch_space(coords)

        # Track auxiliary outputs
        aux_outputs = []
        fusion_idx = 0

        # Process transformer blocks with fusion
        for block_idx, blk in enumerate(self.blocks):
            x = blk(x)

            # Check if this is a fusion point (block indices are 1-indexed in config)
            block_num = block_idx + 1
            if block_num in self.fusion_blocks:
                # Remove CLS token for spatial operations
                cls_token = x[:, :1]
                tokens = x[:, 1:]

                # Reshape to 2D
                pixel_feats = self._reshape_tokens_to_2d(tokens, self.grid_h, self.grid_w)

                if point_feats is not None and patch_coords is not None:
                    # Pixel -> Point fusion
                    mapped_pixel = self.etp.pixel2point(pixel_feats, patch_coords)
                    point_feats = self.point_fusion_layers[fusion_idx](mapped_pixel, point_feats)

                    # Point -> Pixel fusion
                    voxel_coords, cluster_feats = self.etp.point2cluster(point_feats, patch_coords)
                    pixel_from_points = self.etp.cluster2pixel(cluster_feats, voxel_coords, B)
                    pixel_feats = self.pixel_fusion_layers[fusion_idx](pixel_from_points, pixel_feats)

                # Auxiliary head output
                aux_logits = self.aux_heads[fusion_idx](pixel_feats)
                aux_outputs.append(aux_logits)

                # Reshape back to tokens
                tokens = self._reshape_2d_to_tokens(pixel_feats)
                x = torch.cat([cls_token, tokens], dim=1)

                fusion_idx += 1

        # Final normalization
        x = self.norm(x)

        # Remove CLS token and reshape to 2D
        tokens = x[:, 1:]
        pixel_feats = self._reshape_tokens_to_2d(tokens, self.grid_h, self.grid_w)

        return pixel_feats, point_feats, aux_outputs
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fusion_modules.py -v -k "vit_fusion"`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add models/vit_fusion.py tests/test_fusion_modules.py
git commit -m "feat: add VisionTransformerFusion with bidirectional fusion"
```

---

## Task 6: Implement RangeViTFusion Main Model

**Files:**
- Create: `models/rangevit_fusion.py`
- Modify: `tests/test_fusion_modules.py`
- Modify: `models/__init__.py`

**Step 1: Write the failing tests**

Add to `tests/test_fusion_modules.py`:

```python
def test_rangevit_fusion_forward():
    """Test RangeViTFusion end-to-end forward pass."""
    from models.rangevit_fusion import RangeViTFusion

    batch_size = 2
    n_points = 1000
    H, W = 32, 384
    in_channels = 5
    n_classes = 17

    model = RangeViTFusion(
        in_channels=in_channels,
        n_cls=n_classes,
        backbone='vit_small_patch16_384',
        image_size=(H, W),
        patch_size=(2, 8),
        patch_stride=(2, 8),
        conv_stem='none',
        fusion_blocks=[4, 8, 12],
    ).to(DEVICE)

    # Inputs
    images = torch.randn(batch_size, in_channels, H, W, device=DEVICE)
    point_attrs = torch.randn(batch_size * n_points, in_channels, device=DEVICE)

    # Point coords in pixel space
    coords = torch.zeros(batch_size * n_points, 3, dtype=torch.long, device=DEVICE)
    coords[:, 0] = torch.arange(batch_size, device=DEVICE).repeat_interleave(n_points)
    coords[:, 1] = torch.randint(0, H, (batch_size * n_points,), device=DEVICE)
    coords[:, 2] = torch.randint(0, W, (batch_size * n_points,), device=DEVICE)

    # Labels for computing loss
    labels = torch.randint(0, n_classes, (batch_size * n_points,), device=DEVICE)

    outputs = model(images, point_attrs, coords, labels)

    assert 'point_logits' in outputs
    assert outputs['point_logits'].shape == (batch_size * n_points, n_classes)
    assert 'loss' in outputs
    assert 'aux_outputs' in outputs


def test_rangevit_fusion_inference_mode():
    """Test RangeViTFusion in inference mode (no labels)."""
    from models.rangevit_fusion import RangeViTFusion

    batch_size = 1
    n_points = 500
    H, W = 32, 384

    model = RangeViTFusion(
        in_channels=5,
        n_cls=17,
        backbone='vit_small_patch16_384',
        image_size=(H, W),
        patch_size=(2, 8),
        patch_stride=(2, 8),
    ).to(DEVICE)
    model.eval()

    images = torch.randn(batch_size, 5, H, W, device=DEVICE)
    point_attrs = torch.randn(batch_size * n_points, 5, device=DEVICE)
    coords = torch.zeros(batch_size * n_points, 3, dtype=torch.long, device=DEVICE)
    coords[:, 0] = 0
    coords[:, 1] = torch.randint(0, H, (batch_size * n_points,), device=DEVICE)
    coords[:, 2] = torch.randint(0, W, (batch_size * n_points,), device=DEVICE)

    with torch.no_grad():
        outputs = model(images, point_attrs, coords)

    assert 'point_logits' in outputs
    assert outputs['point_logits'].shape == (batch_size * n_points, 17)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion_modules.py::test_rangevit_fusion_forward -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Create `models/rangevit_fusion.py`:

```python
# Copyright 2026 - RangeViT-Fusion
# Main model combining ViT backbone with bidirectional point-pixel fusion

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .vit_fusion import VisionTransformerFusion
from .features_encoder import FeaturesEncoder
from .fusion_head import FusionHead
from .fusion_modules import EfficientTransformationPipeline
from .model_utils import adapt_input_conv, resize_pos_embed


class RangeViTFusion(nn.Module):
    """
    RangeViT with HARP-NeXt style bidirectional point-pixel fusion.

    Architecture:
    1. FeaturesEncoder: Raw point attrs -> point features
    2. VisionTransformerFusion: Range image -> pixel features with fusion at specified blocks
    3. FusionHead: Combined pixel+point features -> per-point predictions

    Args:
        in_channels: Number of input channels (default: 5 for xyz + intensity + range)
        n_cls: Number of semantic classes
        backbone: ViT backbone name
        image_size: (H, W) input range image size
        patch_size: (PH, PW) patch size
        patch_stride: (SH, SW) patch stride
        pretrained_path: Path to pretrained ViT weights
        reuse_pos_emb: Whether to reuse pretrained positional embeddings
        conv_stem: 'none' or 'ConvStem'
        stem_base_channels: Base channels for ConvStem
        stem_hidden_dim: Hidden dimension for ConvStem
        fusion_blocks: List of block indices for fusion (1-indexed)
        aux_loss_weight: Weight for auxiliary pixel losses
    """

    def __init__(
        self,
        in_channels=5,
        n_cls=17,
        backbone='vit_small_patch16_384',
        image_size=(32, 384),
        patch_size=(2, 8),
        patch_stride=None,
        pretrained_path=None,
        reuse_pos_emb=False,
        conv_stem='none',
        stem_base_channels=32,
        stem_hidden_dim=None,
        fusion_blocks=[4, 8, 12],
        aux_loss_weight=0.4,
    ):
        super().__init__()

        self.n_cls = n_cls
        self.aux_loss_weight = aux_loss_weight
        self.fusion_blocks = fusion_blocks

        if patch_stride is None:
            patch_stride = patch_size

        # Get backbone config
        if backbone == 'vit_small_patch16_384':
            n_heads, n_layers, d_model = 6, 12, 384
        elif backbone == 'vit_base_patch16_384':
            n_heads, n_layers, d_model = 12, 12, 768
        elif backbone == 'vit_large_patch16_384':
            n_heads, n_layers, d_model = 16, 24, 1024
        else:
            raise ValueError(f'Unknown backbone: {backbone}')

        self.d_model = d_model

        # Point feature encoder
        self.features_encoder = FeaturesEncoder(
            in_channels=in_channels,
            d_model=d_model,
        )

        # Vision Transformer with fusion
        self.encoder = VisionTransformerFusion(
            image_size=image_size,
            patch_size=patch_size,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_model * 4,
            n_heads=n_heads,
            n_cls=n_cls,
            dropout=0.0,
            drop_path_rate=0.1,
            channels=in_channels,
            patch_stride=patch_stride,
            fusion_blocks=fusion_blocks,
            conv_stem=conv_stem,
            stem_base_channels=stem_base_channels,
            stem_hidden_dim=stem_hidden_dim,
        )

        # Grid size for final pixel2point mapping
        if isinstance(patch_stride, (list, tuple)):
            self.grid_h = image_size[0] // patch_stride[0]
            self.grid_w = image_size[1] // patch_stride[1]
        else:
            self.grid_h = image_size[0] // patch_stride
            self.grid_w = image_size[1] // patch_stride

        self.patch_stride = patch_stride

        # ETP for final pixel2point
        self.etp = EfficientTransformationPipeline(self.grid_h, self.grid_w)

        # Prediction head
        self.head = FusionHead(d_model=d_model, n_classes=n_cls)

        # Load pretrained weights if provided
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path, backbone, reuse_pos_emb,
                                  in_channels, image_size, patch_stride)

    def _load_pretrained(self, pretrained_path, backbone, reuse_pos_emb,
                         in_channels, image_size, patch_stride):
        """Load pretrained ViT weights."""
        print(f'Loading pretrained parameters from {pretrained_path}')

        if pretrained_path == 'timmImageNet21k':
            vit_imagenet = timm.create_model(backbone, pretrained=True)
            pretrained_state_dict = vit_imagenet.state_dict()
            # Add encoder prefix
            pretrained_state_dict = {f'encoder.{k}': v for k, v in pretrained_state_dict.items()}
        else:
            pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'state_dict' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['state_dict']
            if 'model' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['model']

        # Resize positional embeddings if needed
        if reuse_pos_emb and 'encoder.pos_embed' in pretrained_state_dict:
            if isinstance(patch_stride, (list, tuple)):
                gs_new_h = image_size[0] // patch_stride[0]
                gs_new_w = image_size[1] // patch_stride[1]
            else:
                gs_new_h = image_size[0] // patch_stride
                gs_new_w = image_size[1] // patch_stride

            resized_pos_emb = resize_pos_embed(
                pretrained_state_dict['encoder.pos_embed'],
                grid_old_shape=None,
                grid_new_shape=(gs_new_h, gs_new_w),
                num_extra_tokens=1,
            )
            pretrained_state_dict['encoder.pos_embed'] = resized_pos_emb

        msg = self.load_state_dict(pretrained_state_dict, strict=False)
        print(f'Loaded pretrained weights: {msg}')

    def _convert_coords_to_patch_space(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert pixel-space coordinates to patch-space coordinates."""
        patch_coords = coords.clone()
        if isinstance(self.patch_stride, (list, tuple)):
            patch_coords[:, 1] = coords[:, 1] // self.patch_stride[0]
            patch_coords[:, 2] = coords[:, 2] // self.patch_stride[1]
        else:
            patch_coords[:, 1] = coords[:, 1] // self.patch_stride
            patch_coords[:, 2] = coords[:, 2] // self.patch_stride
        return patch_coords

    def forward(self, images: torch.Tensor, point_attrs: torch.Tensor,
                coords: torch.Tensor, labels: torch.Tensor = None) -> dict:
        """
        Forward pass.

        Args:
            images: (B, C, H, W) range images
            point_attrs: (N_total, C) raw point attributes
            coords: (N_total, 3) point coordinates [batch_idx, y, x] in pixel space
            labels: (N_total,) point labels (optional, for training)

        Returns:
            dict with:
                - point_logits: (N_total, n_cls) per-point predictions
                - loss: scalar loss (if labels provided)
                - aux_outputs: list of auxiliary pixel logits
        """
        B = images.shape[0]

        # Encode point features
        point_feats = self.features_encoder(point_attrs)

        # Forward through ViT with fusion
        pixel_feats, point_feats, aux_outputs = self.encoder(images, point_feats, coords)

        # Map final pixel features to points
        patch_coords = self._convert_coords_to_patch_space(coords)
        mapped_pixel_feats = self.etp.pixel2point(pixel_feats, patch_coords)

        # Prediction head
        point_logits = self.head(mapped_pixel_feats, point_feats)

        outputs = {
            'point_logits': point_logits,
            'aux_outputs': aux_outputs,
        }

        # Compute loss if labels provided
        if labels is not None:
            outputs['loss'] = self._compute_loss(point_logits, aux_outputs,
                                                  labels, coords, B)

        return outputs

    def _compute_loss(self, point_logits, aux_outputs, labels, coords, batch_size):
        """Compute combined point and auxiliary pixel losses."""
        # Point-level cross entropy (placeholder for focal + lovasz)
        point_loss = F.cross_entropy(point_logits, labels, ignore_index=0)

        # Auxiliary pixel losses
        aux_loss = 0.0
        if self.training and len(aux_outputs) > 0:
            # Create pixel pseudo-labels from point labels
            pseudo_labels = self._create_pseudo_labels(labels, coords, batch_size)

            for aux_logits in aux_outputs:
                # Resize pseudo_labels to match aux_logits spatial size
                h, w = aux_logits.shape[2], aux_logits.shape[3]
                pseudo_resized = F.interpolate(
                    pseudo_labels.unsqueeze(1).float(),
                    size=(h, w),
                    mode='nearest'
                ).squeeze(1).long()

                aux_loss += F.cross_entropy(aux_logits, pseudo_resized, ignore_index=0)

            aux_loss = aux_loss / len(aux_outputs) * self.aux_loss_weight

        return point_loss + aux_loss

    def _create_pseudo_labels(self, labels, coords, batch_size):
        """Create pixel pseudo-labels from point labels (most frequent class per pixel)."""
        H, W = self.grid_h, self.grid_w
        device = labels.device

        # Initialize with ignore index
        pseudo_labels = torch.zeros(batch_size, H, W, dtype=torch.long, device=device)

        # Convert to patch space
        patch_coords = self._convert_coords_to_patch_space(coords)

        # For simplicity, use the first point's label for each pixel
        # A more accurate implementation would use mode (most frequent)
        batch_idx = patch_coords[:, 0].long()
        y_idx = patch_coords[:, 1].long().clamp(0, H - 1)
        x_idx = patch_coords[:, 2].long().clamp(0, W - 1)

        pseudo_labels[batch_idx, y_idx, x_idx] = labels

        return pseudo_labels

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'encoder.pos_embed', 'encoder.cls_token'}

    def count_parameters(self):
        """Count model parameters."""
        stats = {
            'total': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'encoder': sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            'features_encoder': sum(p.numel() for p in self.features_encoder.parameters() if p.requires_grad),
            'head': sum(p.numel() for p in self.head.parameters() if p.requires_grad),
        }
        return stats
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fusion_modules.py -v -k "rangevit_fusion"`
Expected: PASS (2 tests)

**Step 5: Update models/__init__.py**

```python
from .rangevit import RangeViT
from .rangevit_fusion import RangeViTFusion
```

**Step 6: Commit**

```bash
git add models/rangevit_fusion.py models/__init__.py tests/test_fusion_modules.py
git commit -m "feat: add RangeViTFusion main model class"
```

---

## Task 7: Add Focal + Lovasz Losses to RangeViTFusion

**Files:**
- Modify: `models/rangevit_fusion.py`
- Modify: `tests/test_fusion_modules.py`

**Step 1: Write the failing test**

Add to `tests/test_fusion_modules.py`:

```python
def test_rangevit_fusion_loss_components():
    """Test that loss includes focal, lovasz, and auxiliary components."""
    from models.rangevit_fusion import RangeViTFusion

    batch_size = 2
    n_points = 500
    H, W = 32, 384
    n_classes = 17

    model = RangeViTFusion(
        in_channels=5,
        n_cls=n_classes,
        backbone='vit_small_patch16_384',
        image_size=(H, W),
        patch_size=(2, 8),
        patch_stride=(2, 8),
        aux_loss_weight=0.4,
    ).to(DEVICE)
    model.train()

    images = torch.randn(batch_size, 5, H, W, device=DEVICE)
    point_attrs = torch.randn(batch_size * n_points, 5, device=DEVICE)
    coords = torch.zeros(batch_size * n_points, 3, dtype=torch.long, device=DEVICE)
    coords[:, 0] = torch.arange(batch_size, device=DEVICE).repeat_interleave(n_points)
    coords[:, 1] = torch.randint(0, H, (batch_size * n_points,), device=DEVICE)
    coords[:, 2] = torch.randint(0, W, (batch_size * n_points,), device=DEVICE)
    labels = torch.randint(1, n_classes, (batch_size * n_points,), device=DEVICE)  # Avoid 0 (ignore)

    outputs = model(images, point_attrs, coords, labels)

    assert 'loss' in outputs
    assert 'focal_loss' in outputs
    assert 'lovasz_loss' in outputs
    assert 'aux_loss' in outputs

    # All losses should be positive
    assert outputs['focal_loss'] > 0
    assert outputs['lovasz_loss'] >= 0
    assert outputs['aux_loss'] >= 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion_modules.py::test_rangevit_fusion_loss_components -v`
Expected: FAIL with KeyError for 'focal_loss'

**Step 3: Update implementation**

Modify `models/rangevit_fusion.py` - update the `__init__` and `_compute_loss` methods:

```python
# Add to imports at top of file
from utils.optim.focal_softmax import FocalSoftmaxLoss
from utils.optim.lovasz_softmax import Lovasz_softmax

# Add in __init__ after self.head:
        # Loss functions
        self.focal_loss = FocalSoftmaxLoss(n_classes=n_cls, gamma=2, alpha=0.25)
        self.lovasz_loss = Lovasz_softmax(ignore=0)

# Replace _compute_loss method:
    def _compute_loss(self, point_logits, aux_outputs, labels, coords, batch_size):
        """Compute combined focal, lovasz, and auxiliary pixel losses."""
        # Focal loss on points
        focal = self.focal_loss(point_logits, labels)

        # Lovasz loss on points
        point_probs = F.softmax(point_logits, dim=1)
        lovasz = self.lovasz_loss(point_probs, labels)

        # Point loss = focal + lovasz
        point_loss = focal + lovasz

        # Auxiliary pixel losses
        aux_loss = torch.tensor(0.0, device=point_logits.device)
        if self.training and len(aux_outputs) > 0:
            pseudo_labels = self._create_pseudo_labels(labels, coords, batch_size)

            for aux_logits in aux_outputs:
                h, w = aux_logits.shape[2], aux_logits.shape[3]
                pseudo_resized = F.interpolate(
                    pseudo_labels.unsqueeze(1).float(),
                    size=(h, w),
                    mode='nearest'
                ).squeeze(1).long()

                aux_loss += F.cross_entropy(aux_logits, pseudo_resized, ignore_index=0)

            aux_loss = aux_loss / len(aux_outputs) * self.aux_loss_weight

        total_loss = point_loss + aux_loss

        return {
            'loss': total_loss,
            'focal_loss': focal,
            'lovasz_loss': lovasz,
            'aux_loss': aux_loss,
        }

# Update forward method to unpack loss dict:
        if labels is not None:
            loss_dict = self._compute_loss(point_logits, aux_outputs, labels, coords, B)
            outputs.update(loss_dict)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fusion_modules.py::test_rangevit_fusion_loss_components -v`
Expected: PASS

**Step 5: Commit**

```bash
git add models/rangevit_fusion.py tests/test_fusion_modules.py
git commit -m "feat: add focal and lovasz losses to RangeViTFusion"
```

---

## Task 8: Create Fusion Configuration File

**Files:**
- Create: `config_fusion_nusc.yaml`

**Step 1: Create the config file**

```yaml
# config_fusion_nusc.yaml
# RangeViT-Fusion configuration for nuScenes

# General config
num_workers: 4
id: 'rangevit_fusion_nusc'

# Data config
dataset: 'nuScenes'
n_classes: 17  # 16 + 1 (ignored)

# Train config
has_label: true
val_frequency: 10
n_epochs: 50
warmup_epochs: 5
batch_size: 4
batch_size_val: 1
lr: 0.0008
train_result_frequency: 100

# Model config
model_type: 'fusion'  # 'fusion' or 'original'
vit_backbone: 'vit_small_patch16_384'
in_channels: 5
patch_size: [2, 8]
patch_stride: [2, 8]
image_size: [32, 384]
window_size: [32, 384]
window_stride: [32, 256]
original_image_size: [32, 2048]

# Stem config
conv_stem: 'ConvStem'
stem_base_channels: 32
D_h: 256

# Fusion config
fusion:
  enabled: true
  fusion_blocks: [4, 8, 12]
  aux_loss_weight: 0.4

# Decoder - not used in fusion model
decoder: 'none'
skip_filters: 0

# 3D refiner - not used in fusion model
use_kpconv: false

# Checkpoint
checkpoint: null
pretrained_model: 'timmImageNet21k'

# Pre-trained embeddings
reuse_pos_emb: true
reuse_patch_emb: false

# Data augmentation config
augmentation:
  p_flipx: 0.
  p_flipy: 0.5
  p_transx: 0.5
  trans_xmin: -5
  trans_xmax: 5
  p_transy: 0.5
  trans_ymin: -3
  trans_ymax: 3
  p_transz: 0.5
  trans_zmin: -1
  trans_zmax: 0.
  p_rot_roll: 0.5
  rot_rollmin: -5
  rot_rollmax: 5
  p_rot_pitch: 0.5
  rot_pitchmin: -5
  rot_pitchmax: 5
  p_rot_yaw: 0.5
  rot_yawmin: 5
  rot_yawmax: -5

# Sensor config
sensor:
  name: 'HDL64'
  type: 'spherical'
  proj_h: 32
  proj_w: 2048
  fov_up: 10.
  fov_down: -30.
  fov_left: -180
  fov_right: 180
  img_mean:
    - 12.12
    - 10.88
    - 0.23
    - -1.04
    - 0.21
  img_stds:
    - 12.32
    - 11.47
    - 6.91
    - 0.86
    - 0.16
```

**Step 2: Commit**

```bash
git add config_fusion_nusc.yaml
git commit -m "config: add RangeViT-Fusion configuration for nuScenes"
```

---

## Task 9: Update Option Parser for Fusion Config

**Files:**
- Modify: `option.py`

**Step 1: Read current option.py**

Run: Read the file to understand current structure.

**Step 2: Add fusion config parsing**

Add after the existing config parsing (around line 50-60):

```python
# Fusion config
self.model_type = self.config.get('model_type', 'original')
self.fusion_enabled = self.config.get('fusion', {}).get('enabled', False)
self.fusion_blocks = self.config.get('fusion', {}).get('fusion_blocks', [4, 8, 12])
self.aux_loss_weight = self.config.get('fusion', {}).get('aux_loss_weight', 0.4)
```

**Step 3: Run existing tests to ensure no regression**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add option.py
git commit -m "config: add fusion config parsing to Option class"
```

---

## Task 10: Update main.py for Fusion Model

**Files:**
- Modify: `main.py`

**Step 1: Add fusion model import and building logic**

Add to imports:

```python
from models.rangevit_fusion import RangeViTFusion
```

**Step 2: Add build function for fusion model**

Add after `build_rangevit_model` function:

```python
def build_rangevit_fusion_model(settings, pretrained_path=None):
    """Build RangeViT-Fusion model."""
    model = RangeViTFusion(
        in_channels=settings.in_channels,
        n_cls=settings.n_classes,
        backbone=settings.vit_backbone,
        image_size=settings.image_size,
        patch_size=tuple(settings.patch_size),
        patch_stride=tuple(settings.patch_stride) if settings.patch_stride else None,
        pretrained_path=pretrained_path,
        reuse_pos_emb=settings.reuse_pos_emb,
        conv_stem=settings.conv_stem,
        stem_base_channels=settings.stem_base_channels,
        stem_hidden_dim=settings.D_h,
        fusion_blocks=settings.fusion_blocks,
        aux_loss_weight=settings.aux_loss_weight,
    )
    return model
```

**Step 3: Update _initModel to use fusion model when configured**

In the `_initModel` method, add condition:

```python
def _initModel(self, settings):
    if settings.model_type == 'fusion':
        self.model = build_rangevit_fusion_model(settings, settings.pretrained_model)
    else:
        self.model = build_rangevit_model(settings, settings.pretrained_model)
    # ... rest of the method
```

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add RangeViT-Fusion model support to main.py"
```

---

## Task 11: Update DataLoader for Point Features

**Files:**
- Modify: `dataset/range_view_loader.py`

**Step 1: Update data loader to return point attributes and coordinates**

The data loader needs to return:
- Range image (existing)
- Point attributes (xyz, intensity, range)
- Point coordinates (batch_idx, y, x)
- Point labels

Add a new method or modify existing `__getitem__` to return these additional tensors.

**Step 2: Test data loading**

Create a simple test to verify data format.

**Step 3: Commit**

```bash
git add dataset/range_view_loader.py
git commit -m "feat: update dataloader to return point attributes and coordinates"
```

---

## Task 12: Update Trainer for Fusion Model

**Files:**
- Modify: `train.py`

**Step 1: Add fusion training loop**

Add a new method `run_fusion()` that:
- Handles the fusion model's different input format
- Logs the individual loss components (focal, lovasz, aux)
- Handles the per-point output format

**Step 2: Update metric computation for per-point predictions**

The fusion model outputs per-point predictions, not pixel predictions. Update the IoU computation accordingly.

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add fusion model training support to Trainer"
```

---

## Task 13: Run Full Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
import pytest
import torch
from models.rangevit_fusion import RangeViTFusion

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_full_training_step():
    """Test a complete forward-backward pass."""
    model = RangeViTFusion(
        in_channels=5,
        n_cls=17,
        backbone='vit_small_patch16_384',
        image_size=(32, 384),
        patch_size=(2, 8),
        patch_stride=(2, 8),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Simulate training step
    model.train()
    batch_size = 2
    n_points = 500

    images = torch.randn(batch_size, 5, 32, 384, device=DEVICE)
    point_attrs = torch.randn(batch_size * n_points, 5, device=DEVICE)
    coords = torch.zeros(batch_size * n_points, 3, dtype=torch.long, device=DEVICE)
    coords[:, 0] = torch.arange(batch_size, device=DEVICE).repeat_interleave(n_points)
    coords[:, 1] = torch.randint(0, 32, (batch_size * n_points,), device=DEVICE)
    coords[:, 2] = torch.randint(0, 384, (batch_size * n_points,), device=DEVICE)
    labels = torch.randint(1, 17, (batch_size * n_points,), device=DEVICE)

    # Forward
    outputs = model(images, point_attrs, coords, labels)
    loss = outputs['loss']

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check gradients flowed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_inference_speed():
    """Basic inference speed sanity check."""
    import time

    model = RangeViTFusion(
        in_channels=5,
        n_cls=17,
        backbone='vit_small_patch16_384',
        image_size=(32, 384),
        patch_size=(2, 8),
        patch_stride=(2, 8),
    ).to(DEVICE)
    model.eval()

    images = torch.randn(1, 5, 32, 384, device=DEVICE)
    point_attrs = torch.randn(10000, 5, device=DEVICE)
    coords = torch.zeros(10000, 3, dtype=torch.long, device=DEVICE)
    coords[:, 1] = torch.randint(0, 32, (10000,), device=DEVICE)
    coords[:, 2] = torch.randint(0, 384, (10000,), device=DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(images, point_attrs, coords)

    # Time inference
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    n_runs = 10
    with torch.no_grad():
        for _ in range(n_runs):
            model(images, point_attrs, coords)

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = (time.time() - start) / n_runs
    print(f"Inference time: {elapsed*1000:.2f}ms")

    # Should be reasonably fast (< 500ms per sample on GPU)
    if DEVICE.type == 'cuda':
        assert elapsed < 0.5, f"Inference too slow: {elapsed}s"
```

**Step 2: Run integration tests**

Run: `python -m pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add full integration tests for RangeViT-Fusion"
```

---

## Task 14: Final Cleanup and Documentation

**Files:**
- Update: `models/__init__.py`
- Update: `README.md` or create `docs/FUSION.md`

**Step 1: Ensure all exports are correct**

```python
# models/__init__.py
from .rangevit import RangeViT
from .rangevit_fusion import RangeViTFusion
from .features_encoder import FeaturesEncoder
from .fusion_modules import (
    EfficientTransformationPipeline,
    PointFusionLayer,
    PixelFusionLayer,
    AuxiliaryHead,
)
from .fusion_head import FusionHead
from .vit_fusion import VisionTransformerFusion
```

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 3: Final commit**

```bash
git add models/__init__.py
git commit -m "chore: finalize exports and cleanup"
```

---

## Summary

| Task | Description | New Files | Modified Files |
|------|-------------|-----------|----------------|
| 1 | Test infrastructure | `tests/__init__.py`, `tests/test_fusion_modules.py` | - |
| 2 | FeaturesEncoder | `models/features_encoder.py` | tests |
| 3 | EfficientTransformationPipeline | `models/fusion_modules.py` | tests |
| 4 | FusionHead | `models/fusion_head.py` | tests |
| 5 | VisionTransformerFusion | `models/vit_fusion.py` | tests |
| 6 | RangeViTFusion | `models/rangevit_fusion.py` | `models/__init__.py`, tests |
| 7 | Focal + Lovasz losses | - | `models/rangevit_fusion.py` |
| 8 | Config file | `config_fusion_nusc.yaml` | - |
| 9 | Option parser | - | `option.py` |
| 10 | Main entry point | - | `main.py` |
| 11 | DataLoader updates | - | `dataset/range_view_loader.py` |
| 12 | Trainer updates | - | `train.py` |
| 13 | Integration tests | `tests/test_integration.py` | - |
| 14 | Cleanup | - | `models/__init__.py` |

**Estimated total: 14 tasks, ~50 commits**
