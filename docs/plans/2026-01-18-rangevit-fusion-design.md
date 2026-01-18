# RangeViT-Fusion Design Document

**Date:** 2026-01-18
**Status:** Approved
**Branch:** fusion_point

## Overview

RangeViT-Fusion combines RangeViT's Vision Transformer backbone with HARP-NeXt's bidirectional point-pixel fusion. The goal is to maintain 3D geometric information throughout the network via continuous fusion, rather than only refining at the end with KPConv.

### Architecture Diagram

```
Input: LiDAR Point Cloud
    │
    ├──► Range Projection ──► ConvStem ──► ViT Tokens
    │                                          │
    └──► FeaturesEncoder ──► Point Features    │
                                   │           │
                         ┌─────────┴───────────┘
                         │
                    Fusion Loop (3x):
                         │
           ┌─────────────┼─────────────┐
           │             │             │
      [ViT Blocks]  [Pt↔Px Fusion]  [Point Linear]
           │             │             │
           └─────────────┼─────────────┘
                         │
                    After blocks 4, 8, 12
                         │
                         ▼
              HARP-NeXt Style Head
              (pixel2point + concat + MLP)
                         │
                         ▼
              Per-Point Predictions
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fusion points | Blocks 4, 8, 12 | Even spacing, captures low/mid/high features |
| Point branch | Linear layers | Lightweight, proven in HARP-NeXt |
| Point initialization | FeaturesEncoder | Preserves full 3D info, no projection loss |
| Prediction head | HARP-NeXt style | Direct per-point output, no decoder needed |
| KPConv | Removed | Fusion handles 3D refinement throughout |
| Losses | Focal + Lovász + aux pixel | Best of RangeViT and HARP-NeXt |

---

## Component Details

### 1. Fusion Mechanism

At each fusion point (after blocks 4, 8, 12), bidirectional exchange occurs:

**Step 1: Reshape ViT tokens to 2D grid**
```
tokens: (B, N, D) → pixel_feats: (B, D, H, W)
where N = H × W (number of patches)
```

**Step 2: Pixel → Point mapping**
```python
# Look up pixel features for each point using projection coordinates
mapped_pixel_feats = pixel2point(pixel_feats, point_coords, stride)
# Fuse with existing point features
fused = concat(mapped_pixel_feats, point_feats)
point_feats = point_linear(fused)  # Linear → BN → ReLU
```

**Step 3: Point → Pixel mapping**
```python
# Aggregate point features into pixel grid (max pooling)
cluster_feats = point2cluster(point_feats, point_coords, stride)
pixel_from_points = cluster2pixel(cluster_feats, coords, batch_size, stride)
# Fuse with ViT pixel features
fused = concat(pixel_from_points, pixel_feats)
pixel_feats = pixel_fusion(fused)  # Conv → BN → Hardswish
```

**Step 4: Flatten back to tokens**
```
pixel_feats: (B, D, H, W) → tokens: (B, N, D)
Continue to next ViT blocks...
```

**Stride handling:** Since ViT maintains constant resolution (no downsampling), all fusion uses stride=1.

---

### 2. Point Feature Initialization (FeaturesEncoder)

Processes raw 3D point attributes before any fusion occurs.

**Input features per point:**
- xyz coordinates (3)
- intensity (1)
- range/distance (1)
- Total: 5 channels

**Architecture:**
```
Raw Point Attrs (B×N_pts, 5)
    │
    ▼
Linear(5, 64) → BN → ReLU
    │
    ▼
Linear(64, 128) → BN → ReLU
    │
    ▼
Linear(128, D) → BN → ReLU
    │
    ▼
Point Features (B×N_pts, D)

where D = d_model (384 for ViT-Small)
```

**Key points:**
- Processes each point independently
- Output dimension matches ViT's d_model for easy fusion
- Lightweight: ~0.1M parameters
- Runs once at the start, before ViT encoding begins

---

### 3. Prediction Head

After final fusion at block 12, combines pixel and point features for per-point classification.

```
ViT Output (B, N, D)
    │
    ▼
Reshape to (B, D, H, W)
    │
    ▼
pixel2point mapping
    │
    ▼
Mapped Pixel Feats (B×N_pts, D)
    │
    ├─── concat ◄─── Point Feats (B×N_pts, D)
    │
    ▼
Combined (B×N_pts, 2D)
    │
    ▼
MLP Head:
  Linear(2D, D) → BN → ReLU
  Linear(D, D//2) → BN → ReLU
  Linear(D//2, n_classes)
    │
    ▼
Per-Point Logits (B×N_pts, n_classes)
```

**Key differences from RangeViT:**
- No decoder upsampling (UpConv/PixelShuffle removed)
- No KPConv 3D refiner
- Direct per-point prediction from fused features

---

### 4. Loss Functions

**Primary loss (on final per-point predictions):**
```
L_point = λ₁ · FocalLoss(point_logits, labels)
        + λ₂ · LovászSoftmax(point_logits, labels)

where λ₁ = 1.0, λ₂ = 1.0
```

**Auxiliary losses (at fusion points 4, 8, 12):**
```
L_pixel_aux = Σᵢ βᵢ · CrossEntropy(pixel_feats_i, pseudo_labels)

where i ∈ {4, 8, 12}, βᵢ = 0.4
```

**Pixel pseudo-labels:** For each pixel, assign the most frequent class among points projecting to it.

**Total loss:**
```
L_total = L_point + L_pixel_aux
```

**Auxiliary head structure (at each fusion point, training only):**
```
pixel_feats (B, D, H, W)
    │
    ▼
Conv2d(D, n_classes, 1×1)
    │
    ▼
pixel_logits (B, n_classes, H, W)
```

---

### 5. Pre-training & Training Strategy

**Pre-trained weight handling:**

```
Loadable from pre-trained (ImageNet):
  ✓ ViT attention blocks (all 12)
  ✓ ViT layer norms
  ✓ Positional embeddings (with resize)
  ✓ ConvStem (if architecture matches)

Random initialization:
  ✗ FeaturesEncoder
  ✗ Point fusion linear layers
  ✗ Pixel fusion conv layers
  ✗ Prediction head MLP
  ✗ Auxiliary heads
```

**Training strategy:**

| Phase | Epochs | What's trained | LR |
|-------|--------|----------------|-----|
| Warmup | 5 | New components only (freeze ViT) | 1e-3 |
| Full | 45 | Everything | 1e-4 (ViT), 1e-3 (new) |

**Hyperparameters:**
- Batch size: 4-8 per GPU
- Optimizer: AdamW
- Weight decay: 0.01
- Scheduler: Cosine annealing

---

## Implementation Plan

### New Files

```
models/
├── rangevit_fusion.py      # Main model class
├── fusion_modules.py       # Pt↔Px fusion, EfficientTransformationPipeline
├── features_encoder.py     # Point feature initialization
└── fusion_head.py          # HARP-NeXt style prediction head
```

### Config Additions

```yaml
fusion:
  enabled: true
  fusion_blocks: [4, 8, 12]
  point_dim: 384
  aux_loss_weight: 0.4
```

### Parameter Estimates

| Component | Parameters (est.) |
|-----------|-------------------|
| ConvStem | ~0.5M |
| ViT Encoder (12 blocks) | ~21M (ViT-S) |
| FeaturesEncoder | ~0.1M |
| Point Fusion Layers (×3) | ~0.9M |
| Pixel Fusion Layers (×3) | ~0.9M |
| Prediction Head | ~0.3M |
| **Total** | **~24M** |

Comparison: RangeViT+KPConv (~25M), HARP-NeXt (~5.4M)

---

## References

- RangeViT: Vision Transformers for Range-Image-Based 3D Semantic Segmentation (CVPR 2023)
- HARP-NeXt: High-Speed and Accurate Range-Point Fusion Network (arXiv:2510.06876)
