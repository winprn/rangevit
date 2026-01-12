# Fusion Model Implementation Documentation

This document describes the implementation of the learnable voxel branch with multi-stage fusion for RangeViT. This serves as a reference for debugging and future development.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Files Created](#files-created)
3. [Files Modified](#files-modified)
4. [Data Flow](#data-flow)
5. [Key Design Decisions](#key-design-decisions)
6. [Configuration Options](#configuration-options)
7. [Checkpoint Format](#checkpoint-format)
8. [Common Issues & Debugging](#common-issues--debugging)

---

## Architecture Overview

The fusion model combines a **ViT-based range branch** with a **MinkUNet-based voxel branch**, using **point representation as a communication hub** for bidirectional feature exchange.

### Architecture Diagram

```
              Fusion 1                  Fusion 2                  Fusion 3
              (After Stem)              (After Encoder)           (After Decoder)
                  │                         │                         │
Range:  ConvStem ►│──► ViT Encoder ────────►│──► DecoderUpConv ──────►│
                  │         ▲               │         ▲               │
                  │    R2P  │   P2R         │    R2P  │   P2R         │    R2P
                  ▼         │               ▼         │               ▼
Point:  [x,y,z,i]► z0 ─────────────────────► z1 ─────────────────────► z2 ─►Classifier
                  ▲         │               ▲         │               ▲
                  │    V2P  │   P2V         │    V2P  │   P2V         │    V2P
                  │         ▼               │         ▼               │
Voxel:  MinkStem ►│──► MinkStages1-4 ──────►│──► MinkUp1-4 ──────────►│

Final: classifier(concat(z1, z2)) → [N, n_cls]   (last 2 stages only)
```

### Key Components

1. **Range Branch**: ViT encoder with ConvStem and DecoderUpConv
2. **Voxel Branch**: MinkUNet (sparse 3D U-Net using torchsparse)
3. **Fusion Modules**: FusionMLP (concatenation + MLP) at 3 fusion points
4. **Point Transforms**: Channel dimension matching at each fusion stage
5. **Classifier**: Linear layer on concatenated z1 + z2 features

---

## Files Created

### 1. `models/fusion/__init__.py`
Exports all fusion module components:
```python
from .fusion_rangevit import FusionRangeViT
from .minkunet_voxel import MinkUNetVoxelEncoder
from .fusion_modules import FusionMLP, PointTransform
from .representation_utils import (
    initial_voxelize, voxel_to_point, point_to_voxel,
    range_to_point, point_to_range,
)
```

### 2. `models/fusion/representation_utils.py`
Implements view conversion utilities:

| Function | Description | Input → Output |
|----------|-------------|----------------|
| `initial_voxelize(z, pres, vres)` | Convert PointTensor to SparseTensor | PointTensor → SparseTensor |
| `voxel_to_point(x, z, nearest)` | Sample voxel features at point locations | SparseTensor, PointTensor → PointTensor |
| `point_to_voxel(x, z)` | Aggregate point features to voxels | SparseTensor, PointTensor → SparseTensor |
| `range_to_point(feat_map, pxpy, batch_indices, B)` | Sample 2D features at point projection coords | [B,C,H,W], [N,2] → [N,C] |
| `range_to_point_from_tokens(tokens, pxpy, ...)` | Sample from ViT tokens (reshaped to 2D) | [B,T,C], [N,2] → [N,C] |
| `point_to_range_fast(point_feats, pxpy, ...)` | Scatter point features to range image | [N,C], [N,2] → [B,C,H,W] |

**Key Implementation Details:**
- Uses `F.grid_sample` with `align_corners=True` for R2P
- V2P uses trilinear interpolation via torchsparse's `voxel_to_point`
- P2V uses mean aggregation via torchsparse's `point_to_voxel`
- pxpy coordinates are normalized to [-1, 1] for grid_sample

### 3. `models/fusion/minkunet_voxel.py`
MinkUNet sparse 3D U-Net encoder with:
- **Stem**: Initial convolution (in_channels → cs[0])
- **Stages 1-4**: Downsampling encoder stages
- **Up blocks 1-4**: Upsampling decoder stages with skip connections

**Fusion Point Outputs** (with default config, Bottleneck, cr=1.0):
- `stem_out`: 128 channels (cs[0] * expansion)
- `bottleneck_out`: 1024 channels (cs[4] * expansion)
- `final_out`: 384 channels (cs[8] * expansion)

**Channel Computation:**
```python
expansion = 4 if block_type == 'Bottleneck' else 1
cs = [int(c * cr) for c in planes]  # planes = [32, 32, 64, 128, 256, 256, 128, 96, 96]
# stem_out: cs[0] * expansion = 32 * 4 = 128
# bottleneck_out: cs[4] * expansion = 256 * 4 = 1024
# final_out: cs[8] * expansion = 96 * 4 = 384
```

### 4. `models/fusion/fusion_modules.py`
Implements fusion primitives:

**FusionMLP**: Concatenation + MLP fusion
```python
# Input: range_feats [N, C_r], voxel_feats [N, C_v], point_feats [N, C_p]
concat = torch.cat([range_feats, voxel_feats, point_feats], dim=1)
output = mlp(concat)  # [N, C_out]
```

**PointTransform**: Channel dimension projection
```python
# Input: [N, C_in] → Output: [N, C_out]
output = Linear + BatchNorm + ReLU
```

### 5. `models/fusion/fusion_rangevit.py`
Main fusion model class `FusionRangeViT`:

**Key Attributes:**
- `range_stem`: ConvStem from original RangeViT
- `range_encoder`: ViT encoder
- `range_decoder`: DecoderUpConv
- `voxel_branch`: MinkUNetVoxelEncoder
- `point_transforms`: ModuleList of 3 PointTransform layers
- `fusion_modules`: ModuleList of 3 FusionMLP layers
- `classifier`: Linear layer (enc_dim + dec_dim → n_cls)

**Forward Pass:**
```python
def forward(self, range_image, point_features, point_coords, batch_indices, range_pxpy):
    # range_image: [B, 5, H, W]
    # point_features: [N, 4] (x, y, z, intensity)
    # point_coords: [N, 3]
    # batch_indices: [N]
    # range_pxpy: [N, 2] normalized to [-1, 1]

    # Returns: [N, n_cls] per-point logits
```

**Checkpoint Methods:**
- `get_checkpoint_state()`: Returns dict with separate component state_dicts
- `load_checkpoint_state(checkpoint, strict)`: Loads component states with optional strict mode

### 6. `config_fusion_kitti.yaml`
Configuration file for fusion model training on SemanticKITTI.

---

## Files Modified

### 1. `models/__init__.py`
Added export:
```python
from .fusion import FusionRangeViT
```

### 2. `dataset/__init__.py`
Added export:
```python
from .range_view_loader import RangeViewLoader, custom_collate_kpconv_fn, custom_collate_fusion_fn
```

### 3. `dataset/range_view_loader.py`
**Changes:**
- Added `use_fusion_voxel` parameter to `__init__`
- Added `get_item_for_fusion()` method returning:
  ```python
  {
      'range_image': [5, H, W],      # 5-channel range image
      'range_label': [H, W],          # Per-pixel labels
      'range_mask': [H, W],           # Valid pixel mask
      'point_features': [N, 4],       # x, y, z, intensity
      'point_coords': [N, 3],         # x, y, z coordinates
      'point_labels': [N],            # Per-point labels
      'range_pxpy': [N, 2],           # Normalized projection coords
      'num_points': int,
      'index': int,
  }
  ```
- Added `custom_collate_fusion_fn()` for batching variable-length point clouds

**Collate Function Output:**
```python
{
    'range_image': [B, 5, H, W],
    'range_label': [B, H, W],
    'range_mask': [B, H, W],
    'point_features': [N_total, 4],   # Concatenated across batch
    'point_coords': [N_total, 3],
    'point_labels': [N_total],
    'batch_indices': [N_total],       # Batch index per point
    'range_pxpy': [N_total, 2],
    'num_points': [B],
    'index': [B],
}
```

### 4. `option.py`
**Added Config Parsing:**
```python
# Fusion model config
self.use_fusion_voxel = self.config.get('use_fusion_voxel', False)

# Voxel branch config
voxel_branch_cfg = self.config.get('voxel_branch', {})
self.voxel_in_channels = voxel_branch_cfg.get('in_channels', 4)
self.voxel_num_layer = voxel_branch_cfg.get('num_layer', [2, 3, 4, 6, 2, 2, 2, 2])
self.voxel_block_type = voxel_branch_cfg.get('block_type', 'Bottleneck')
self.voxel_cr = voxel_branch_cfg.get('cr', 1.0)
self.voxel_planes = voxel_branch_cfg.get('planes', [32, 32, 64, 128, 256, 256, 128, 96, 96])
self.voxel_pres = voxel_branch_cfg.get('pres', 0.05)
self.voxel_vres = voxel_branch_cfg.get('vres', 0.05)
self.voxel_dropout_p = voxel_branch_cfg.get('dropout_p', 0.3)

# Fusion config
fusion_cfg = self.config.get('fusion', {})
self.fusion_hidden_ratio = fusion_cfg.get('hidden_ratio', 2.0)

# Separate pretrained paths
self.range_pretrained_model = self.config.get('range_pretrained_model', None)
self.voxel_pretrained_model = self.config.get('voxel_pretrained_model', None)
```

**Added Validation:**
```python
if self.use_fusion_voxel:
    assert self.decoder == 'up_conv', "Fusion model requires up_conv decoder"
    assert not self.use_kpconv, "Fusion model is incompatible with KPConv"
    assert not self.use_voxel_features, "Fusion model has its own voxel branch"
```

### 5. `train.py`
**Changes:**
- Updated `_initDataloader()` to pass `use_fusion_voxel` to RangeViewLoader
- Updated collate function selection:
  ```python
  if self.settings.use_fusion_voxel:
      collate_fn = dataset.custom_collate_fusion_fn
  elif self.settings.use_kpconv:
      collate_fn = dataset.custom_collate_kpconv_fn
  else:
      collate_fn = None
  ```
- Updated `run()` to dispatch to `run_with_fusion()` when fusion enabled
- Added `run_with_fusion()` method (~260 lines) for point-level training

**run_with_fusion Key Details:**
- Extracts fusion inputs from batch_dict
- Reshapes output for loss computation: `[N, C] → [N, C, 1, 1]`
- Uses point-level labels: `[N] → [N, 1, 1]`
- Metrics computed on point predictions

### 6. `main.py`
**Changes:**
- Added `build_fusion_model(settings)` function
- Updated `_initModel()` to select model based on `use_fusion_voxel`
- Updated `_loadCheckpoint()` to handle fusion checkpoint format
- Added `_get_checkpoint_data()` helper for consistent checkpoint saving
- Updated checkpoint saving to use helper method

---

## Data Flow

### Training Data Flow

```
1. RangeViewLoader.get_item_for_fusion()
   ├── Load point cloud and labels
   ├── Apply augmentations (flip, translate, rotate)
   ├── Project to range image (5 channels)
   ├── Compute pxpy projection coordinates
   └── Return dict with all tensors

2. custom_collate_fusion_fn()
   ├── Stack range images: [B, 5, H, W]
   ├── Concatenate point data with batch indices
   └── Return batched dict

3. run_with_fusion() training loop
   ├── Move tensors to GPU
   ├── Forward pass through FusionRangeViT
   ├── Compute point-level loss (focal + lovasz)
   ├── Backward pass and optimizer step
   └── Update metrics
```

### Model Forward Pass

```
1. Initialize PointTensor from point_features and coords

2. FUSION 1 (After Stem):
   ├── Range: ConvStem → skip [B, D_h, H, W]
   ├── Voxel: initial_voxelize → stem
   ├── Fusion: R2P(skip) + V2P(voxel_stem) + PointTransform(raw) → z0
   └── Update voxel: P2V(z0)

3. FUSION 2 (After Encoder):
   ├── Range: ViT encoder → tokens [B, T, d_model]
   ├── Voxel: stages 1-4 → bottleneck
   ├── Fusion: R2P(tokens) + V2P(bottleneck) + PointTransform(z0) → z1
   └── Update voxel: P2V(z1)

4. FUSION 3 (After Decoder):
   ├── Range: DecoderUpConv → features [B, D_h, H, W]
   ├── Voxel: up blocks 1-4 → final
   ├── Fusion: R2P(features) + V2P(final) + PointTransform(z1) → z2
   └── No voxel update (final stage)

5. Classification:
   └── classifier(concat(z1, z2)) → [N, n_cls]
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Voxel Encoder | MinkUNet (Bottleneck) | Proven architecture from RPVNet |
| Fusion Strategy | Concat + MLP | Simple baseline, easy to debug |
| Fusion Points | 3 stages | Matches RangeViT's single-stage decoder |
| Classifier Input | z1 + z2 | Last 2 fusion stages (similar to RPVNet) |
| Communication Hub | Point representation | Natural bridge between 3D voxels and 2D range |
| Range Input | 5 channels | Voxel branch handles 3D features separately |
| Checkpoints | Separate by component | Enables independent encoder loading |
| Output | Per-point logits [N, n_cls] | Direct point classification |

---

## Configuration Options

### config_fusion_kitti.yaml Key Options

```yaml
# Enable fusion model
use_fusion_voxel: true

# Range branch (5 channels only)
in_channels: 5

# Voxel branch
voxel_branch:
  in_channels: 4          # x, y, z, intensity
  num_layer: [2, 3, 4, 6, 2, 2, 2, 2]  # Layers per stage
  block_type: "Bottleneck"  # or "BasicBlock"
  cr: 1.0                 # Channel ratio
  planes: [32, 32, 64, 128, 256, 256, 128, 96, 96]
  pres: 0.05              # Point resolution (5cm)
  vres: 0.05              # Voxel resolution (5cm)
  dropout_p: 0.3

# Fusion
fusion:
  hidden_ratio: 2.0       # Hidden layer ratio in FusionMLP

# Pretrained paths (optional)
range_pretrained_model: null  # Path to RangeViT checkpoint
voxel_pretrained_model: null  # Path to MinkUNet checkpoint
```

---

## Checkpoint Format

### Fusion Checkpoint Structure

```python
checkpoint = {
    'model': state_dict,           # Full model state (backward compat)
    'fusion_state': {              # Component-based format
        'range_stem': state_dict,
        'range_encoder': state_dict,
        'range_decoder': state_dict,
        'voxel_branch': state_dict,
        'point_transforms': state_dict,
        'fusion_modules': state_dict,
        'classifier': state_dict,
    },
    'optimizer': state_dict,
    'epoch': int,
    'fp16_scaler': state_dict,     # Optional
}
```

### Loading Pretrained Encoders

Set in config or pass via constructor:
```yaml
range_pretrained_model: "/path/to/rangevit_checkpoint.pth"
voxel_pretrained_model: "/path/to/minkunet_checkpoint.pth"
```

The model will attempt to load these during initialization with `strict=False`.

---

## Common Issues & Debugging

### 1. CUDA Out of Memory

**Symptoms:** OOM error during training

**Solutions:**
- Reduce `batch_size` in config
- Reduce `voxel_branch.planes` (e.g., multiply all by 0.5)
- Reduce `voxel_branch.cr` (channel ratio)
- Increase `voxel_branch.vres` (coarser voxels)

### 2. torchsparse Import Error

**Symptoms:** `ModuleNotFoundError: No module named 'torchsparse'`

**Solution:**
```bash
pip install torchsparse
# Or for specific CUDA version:
pip install git+https://github.com/mit-han-lab/torchsparse.git
```

### 3. Shape Mismatch in Fusion

**Symptoms:** Tensor shape mismatch in FusionMLP

**Debug Steps:**
1. Print shapes at each fusion point:
   ```python
   print(f"z_range: {z_range.shape}, z_voxel: {z_voxel.shape}, z_point: {z_point.shape}")
   ```
2. Verify channel dimensions match expected values
3. Check that batch_indices correctly separates samples

### 4. NaN Loss

**Symptoms:** Loss becomes NaN during training

**Possible Causes:**
- Learning rate too high (try 1e-4 or lower)
- Empty voxels causing division by zero
- Invalid pxpy coordinates (outside [-1, 1])

**Debug:**
```python
# Check for NaN in features
print(f"range_feats has NaN: {torch.isnan(range_feats).any()}")
print(f"voxel_feats has NaN: {torch.isnan(voxel_feats).any()}")
```

### 5. Poor Convergence

**Symptoms:** Model doesn't improve or oscillates

**Solutions:**
- Verify data augmentation is reasonable
- Check class weights in loss function
- Try loading pretrained range/voxel encoders
- Reduce fusion_hidden_ratio for simpler fusion

### 6. Checkpoint Loading Issues

**Symptoms:** Warnings about missing/unexpected keys

**Debug:**
```python
msg = model.load_state_dict(checkpoint['model'], strict=False)
print(f"Missing keys: {msg.missing_keys}")
print(f"Unexpected keys: {msg.unexpected_keys}")
```

### 7. pxpy Coordinate Issues

**Symptoms:** Features not sampled correctly from range image

**Verify:**
- pxpy should be normalized to [-1, 1] for F.grid_sample
- Check: `pxpy[:, 0]` is x (width), `pxpy[:, 1]` is y (height)
- grid_sample expects (x, y) order, not (row, col)

### 8. Distributed Training Issues

**Symptoms:** Hangs or crashes in multi-GPU training

**Solutions:**
- Ensure `SyncBatchNorm` is used (automatic via train.py)
- Check that all processes reach barriers
- Verify NCCL backend is available

---

## Usage Examples

### Training

```bash
# Single GPU
python main.py 'config_fusion_kitti.yaml' \
    --data_root '/path/to/semantic_kitti/dataset/sequences/' \
    --save_path '/path/to/logs'

# Multi-GPU (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63545 \
    --use_env main.py 'config_fusion_kitti.yaml' \
    --data_root '/path/to/semantic_kitti/dataset/sequences/' \
    --save_path '/path/to/logs'
```

### Validation Only

```bash
python main.py 'config_fusion_kitti.yaml' \
    --data_root '/path/to/semantic_kitti/dataset/sequences/' \
    --save_path '/path/to/logs' \
    --checkpoint '/path/to/checkpoint.pth' \
    --val_only
```

### With Pretrained Encoders

Modify config:
```yaml
range_pretrained_model: "/path/to/rangevit_best_IOU_model.pth"
voxel_pretrained_model: "/path/to/minkunet_pretrained.pth"
```

---

## File Locations Summary

```
D:\rangevit\
├── models/
│   ├── __init__.py                    # Modified: added FusionRangeViT export
│   └── fusion/
│       ├── __init__.py                # Created: module exports
│       ├── representation_utils.py    # Created: V2P, P2V, R2P, P2R
│       ├── minkunet_voxel.py          # Created: MinkUNet encoder
│       ├── fusion_modules.py          # Created: FusionMLP, PointTransform
│       └── fusion_rangevit.py         # Created: main fusion model
├── dataset/
│   ├── __init__.py                    # Modified: added collate export
│   └── range_view_loader.py           # Modified: fusion data loading
├── option.py                          # Modified: fusion config parsing
├── train.py                           # Modified: run_with_fusion method
├── main.py                            # Modified: fusion model building
├── config_fusion_kitti.yaml           # Created: fusion config
└── FUSION_IMPLEMENTATION.md           # This file
```

---

## Version History

- **v1.0** (2024): Initial implementation
  - 3-stage fusion with point hub
  - MinkUNet voxel encoder
  - Concat + MLP fusion
  - Separate checkpoint handling
