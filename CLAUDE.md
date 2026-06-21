# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**RangeViT** is a Vision Transformer-based approach for 3D semantic segmentation in autonomous driving (CVPR 2023). It processes LiDAR data as range images using Vision Transformers adapted for 3D semantic segmentation on nuScenes and SemanticKITTI datasets.

**TinyViM Integration**: This repository now integrates TinyViM (Tiny Vision Mamba), a state-space model backbone that uses selective scan mechanisms for efficient long-range modeling. TinyViM replaces the standard ViT encoder while maintaining compatibility with RangeViT's projection-based segmentation pipeline.

## Repository Structure

```
rangevit/
├── config/
│   ├── kitti/
│   │   ├── main/                  # Main experiments (SemanticKITTI)
│   │   ├── ablation/              # Ablation studies
│   │   │   ├── backbone/          # Backbone variants
│   │   │   ├── decoder/           # Decoder variants
│   │   │   ├── window/            # Sliding window studies
│   │   │   └── robustness/        # Robustness sensitivity
│   │   └── reproduce/              # Reproduction configs
│   └── nusc/
│       └── reproduce/              # nuScenes configs
├── main.py                    # Training/evaluation entry point
├── train.py                   # Trainer class with train/val loops
├── option.py                  # Configuration parsing
│
├── models/
│   ├── rangevit.py           # Main RangeViT model wrapper
│   ├── rangevit_kpconv.py    # RangeViT + KPConv 3D refiner
│   ├── tinyvim_adapter.py    # TinyViM integration adapter
│   ├── stems.py              # Patch embedding and ConvStem
│   ├── decoders.py           # Linear and UpConv decoders
│   ├── blocks.py             # Transformer blocks
│   ├── swin_transformer_v2.py # Swin Transformer support
│   │
│   ├── tinyvim/              # TinyViM backbone
│   │   ├── tinyvim.py        # TinyViM model (S/B/L variants)
│   │   ├── tvimblock.py      # SSM-based blocks (TViMBlock, LocalBlock)
│   │   └── fpn_decoder.py    # Feature Pyramid Network decoder
│   │
│   └── kpconv/               # KPConv 3D refiner
│       ├── blocks.py
│       ├── modules.py
│       └── ...
│
├── dataset/
│   ├── range_view_loader.py  # Main range image data loader
│   ├── nuScenes/             # nuScenes dataset handling
│   ├── semantic_kitti/       # SemanticKITTI dataset handling
│   └── preprocess/           # Augmentation and projection
│
├── tests/
│   ├── test_tinyvim_integration.py  # TinyViM tests
│   └── test_range_image_plot.py     # Visualization tests
│
└── utils/
    ├── inference/            # Inference utilities
    ├── metrics/              # Evaluation metrics (mIoU)
    ├── optim/                # Optimizers and schedulers
    ├── postproc/             # Post-processing (KNN)
    └── tools/                # Logging, MLflow
```

## Core Architecture

### Overall Pipeline

```
LiDAR Point Cloud [N, 4]
    ↓ (spherical projection)
Range Image [B, 5, H, W]
    ↓
┌─────────────────────────────────────┐
│ ENCODER (ViT / Swin / TinyViM)      │
│  ├─ Stem/Patch Embedding            │
│  ├─ Backbone Blocks                 │
│  └─ Output: tokens + skip features  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ DECODER (Linear / UpConv / FPN)     │
│  └─ Upsamples to [B, C, H, W]       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3D REFINER (KPConv / KNN / None)    │
│  └─ Post-processing in 3D space     │
└─────────────────────────────────────┘
    ↓
Segmentation Output [B, n_classes, H, W]
```

### TinyViM-Specific Architecture

TinyViM uses a **4-stage architecture** with asymmetric stride for range images:

```
Input Range Image [B, 5, H, W]
    ↓
Stem (stride 1,2)
    [B, 5, H, W] → [B, C0, H, W/2]
    - Conv2d_BN(kernel=3, stride=(1,2)) + GELU
    - Conv2d_BN(kernel=3, stride=(1,1)) + GELU
    ↓
Stage 0: LocalBlocks + TViMBlock
    [B, 48, H, W/2]  (small variant)
    ↓ (Embedding layer, stride 1,2)
Stage 1: LocalBlocks + TViMBlock
    [B, 96, H, W/4]
    ↓ (Embedding layer, stride 1,2)
Stage 2: LocalBlocks + TViMBlocks
    [B, 192, H, W/8]
    ↓ (Embedding layer, stride 1,2)
Stage 3: LocalBlocks + TViMBlocks
    [B, 384, H, W/16]
    ↓
Output:
  - Final tokens: [B, N, D] (with dummy CLS token for compatibility)
  - Multi-scale features: [stage0, stage1, stage2, stage3] for FPN
```

**Key Design Decision: Asymmetric Stride (1, 2)**

The stem uses stride `(1, 2)` to:
- **Preserve height**: No downsampling vertically (critical for 64-channel range images)
- **Downsample width**: 2× reduction horizontally (efficient for 2048-wide images)
- **Stage strides**: `/2, /4, /8, /16` (width-only downsample)

This optimizes for the elongated aspect ratio of range images (64×2048 or 32×2048).

### Model Variants

**TinyViM Sizes:**
- `tinyvim_small`: 48/64/168/224 channels, 3/3/9/6 blocks per stage
- `tinyvim_base`: 48/96/192/384 channels, 4/3/10/5 blocks per stage
- `tinyvim_large`: 64/128/384/512 channels, 4/4/12/6 blocks per stage

**Other Backbones:**
- `vit_small_patch16_384`: Standard ViT-S/16
- `swin_v2_small_window16_256`: Swin Transformer V2
- Custom ViT sizes via configuration

## Key Components

### 1. TinyViM Integration (`models/tinyvim_adapter.py`)

**Purpose:** Bridges TinyViM with RangeViT's encoder interface.

**Key responsibilities:**
- Parses backbone name (e.g., `tinyvim_base`)
- Adapts input channels from 3 (RGB) to 5 (LiDAR: r, x, y, z, intensity)
- Configures stem stride: `(1, 2)` by default
- Adds dummy CLS token for compatibility (TinyViM has no CLS token)
- Returns tokens and skip connections for decoder

**Usage in code:**
```python
# models/rangevit.py
if 'tinyvim' in args.vit_backbone:
    from models.tinyvim_adapter import TinyViMAdapter
    self.encoder = TinyViMAdapter(
        backbone_name=args.vit_backbone,
        in_channels=args.in_channels,
        num_classes=n_classes,
        stem_stride=(1, 2),
        pretrained=args.pretrained_model
    )
```

### 2. TinyViM Core (`models/tinyvim/`)

**`tinyvim.py`** - Main model implementation
- Defines S/B/L variants with width/depth configs
- 4-stage architecture with embedding layers between stages
- Supports asymmetric stem stride for range images

**`tvimblock.py`** - Building blocks
- `TViMBlock`: SSM-based block using selective scan (Mamba-style)
- `LocalBlock`: Convolutional block with RepDW and FFN
- `Conv2d_BN`: Fused Conv2d + BatchNorm for efficiency
- `RepDW`: Reparameterizable depthwise convolution

**`fpn_decoder.py`** - Feature Pyramid Network decoder
- Multi-scale feature fusion from all 4 stages
- Lateral connections and top-down pathway
- Default: 256 feature channels, 128 head channels

### 3. Decoders (`models/decoders.py`)

**Linear Decoder:**
- Simple linear projection per patch
- No skip connections
- Fastest but lowest accuracy

**UpConv Decoder:**
- Upconvolution with PixelShuffle
- Skip connection from stem features
- Good balance of speed/accuracy

**FPN Decoder** (`models/tinyvim/fpn_decoder.py`):
- Multi-scale feature fusion
- Best for TinyViM (uses all stage outputs)
- Higher accuracy, more parameters

### 4. Post-Processing Options

**KPConv** (`models/kpconv/`):
- 3D point convolution refiner
- Learned post-processing in 3D space
- Requires unprojection to points
- Best accuracy, higher compute

**KNN** (`utils/postproc/`):
- K-nearest neighbor smoothing in 3D
- Non-learned, fast
- Good balance for inference

**None:**
- No post-processing
- Fastest, lowest accuracy

### 5. Dataset Handling (`dataset/range_view_loader.py`)

**Range Image Projection:**
- Projects 3D point cloud to 2D range image via spherical projection
- 5 channels: `[range, x, y, z, intensity]`
- Handles empty pixels (zero-filled) and overlap (keeps closest point)

**Data Augmentation:**
- Random horizontal flip (azimuth wrap-around)
- Random translation (range shift)
- Random rotation (yaw)
- Normalization using sensor-specific mean/std

**Training Modes:**
- **Full-image mode**: `train_full_image: true` (no random crop)
- **Crop mode**: Random crops during training for memory efficiency

**SemanticKITTI:** 64×2048 range images (64 vertical channels)
**nuScenes:** 32×2048 range images (32 vertical channels)

## Configuration System

Configurations are defined in YAML files. Key sections:

### Model Configuration

```yaml
# Encoder (backbone selection)
vit_backbone: "tinyvim_base"      # or "vit_small_patch16_384", "swin_v2_small_window16_256"
in_channels: 5                    # LiDAR channels: r, x, y, z, intensity
patch_size: [2, 2]                # Overall effective stride after stem
patch_stride: [2, 2]
image_size: [64, 2048]            # H × W (SemanticKITTI)

# Decoder
decoder: "fpn"                    # or "linear", "up_conv"
decoder_skip: true                # Skip connection from stem (for up_conv)

# 3D Refiner
use_kpconv: false                 # KPConv refiner (if true)
point_postproc: "knn"             # or "kpconv", null
```

### Training Configuration

```yaml
# Training settings
num_epochs: 50
batch_size: 16                    # Per GPU
train_full_image: true            # No random crop
use_sliding_window: false         # Full-frame inference

# Optimization
learning_rate: 0.001
weight_decay: 0.0001
warmup_epochs: 5
scheduler: "warmupcosinelr"

# Data augmentation
train_augment: true
flip_prob: 0.5
translation_range: [0.0, 0.1]
rotation_range: [-3.14, 3.14]
```

### Sensor Specifications

```yaml
# SemanticKITTI sensor params
sensor:
  fov_up: 3.0
  fov_down: -25.0
  proj_h: 64
  proj_w: 2048
  max_range: 50.0
  min_range: 3.0
```

## Development Commands

### Environment Setup

```bash
pip install -r requirements.txt
pip install nuscenes-devkit
```

**Key dependencies:**
- PyTorch >= 1.10
- timm (for ViT models)
- selective_scan_cuda (for TinyViM SSM operations - requires CUDA)
- MLflow (for experiment tracking)
- nuscenes-devkit (for nuScenes dataset)

### Training

**SemanticKITTI with TinyViM:**
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63545 \
    --use_env main.py config/kitti/main/config.yaml \
    --data_root /path/to/semantic_kitti/dataset/sequences/ \
    --save_path ./logs/tinyvim_kitti \
    --pretrained_model /path/to/tinyvim_checkpoint.pth
```

**nuScenes with standard ViT:**
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63545 \
    --use_env main.py config/nusc/config.yaml \
    --data_root /path/to/nuscenes/ \
    --save_path ./logs/rangevit_nusc \
    --pretrained_model /path/to/vit_checkpoint.pth
```

### Evaluation

**Validation:**
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63545 \
    --use_env main.py config/kitti/main/config.yaml \
    --data_root /path/to/dataset/ \
    --save_path ./logs/eval \
    --checkpoint /path/to/trained_model.pth \
    --val_only
```

**Test set (SemanticKITTI):**
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63545 \
    --use_env main.py config/kitti/main/config.yaml \
    --data_root /path/to/dataset/ \
    --save_path ./logs/test \
    --checkpoint /path/to/trained_model.pth \
    --test_split --save_eval_results
```

### Testing Integration

**Test TinyViM model instantiation:**
```bash
python tests/test_tinyvim_integration.py
```

**Test inference on real range image:**
```bash
python tests/test_range_image_plot.py
```

## Key Implementation Details

### 1. Input Channel Adaptation (RGB → LiDAR)

Standard vision models expect 3 RGB channels. RangeViT uses 5 LiDAR channels.

**Solution:** Modify the first conv/embedding layer:
```python
# models/tinyvim/tinyvim.py
if in_channels != 3:
    self.stem[0] = Conv2d_BN(in_channels, self.width[0], ...)
```

### 2. CLS Token Compatibility

- **ViT:** Has CLS token at position 0 for classification
- **TinyViM:** No CLS token (pure spatial features)

**Solution:** Add dummy CLS token in adapter:
```python
# models/tinyvim_adapter.py
B, N, C = x.shape
cls_token = torch.zeros(B, 1, C, device=x.device)
x = torch.cat([cls_token, x], dim=1)  # [B, 1+N, C]
```

Decoders remove it: `x[:, num_extra_tokens:]`

### 3. Pretrained Model Transfer

TinyViM can use ImageNet-pretrained weights (RGB → LiDAR transfer):

```yaml
pretrained_model: /path/to/tinyvim_imagenet.pth
```

**Weight adaptation:**
- RGB stem (3 channels) → LiDAR stem (5 channels)
- Strategy: Inflate 3→5 by replicating + scaling, or random init new channels
- Rest of the model (stages, blocks) loads directly

### 4. Sliding Window vs Full-Image Inference

**Full-image mode** (current default):
```yaml
train_full_image: true
use_sliding_window: false
```
- Processes entire range image at once
- More memory but faster
- Better for small images (64×2048)

**Sliding window mode:**
```yaml
train_full_image: false
use_sliding_window: true
window_size: [64, 512]
window_stride: [64, 256]
```
- Processes overlapping crops
- Lower memory, slower
- Better for very large images

### 5. Aspect Ratio and Wrap-Around

Range images have **extreme aspect ratio** (64×2048 ≈ 1:32) due to:
- **Height:** Vertical laser channels (64 for Velodyne HDL-64E)
- **Width:** Azimuth bins (360° coverage, discretized)

**Horizontal wrap-around:** Width dimension is **circular** (azimuth):
- Left edge connects to right edge
- Data augmentation (flip) wraps around correctly

### 6. Loss Functions

**Focal Loss:**
- Handles class imbalance (e.g., road vs. pedestrian)
- Focuses on hard examples

**Lovász-Softmax:**
- Optimizes IoU directly
- Better for segmentation metrics

Combined loss: `λ_focal * FocalLoss + λ_lovasz * LovászLoss`

### 7. Distributed Training

Uses PyTorch `DistributedDataParallel`:
```python
model = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[args.gpu], find_unused_parameters=False
)
```

**Important:** Set `CUDA_VISIBLE_DEVICES` or `--nproc_per_node` correctly.

## Common Workflows

### Adding a New Backbone

1. **Create model file:** `models/my_backbone.py`
2. **Create adapter:** `models/my_backbone_adapter.py` (if needed)
3. **Register in RangeViT:** Edit `models/rangevit.py`:
   ```python
   if 'my_backbone' in args.vit_backbone:
       from models.my_backbone_adapter import MyBackboneAdapter
       self.encoder = MyBackboneAdapter(...)
   ```
4. **Add config:** Create `config_my_backbone.yaml`
5. **Test:** Write `tests/test_my_backbone.py`

### Adding a New Decoder

1. **Create decoder class:** Add to `models/decoders.py` or new file
2. **Register in RangeViT:** Edit `models/rangevit.py`:
   ```python
   if args.decoder == 'my_decoder':
       self.decoder = MyDecoder(...)
   ```
3. **Update config:** Add `decoder: "my_decoder"` in YAML

### Modifying TinyViM Block Structure

**File to edit:** `models/tinyvim/tvimblock.py`

Example: Change SSM expansion ratio:
```python
class TViMBlock(nn.Module):
    def __init__(self, dim, ...):
        self.expand = 2  # Change from 2 to 4 for more capacity
        self.ssm = SelectiveScan(dim * self.expand, ...)
```

### Adjusting Stem Stride

**File to edit:** `models/tinyvim_adapter.py`

Example: Change from (1,2) to (2,2):
```python
self.model = TinyViM(
    stem_stride=(2, 2),  # Now downsamples both H and W
    ...
)
```

**Caution:** Affects all downstream dimensions and memory usage.

### Enabling/Disabling Post-Processing

**In config YAML:**
```yaml
# Option 1: KNN post-processing
point_postproc: "knn"
knn_neighbors: 5

# Option 2: KPConv refiner
use_kpconv: true
point_postproc: "kpconv"

# Option 3: No post-processing
use_kpconv: false
point_postproc: null
```

## Current Branch: `feat/tinyvim`

**Recent changes:**
- Changed stem stride to `(1, 2)` for better height preservation (commit 46a0094)
- Optimized stage downsample strategy for memory efficiency
- Integrated FPN decoder for multi-scale feature fusion
- Added KNN post-processing option
- Enabled full-image training mode (no random crops)

**Testing status:**
- TinyViM model instantiation: ✓ Tested
- Forward pass with dummy data: ✓ Tested
- Inference on real range image: ✓ Tested (see `tests/test_range_image_plot.py`)

## Troubleshooting

### CUDA Out of Memory

**Solutions:**
1. Reduce batch size: `batch_size: 8` (in config)
2. Enable gradient checkpointing (if implemented)
3. Use sliding window mode: `use_sliding_window: true`
4. Reduce model size: `tinyvim_small` instead of `tinyvim_base`

### `selective_scan_cuda` Import Error

TinyViM requires selective scan CUDA kernels (from Mamba).

**Solutions:**
1. Install mamba-ssm: `pip install mamba-ssm`
2. Or compile from source: See TinyViM/Mamba repo instructions
3. Ensure CUDA toolkit matches PyTorch CUDA version

### Pretrained Model Shape Mismatch

When loading RGB-pretrained model for LiDAR (3→5 channels):

**Solutions:**
1. Set `strict=False` in `load_state_dict()`
2. Manually inflate stem weights (see `models/tinyvim_adapter.py`)
3. Or train from scratch (slower convergence)

### NaN Loss During Training

**Common causes:**
1. Learning rate too high → Reduce by 10×
2. Gradient explosion → Enable gradient clipping
3. Empty range image regions → Check data preprocessing

## File Reference (Critical Paths)

**Entry points:**
- `main.py:L1` - Main training/eval script
- `train.py:L1` - Trainer class

**Model core:**
- `models/rangevit.py:L1` - RangeViT wrapper
- `models/tinyvim_adapter.py:L1` - TinyViM integration
- `models/tinyvim/tinyvim.py:L1` - TinyViM model
- `models/tinyvim/tvimblock.py:L1` - SSM blocks

**Data:**
- `dataset/range_view_loader.py:L1` - Main data loader
- `dataset/semantic_kitti/parser.py:L1` - SemanticKITTI parser

**Configuration:**
- `option.py:L1` - Config parsing
- `config/kitti/main/config.yaml:L1` - SemanticKITTI main config
- `config/nusc/config.yaml:L1` - nuScenes config

## Research Context

**RangeViT paper:** arXiv:2301.10222 (CVPR 2023)
- First work to effectively transfer RGB-pretrained ViTs to LiDAR segmentation
- Key insight: ConvStem + Skip decoder + 3D refiner

**TinyViM paper:** arXiv:2411.17473
- Frequency decoupling for efficient Mamba-style vision models
- Laplace mixer: Low-freq → SSM, high-freq → Conv
- Achieves better efficiency-accuracy tradeoff than CNN/ViT at tiny scale

**This integration:** Combines RangeViT's projection-based pipeline with TinyViM's efficient state-space modeling for long-range context in autonomous driving scenarios.

## Citation

If using this codebase, please cite both papers:

```bibtex
@inproceedings{banerjee2023rangevit,
  title={RangeViT: Towards Vision Transformers for 3D Semantic Segmentation in Autonomous Driving},
  author={Banerjee, Angelina and others},
  booktitle={CVPR},
  year={2023}
}

@article{tinyvim2024,
  title={TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba},
  author={Authors},
  journal={arXiv preprint arXiv:2411.17473},
  year={2024}
}
```
