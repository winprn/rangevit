# Configuration Files

All experiment configurations are organized under this directory.

## Structure

```
config/
├── README.md                     ← This file
├── kitti/
│   ├── main/                    Main SemanticKITTI experiments
│   ├── ablation/                 Ablation studies
│   └── reproduce/                Paper reproduction configs
└── nusc/
    └── reproduce/                nuScenes experiment configs
```

---

## SemanticKITTI — `kitti/`

### Main experiments (`main/`)

| File | Backbone | Decoder | Epochs | Batch | Resolution | Augmentation | Notes |
|------|----------|---------|--------|-------|------------|-------------|-------|
| `config.yaml` | TinyViM-Base | FPN | 60 | 6 | 64×2048 full-frame | No | Base validation config |
| `config_aug.yaml` | TinyViM-Base | FPN | 60 | 6 | 64×2048 full-frame | Yes | Main paper result (65.96% val mIoU) |
| `config_tinyvim.yaml` | TinyViM-Base | FPN | 60 | 1 | 64×2048 full-frame | No | For test submission (train on 00–10) |
| `config_tinyvim_aug.yaml` | TinyViM-Base | FPN | 70 | 6 | 64×2048 full-frame | Yes | Extended training run |
| `config_trainval.yaml` | TinyViM-Base | FPN | 60 | 1 | 64×2048 full-frame | Yes | Test submission, train on 00–10 |

### Ablation studies (`ablation/`)

#### Backbone variants (`backbone/`)

| File | Backbone | Decoder | Resolution | Notes |
|------|----------|---------|------------|-------|
| `config_tinyvim_small.yaml` | TinyViM-Small | FPN | 64×2048 full-frame | 8.49M params, 65.00% val mIoU |
| `config_tinyvim_large.yaml` | TinyViM-Large | FPN | 64×2048 full-frame | 24.28M params, 65.07% val mIoU |
| `config_metaformer.yaml` | caformer/convformer | FPN | 64×2048 full-frame | Backbone comparison |

#### Decoder variants (`decoder/`)

| File | Decoder | Notes |
|------|---------|-------|
| `config_fpn_gated.yaml` | FPN-Gated | Learnable weighted fusion of stage features |
| `config_fpn_gated_detail.yaml` | FPN-Gated-Detail | Adds shallow detail reinjection branch |
| `config_fpn_residual.yaml` | FPN-Residual | Cross-stage residual connection |
| `config_fpn_cross_attn.yaml` | FPN-CrossAttn | Cross-attention gate instead of 1×1 conv gate |
| `config_fpn_residual_cross_attn.yaml` | FPN-Residual-CrossAttn | Both residual + cross-attention |
| `config_fuse_aux.yaml` | Fuse-Aux | Simple concatenation + refinement baseline |

#### Sliding window studies (`window/`)

| File | Resolution | Mode | Notes |
|------|------------|------|-------|
| `config_64x1024.yaml` | 64×1024 | Sliding window | Overlapping (stride 512) and non-overlapping |
| `config_64x512.yaml` | 64×512 | Sliding window | 64×512 crops with overlap |

#### Robustness sensitivity (`robustness/`)

| File | Corruption | Severity |
|------|-----------|----------|
| `config_robust_point_dropout.yaml` | Projected-pixel dropout (p=0.10) | Validation-time only |
| `config_robust_beam_dropout.yaml` | Beam dropout (p=0.10) | Validation-time only |
| `config_robust_range_noise.yaml` | Normalized range-coordinate noise (σ=0.03) | Validation-time only |

### Paper reproduction (`reproduce/`)

| File | Purpose | Expected Result |
|------|---------|----------------|
| `config_knn7.yaml` | Main config with KNN search window = 7 (paper value) | ~65.96% val mIoU |

---

## nuScenes — `nusc/`

| File | Backbone | Decoder | Epochs | Batch | Resolution | Notes |
|------|----------|---------|--------|-------|------------|-------|
| `config.yaml` | ViT-Small | UpConv | 150 | 8 | 32×384 sliding | Original RangeViT baseline |
| `config_tinyvim.yaml` | TinyViM-Base | FPN | 120 | 8 | 32×2048 full-frame | TinyViM on nuScenes |
| `config_full.yaml` | TinyViM-Base | FPN | 120 | 4 | 32×2048 full-frame | Main paper result (76.88% val mIoU) |
| `config_swin.yaml` | Swin-Small | UpConv | 150 | 8 | 32×384 sliding | Swin Transformer variant |

---

## Usage

### Training

```bash
# SemanticKITTI main experiment
python main.py config/kitti/main/config_aug.yaml \
    --data_root <SEMANTIC_KITTI_ROOT>/sequences/ \
    --save_path ./logs/rangevim_kitti

# nuScenes
python main.py config/nusc/config_full.yaml \
    --data_root <NUSCENES_ROOT> \
    --save_path ./logs/rangevim_nusc
```

### Evaluation

```bash
python main.py config/kitti/main/config_aug.yaml \
    --data_root <SEMANTIC_KITTI_ROOT>/sequences/ \
    --save_path ./logs/eval \
    --checkpoint <CHECKPOINT_PATH> \
    --val_only
```

### Profiling

```bash
python tools/profile_metrics.py config/kitti/main/config.yaml \
    --device cuda --amp --validation_style \
    --batch_size 1 --warmup 20 --iters 50
```

---

## Key hyperparameters (matching the paper)

| Parameter | SemanticKITTI | nuScenes |
|-----------|-------------|----------|
| Epochs | 60 | 120 |
| Batch size (per GPU) | 6 | 4 |
| Peak learning rate | 3e-4 | 3e-4 |
| Warmup epochs | 6 | 10 |
| KNN search window | **7** | **7** |
| KNN k neighbors | 5 | 5 |
| KNN Gaussian σ | 1.0 | 1.0 |
| KNN cutoff | 1.0 | 1.0 |
