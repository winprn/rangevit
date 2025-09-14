# Swin Transformer Training Commands for RangeViT

This document provides ready-to-use training commands for RangeViT with Swin Transformer backbones.

## Prerequisites

1. **Pretrained Models**: Place Swin pretrained models in `./pretrained_model/`:
   - `swin_tiny_patch4_window7_224_22kto1k_finetune.pth`
   - `swinv2_tiny_patch4_window16_256.pth` (optional)

2. **Dataset Preparation**: Set up your dataset paths according to the original RangeViT instructions.

## Configuration Files

- `config/config_swin_nusc.yaml` - Swin Transformer for nuScenes
- `config/config_swin_kitti.yaml` - Swin Transformer for SemanticKITTI
- `config/config_swinv2_nusc.yaml` - SwinV2 for nuScenes (experimental)

## Training Commands

### nuScenes with Swin Transformer

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63545 \
    --use_env main.py 'config/config_swin_nusc.yaml' \
    --data_root '<path_to_nuscenes_dataset>' \
    --save_path '<path_to_log>' \
    --pretrained_model './pretrained_model/swin_tiny_patch4_window7_224_22kto1k_finetune.pth'
```

### SemanticKITTI with Swin Transformer

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63545 \
    --use_env main.py 'config/config_swin_kitti.yaml' \
    --data_root '<path_to_semantic_kitti_dataset>/sequences/' \
    --save_path '<path_to_log>' \
    --pretrained_model './pretrained_model/swin_tiny_patch4_window7_224_22kto1k_finetune.pth'
```

### Alternative: SwinV2 for nuScenes (Experimental)

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63545 \
    --use_env main.py 'config/config_swinv2_nusc.yaml' \
    --data_root '<path_to_nuscenes_dataset>' \
    --save_path '<path_to_log>' \
    --pretrained_model './pretrained_model/swinv2_tiny_patch4_window16_256.pth'
```

## Evaluation Commands

### nuScenes Validation

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63545 \
    --use_env main.py 'config/config_swin_nusc.yaml' \
    --data_root '<path_to_nuscenes_dataset>' \
    --save_path '<path_to_log>' \
    --checkpoint '<path_to_trained_swin_model.pth>' \
    --val_only
```

### SemanticKITTI Validation

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63545 \
    --use_env main.py 'config/config_swin_kitti.yaml' \
    --data_root '<path_to_semantic_kitti_dataset>' \
    --save_path '<path_to_log>' \
    --checkpoint '<path_to_trained_swin_model.pth>' \
    --val_only
```

### SemanticKITTI Test Split

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63545 \
    --use_env main.py 'config/config_swin_kitti.yaml' \
    --data_root '<path_to_semantic_kitti_dataset>' \
    --save_path '<path_to_log>' \
    --checkpoint '<path_to_trained_swin_model.pth>' \
    --val_only --test_split --save_eval_results
```

## Key Configuration Differences from ViT

### Model Changes
- **Backbone**: `swin_tiny_patch4_window7_224` instead of `vit_small_patch16_384`
- **Skip connections**: Disabled (`skip_filters: 0`) since Swin doesn't provide ConvStem skip
- **Weight loading**: `reuse_pos_emb: false` (Swin uses relative position bias)

### Expected Performance
Based on the implementation:
- **nuScenes**: Expected ~75-76% mIoU (comparable to ViT-Small)
- **SemanticKITTI**: Expected ~60-62% mIoU (comparable to ViT-Small)

### Training Tips
1. **Learning Rate**: Start with the same LR as ViT (0.0008 for nuScenes, 0.0004 for KITTI)
2. **Memory Usage**: Swin may use more memory due to hierarchical features
3. **Convergence**: May converge slightly faster than ViT due to better inductive bias
4. **Batch Size**: Reduce if memory issues occur

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**: Ensure image sizes are divisible by 32 (Swin requirement)
2. **Memory Error**: Reduce batch size or use gradient checkpointing
3. **Weight Loading**: Use `strict=False` - some keys may not match between checkpoint formats

### Known Limitations

1. **Range Image Compatibility**: Current implementation works best with standard dimensions
2. **KPConv Integration**: Fully tested with `use_kpconv: false`
3. **Multi-Scale**: Phase 1 uses single-scale (F2 stage), multi-scale decoder available in future phases

## Performance Comparison

| Model | nuScenes mIoU | SemanticKITTI mIoU | Notes |
|-------|---------------|-------------------|-------|
| RangeViT + ViT-S/16 | 75.2% | 60.8% | Original |
| RangeViT + Swin-Tiny | ~75-76% | ~60-62% | Expected (Phase 1) |
| RangeViT + Swin-Tiny (Multi-scale) | ~76-78% | ~62-64% | Future (Phase 2) |

*Performance numbers are estimates based on backbone capabilities and will be updated with actual results.*