# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RangeViT is a Vision Transformer-based approach for 3D semantic segmentation in autonomous driving, published at CVPR 2023. It processes LiDAR data as range images using Vision Transformers adapted for 3D semantic segmentation on nuScenes and SemanticKITTI datasets.

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
pip install nuscenes-devkit
```

### Training Commands
For nuScenes training:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63545 \
    --use_env main.py 'config_nusc.yaml' \
    --data_root '<path_to_nuscenes_dataset>' \
    --save_path '<path_to_log>' \
    --pretrained_model '<path_to_image_pretrained_model.pth>'
```

For SemanticKITTI training:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63545 \
    --use_env main.py 'config_kitti.yaml' \
    --data_root '<path_to_semantic_kitti_dataset>/dataset/sequences/' \
    --save_path '<path_to_log>' \
    --pretrained_model '<path_to_image_pretrained_model.pth>'
```

### Evaluation Commands
For validation:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63545 \
    --use_env main.py 'config_nusc.yaml' \
    --data_root '<path_to_dataset>' \
    --save_path '<path_to_log>' \
    --checkpoint '<path_to_pretrained_rangevit_model.pth>' \
    --val_only
```

For SemanticKITTI test evaluation, add `--test_split --save_eval_results` flags.

## Code Architecture

### Core Components
- **main.py**: Entry point handling distributed training setup and model building
- **train.py**: Contains the main Trainer class with training/validation loops
- **models/rangevit.py**: Core RangeViT model implementation
- **models/rangevit_kpconv.py**: RangeViT variant with KPConv 3D refiner
- **option.py**: Configuration parsing and argument handling

### Model Architecture
- **Stem**: Convolutional stem (`models/stems.py`) for patch embedding
- **Backbone**: ViT encoder with configurable patch sizes and pre-trained initialization
- **Decoder**: Either linear or up-convolution decoder (`models/decoders.py`)
- **3D Refiner**: Optional KPConv-based post-processing (`models/kpconv/`)

### Dataset Handling
- **dataset/range_view_loader.py**: Main data loader for range image processing
- **dataset/semantic_kitti/**: SemanticKITTI-specific parsers and configs
- **dataset/nuScenes/**: nuScenes-specific dataset handling
- **dataset/preprocess/**: Data augmentation and projection utilities

### Configuration System
Uses YAML configs (`config_*.yaml`) with these key sections:
- Model config (ViT backbone, patch sizes, decoder type)
- Training config (epochs, batch size, learning rate)
- Data augmentation parameters
- Sensor specifications (projection parameters, normalization)

### Pre-trained Model Support
Supports multiple ViT initialization strategies:
- ImageNet21k supervised (`"timmImageNet21k"`)
- Cityscapes segmentation pre-trained
- DINO self-supervised
- Random initialization

## Key Implementation Details

- Uses distributed training with `torch.distributed.launch`
- Range images are processed with sliding windows for memory efficiency
- Supports both ConvStem and standard patch embedding
- Optional KPConv 3D refiner for post-processing predictions
- Configurable skip connections between encoder and decoder