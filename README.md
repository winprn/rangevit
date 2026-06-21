# RangeViM

**Full-Resolution Range-View LiDAR Segmentation with a Lightweight Hybrid Vision Mamba Backbone**

Tuan-Kiet Huynh-Cao, Minh-Man Ly-Dinh, Ngoc-Thao Nguyen
Faculty of Information Technology, University of Science, VNU-HCM
**KES 2026**

[Paper](#) · [BibTeX](#citation) · [Google Drive (checkpoints)](#pretrained-models)

---

## Overview

RangeViM is a LiDAR semantic segmentation framework that processes range images with TinyViM, a lightweight hybrid backbone pairing convolutional blocks (local detail) with state-space blocks (long-range azimuthal context). This enables **direct full-frame inference** on the complete 64x2048 range image without sliding-window decomposition, preserving 360-degree scene context.

See the [paper](#) for full method details. This README focuses on **reproducing the results**.

---

## Key Results

| Dataset | Metric | Result |
|---------|--------|--------|
| SemanticKITTI test (single-scan) | mIoU | **67.8%** |
| nuScenes validation | mIoU | **76.88%** |

### SemanticKITTI per-class IoU (%)

| Class | IoU | Class | IoU |
|-------|-----|-------|-----|
| car | 95.9 | road | 93.1 |
| bicycle | 55.5 | parking | 73.4 |
| motorcycle | 50.8 | sidewalk | 80.0 |
| truck | 40.9 | other-ground | 35.6 |
| other-vehicle | 51.8 | building | 92.9 |
| pedestrian | 71.5 | fence | 71.3 |
| bicyclist | 64.2 | vegetation | 85.6 |
| motorcyclist | 52.7 | trunk | 70.0 |
| trunk | 70.8 | terrain | 63.2 |
| traffic sign | 68.3 | | |

### nuScenes per-class IoU (%)

| Class | IoU | Class | IoU |
|-------|-----|-------|-----|
| barrier | 74.80 | driveable surface | 96.46 |
| bicycle | 45.68 | other-flat | 71.87 |
| bus | 93.77 | sidewalk | 75.41 |
| car | 90.05 | terrain | 74.00 |
| construction vehicle | 54.19 | man-made | 88.53 |
| motorcycle | 83.36 | vegetation | 86.36 |
| pedestrian | 74.66 | | |
| cone | 64.80 | | |
| trailer | 72.30 | | |
| truck | 83.76 | | |

---

## Installation

### 1. Install Python dependencies

**CUDA 12.1 (verified on WSL / Python 3.10):**

```bash
# Install PyTorch with CUDA 12.1 support
python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

**mamba-ssm** (state-space operations — required for TinyViM):

```bash
python -m pip install mamba-ssm==2.3.0 --no-build-isolation
python -c "import torch; import selective_scan_cuda; print('OK:', torch.__version__, torch.version.cuda)"
```

If the `selective_scan_cuda` import fails, ensure your CUDA toolkit version matches PyTorch's built-in version and reinstall mamba-ssm from source.

For a fully pinned install:

```bash
python -m pip install -r requirements-cu121-lock.txt
python -m pip install mamba-ssm==2.3.0 --no-build-isolation
```

### 2. Clone and set up the repository

```bash
git clone https://github.com/<your-org>/rangevim.git
cd rangevim
```

### 3. Download pretrained TinyViM backbone weights

RangeViM uses TinyViM pretrained on ImageNet as the encoder backbone. Download the weights from the [TinyViM official repository](https://github.com/Zhangour24/TinyViM) and place them at a path of your choice, then pass the path via `--pretrained_model`.

---

## Dataset Preparation

### SemanticKITTI

1. Download the KITTI Odometry dataset (velodyne point clouds) and SemanticKITTI labels from [semantic-kitti.org](https://www.semantic-kitti.org).
2. Arrange as follows:

```
<SEMANTIC_KITTI_ROOT>/
└── sequences/
    ├── 00/  01/  ...  21/
    │   ├── velodyne/
    │   │   ├── 000000.bin
    │   │   ├── 000001.bin
    │   │   └── ...
    │   └── labels/          # not available for test sequences
    │       ├── 000000.label
    │       └── ...
```

**Official splits:** sequences 00–07, 09–10 for training; sequence 08 for validation; sequences 11–21 for test.

### nuScenes

1. Download the nuScenes dataset (v1.0-trainval) from [nuscenes.org](https://www.nuscenes.org/nuscenes.html).
2. Install the devkit:

```bash
pip install nuscenes-devkit
```

3. Point `data_root` in `config/nusc/config.yaml` to the nuScenes root directory.

---

## Training

### SemanticKITTI

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=63545 \
    --use_env \
    main.py config/kitti/main/config.yaml \
    --data_root <SEMANTIC_KITTI_ROOT>/sequences/ \
    --save_path ./logs/rangevim_kitti \
```

### nuScenes

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=63545 \
    --use_env \
    main.py config/nusc/config.yaml \
    --data_root <NUSCENES_ROOT> \
    --save_path ./logs/rangevim_nusc \
```

### Key training hyperparameters

| Parameter | SemanticKITTI | nuScenes |
|-----------|-------------|----------|
| Epochs | 60 | 120 |
| Batch size (per GPU) | 6 | 4 |
| Peak learning rate | 3e-4 | 3e-4 |
| Warmup epochs | 6 | 10 |
| Precision | AMP (FP16) | AMP (FP16) |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0.01 |

---

## Evaluation

### Validation

```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=63545 \
    --use_env \
    main.py config/kitti/main/config.yaml \
    --data_root <SEMANTIC_KITTI_ROOT>/sequences/ \
    --save_path ./logs/eval \
    --checkpoint <trained_model.pth> \
    --val_only
```

### Test-set prediction dump (SemanticKITTI)

```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=63545 \
    --use_env \
    main.py config/kitti/main/config_trainval.yaml \
    --data_root <SEMANTIC_KITTI_ROOT>/sequences/ \
    --save_path ./logs/test \
    --checkpoint <trained_model.pth> \
    --test_split --save_eval_results
```

### KNN post-processing

Point-wise predictions are recovered from range-image outputs via KNN-based post-processing. The settings used in the paper:

| Parameter | Value |
|----------|-------|
| Search window | 7 |
| k neighbors | 5 |
| Gaussian sigma | 1.0 |
| Cutoff | 1.0 |

These are configured in the YAML files under `knn_search`, `knn_k`, `knn_sigma`, and `knn_cutoff`.

---

## Ablation Highlights

### Backbone capacity (SemanticKITTI val, A100)

| Backbone | mIoU (%) | Params (M) | GFLOPs | Latency (ms) | VRAM (MB) |
|----------|----------|------------|--------|--------------|-----------|
| RangeViT (ViT) | 60.80 | 26.99 | 1873.80 | 97.97 | 995.76 |
| TinyViM-Small | 65.00 | 8.49 | 726.01 | 32.51 | 1032.42 |
| **TinyViM-Base** | **65.96** | **13.83** | **952.47** | **38.50** | **1083.79** |
| TinyViM-Large | 65.07 | 24.28 | 1415.38 | 39.94 | 1174.71 |

**TinyViM-Base** is the recommended operating point — highest accuracy among TinyViM variants with moderate cost.

### Component ablations (SemanticKITTI val, TinyViM-Base)

| Setting | Point mIoU (%) |
|---------|---------------|
| Full model + range-image augmentation + FPN | **65.96** |
| Without range-image augmentation | 63.18 |
| Without FPN (simple fusion decoder) | 64.76 |

Range-image augmentation contributes **+2.78 pp** and the FPN contributes **+1.20 pp** relative to a simple fusion decoder.

### Inference window size (SemanticKITTI val)

| Window | Mode | Stride | Point mIoU (%) |
|---------|------|--------|----------------|
| **64x2048** | **Full frame** | — | **65.96** |
| 64x1024 | Non-overlap | 1024 | 62.38 |
| 64x1024 | Overlap | 512 | 65.48 |
| 64x512 | Non-overlap | 512 | 62.33 |
| 64x512 | Overlap | 256 | 62.61 |

Full-frame inference outperforms all sliding-window settings. Overlapping windows recover some accuracy but at higher latency and memory cost.

### Robustness to input corruptions (SemanticKITTI val)

| Corruption | Point mIoU (%) |
|------------|---------------|
| Clean | 65.96 |
| Projected-pixel dropout (p=0.10) | 65.37 |
| Beam dropout (p=0.10) | 65.00 |
| Normalized range-coordinate noise (sigma=0.03) | 62.01 |

RangeViM is most sensitive to incoherent geometric feature perturbations.

---

## Pretrained Models

All checkpoints are hosted on [Google Drive](https://drive.google.com/drive/folders/17zkW0KQPqzc87A2Ws30D25QHIVFqzpW5?usp=sharing).

| Train data | Test data | mIoU (%) | Download |
|-----------|-----------|----------|----------|
| SemanticKITTI train | SemanticKITTI val | 65.96 | [SemanticKITTI](https://drive.google.com/drive/folders/17zkW0KQPqzc87A2Ws30D25QHIVFqzpW5?usp=sharing) |
| SemanticKITTI train+val | SemanticKITTI test | 67.8 | [SemanticKITTI](https://drive.google.com/drive/folders/17zkW0KQPqzc87A2Ws30D25QHIVFqzpW5?usp=sharing) |
| nuScenes train | nuScenes val | 76.88 | [nuScenes](https://drive.google.com/drive/folders/17zkW0KQPqzc87A2Ws30D25QHIVFqzpW5?usp=sharing) |

---

## Repository Structure

```
rangevim/
├── config/                       Experiment configurations (see `config/README.md`)
│   ├── kitti/
│   │   ├── main/                Main experiments
│   │   ├── ablation/            Ablation studies
│   │   └── reproduce/           Paper reproduction configs
│   └── nusc/
│       └── reproduce/           nuScenes experiment configs
├── main.py                        Training/evaluation entry point
├── train.py                       Trainer class
├── option.py                      Configuration parser
├── models/
│   ├── rangevit.py                Main model (encoder + decoder factory)
│   ├── tinyvim_adapter.py         TinyViM backbone wrapper
│   ├── tinyvim/
│   │   ├── tinyvim.py             TinyViM backbone (SSM + conv)
│   │   ├── tvimblock.py          SS2D / TViMBlock / FFN / RepDW
│   │   └── fpn_decoder.py         FPN and variant decoders
│   └── kpconv/                    Optional KPConv 3D refiner
├── dataset/
│   ├── range_view_loader.py       Range-image dataloader
│   ├── preprocess/
│   │   ├── projection.py          Spherical and scan-unfolded projection
│   │   ├── augmentor.py          Point-cloud augmentation
│   │   └── rangeaug.py           Range-image-level augmentation
│   ├── semantic_kitti/            SemanticKITTI parser
│   └── nuScenes/                 nuScenes parser
├── utils/
│   ├── optim/                     Focal, Lovasz, boundary losses; LR scheduler
│   ├── metrics/                  IoU, accuracy evaluation; TensorBoard logging
│   ├── postproc/knn.py           KNN point-cloud post-processing
│   └── tools/                    Recorder, MLflow logger, utilities
└── tools/
    ├── profile_metrics.py         Throughput / VRAM profiling
    └── calc_gflops.py            GFLOPs calculator
```

---

## Optional: MLflow Logging

MLflow is disabled by default. Enable it in the YAML config:

```yaml
mlflow:
  enable: true
  tracking_uri: "file:./mlruns"   # or your MLflow server URL
  experiment_name: "rangevim"
  run_name: "my_run"
```

---

## Acknowledgements

RangeViM builds on [RangeViT](https://github.com/valeoai/rangevit) (Ando et al., CVPR 2023) and the [TinyViM](https://github.com/Zhangour24/TinyViM) backbone (Ma et al., ICCV 2025). We also thank the authors of SalsaNext, Segmenter, timm, and the Mamba / selective-scan project.

---

## Citation

```bibtex
@inproceedings{huynhcao2026rangevim,
  title     = {{RangeViM}: Full-Resolution Range-View {LiDAR} Segmentation with a Lightweight Hybrid Vision Mamba Backbone},
  author    = {Huynh-Cao, Tuan-Kiet and Ly-Dinh, Minh-Man and Nguyen, Ngoc-Thao},
  booktitle = {Procedia Computer Science (KES)},
  year      = {2026}
}

@inproceedings{ando2023rangevit,
  title     = {{RangeViT}: Towards Vision Transformers for 3D Semantic Segmentation in Autonomous Driving},
  author    = {Ando, Angelika and Gidaris, Spyros and Bursuc, Andrei and Puy, Gilles and Boulch, Alexandre and Marlet, Renaud},
  booktitle = {CVPR},
  year      = {2023}
}
```

## License

See [`LICENSE`](LICENSE).
