# Reproducing the RangeViM Paper Results

This document records the commands and expected outputs for the experiments reported in the KES 2026 RangeViM manuscript. It is intended as a reproducibility checklist for the current repository state.

The most important scripts are:

- `main.py`: training, validation, test prediction export, and point-level mIoU evaluation.
- `tools/profile_metrics.py`: parameters, FLOPs, latency, and peak inference VRAM.
- `utils/metrics/iou_eval.py`: mIoU computation from the confusion matrix.
- `utils/inference/infer_knn.py` and `utils/postproc/knn.py`: KNN point-level re-projection/post-processing.

Use the checkpoint-local config files when available, because some root-level configs may have changed during later revisions.

## 1. Environment

Install the project dependencies, CUDA-enabled PyTorch, and the TinyViM selective-scan extension used by the model. The profiling script will warn if the real selective-scan CUDA extension is not installed.

Example environment variables:

```bash
export CUDA_VISIBLE_DEVICES=0
export KITTI_ROOT=/path/to/SemanticKITTI
export NUSC_ROOT=/path/to/nuScenes
```

On Windows PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
$env:KITTI_ROOT="E:\path\to\SemanticKITTI"
$env:NUSC_ROOT="E:\path\to\nuScenes"
```

All validation commands below use batch size 1 during evaluation, KNN point post-processing where configured, and report point-level mIoU as `IOU_point`.

Unless otherwise stated, the paper experiments and profiling are conducted on NVIDIA A100 GPUs.

## 2. Main SemanticKITTI Validation Result

Paper value: `65.96%` point mIoU for TinyViM-Base, FPN, full-frame `64x2048`, with range-image augmentation and KNN post-processing.

```bash
python main.py config/kitti/main/config_aug.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_kitti_base" \
  --checkpoint checkpoint/kitti_base/train/best_miou_model.pth \
  --val_only
```

Expected final log line contains approximately:

```text
IOU_point 0.6596
```

The final paper reports this as `65.96%`.

## 3. SemanticKITTI Test Submission

Paper value: `67.8%` mIoU on the official SemanticKITTI test server.

The SemanticKITTI test labels are hidden, so this value cannot be computed locally. To reproduce the submitted prediction files, run test-split inference and upload the generated `sequences/*/predictions/*.label` files to the official benchmark server.

```bash
python main.py config/kitti/main/config_aug.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_kitti_test" \
  --checkpoint checkpoint/kitti_base/train/best_miou_model.pth \
  --val_only \
  --test_split \
  --save_eval_results
```

Notes:

- The final manuscript reports KNN search window `7`, `k=5`, `sigma=1.0`, and cutoff `1.0`.
- The official benchmark computes the final test mIoU.

## 4. nuScenes Validation Result

Paper value: `76.88%` point mIoU and `88.31%` mean accuracy on nuScenes validation.

```bash
python main.py config/nusc/config_full.yaml \
  --data_root "$NUSC_ROOT" \
  --save_path "runs/reproduce_nusc_base" \
  --checkpoint checkpoint/nuscene_base/best_miou_model.pth \
  --val_only
```

Expected final log contains approximately:

```text
Acc_point 0.8831 IOU_point 0.7688
```

The final paper reports this as `76.88%` mIoU and `88.31%` mean accuracy.

## 5. Backbone Capacity and Efficiency Ablation

Paper table:

| Model | Validation mIoU (%) | Params (M) | GFLOPs | Latency (ms) | Peak VRAM (MB) |
|---|---:|---:|---:|---:|---:|
| RangeViT | 60.8 | 26.99 | 1873.80 | `97.97 +/- 0.24` | 995.76 |
| TinyViM-Small | 65.00 | 8.49 | 726.01 | `32.51 +/- 3.33` | 1032.42 |
| TinyViM-Base | 65.96 | 13.83 | 952.47 | `38.50 +/- 3.79` | 1083.79 |
| TinyViM-Large | 65.07 | 24.28 | 1415.38 | `39.94 +/- 0.25` | 1174.71 |

### 5.1 Validation mIoU

TinyViM-Small:

```bash
python main.py config/kitti/ablation/backbone/config_tinyvim_small.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_kitti_small" \
  --checkpoint checkpoint/kitti_small/best_miou_model.pth \
  --val_only
```

TinyViM-Base:

```bash
python main.py config/kitti/main/config_aug.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_kitti_base" \
  --checkpoint checkpoint/kitti_base/train/best_miou_model.pth \
  --val_only
```

TinyViM-Large:

```bash
python main.py config/kitti/ablation/backbone/config_tinyvim_large.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_kitti_large" \
  --checkpoint checkpoint/kitti_large/best_miou_model.pth \
  --val_only
```

RangeViT baseline uses the original RangeViT configuration/checkpoint if available locally. In the paper, the RangeViT validation mIoU and profile are used as an efficiency reference.

### 5.2 Parameters, GFLOPs, latency, and peak VRAM

Use `tools/profile_metrics.py`. These measurements are network inference on projected tensors. Projection and KNN point post-processing are not included.

RangeViT:

```bash
python tools/profile_metrics.py config/kitti/main/config.yaml \
  --device cuda \
  --amp \
  --validation_style \
  --batch_size 1 \
  --warmup 20 \
  --iters 50
```

TinyViM-Small:

```bash
python tools/profile_metrics.py config/kitti/ablation/backbone/config_tinyvim_small.yaml \
  --device cuda \
  --amp \
  --validation_style \
  --batch_size 1 \
  --warmup 20 \
  --iters 50
```

TinyViM-Base:

```bash
python tools/profile_metrics.py config/kitti/main/config_aug.yaml \
  --device cuda \
  --amp \
  --validation_style \
  --batch_size 1 \
  --warmup 20 \
  --iters 50
```

TinyViM-Large:

```bash
python tools/profile_metrics.py config/kitti/ablation/backbone/config_tinyvim_large.yaml \
  --device cuda \
  --amp \
  --validation_style \
  --batch_size 1 \
  --warmup 20 \
  --iters 50
```

Read these output fields:

```text
Parameters
FLOPs
Latency mean
Peak VRAM (inference)
```

The Table 4 latency values are summarized from profiler measurements on the shared A100 server. Because latency is sensitive to server load, use the same GPU and verify that repeated runs are taken under comparable conditions.

The JSON form is useful for scripts:

```bash
python tools/profile_metrics.py config/kitti/main/config_aug.yaml \
  --device cuda --amp --validation_style --batch_size 1 --warmup 20 --iters 50 --json
```

Relevant implementation:

- Params: `tools/profile_metrics.py::count_parameters`
- FLOPs: `tools/profile_metrics.py::profile_flops`
- Latency: `tools/profile_metrics.py::benchmark_latency`
- Peak VRAM: `tools/profile_metrics.py::measure_inference_vram`

## 6. Component Ablation

Paper table:

| Setting | Decoder | Point mIoU (%) |
|---|---|---:|
| Full model with range-image augmentation | FPN | 65.96 |
| Without range-image augmentation | FPN | 63.18 |
| Full model with range-image augmentation | Fuse-Aux | 64.76 |

With range-image augmentation:

```bash
python main.py config/kitti/main/config_aug.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_aug_on" \
  --checkpoint checkpoint/kitti_base/train/best_miou_model.pth \
  --val_only
```

Without range-image augmentation:

```bash
python main.py config/kitti/main/config_tinyvim.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_aug_off" \
  --checkpoint checkpoint/kitti_base_noaug/best_miou_model.pth \
  --val_only
```

Fuse-Aux decoder ablation:

```bash
python main.py config/kitti/ablation/decoder/config_fuse_aux.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_fuse_aux" \
  --checkpoint /path/to/fuse_aux/best_miou_model.pth \
  --val_only
```

Expected validation point metric: `IOU_point ~= 0.6476`.

This Fuse-Aux run is a decoder-topology ablation against the FPN setting. The config keeps the same TinyViM-Base backbone, full-frame `64x2048` input, range-image augmentation, and final KNN settings (`knn_search: 7`, `knn_k: 5`, `knn_sigma: 1.0`, `knn_cutoff: 1.0`). Technically, this decoder projects and upsamples TinyViM stage features to the highest-resolution stage, concatenates them across channels, and applies two convolutional refinement blocks before classification. It serves as a direct-fusion baseline; the FPN decoder improves over it by `1.20` percentage points through structured top-down multi-scale fusion.

Expected point mIoU:

- Augmentation on: `65.96%`
- Augmentation off: `63.18%`
- Fuse-Aux decoder: `64.76%`

## 7. Inference Window Ablation

Paper table:

| Model input window | Inference strategy | Point mIoU (%) | Params (M) | GFLOPs | Latency (ms) | Peak VRAM (MiB) |
|---|---|---:|---:|---:|---:|---:|
| `64x2048` | Full frame | 65.96 | 13.83 | 952.47 | `38.50 +/- 3.79` | 1083.79 |
| `64x1024` | Sliding window | 65.48 | 13.83 | 1428.71 | `61.69 +/- 3.75` | 1582.92 |
| `64x512` | Sliding window | 62.61 | 13.83 | 1666.83 | `73.78 +/- 5.34` | 1832.92 |

Full-frame `64x2048`:

```bash
python main.py config/kitti/main/config_aug.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_window_2048" \
  --checkpoint checkpoint/kitti_base/train/best_miou_model.pth \
  --val_only
```

Sliding-window `64x512`:

```bash
python main.py config/kitti/ablation/window/config_64x512.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_window_512" \
  --checkpoint checkpoint/kitti_base/64x512/best_miou_model.pth \
  --val_only
```

Sliding-window `64x1024`:

```bash
python main.py config/kitti/ablation/window/config_64x1024.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_window_1024" \
  --checkpoint /path/to/64x1024/best_miou_model.pth \
  --val_only
```

The current repository checkpoint folder does not include a visible `64x1024` checkpoint file. Use the checkpoint from the corresponding training run, or retrain this config.

Training command template:

```bash
python main.py config/kitti/ablation/window/config_64x1024.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/train_window_1024"
```

Profiling commands for the window ablation:

```bash
python tools/profile_metrics.py config/kitti/main/config_aug.yaml \
  --device cuda --amp --validation_style --batch_size 1 --warmup 50 --iters 200 --json

python tools/profile_metrics.py config/kitti/ablation/window/config_64x1024.yaml \
  --device cuda --amp --validation_style --batch_size 1 --warmup 50 --iters 200 --json

python tools/profile_metrics.py config/kitti/ablation/window/config_64x512.yaml \
  --device cuda --amp --validation_style --batch_size 1 --warmup 50 --iters 200 --json
```

Use `validation_style: true`, `selective_scan_mode: real`, and the same GPU for all rows. These numbers profile full-scan model execution; projection and KNN point post-processing are excluded. Latency is indicative on the shared A100 server and is reported as mean +/- standard deviation. For sliding-window rows, verify that the profiling JSON reports `use_sliding_window: true` and the expected `window_size`; otherwise the run did not measure the intended inference strategy.

## 8. Robustness Sensitivity Analysis

Paper table:

| Validation setting | Point mIoU (%) |
|---|---:|
| Clean | 65.96 |
| Projected-pixel dropout, `p=0.10` | `65.37 +/- 0.01` |
| Beam dropout, `p=0.10` | `65.00 +/- 0.09` |
| Normalized range-coordinate noise, `sigma=0.03` | `62.01 +/- 0.05` |

Clean:

```bash
python main.py config/kitti/main/config_aug.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_robust_clean" \
  --checkpoint checkpoint/kitti_base/train/best_miou_model.pth \
  --val_only
```

Projected-pixel dropout:

```bash
python main.py config/kitti/ablation/robustness/config_robust_point_dropout.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_robust_point_dropout" \
  --checkpoint checkpoint/kitti_base/train/best_miou_model.pth \
  --val_only
```

Beam dropout:

```bash
python main.py config/kitti/ablation/robustness/config_robust_beam_dropout.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_robust_beam_dropout" \
  --checkpoint checkpoint/kitti_base/train/best_miou_model.pth \
  --val_only
```

Normalized range-coordinate noise:

```bash
python main.py config/kitti/ablation/robustness/config_robust_range_noise.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_robust_range_noise" \
  --checkpoint checkpoint/kitti_base/train/best_miou_model.pth \
  --val_only
```

The robustness table is an input-sensitivity analysis only. Corruptions are applied to normalized range-image input features; labels and KNN re-projection geometry are unchanged.

## 9. Decoder/Fuse-Aux Ablation

The repository contains `config/kitti/ablation/decoder/config_fuse_aux.yaml` and `models/tinyvim/fuse_aux_decoder.py` for the simpler decoder ablation requested during review. The completed validation result is `64.76%` point mIoU.

Training command:

```bash
python main.py config/kitti/ablation/decoder/config_fuse_aux.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/train_fuse_aux"
```

Evaluation command:

```bash
python main.py config/kitti/ablation/decoder/config_fuse_aux.yaml \
  --data_root "$KITTI_ROOT" \
  --save_path "runs/reproduce_fuse_aux" \
  --checkpoint /path/to/fuse_aux/best_miou_model.pth \
  --val_only
```

Report this row only for the checkpoint evaluated under the final KNN setting used throughout the revised manuscript: search window 7, `k = 5`, `sigma = 1.0`, and cutoff 1.0.

## 10. Where Each Reported Quantity Comes From

| Paper quantity | Source |
|---|---|
| Validation mIoU | `main.py --val_only`, logged as `IOU_point` when KNN is enabled |
| Pixel mIoU | `train.py`, logged as `IOU` / `miou_pixel` |
| Point mIoU | `train.py`, `metrics_3d`, logged as `IOU_point` / `miou_point` |
| Class-wise IoU LaTeX row | `utils/metrics/eval_results.py`, printed as `Latext Format String` |
| Parameters | `tools/profile_metrics.py::count_parameters` |
| GFLOPs | `tools/profile_metrics.py::profile_flops` |
| Latency | `tools/profile_metrics.py::benchmark_latency` |
| Peak inference VRAM | `tools/profile_metrics.py::measure_inference_vram` |
| SemanticKITTI test mIoU | Official SemanticKITTI benchmark server |

## 11. Important Caveats

- Official SemanticKITTI test mIoU cannot be computed locally because test labels are hidden.
- Profiling numbers use dummy projected tensors and do not include projection or KNN post-processing latency.
- `tools/profile_metrics.py` notes that hook-based FLOPs may undercount custom TinyViM selective-scan kernels.
- Latency and VRAM depend on GPU model, CUDA version, PyTorch version, AMP setting, and whether the real selective-scan CUDA extension is installed.
- For paper consistency, use the checkpoint-local config files in `checkpoint/` where provided.
