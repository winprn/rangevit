# SemanticPOSS Dataset Integration — Design

**Date:** 2026-05-13
**Branch:** feat/semanticposs
**Goal:** Add SemanticPOSS as a first-class training and evaluation target for ablation studies, alongside the existing SemanticKITTI and nuScenes pipelines.

## Scope

Train and evaluate the existing TinyViM/RangeViT pipeline on SemanticPOSS from scratch. No cross-dataset evaluation, no shared label space with KITTI. Native 14-class POSS taxonomy, native sensor resolution (40 × 1800).

This is an additive integration — KITTI and nuScenes behavior must remain unchanged.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Train + eval from scratch | Cleanest ablation; no transfer-learning confounds |
| Projection resolution | Native 40 × 1800 | Faithful to sensor; matches POSS canonical view |
| Class space | Native 14 classes | Standard POSS benchmark protocol |
| Width stride budget | Total /8 (was /16 on KITTI) | 1800 is not divisible by 16; /8 makes 1800 → 225 cleanly |
| Stride placement | Skip the stage 0 → 1 Embedding downsample | First inter-stage transition keeps width unchanged; stages 1→2 and 2→3 keep /2 each |
| Train/val split | seq {00,01,02,04,05} train, seq 03 val | Standard SemanticPOSS protocol |
| Projection method | Tag-file scatter (use POSS `tag/*` masks) | Matches the dataset's canonical projection; 1-to-1 point→cell mapping |
| Augmentation | Disable point-cloud geometric aug for POSS | Tag mask is precomputed; geometric aug would invalidate it. Range-image 2D aug only. |
| Sensor stats (img_mean/std) | Compute from POSS train set via dedicated script | No public values exist; KITTI stats are wrong sensor |
| YAML location | `dataset/semantic_poss/semantic-poss.yaml` | Mirrors KITTI layout |

## Architecture

### Component map

```
config_poss.yaml                            (new, top-level config)
    │
    ├── data.dataset: "SemanticPOSS"
    │
    ▼
train.py                                    (new branch for SemanticPOSS)
    │
    ▼
dataset/semantic_poss/                      (new module)
    ├── __init__.py                         exports SemanticPOSS
    ├── parser.py                           SemanticPOSS class
    └── semantic-poss.yaml                  learning_map, content, class names
    │
    ▼
dataset/range_view_loader.py                (modified: projection mode switch)
    │
    ▼
dataset/preprocess/projection.py            (modified: tag-based projection helper)
    │
    ▼
models/tinyvim_adapter.py                   (modified: configurable inter-stage stride)
models/tinyvim/tinyvim.py                   (modified: skip-first-embedding-downsample option)
    │
    ▼
tools/compute_poss_stats.py                 (new: compute img_mean/img_stds)
```

### Data flow

```
POSS raw frame
    velodyne/<id>.bin        (N, 4 — x, y, z, intensity)
    labels/<id>.label        (N — sem|inst packed)
    tag/<id>.tag             (40*1800 bool — which cells are occupied)
        │
        ▼
SemanticPOSS.loadDataByIndex(idx)
    returns: points (N, 4), sem_label (N,), inst_label (N,), tag (H*W,)
        │
        ▼
RangeViewLoader (POSS branch)
    scatter via tag mask:
        proj_image[H*W, 5] = 0
        proj_image[tag] = stack(range, x, y, z, intensity)
        reshape → (5, H, W)
    apply 2D range-image aug only
        │
        ▼
TinyViM encoder (stride config: stem /2 + stage1→2 /2 + stage2→3 /2 = /8)
    input  40 × 1800
    stem   40 × 900
    stage0 40 × 900
    stage1 40 × 900    ← first Embedding skips width downsample
    stage2 40 × 450
    stage3 40 × 225
        │
        ▼
FPN decoder → 14 logits per pixel → KNN postproc → 3D labels
```

## Detailed component design

### 1. `dataset/semantic_poss/parser.py` — `SemanticPOSS` class

Public interface mirrors `SemanticKitti` exactly so `train.py` can swap in/out:

- `__init__(root, sequences, config_path, has_label=True)`
- `loadDataByIndex(idx)` → `(pointcloud, sem_label, inst_label, tag)` — returns one extra item versus KITTI (the tag mask). Loader handles either tuple shape.
- `loadLabelByIndex(idx)` → `(sem_label, inst_label)`
- `labelMapping(label)` → mapped label via `class_map_lut`
- `parsePathInfoByIndex(idx)` → `(seq_id, frame_id)`
- `class_map_lut`, `class_map_lut_inv`, `cls_freq`, `mapped_cls_name` populated from YAML

POSS frames are stored as `sequences/<id>/velodyne/<frame>.bin`, `sequences/<id>/labels/<frame>.label`, `sequences/<id>/tag/<frame>.tag`. The parser sorts files per sequence and asserts triplet alignment.

`readTag(path)` reads a uint8/bool file of length 40 × 1800 and returns a `np.bool_` array.

### 2. `dataset/semantic_poss/semantic-poss.yaml`

```yaml
name: "semantic-poss"
labels:
  0: "unlabeled"
  4: "person"
  5: "person (2+)"
  6: "rider"
  7: "car"
  8: "trunk"
  9: "plants"
  10: "traffic-sign-1"
  11: "traffic-sign-2"
  12: "traffic-sign-3"
  13: "pole"
  14: "trashcan"
  15: "building"
  16: "cone/stone"
  17: "fence"
  21: "bike"
  22: "ground"

learning_map:
  0: 0
  4: 1
  5: 1
  6: 2
  7: 3
  8: 4
  9: 5
  10: 6
  11: 6
  12: 6
  13: 7
  14: 8
  15: 9
  16: 10
  17: 11
  21: 12
  22: 13

learning_map_inv:
  0: 0
  1: 4
  2: 6
  3: 7
  4: 8
  5: 9
  6: 10
  7: 13
  8: 14
  9: 15
  10: 16
  11: 17
  12: 21
  13: 22

mapped_class_name:
  0: "unlabeled"
  1: "person"
  2: "rider"
  3: "car"
  4: "trunk"
  5: "plants"
  6: "traffic-sign"
  7: "pole"
  8: "trashcan"
  9: "building"
  10: "cone/stone"
  11: "fence"
  12: "bike"
  13: "ground"

content:  # filled in by compute_poss_stats.py; placeholder uniform values acceptable for first run
  0: 0.01
  4: 0.005
  5: 0.005
  6: 0.005
  7: 0.05
  8: 0.02
  9: 0.30
  10: 0.002
  11: 0.002
  12: 0.002
  13: 0.01
  14: 0.002
  15: 0.20
  16: 0.002
  17: 0.05
  21: 0.005
  22: 0.32

split:
  train: [0, 1, 2, 4, 5]
  val: [3]
  test: []
```

### 3. `dataset/range_view_loader.py` — projection mode switch

Add a `projection_mode` setting (read from `sensor.projection_mode`, default `"spherical"`). When `"tag"`, the loader expects the parser to return a tag mask and constructs the range image by scatter:

```python
def _project_tag(self, points, tag, H, W):
    # points: (N, 4) — x, y, z, intensity
    # tag:    (H*W,) bool, with tag.sum() == N
    img = np.zeros((H * W, 5), dtype=np.float32)
    rng = np.linalg.norm(points[:, :3], axis=1)
    feats = np.stack([rng, points[:, 0], points[:, 1], points[:, 2], points[:, 3]], axis=1)
    img[tag] = feats
    return img.reshape(H, W, 5)
```

When `projection_mode == "tag"`:
- Skip `Augmentor` (point-cloud geometric aug) entirely
- Skip `PolarMix`, `InstanceCutMix`, `InstanceCopy`, `ClusterMix` (all operate on points and re-project)
- Keep range-image-level 2D aug (`range_aug` if enabled)
- Skip `PointSampler` (would invalidate tag length)

The loader detects which projection mode applies based on config; KITTI/nuScenes paths are untouched (their default is `"spherical"`).

### 4. `models/tinyvim/tinyvim.py` and `models/tinyvim_adapter.py` — stride config

Add a new construction arg, `stage_embedding_strides`, a list of 3 (h, w) tuples controlling the Embedding layers between stages 0→1, 1→2, 2→3. Defaults to `[(1,2), (1,2), (1,2)]` (current behavior). POSS config passes `[(1,1), (1,2), (1,2)]`.

In `tinyvim.py`, wherever the `Embedding` is constructed between stages, read the corresponding tuple from this list rather than hard-coding `stride=(1,2)`.

`TinyViMAdapter` forwards the arg through `model.create_tinyvim(...)`. The YAML key lives under `model.stage_embedding_strides` so it's user-configurable per dataset.

### 5. `config_poss.yaml` — top-level training config

Clone `config_kitti.yaml`, then change:

```yaml
data:
  dataset: "SemanticPOSS"
  n_classes: 14
  data_root: "../dataset/SemanticPOSS/sequences"

model:
  vit_backbone: "tinyvim_base"
  image_size: [40, 1800]
  window_size: [40, 1800]
  window_stride: [40, 1800]
  original_image_size: [40, 1800]
  patch_size: [1, 2]               # documents the effective stem stride
  patch_stride: [1, 2]
  stage_embedding_strides:         # /8 total width stride: stem /2 + skip + /2 + /2
    - [1, 1]
    - [1, 2]
    - [1, 2]

sensor:
  name: "Pandora40"
  type: "spherical"                # informational; projection mode below overrides
  projection_mode: "tag"           # NEW key consumed by loader
  proj_h: 40
  proj_w: 1800
  fov_up: 7.0
  fov_down: -16.0
  fov_left: -180
  fov_right: 180
  img_mean: [0.0, 0.0, 0.0, 0.0, 0.0]   # placeholder; replace with output of compute_poss_stats.py
  img_stds: [1.0, 1.0, 1.0, 1.0, 1.0]   # placeholder

# Disable geometric aug — incompatible with tag-based projection
augmentation:
  p_flipx: 0.0
  p_flipy: 0.0
  p_transx: 0.0
  p_transy: 0.0
  p_transz: 0.0
  p_rot_roll: 0.0
  p_rot_pitch: 0.0
  p_rot_yaw: 0.0

# Disable mix augmentations — also incompatible with tag-based projection
adapted_augmentation:
  use_mapped_labels: true
  pointsample: { enable: false }
  polarmix: { enable: false }
  instance_cutmix: { enable: false }
  clustermix: { enable: false }
  instance_copy: { enable: false }

# Optional: range_aug stays available
range_aug: false
```

Adjust `training.loss.focal_loss.ignore_index: 0` (already correct for POSS — index 0 is unlabeled).

### 6. `train.py` — SemanticPOSS branch

Three sites currently key on `self.settings.dataset`:

- ~line 116 (dataset construction)
- ~line 233 (post-init dataset-specific config)
- ~line 923 (eval-time logic)

Add a `elif self.settings.dataset == 'SemanticPOSS':` branch at each site, mirroring the KITTI branch:

```python
elif self.settings.dataset == 'SemanticPOSS':
    config_path = os.path.join(
        os.path.dirname(dataset.semantic_poss.__file__),
        'semantic-poss.yaml',
    )
    trainset = dataset.semantic_poss.SemanticPOSS(
        root=self.settings.data_root,
        sequences=[0, 1, 2, 4, 5],
        config_path=config_path,
        has_label=True,
    )
    valset = dataset.semantic_poss.SemanticPOSS(
        root=self.settings.data_root,
        sequences=[3],
        config_path=config_path,
        has_label=True,
    )
```

The eval-time branch (line 923) inherits the KITTI behavior; only the dataset string comparison changes.

### 7. `dataset/__init__.py`

Add `from . import semantic_poss`.

### 8. `tools/compute_poss_stats.py`

Small standalone script:

- CLI: `python tools/compute_poss_stats.py --data_root <path> --sequences 0 1 2 4 5`
- Iterates frames, tag-projects to (40, 1800, 5), accumulates per-channel sum and sum-of-squares over **valid pixels only** (where `tag == True`).
- Prints YAML-ready `img_mean` and `img_stds` arrays.
- Also computes class frequency from labels for the `content:` section, and prints that as YAML.

User runs once and pastes results into `semantic-poss.yaml` and `config_poss.yaml`.

## Error handling and edge cases

- **Empty cells in tag projection:** zero-fill the 5-channel image; this matches the KITTI loader's handling of empty range-image pixels.
- **Tag length mismatch:** assert `tag.sum() == len(points)` inside `_project_tag`. If a frame violates this (corrupted), raise a clear `ValueError` with frame id.
- **Label > max in `learning_map`:** the `class_map_lut` is sized to `max_key + 100` (KITTI pattern). Any unexpected raw label outside the map collapses to 0 (unlabeled).
- **Pretrained ViT weights:** none expected; POSS is small enough to train from scratch. `pretrained_model: null` in `config_poss.yaml`.
- **`instance_bank_root`:** point to a POSS-specific path (e.g., `cache/instance_bank_poss`), but since instance-cutmix is disabled, the bank isn't built.

## Testing strategy

- **Unit:** `tests/test_semantic_poss_parser.py` — load 1–2 frames, verify shapes (`points.shape[1] == 4`, `tag.shape == (40*1800,)`, `tag.sum() == len(points)`), verify `labelMapping` collapses 10/11/12 → 6 and 4/5 → 1.
- **Unit:** `tests/test_tag_projection.py` — feed synthetic points + a tag mask of 5 random cells, verify scatter places r/x/y/z/intensity into the right cells.
- **Integration:** `tests/test_semanticposs_integration.py` — instantiate `RangeViewLoader` over a tiny POSS sample dir, iterate one batch, run TinyViM forward, assert output `(B, 14, 40, 1800)`.
- **Regression:** existing `tests/test_tinyvim_integration.py` and KITTI training must still work — verified by running once with `config_kitti.yaml` after the TinyViM stride refactor.

## Risks and unknowns

- **Whether the `tag` files use bool or uint8 packing** — the snippet uses `dtype=np.bool`. Parser should handle both. (Mitigation: try bool first, fall back to uint8.)
- **`stage_embedding_strides` ripple effects on TinyViM** — the inter-stage Embedding also changes channel count, not just spatial size. Need to verify channel projection still works with stride (1,1). Expected to work since `Conv2d_BN` handles any stride, but worth a forward-pass smoke test.
- **Class imbalance:** ground + plants + building ≈ 80% of pixels. Focal loss handles this, but rare classes (cone, trashcan, traffic-sign) may need extra weighting. Initial run uses focal+lovász; revisit class_weights if rare-class IoU is near zero.
- **`img_mean`/`img_stds` placeholders:** training will misbehave until real values are computed. The compute_poss_stats.py script must be run before first training.

## Out of scope

- Cross-dataset evaluation (KITTI ↔ POSS).
- Standard spherical projection mode for POSS.
- Pretrained weight transfer between datasets.
- Test-set submission (POSS has no held-out test set in the standard protocol).
- Data augmentation parity with KITTI — geometric and mix augmentations are explicitly disabled for POSS in this design.
