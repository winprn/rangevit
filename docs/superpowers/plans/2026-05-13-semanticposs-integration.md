# SemanticPOSS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SemanticPOSS as a first-class training/eval dataset (native 14-class taxonomy, 40×1800 tag-based projection) for ablation studies, without disturbing the existing SemanticKITTI and nuScenes pipelines.

**Architecture:** New `dataset/semantic_poss/` module mirrors the KITTI layout (parser + YAML). `RangeViewLoader` gains a `projection_mode` switch — when `"tag"`, it scatters points into the 40×1800 grid using the dataset's precomputed `tag/*` boolean masks and skips point-cloud geometric/mix augmentations. TinyViM gets a configurable `stage_embedding_strides` argument so POSS can run at a /8 total width budget (1800 → 225) without altering KITTI behavior.

**Tech Stack:** PyTorch, NumPy, PyYAML, pytest. TinyViM (existing). RangeViT/TinyViMAdapter pipeline (existing).

**Spec:** `docs/superpowers/specs/2026-05-13-semanticposs-integration-design.md`

---

## File Structure

**New files:**
- `dataset/semantic_poss/__init__.py` — module export
- `dataset/semantic_poss/parser.py` — `SemanticPOSS` class, tag-file aware
- `dataset/semantic_poss/semantic-poss.yaml` — class taxonomy, learning map, splits, frequencies
- `config_poss.yaml` — top-level training config
- `tools/compute_poss_stats.py` — channel mean/std and class-frequency calculator
- `tests/test_semantic_poss_parser.py` — parser unit tests
- `tests/test_tag_projection.py` — projection helper unit test
- `tests/test_semanticposs_integration.py` — end-to-end loader → encoder smoke test

**Modified files:**
- `dataset/__init__.py` — add `from . import semantic_poss`
- `dataset/preprocess/projection.py` — add `TagProjection` helper
- `dataset/range_view_loader.py` — projection-mode switch; skip incompatible augs when `"tag"`
- `models/tinyvim/tinyvim.py` — accept `stage_embedding_strides` list overriding `down_stride` per inter-stage Embedding
- `models/tinyvim_adapter.py` — forward `stage_embedding_strides` from kwargs
- `models/rangevit.py` — read `stage_embedding_strides` from config and pass it to the adapter
- `train.py` — add `elif self.settings.dataset == 'SemanticPOSS':` branches at the three sites that key on dataset name (lines ~129, ~233, ~923)

---

## Task 1: SemanticPOSS class YAML

**Files:**
- Create: `dataset/semantic_poss/__init__.py`
- Create: `dataset/semantic_poss/semantic-poss.yaml`

- [ ] **Step 1: Create the YAML class config**

Create `dataset/semantic_poss/semantic-poss.yaml`:

```yaml
name: "semantic-poss"

labels:
  0: "unlabeled"
  4: "person"
  5: "person_2plus"
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
  16: "cone-stone"
  17: "fence"
  21: "bike"
  22: "ground"

# Raw label id -> mapped class id in [0..13]; 0 = unlabeled (ignored)
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

# Mapped id -> a canonical raw id (for prediction export)
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
  10: "cone-stone"
  11: "fence"
  12: "bike"
  13: "ground"

# 14 entries indexed by mapped class id; only id 0 is ignored
learning_ignore:
  0: true
  1: false
  2: false
  3: false
  4: false
  5: false
  6: false
  7: false
  8: false
  9: false
  10: false
  11: false
  12: false
  13: false

# Raw-label frequencies (placeholder, replace via tools/compute_poss_stats.py).
# Must include every raw id present in learning_map.
content:
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
  valid: [3]
  test: []
```

- [ ] **Step 2: Create the module init**

Create `dataset/semantic_poss/__init__.py`:

```python
from .parser import SemanticPOSS

__all__ = ["SemanticPOSS"]
```

- [ ] **Step 3: Commit**

```bash
git add dataset/semantic_poss/__init__.py dataset/semantic_poss/semantic-poss.yaml
git commit -m "feat(poss): add SemanticPOSS class taxonomy YAML"
```

---

## Task 2: SemanticPOSS parser

**Files:**
- Create: `dataset/semantic_poss/parser.py`
- Test: `tests/test_semantic_poss_parser.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_semantic_poss_parser.py`:

```python
import os
import numpy as np
import pytest

from dataset.semantic_poss.parser import SemanticPOSS


H, W = 40, 1800


@pytest.fixture
def tiny_poss_root(tmp_path):
    """Build a 1-sequence, 1-frame mock POSS directory."""
    seq = "00"
    seq_dir = tmp_path / "sequences" / seq
    (seq_dir / "velodyne").mkdir(parents=True)
    (seq_dir / "labels").mkdir()
    (seq_dir / "tag").mkdir()

    # 5 points, all "ground" (raw id 22) except one "person" (raw id 4)
    n_points = 5
    pts = np.array(
        [[1.0, 2.0, 0.1, 0.5],
         [2.0, 1.0, 0.0, 0.3],
         [-1.0, 0.5, 0.2, 0.7],
         [0.5, -1.5, 0.1, 0.4],
         [3.0, 0.0, 1.5, 0.9]],
        dtype=np.float32,
    )
    pts.tofile(seq_dir / "velodyne" / "000000.bin")

    sem_raw = np.array([22, 22, 22, 22, 4], dtype=np.uint32)
    inst = np.zeros(n_points, dtype=np.uint32)
    packed = (inst << 16) | sem_raw
    packed.astype(np.uint32).tofile(seq_dir / "labels" / "000000.label")

    # Tag mask: 5 cells set true, rest false
    tag = np.zeros(H * W, dtype=np.bool_)
    tag[[0, 5, 100, 1000, H * W - 1]] = True
    tag.tofile(seq_dir / "tag" / "000000.tag")

    return str(tmp_path / "sequences")


def test_parser_loads_frame(tiny_poss_root):
    config = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "semantic_poss", "semantic-poss.yaml"
    )
    parser = SemanticPOSS(root=tiny_poss_root, sequences=[0], config_path=config)
    assert len(parser) == 1
    pc, sem, inst = parser.loadDataByIndex(0)
    assert pc.shape == (5, 4)
    assert sem.shape == (5,)
    assert inst.shape == (5,)


def test_parser_returns_tag_mask(tiny_poss_root):
    config = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "semantic_poss", "semantic-poss.yaml"
    )
    parser = SemanticPOSS(root=tiny_poss_root, sequences=[0], config_path=config)
    tag = parser.loadTagByIndex(0)
    assert tag.dtype == np.bool_
    assert tag.shape == (H * W,)
    assert int(tag.sum()) == 5


def test_label_mapping_collapses_traffic_signs_and_persons(tiny_poss_root):
    config = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "semantic_poss", "semantic-poss.yaml"
    )
    parser = SemanticPOSS(root=tiny_poss_root, sequences=[0], config_path=config)
    raw = np.array([0, 4, 5, 10, 11, 12, 22], dtype=np.int32)
    mapped = parser.labelMapping(raw)
    # 4,5 -> 1; 10,11,12 -> 6; 22 -> 13; 0 -> 0
    assert mapped.tolist() == [0, 1, 1, 6, 6, 6, 13]


def test_parse_path_info(tiny_poss_root):
    config = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "semantic_poss", "semantic-poss.yaml"
    )
    parser = SemanticPOSS(root=tiny_poss_root, sequences=[0], config_path=config)
    seq_id, frame_id = parser.parsePathInfoByIndex(0)
    assert seq_id == "00"
    assert frame_id == "000000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_semantic_poss_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dataset.semantic_poss.parser'`.

- [ ] **Step 3: Implement the parser**

Create `dataset/semantic_poss/parser.py`:

```python
import os
import yaml
import numpy as np


class SemanticPOSS(object):
    """SemanticPOSS dataset parser.

    Layout: <root>/<seq>/velodyne/*.bin, labels/*.label, tag/*.tag.
    Tag files are H*W boolean masks identifying which range-image cells
    a 1-to-1 point->cell projection occupies.
    """

    H = 40
    W = 1800

    def __init__(self, root, sequences, config_path, has_label=True):
        self.root = root
        self.sequences = sorted(int(s) for s in sequences)
        self.has_label = has_label

        if not os.path.isfile(config_path):
            raise ValueError(f"Config file not found: {config_path}")
        self.data_config = yaml.safe_load(open(config_path, "r"))

        if not os.path.isdir(self.root):
            raise ValueError(f"Dataset not found: {self.root}")

        self.pointcloud_files = []
        self.label_files = []
        self.tag_files = []
        for seq in self.sequences:
            seq_str = f"{seq:02d}"
            seq_dir = os.path.join(self.root, seq_str)
            pc_dir = os.path.join(seq_dir, "velodyne")
            tag_dir = os.path.join(seq_dir, "tag")
            pc_files = sorted(
                os.path.join(pc_dir, f) for f in os.listdir(pc_dir) if f.endswith(".bin")
            )
            tag_files = sorted(
                os.path.join(tag_dir, f) for f in os.listdir(tag_dir) if f.endswith(".tag")
            )
            assert len(pc_files) == len(tag_files), (
                f"Seq {seq_str}: {len(pc_files)} bins vs {len(tag_files)} tags"
            )

            if self.has_label:
                lbl_dir = os.path.join(seq_dir, "labels")
                lbl_files = sorted(
                    os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith(".label")
                )
                assert len(pc_files) == len(lbl_files), (
                    f"Seq {seq_str}: {len(pc_files)} bins vs {len(lbl_files)} labels"
                )
                self.label_files.extend(lbl_files)

            self.pointcloud_files.extend(pc_files)
            self.tag_files.extend(tag_files)

        print(f"Using {len(self.pointcloud_files)} POSS frames from sequences {self.sequences}")

        learning_map = self.data_config["learning_map"]
        max_key = max(learning_map.keys())
        self.class_map_lut = np.zeros((max_key + 100,), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut[k] = v

        learning_map_inv = self.data_config["learning_map_inv"]
        max_inv = max(learning_map_inv.keys())
        self.class_map_lut_inv = np.zeros((max_inv + 100,), dtype=np.int32)
        for k, v in learning_map_inv.items():
            self.class_map_lut_inv[k] = v

        cls_content = self.data_config["content"]
        n_mapped = len(self.data_config["learning_map_inv"])
        content = np.zeros(n_mapped, dtype=np.float32)
        for raw_id, freq in cls_content.items():
            content[self.class_map_lut[raw_id]] += freq
        self.cls_freq = content

        self.mapped_cls_name = self.data_config["mapped_class_name"]

    @staticmethod
    def readPCD(path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def readLabel(path):
        raw = np.fromfile(path, dtype=np.uint32)
        sem = (raw & 0xFFFF).astype(np.int32)
        inst = (raw >> 16).astype(np.int32)
        return sem, inst

    @classmethod
    def readTag(cls, path):
        # Files are written as np.bool_ (1 byte). Read as uint8 then cast.
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size != cls.H * cls.W:
            raise ValueError(
                f"Tag length {raw.size} != expected {cls.H * cls.W} for {path}"
            )
        return raw.astype(np.bool_)

    def parsePathInfoByIndex(self, index):
        path = self.pointcloud_files[index]
        parts = path.replace("\\", "/").split("/")
        seq_id = parts[-3]
        frame_id = parts[-1].split(".")[0]
        return seq_id, frame_id

    def labelMapping(self, label):
        return self.class_map_lut[label]

    def loadLabelByIndex(self, index):
        return self.readLabel(self.label_files[index])

    def loadDataByIndex(self, index):
        pc = self.readPCD(self.pointcloud_files[index])
        if self.has_label:
            sem, inst = self.readLabel(self.label_files[index])
        else:
            sem = np.zeros(pc.shape[0], dtype=np.int32)
            inst = np.zeros(pc.shape[0], dtype=np.int32)
        return pc, sem, inst

    def loadTagByIndex(self, index):
        return self.readTag(self.tag_files[index])

    def __len__(self):
        return len(self.pointcloud_files)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_semantic_poss_parser.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add dataset/semantic_poss/parser.py tests/test_semantic_poss_parser.py
git commit -m "feat(poss): add SemanticPOSS parser with tag-mask loader"
```

---

## Task 3: dataset module export

**Files:**
- Modify: `dataset/__init__.py`

- [ ] **Step 1: Edit `dataset/__init__.py`**

Apply this exact change. Current content:

```python
from . import semantic_kitti
from . import nuScenes

from .range_view_loader import RangeViewLoader, custom_collate_kpconv_fn
```

New content:

```python
from . import semantic_kitti
from . import nuScenes
from . import semantic_poss

from .range_view_loader import RangeViewLoader, custom_collate_kpconv_fn
```

- [ ] **Step 2: Verify import works**

Run: `python -c "import dataset; print(dataset.semantic_poss.SemanticPOSS)"`
Expected: `<class 'dataset.semantic_poss.parser.SemanticPOSS'>`

- [ ] **Step 3: Commit**

```bash
git add dataset/__init__.py
git commit -m "feat(poss): export semantic_poss from dataset package"
```

---

## Task 4: Tag-based projection helper

**Files:**
- Modify: `dataset/preprocess/projection.py`
- Test: `tests/test_tag_projection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tag_projection.py`:

```python
import numpy as np

from dataset.preprocess.projection import TagProjection


H, W = 40, 1800


def test_tag_projection_scatters_features_into_correct_cells():
    pts = np.array(
        [[1.0, 2.0, 0.5, 0.1],
         [3.0, -1.0, 0.0, 0.7],
         [-2.0, 1.5, 0.2, 0.4]],
        dtype=np.float32,
    )
    cells = [0, 250, H * W - 1]
    tag = np.zeros(H * W, dtype=np.bool_)
    tag[cells] = True

    proj = TagProjection(proj_h=H, proj_w=W)
    proj_xyz_i, proj_range, proj_idx, proj_mask = proj.doProjection(pts, tag)

    assert proj_xyz_i.shape == (H, W, 4)
    assert proj_range.shape == (H, W)
    assert proj_idx.shape == (H, W)
    assert proj_mask.shape == (H, W)

    flat_mask = proj_mask.reshape(-1)
    assert flat_mask.dtype == bool or flat_mask.dtype == np.bool_
    assert int(flat_mask.sum()) == 3
    for c in cells:
        assert flat_mask[c]

    flat_xyz_i = proj_xyz_i.reshape(-1, 4)
    flat_range = proj_range.reshape(-1)
    flat_idx = proj_idx.reshape(-1)
    for i, c in enumerate(cells):
        np.testing.assert_allclose(flat_xyz_i[c], pts[i])
        np.testing.assert_allclose(flat_range[c], np.linalg.norm(pts[i, :3]))
        assert flat_idx[c] == i


def test_tag_projection_rejects_length_mismatch():
    pts = np.zeros((2, 4), dtype=np.float32)
    tag = np.zeros(H * W, dtype=np.bool_)
    tag[0] = True  # only 1 occupied cell but 2 points
    proj = TagProjection(proj_h=H, proj_w=W)
    import pytest
    with pytest.raises(ValueError):
        proj.doProjection(pts, tag)


def test_tag_projection_caches_uproj_indices():
    pts = np.array([[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.6]], dtype=np.float32)
    tag = np.zeros(H * W, dtype=np.bool_)
    tag[[3, 9]] = True
    proj = TagProjection(proj_h=H, proj_w=W)
    proj.doProjection(pts, tag)
    cached = proj.cached_data
    assert "uproj_x_idx" in cached and "uproj_y_idx" in cached and "uproj_depth" in cached
    # First point lives in cell 3 -> row 0, col 3; second in cell 9 -> row 0, col 9
    assert cached["uproj_y_idx"].tolist() == [0, 0]
    assert cached["uproj_x_idx"].tolist() == [3, 9]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tag_projection.py -v`
Expected: FAIL — `ImportError: cannot import name 'TagProjection'`.

- [ ] **Step 3: Implement `TagProjection`**

Append to `dataset/preprocess/projection.py`:

```python
class TagProjection(object):
    """Scatter points into a fixed-size range image using a precomputed
    boolean tag mask (1-to-1 point->cell mapping). Output mirrors
    RangeProjection: (proj_xyz_i [H,W,4], proj_range [H,W], proj_idx [H,W],
    proj_mask [H,W]).
    """

    def __init__(self, proj_h, proj_w):
        self.proj_h = proj_h
        self.proj_w = proj_w
        self.cached_data = {}

    def doProjection(self, pointcloud, tag):
        H, W = self.proj_h, self.proj_w
        flat_n = H * W
        if tag.shape[0] != flat_n:
            raise ValueError(f"Tag length {tag.shape[0]} != expected {flat_n}")
        n_points = pointcloud.shape[0]
        n_occupied = int(tag.sum())
        if n_occupied != n_points:
            raise ValueError(
                f"Tag occupied cells ({n_occupied}) != number of points ({n_points})"
            )

        proj_xyz_i = np.zeros((flat_n, 4), dtype=np.float32)
        proj_range = np.zeros((flat_n,), dtype=np.float32)
        proj_idx = -1 * np.ones((flat_n,), dtype=np.int32)

        ranges = np.linalg.norm(pointcloud[:, :3], axis=1).astype(np.float32)
        proj_xyz_i[tag] = pointcloud
        proj_range[tag] = ranges
        proj_idx[tag] = np.arange(n_points, dtype=np.int32)

        proj_xyz_i = proj_xyz_i.reshape(H, W, 4)
        proj_range = proj_range.reshape(H, W)
        proj_idx = proj_idx.reshape(H, W)
        proj_mask = tag.reshape(H, W).astype(np.bool_)

        # Cache unprojection indices for KNN/3D postproc compatibility.
        flat_cells = np.flatnonzero(tag).astype(np.int32)
        uproj_y = (flat_cells // W).astype(np.int32)
        uproj_x = (flat_cells %  W).astype(np.int32)
        self.cached_data = {
            "px": proj_xyz_i,  # placeholder; loader uses uproj_*
            "py": proj_xyz_i,
            "uproj_x_idx": uproj_x,
            "uproj_y_idx": uproj_y,
            "uproj_depth": ranges.astype(np.float32),
        }

        return proj_xyz_i, proj_range, proj_idx, proj_mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tag_projection.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add dataset/preprocess/projection.py tests/test_tag_projection.py
git commit -m "feat(poss): add TagProjection helper for precomputed range masks"
```

---

## Task 5: RangeViewLoader — projection-mode switch

**Files:**
- Modify: `dataset/range_view_loader.py`

- [ ] **Step 1: Wire projection mode into `__init__`**

In `dataset/range_view_loader.py`, locate the projection-setup block (currently around lines 123–135):

```python
        projection_config = self.config['sensor']
        self.scan_proj = projection_config.get('scan_proj', False)
        if self.scan_proj:
            print('Use scan-based range projection.')
            self.projection = projection.ScanProjection(
                proj_h=projection_config['proj_h'], proj_w=projection_config['proj_w'],
            )
        else:
            self.projection = projection.RangeProjection(
                fov_up=projection_config['fov_up'], fov_down=projection_config['fov_down'],
                fov_left=projection_config['fov_left'], fov_right=projection_config['fov_right'],
                proj_h=projection_config['proj_h'], proj_w=projection_config['proj_w'],
            )
```

Replace with:

```python
        projection_config = self.config['sensor']
        self.scan_proj = projection_config.get('scan_proj', False)
        self.projection_mode = projection_config.get('projection_mode', 'spherical')
        if self.projection_mode == 'tag':
            print('Use tag-mask range projection (precomputed cell map).')
            self.projection = projection.TagProjection(
                proj_h=projection_config['proj_h'], proj_w=projection_config['proj_w'],
            )
        elif self.scan_proj:
            print('Use scan-based range projection.')
            self.projection = projection.ScanProjection(
                proj_h=projection_config['proj_h'], proj_w=projection_config['proj_w'],
            )
        else:
            self.projection = projection.RangeProjection(
                fov_up=projection_config['fov_up'], fov_down=projection_config['fov_down'],
                fov_left=projection_config['fov_left'], fov_right=projection_config['fov_right'],
                proj_h=projection_config['proj_h'], proj_w=projection_config['proj_w'],
            )
```

- [ ] **Step 2: Force-disable incompatible augmentations in tag mode**

In the same `__init__`, immediately AFTER the projection block above (right before the line `self.train_full_image = ...`), add:

```python
        if self.projection_mode == 'tag':
            # Geometric and mix augmentations re-arrange or replace points; the
            # precomputed tag mask becomes invalid. Disable them at construction.
            self.augmentor = None
            self.point_sampler = None
            self.polarmix = None
            self.instance_cutmix = None
            self.clustermix = None
            self.instance_copy = None
```

- [ ] **Step 3: Add tag-mode `TagProjection` import (if not already in the wildcard)**

The file already imports `projection` from `.preprocess`. Confirm by inspecting:

Run (read-only): `grep -n "from .preprocess" dataset/range_view_loader.py`
Expected: line 25 shows `from .preprocess import augmentor, projection, ...`. No change needed — `projection.TagProjection` resolves through the package.

- [ ] **Step 4: Branch projection call in `__getitem__`**

In `dataset/range_view_loader.py`, locate the line in `__getitem__` (~line 400):

```python
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)
```

Replace with:

```python
        if self.projection_mode == 'tag':
            tag = self.dataset.loadTagByIndex(index)
            proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud, tag)
        else:
            proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)
```

Apply the same replacement in `get_item_for_kpconv` (the matching line near ~line 292):

```python
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)
```

becomes:

```python
        if self.projection_mode == 'tag':
            tag = self.dataset.loadTagByIndex(index)
            proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud, tag)
        else:
            proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)
```

- [ ] **Step 5: Smoke-test backwards compatibility**

Run: `pytest tests/test_tag_projection.py tests/test_semantic_poss_parser.py -v`
Expected: all previous tests still PASS.

Run: `python -c "from dataset.range_view_loader import RangeViewLoader; print('ok')"`
Expected: `ok` (no import errors from the edited file).

- [ ] **Step 6: Commit**

```bash
git add dataset/range_view_loader.py
git commit -m "feat(poss): projection_mode switch in RangeViewLoader (tag vs spherical)"
```

---

## Task 6: TinyViM configurable per-stage embedding strides

**Files:**
- Modify: `models/tinyvim/tinyvim.py`

- [ ] **Step 1: Add the `stage_embedding_strides` constructor argument**

In `models/tinyvim/tinyvim.py`, find the `TinyViM.__init__` signature (around line 139):

```python
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 num_classes=1000,
                 down_patch_size=3, down_stride=(1, 2), down_pad=1,
                 height_downsample_stage=1,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 ssm_num=1,
                 distillation=True,
                 stem_stride=(1, 1),
                 **kwargs):
```

Insert `stage_embedding_strides=None,` after `down_stride=(1, 2),`:

```python
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 num_classes=1000,
                 down_patch_size=3, down_stride=(1, 2), down_pad=1,
                 stage_embedding_strides=None,
                 height_downsample_stage=1,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 ssm_num=1,
                 distillation=True,
                 stem_stride=(1, 1),
                 **kwargs):
```

- [ ] **Step 2: Use per-stage strides when building the network**

In the same file, locate the network-building loop (around lines 160–178):

```python
        network = []
        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers, mlp_ratio=mlp_ratios,
                          use_layer_scale=use_layer_scale,
                          layer_scale_init_value=layer_scale_init_value,
                          ssm_num=ssm_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # Optionally downsample height at one selected transition.
                stride = (2, 2) if height_downsample_stage is not None and i == height_downsample_stage else down_stride
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )
```

Replace the inner `if downsamples[i] or ...:` block so that `stage_embedding_strides`, when provided, takes precedence:

```python
        network = []
        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers, mlp_ratio=mlp_ratios,
                          use_layer_scale=use_layer_scale,
                          layer_scale_init_value=layer_scale_init_value,
                          ssm_num=ssm_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                if stage_embedding_strides is not None:
                    # Caller-supplied per-transition stride (i is the
                    # 0-indexed transition between stage i and stage i+1).
                    stride = tuple(stage_embedding_strides[i])
                elif height_downsample_stage is not None and i == height_downsample_stage:
                    stride = (2, 2)
                else:
                    stride = down_stride
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )
```

- [ ] **Step 3: Smoke-test instantiation in both modes**

Run:

```bash
python -c "
import torch
from models.tinyvim.tinyvim import TinyViM, TinyViM_depth, TinyViM_width

# Default (KITTI behaviour unchanged): /16 width
m = TinyViM(
    layers=TinyViM_depth['B'], embed_dims=TinyViM_width['B'],
    downsamples=[True, True, True, True], num_classes=0, fork_feat=False,
    stem_stride=(1, 2),
)
y = m(torch.randn(1, 3, 64, 2048))
print('default ok, last stage out:', y[0].shape if isinstance(y, tuple) else y.shape)

# POSS-style: stem /2 + (1,1) + (1,2) + (1,2) => total /8 width
m2 = TinyViM(
    layers=TinyViM_depth['B'], embed_dims=TinyViM_width['B'],
    downsamples=[True, True, True, True], num_classes=0, fork_feat=False,
    stem_stride=(1, 2),
    stage_embedding_strides=[(1, 1), (1, 2), (1, 2)],
)
"
```

Expected: prints `default ok, last stage out: ...` and completes without error. (The classification head produces a tuple from `dist=True`; only the existence of the call matters here.)

- [ ] **Step 4: Commit**

```bash
git add models/tinyvim/tinyvim.py
git commit -m "feat(tinyvim): configurable per-stage embedding strides"
```

---

## Task 7: TinyViMAdapter forwards stage strides

**Files:**
- Modify: `models/tinyvim_adapter.py`

- [ ] **Step 1: Read the kwarg and pass it through**

In `models/tinyvim_adapter.py`, locate this block (lines 44–46):

```python
        stem_stride = kwargs.pop('stem_stride', (1, 1))
        down_stride = kwargs.pop('down_stride', (1, 2))
        height_downsample_stage = kwargs.pop('height_downsample_stage', None)
```

Add a new line:

```python
        stem_stride = kwargs.pop('stem_stride', (1, 1))
        down_stride = kwargs.pop('down_stride', (1, 2))
        height_downsample_stage = kwargs.pop('height_downsample_stage', None)
        stage_embedding_strides = kwargs.pop('stage_embedding_strides', None)
```

Then in the `TinyViM(...)` constructor call (lines 55–65):

```python
        self.model = TinyViM(
            layers=layers,
            embed_dims=embed_dims,
            downsamples=downsamples,
            vit_num=1,
            num_classes=0, # No classification head
            fork_feat=False, # We handle feature extraction manually or change this
            stem_stride=stem_stride,
            down_stride=down_stride,
            height_downsample_stage=height_downsample_stage,
        )
```

Add the new kwarg:

```python
        self.model = TinyViM(
            layers=layers,
            embed_dims=embed_dims,
            downsamples=downsamples,
            vit_num=1,
            num_classes=0, # No classification head
            fork_feat=False, # We handle feature extraction manually or change this
            stem_stride=stem_stride,
            down_stride=down_stride,
            stage_embedding_strides=stage_embedding_strides,
            height_downsample_stage=height_downsample_stage,
        )
```

- [ ] **Step 2: Smoke-test that POSS shape flows through the adapter**

Run:

```bash
python -c "
import torch
from models.tinyvim_adapter import TinyViMAdapter

adapter = TinyViMAdapter(
    backbone_name='tinyvim_base',
    in_channels=5,
    stem_stride=(1, 2),
    stage_embedding_strides=[(1, 1), (1, 2), (1, 2)],
    use_fpn_decoder=True,
)
x = torch.randn(1, 5, 40, 1800)
feats, _ = adapter(x, return_features=True)
shapes = [tuple(f.shape) for f in feats]
print(shapes)
assert all(s[2] == 40 for s in shapes), shapes  # height preserved
# Widths: 900, 900, 450, 225
assert [s[3] for s in shapes] == [900, 900, 450, 225], shapes
print('POSS shape budget ok')
"
```

Expected: prints the four stage shapes and `POSS shape budget ok`.

- [ ] **Step 3: Commit**

```bash
git add models/tinyvim_adapter.py
git commit -m "feat(tinyvim): adapter forwards stage_embedding_strides"
```

---

## Task 8: RangeViT consumes `stage_embedding_strides` from config

**Files:**
- Modify: `models/rangevit.py`

- [ ] **Step 1: Locate the TinyViMAdapter construction site**

Find the section in `models/rangevit.py` where `TinyViMAdapter` is instantiated (search for `TinyViMAdapter`). The instantiation passes config-driven kwargs.

- [ ] **Step 2: Read `stage_embedding_strides` from the model config and forward it**

Wherever the adapter kwargs dict is built, add (before the `TinyViMAdapter(...)` call):

```python
        stage_embedding_strides = self.model_cfg.get('stage_embedding_strides', None)
```

and include `stage_embedding_strides=stage_embedding_strides,` in the `TinyViMAdapter(...)` keyword arguments. If the file uses a different variable name for the model config dict, use that instead — preserve existing style.

If the file does not currently expose `self.model_cfg`, derive it inline:

```python
        model_cfg = getattr(args, 'model', None) or {}
        stage_embedding_strides = model_cfg.get('stage_embedding_strides', None) if isinstance(model_cfg, dict) else None
```

(Match the actual config access pattern already in `models/rangevit.py` — do not introduce a new style.)

- [ ] **Step 3: Smoke-test RangeViT instantiation with POSS-like config**

Run (after writing `config_poss.yaml` in Task 10, or now with a minimal stub):

```bash
python -c "
import yaml
from option import Option
import sys
sys.argv = ['x', 'config_kitti.yaml']
# Sanity: KITTI still works (no stage_embedding_strides set)
print('KITTI option load ok')
"
```

Expected: prints `KITTI option load ok` with no errors.

- [ ] **Step 4: Commit**

```bash
git add models/rangevit.py
git commit -m "feat(tinyvim): forward stage_embedding_strides from config"
```

---

## Task 9: train.py SemanticPOSS branch

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add dataset-construction branch (~line 168)**

In `train.py`, locate the `_initDataloader` method. After the `elif self.settings.dataset == 'SemanticKitti':` block (which ends just before `else: raise ValueError(...)` around line 168), insert a new branch:

```python
        # SemanticPOSS dataset
        elif self.settings.dataset == 'SemanticPOSS':
            data_config_path = 'dataset/semantic_poss/semantic-poss.yaml'
            data_config = yaml.safe_load(open(data_config_path, 'r'))

            if self.settings.use_mini_version:
                train_sequences = [0]
            elif self.settings.use_trainval:
                train_sequences = data_config['split']['train'] + data_config['split']['valid']
            else:
                train_sequences = data_config['split']['train']

            trainset = dataset.semantic_poss.SemanticPOSS(
                root=self.settings.data_root,
                sequences=train_sequences,
                config_path=data_config_path)

            self.cls_weight = 1 / (trainset.cls_freq + 1e-3)
            self.ignore_class = []
            for cl, _ in enumerate(self.cls_weight):
                if trainset.data_config['learning_ignore'][cl]:
                    self.cls_weight[cl] = 0
                if self.cls_weight[cl] < 1e-10:
                    self.ignore_class.append(cl)
            if self.recorder is not None:
                self.recorder.logger.info('weight: {}'.format(self.cls_weight))
            self.mapped_cls_name = trainset.mapped_cls_name

            test_sequences = (
                data_config['split']['test'] if self.settings.test_split else
                data_config['split']['valid'])

            valset = dataset.semantic_poss.SemanticPOSS(
                root=self.settings.data_root,
                sequences=test_sequences,
                config_path=data_config_path,
                has_label=(self.settings.test_split is False),
            )
```

- [ ] **Step 2: Add focal-loss alpha branch (~line 233 and ~line 240)**

Find the `_initCriterion` method. Two `elif` chains key on `self.settings.dataset`. After each `elif self.settings.dataset == 'nuScenes':` block (lines ~236 and ~244), add a SemanticPOSS branch.

First chain (`class_weighted_focal`, line ~230–238) — after the nuScenes branch, before the trailing `else:`:

```python
            elif self.settings.dataset == 'SemanticPOSS':
                alpha = self.cls_weight.astype(np.float32)
```

Second chain (default path, line ~240–246) — after the nuScenes branch:

```python
            elif self.settings.dataset == 'SemanticPOSS':
                alpha = np.log(1 + self.cls_weight)
                alpha = alpha / max(alpha.max(), 1e-6)
```

- [ ] **Step 3: Add prediction-export branch (~line 923)**

In the validation-output block (around line 914–931), the existing chain handles `nuScenes` and `SemanticKitti`. After the `SemanticKitti` block, append:

```python
                elif self.settings.dataset == 'SemanticPOSS':
                    poss_dataset = self.val_loader.dataset.dataset
                    pred_np_origin = poss_dataset.class_map_lut_inv[pred_np]
                    seq_id, frame_id = poss_dataset.parsePathInfoByIndex(index)
                    pred_path = os.path.join(save_results_path, 'sequences', seq_id, 'predictions')
                    if not os.path.isdir(pred_path):
                        os.makedirs(pred_path)
                    pred_result_path = os.path.join(pred_path, '{}.label'.format(frame_id))
                    pred_np_origin.tofile(pred_result_path)
```

- [ ] **Step 4: Verify the file still parses**

Run: `python -c "import ast; ast.parse(open('train.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "feat(poss): wire SemanticPOSS into train.py dataset/loss/eval branches"
```

---

## Task 10: config_poss.yaml

**Files:**
- Create: `config_poss.yaml`

- [ ] **Step 1: Write the config**

Create `config_poss.yaml`:

```yaml
# General config
num_workers: 4
id: "exp_poss"
save_path: "save_rangevim_poss"

# MLflow config
mlflow:
  enable: true
  tracking_uri: "http://140.245.117.232:5000"
  experiment_name: "rangevit"
  run_name: "[train] rangevim_base_poss"
  nested: false
  log_checkpoints: false
  log_code_snapshot: false

# Data config
data:
  dataset: "SemanticPOSS"
  n_classes: 14
  data_root: "../dataset/SemanticPOSS/sequences"
  has_label: true
  use_trainval: false

knni:
  enable: false
  window_size: 5

# Train config
training:
  val_frequency: 1
  n_epochs: 60
  warmup_epochs: 6
  batch_size: 8
  batch_size_val: 1
  lr: 0.0003
  train_result_frequency: 100
  loss:
    focal_loss:
      type: "focal"
      gamma: 2.0
      ignore_index: 0
      weight: 1.0
    lovasz_loss:
      weight: 1.0
    boundary_loss:
      weight: 0.0
    aux_loss:
      weight: 0.3
  save_epochs_at: [47, 50, 55, 57]

# Model config
model:
  vit_backbone: "tinyvim_base"
  in_channels: 5
  patch_size: [1, 2]
  patch_stride: [1, 2]
  image_size: [40, 1800]
  window_size: [40, 1800]
  window_stride: [40, 1800]
  original_image_size: [40, 1800]
  train_full_image: true
  use_sliding_window: false
  # /8 total width budget: stem /2 + (1,1) + (1,2) + (1,2). 1800 -> 900 -> 900 -> 450 -> 225.
  stage_embedding_strides:
    - [1, 1]
    - [1, 2]
    - [1, 2]

# Stem
conv_stem: "ConvStem"
stem_base_channels: 32
D_h: 48

# Decoder
decoder: "fpn"
skip_filters: 0

# 3D refiner / post-processing
point_postproc: "knn"
knn_search: 13
knn_k: 5
knn_sigma: 1.0
knn_cutoff: 1.0

# Checkpoint model
checkpoint: null
pretrained_model: null
reuse_pos_emb: false
reuse_patch_emb: false

pretrained_channel_adaptation: "repeat"

# 2D range-image augmentation OK; 3D geometric and mix augs disabled below.
range_aug: false

augmentation:
  p_flipx: 0.0
  p_flipy: 0.0
  p_transx: 0.0
  trans_xmin: 0
  trans_xmax: 0
  p_transy: 0.0
  trans_ymin: 0
  trans_ymax: 0
  p_transz: 0.0
  trans_zmin: 0
  trans_zmax: 0
  p_rot_roll: 0.0
  rot_rollmin: 0
  rot_rollmax: 0
  p_rot_pitch: 0.0
  rot_pitchmin: 0
  rot_pitchmax: 0
  p_rot_yaw: 0.0
  rot_yawmin: 0
  rot_yawmax: 0

adapted_augmentation:
  # POSS mapped class ids:
  # 0=unlabeled, 1=person, 2=rider, 3=car, 4=trunk, 5=plants,
  # 6=traffic-sign, 7=pole, 8=trashcan, 9=building, 10=cone-stone,
  # 11=fence, 12=bike, 13=ground
  use_mapped_labels: true
  pointsample:
    enable: false
    num_points: 0
    replace: false
  polarmix:
    enable: false
    prob: 0.0
    classes: []
  instance_cutmix:
    enable: false
    prob: 0.0
    instance_bank_root: "cache/instance_bank_poss"
    classes: []
    min_points: 20
    num_to_add: [0, 0]
  clustermix:
    enable: false
    prob: 0.0
    vertical_prob: 0.0
    sector_width: 1.570796
    height_ratio: 0.5
  instance_copy:
    enable: false
    prob: 0.0
    classes: []
    max_instances_per_class: 0

sensor:
  name: "Pandora40"
  type: "spherical"
  projection_mode: "tag"     # consumed by RangeViewLoader (Task 5)
  scan_proj: false
  proj_h: 40
  proj_w: 1800
  fov_up: 7.0
  fov_down: -16.0
  fov_left: -180
  fov_right: 180
  # TODO replace with output of tools/compute_poss_stats.py before first real training run
  img_mean: [0.0, 0.0, 0.0, 0.0, 0.0]
  img_stds: [1.0, 1.0, 1.0, 1.0, 1.0]
```

- [ ] **Step 2: Verify YAML parses**

Run: `python -c "import yaml; print(list(yaml.safe_load(open('config_poss.yaml')).keys()))"`
Expected: prints a list including `data`, `model`, `sensor`, `augmentation`, `adapted_augmentation`.

- [ ] **Step 3: Commit**

```bash
git add config_poss.yaml
git commit -m "feat(poss): add config_poss.yaml (native 40x1800, /8 width, tag projection)"
```

---

## Task 11: Stats / class-frequency tool

**Files:**
- Create: `tools/compute_poss_stats.py`

- [ ] **Step 1: Write the script**

Create `tools/compute_poss_stats.py`:

```python
"""Compute per-channel mean/std and raw-label frequencies for SemanticPOSS.

Usage:
    python tools/compute_poss_stats.py \
        --data_root /path/to/SemanticPOSS/sequences \
        --sequences 0 1 2 4 5

Prints YAML-ready img_mean, img_stds, and `content:` entries for
config_poss.yaml and dataset/semantic_poss/semantic-poss.yaml.
"""
import argparse
import os

import numpy as np


H, W = 40, 1800
N_CHANNELS = 5  # range, x, y, z, intensity


def iter_frames(data_root, sequences):
    for seq in sequences:
        seq_str = f"{int(seq):02d}"
        seq_dir = os.path.join(data_root, seq_str)
        pc_dir = os.path.join(seq_dir, "velodyne")
        lbl_dir = os.path.join(seq_dir, "labels")
        tag_dir = os.path.join(seq_dir, "tag")
        pc_files = sorted(f for f in os.listdir(pc_dir) if f.endswith(".bin"))
        for f in pc_files:
            stem = f.rsplit(".", 1)[0]
            yield (
                os.path.join(pc_dir, f),
                os.path.join(lbl_dir, stem + ".label"),
                os.path.join(tag_dir, stem + ".tag"),
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sequences", type=int, nargs="+", required=True)
    args = ap.parse_args()

    total = 0
    sum_ = np.zeros(N_CHANNELS, dtype=np.float64)
    sumsq = np.zeros(N_CHANNELS, dtype=np.float64)
    label_counts = {}

    for pc_path, lbl_path, tag_path in iter_frames(args.data_root, args.sequences):
        pts = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        tag = np.fromfile(tag_path, dtype=np.uint8).astype(bool)
        if tag.sum() != pts.shape[0]:
            print(f"WARN: tag mismatch for {pc_path}; skipping")
            continue
        rng = np.linalg.norm(pts[:, :3], axis=1).astype(np.float32)
        feats = np.stack([rng, pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]], axis=1).astype(np.float64)
        total += feats.shape[0]
        sum_ += feats.sum(axis=0)
        sumsq += (feats ** 2).sum(axis=0)

        raw = np.fromfile(lbl_path, dtype=np.uint32) & 0xFFFF
        ids, counts = np.unique(raw, return_counts=True)
        for i, c in zip(ids.tolist(), counts.tolist()):
            label_counts[int(i)] = label_counts.get(int(i), 0) + int(c)

    mean = sum_ / max(total, 1)
    var = sumsq / max(total, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))

    print("# Paste into config_poss.yaml under sensor:")
    print(f"img_mean: {[float(round(m, 4)) for m in mean.tolist()]}")
    print(f"img_stds: {[float(round(s, 4)) for s in std.tolist()]}")

    total_labels = sum(label_counts.values())
    print("\n# Paste into dataset/semantic_poss/semantic-poss.yaml under content:")
    print("content:")
    for raw_id in sorted(label_counts.keys()):
        freq = label_counts[raw_id] / max(total_labels, 1)
        print(f"  {raw_id}: {freq:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses**

Run: `python -c "import ast; ast.parse(open('tools/compute_poss_stats.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add tools/compute_poss_stats.py
git commit -m "feat(poss): tool to compute sensor mean/std and class frequencies"
```

---

## Task 12: Integration smoke test

**Files:**
- Test: `tests/test_semanticposs_integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_semanticposs_integration.py`:

```python
import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataset.semantic_poss.parser import SemanticPOSS
from dataset.range_view_loader import RangeViewLoader


H, W = 40, 1800


@pytest.fixture
def tiny_poss_root(tmp_path):
    seq = "03"
    seq_dir = tmp_path / "sequences" / seq
    (seq_dir / "velodyne").mkdir(parents=True)
    (seq_dir / "labels").mkdir()
    (seq_dir / "tag").mkdir()

    n = 6
    pts = np.array(
        [[1, 0, 0, 0.2], [0, 1, 0, 0.3], [-1, 0, 0, 0.4],
         [0, -1, 0, 0.5], [2, 2, 0.5, 0.6], [-2, -2, 0.1, 0.7]],
        dtype=np.float32,
    )
    pts.tofile(seq_dir / "velodyne" / "000000.bin")

    sem = np.array([22, 22, 9, 9, 7, 4], dtype=np.uint32)
    inst = np.zeros(n, dtype=np.uint32)
    ((inst << 16) | sem).astype(np.uint32).tofile(seq_dir / "labels" / "000000.label")

    tag = np.zeros(H * W, dtype=np.uint8)
    tag[[0, 1, 2, 3, 4, 5]] = 1
    tag.astype(np.bool_).tofile(seq_dir / "tag" / "000000.tag")

    return str(tmp_path / "sequences")


def _minimal_config():
    return {
        "model": {
            "image_size": [H, W],
            "original_image_size": [H, W],
            "train_full_image": True,
        },
        "sensor": {
            "name": "Pandora40",
            "type": "spherical",
            "projection_mode": "tag",
            "proj_h": H,
            "proj_w": W,
            "fov_up": 7.0,
            "fov_down": -16.0,
            "fov_left": -180,
            "fov_right": 180,
            "img_mean": [0.0, 0.0, 0.0, 0.0, 0.0],
            "img_stds": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        "augmentation": {
            "p_flipx": 0.0, "p_flipy": 0.0,
            "p_transx": 0.0, "trans_xmin": 0, "trans_xmax": 0,
            "p_transy": 0.0, "trans_ymin": 0, "trans_ymax": 0,
            "p_transz": 0.0, "trans_zmin": 0, "trans_zmax": 0,
            "p_rot_roll": 0.0, "rot_rollmin": 0, "rot_rollmax": 0,
            "p_rot_pitch": 0.0, "rot_pitchmin": 0, "rot_pitchmax": 0,
            "p_rot_yaw": 0.0, "rot_yawmin": 0, "rot_yawmax": 0,
        },
        "adapted_augmentation": {"use_mapped_labels": True},
        "knni": {"enable": False},
    }


def test_loader_returns_correct_shapes(tiny_poss_root):
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "semantic_poss", "semantic-poss.yaml"
    )
    parser = SemanticPOSS(root=tiny_poss_root, sequences=[3], config_path=config_path)
    loader = RangeViewLoader(dataset=parser, config=_minimal_config(), is_train=True)

    feat, label, mask = loader[0]
    assert feat.shape == (5, H, W)
    assert label.shape == (H, W)
    assert mask.shape == (H, W)
    # Mask should mark 6 occupied cells
    assert int(mask.sum().item()) == 6


def test_tinyvim_forward_pass_on_poss_shape():
    from models.tinyvim_adapter import TinyViMAdapter

    adapter = TinyViMAdapter(
        backbone_name="tinyvim_base",
        in_channels=5,
        stem_stride=(1, 2),
        stage_embedding_strides=[(1, 1), (1, 2), (1, 2)],
        use_fpn_decoder=True,
    )
    feats, _ = adapter(torch.randn(1, 5, H, W), return_features=True)
    assert [f.shape[2] for f in feats] == [H, H, H, H]
    assert [f.shape[3] for f in feats] == [900, 900, 450, 225]
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_semanticposs_integration.py -v`
Expected: 2 PASSED. If TinyViM requires CUDA (selective_scan_cuda) and the host lacks it, the second test will be skipped via `pytest.importorskip("torch")` only if torch import fails; otherwise the selective-scan import may raise. If the environment lacks CUDA, mark the second test with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TinyViM SSM")` and re-run.

- [ ] **Step 3: Run the full existing test suite to catch regressions**

Run: `pytest tests/test_tinyvim_integration.py tests/test_semantic_poss_parser.py tests/test_tag_projection.py tests/test_semanticposs_integration.py -v`
Expected: all PASS. If `tests/test_tinyvim_integration.py` was already failing on this branch for unrelated reasons, document that and proceed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_semanticposs_integration.py
git commit -m "test(poss): end-to-end loader -> TinyViM shape check"
```

---

## Task 13: Final regression check on existing pipelines

**Files:** (no edits; verification only)

- [ ] **Step 1: Syntax + import sanity for changed modules**

Run:

```bash
python -c "
import dataset
import dataset.semantic_kitti
import dataset.nuScenes
import dataset.semantic_poss
from dataset.range_view_loader import RangeViewLoader
from models.tinyvim.tinyvim import TinyViM
from models.tinyvim_adapter import TinyViMAdapter
from models.rangevit import *  # noqa
print('all imports ok')
"
```

Expected: `all imports ok`.

- [ ] **Step 2: KITTI config still loads end-to-end**

Run: `python -c "import yaml; cfg = yaml.safe_load(open('config_kitti.yaml')); assert cfg['data']['dataset'] == 'SemanticKitti'; print('kitti config ok')"`
Expected: `kitti config ok`.

- [ ] **Step 3: Default TinyViM stride budget unchanged for KITTI**

Run:

```bash
python -c "
import torch
from models.tinyvim_adapter import TinyViMAdapter
a = TinyViMAdapter(backbone_name='tinyvim_base', in_channels=5, stem_stride=(1, 2), use_fpn_decoder=True)
feats, _ = a(torch.randn(1, 5, 64, 2048), return_features=True)
print('kitti widths:', [f.shape[3] for f in feats])
assert [f.shape[3] for f in feats] == [1024, 512, 256, 128], 'KITTI shape budget regressed'
print('kitti shape budget ok')
"
```

Expected: prints `kitti widths: [1024, 512, 256, 128]` and `kitti shape budget ok`. (No `stage_embedding_strides` arg means the legacy `down_stride=(1,2)` path is used at every transition.)

- [ ] **Step 4: Existing TinyViM tests pass**

Run: `pytest tests/test_tinyvim_integration.py -v`
Expected: same pass/fail status as before this plan started. (If it was passing before, it must still pass.)

- [ ] **Step 5: No commit needed**

This task only verifies — no file changes.

---

## Notes for the executor

- **Real POSS data not required for any task in this plan.** Every test builds a synthetic POSS-shaped directory on disk. The user runs `tools/compute_poss_stats.py` separately against their real data and pastes the output into the two YAMLs before the first real training run.
- **`img_mean`/`img_stds` placeholders ship with the config.** Training will not produce sensible loss until they are replaced. The placeholder is intentional and documented inline in `config_poss.yaml`.
- **Selective-scan CUDA kernel may be missing on dev machines.** If `pytest tests/test_semanticposs_integration.py::test_tinyvim_forward_pass_on_poss_shape` fails to import the SSM kernel, that's an environment issue, not a regression — skip that one test and continue.
- **Augmentations are intentionally disabled for POSS.** Do not re-enable any geometric or mix augmentation in `config_poss.yaml` — the tag mask is the source of truth for point ↔ cell correspondence and any point reordering invalidates it. Range-image-level 2D augmentation (`range_aug`) remains available but is off by default.
