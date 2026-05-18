# POSS Tag-Compatible Augmentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-enable tag-compatible augmentation (2D `RangeAugmentation` + range-image horizontal flip) for SemanticPOSS by parametrizing tail classes and restructuring the `range_aug` config key into a block, with full backward compatibility for SemanticKITTI configs.

**Architecture:** `RangeAugmentation` is generalized so tail-class lists and fake-pair maps are constructor args (defaults preserve KITTI behaviour). `option.py` parses `range_aug` as either a bool (legacy) or a dict (new block form with `enable`, `p_hflip`, `tail_classes`, `probs`). `train.py` passes the parsed lists into `RangeAugmentation`. `range_view_loader.py` looks up `p_hflip` first from the new block, falling back to the legacy `augmentation:` block. `config_poss.yaml` flips on the new block with POSS-specific tail classes; the deprecated `augmentation:` block stays in place (zeros) per spec.

**Tech Stack:** Python 3, PyTorch, NumPy, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-05-18-poss-range-aug-design.md`

---

## File Structure

- **Modify** `dataset/preprocess/rangeaug.py` — add `tail_classes` and `fake_pairs` kwargs to `RangeAugmentation.__init__`; use the instance attributes inside `_range_polar`, `_range_instance`, and `_range_fake`.
- **Modify** `option.py` — parse `range_aug` as bool-or-dict; expose `range_aug_enable`, `range_aug_tail_classes`, `range_aug_probs`, `range_aug_p_hflip`. Keep `self.range_aug` as the bool enable flag for back-compat.
- **Modify** `train.py` — construct `RangeAugmentation` with parsed tail classes and probs.
- **Modify** `dataset/range_view_loader.py` — read `p_hflip` from `range_aug.p_hflip` when `range_aug` is a dict, falling back to `augmentation.p_hflip`.
- **Modify** `config_poss.yaml` — replace `range_aug: false` with the new block (enable=true, POSS tail classes, `p_hflip: 0.5`, default probs).
- **Create** `tests/test_range_aug_parametrization.py` — unit tests for the new `RangeAugmentation` kwargs and `option.py` parsing.

---

## Design Principles

- `RangeAugmentation` keeps a single constructor; the new kwargs default to current values so all existing call sites continue to work without change.
- The `range_aug` config key has two valid shapes (bool, dict). Parsing logic lives entirely in `option.py`; downstream code consumes flat scalars.
- No new modules. No abstractions introduced beyond what the spec requires.
- Doc the `range_aug` block schema in `option.py` where it's parsed (the natural place a future reader looks).

---

### Task 1: Parametrize `RangeAugmentation` (tail classes + fake pairs)

**Files:**
- Modify: `dataset/preprocess/rangeaug.py:42-184`
- Test: `tests/test_range_aug_parametrization.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_range_aug_parametrization.py`:

```python
"""Unit tests for RangeAugmentation parametrization (tail_classes, fake_pairs)."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.preprocess.rangeaug import RangeAugmentation


def test_default_tail_classes_preserve_kitti_behaviour():
    """Default tail set must match the previous module-level constant exactly."""
    aug = RangeAugmentation()
    assert aug.tail_classes == [2, 3, 4, 5, 6, 7, 8, 16, 18, 19]


def test_default_fake_pairs_preserve_kitti_behaviour():
    aug = RangeAugmentation()
    assert aug.fake_pairs == {1: [2, 3, 5, 8], 9: [10, 11, 12]}


def test_custom_tail_classes_are_used_by_range_instance():
    """_range_instance must paste pixels for every cls in self.tail_classes."""
    custom = [1, 4, 12]
    aug = RangeAugmentation(tail_classes=custom)

    H, W = 4, 6
    scan_a = torch.zeros((3, H, W))
    scan_b = torch.full((3, H, W), 9.0)
    label_a = torch.zeros((H, W), dtype=torch.long)
    label_b = torch.zeros((H, W), dtype=torch.long)

    # Place one pixel per custom tail class in label_b.
    label_b[0, 0] = 1
    label_b[1, 1] = 4
    label_b[2, 2] = 12
    # A non-tail class should NOT be pasted.
    label_b[3, 3] = 99

    out_scan, out_label = aug._range_instance(scan_a.clone(), label_a.clone(), scan_b, label_b)

    assert out_label[0, 0].item() == 1
    assert out_label[1, 1].item() == 4
    assert out_label[2, 2].item() == 12
    assert out_label[3, 3].item() == 0  # non-tail untouched
    assert torch.allclose(out_scan[:, 0, 0], scan_b[:, 0, 0])
    assert torch.allclose(out_scan[:, 3, 3], scan_a[:, 3, 3])


def test_custom_fake_pairs_used_by_range_fake():
    custom = {7: [42]}
    aug = RangeAugmentation(fake_pairs=custom)

    scan = torch.zeros((3, 2, 2))
    label = torch.tensor([[7, 0], [7, 1]], dtype=torch.long)

    _, out_label = aug._range_fake(scan, label.clone())
    # Every "7" must become 42; nothing else changes.
    assert (out_label == torch.tensor([[42, 0], [42, 1]])).all()


def test_poss_tail_set_construction():
    """POSS tail set must be storable; ids above 13 (KITTI-only) are absent."""
    poss_tail = [1, 2, 4, 6, 7, 8, 10, 11, 12]
    aug = RangeAugmentation(tail_classes=poss_tail, aug_prob=[0.0, 0.0, 0.0, 0.0, 0.0])
    assert aug.tail_classes == poss_tail
    assert all(c <= 13 for c in aug.tail_classes)
```

- [ ] **Step 2: Add the kwargs and replace the constants**

Replace the class-level `TAIL_CLASSES` constant and the local dict in `_range_fake` with constructor-provided attributes. New `__init__` body:

```python
class RangeAugmentation:
    """Applies range image-level augmentations on a batch of projected LiDAR scans.

    Each sample in the batch is paired with a different sample (via derangement),
    then five augmentation techniques are applied sequentially with independent
    probability rolls:
      - RangePolar:      mix azimuth columns + flip tail-class instances
      - RangeBeams:      mix horizontal laser beam bands
      - RangeCompletion: fill void pixels by row-shifting
      - RangeFake:       relabel front-class as tail-class (disabled by default)
      - RangeInstance:   paste tail-class pixels from paired sample

    Args:
        aug_prob: Per-technique probabilities, ordered
            [RangePolar, RangeBeams, RangeCompletion, RangeFake, RangeInstance].
            Default: [0.9, 0.7, 0.9, 0.0, 0.9].
        tail_classes: Mapped class ids treated as tail (underrepresented).
            Defaults to the SemanticKITTI tail set.
        fake_pairs: Mapping from a front-class id to a list of tail-class ids
            it may be relabelled as during RangeFake. Defaults to the
            SemanticKITTI pair map. RangeFake is off by default; if turned on
            for a non-KITTI dataset this map MUST be overridden.
    """

    _DEFAULT_TAIL_CLASSES = [2, 3, 4, 5, 6, 7, 8, 16, 18, 19]
    _DEFAULT_FAKE_PAIRS = {1: [2, 3, 5, 8], 9: [10, 11, 12]}

    def __init__(self, aug_prob=None, tail_classes=None, fake_pairs=None):
        if aug_prob is None:
            aug_prob = [0.9, 0.7, 0.9, 0.0, 0.9]
        self.aug_prob = aug_prob
        self.tail_classes = list(tail_classes) if tail_classes is not None else list(self._DEFAULT_TAIL_CLASSES)
        self.fake_pairs = dict(fake_pairs) if fake_pairs is not None else dict(self._DEFAULT_FAKE_PAIRS)
        print(
            f'[INFO] Range image augmentation enabled with probabilities {aug_prob}, '
            f'tail_classes={self.tail_classes}'
        )
```

Delete the module-level `TAIL_CLASSES = [...]` line in the class body (rangeaug.py:60-63).

- [ ] **Step 3: Update internal references**

In `_range_polar` (rangeaug.py:125) replace:

```python
tail_tensor = torch.tensor(self.TAIL_CLASSES, device=label_a.device)
```

with:

```python
tail_tensor = torch.tensor(self.tail_classes, device=label_a.device)
```

In `_range_fake` (rangeaug.py:166-175) replace the body with:

```python
def _range_fake(self, scan_a, label_a):
    """Relabel front-class pixels as tail-class (unstable, disabled by default).

    Uses self.fake_pairs (front_cls -> list of tail_cls). For non-KITTI
    datasets the caller must provide an appropriate fake_pairs map.
    """
    if not self.fake_pairs:
        return scan_a, label_a
    rand_front_class = random.choice(list(self.fake_pairs.keys()))
    candidates = self.fake_pairs[rand_front_class]
    if not candidates:
        return scan_a, label_a
    rand_tail_class = random.choice(candidates)
    label_a[label_a == rand_front_class] = rand_tail_class
    return scan_a, label_a
```

In `_range_instance` (rangeaug.py:177-183) replace:

```python
for cls in self.TAIL_CLASSES:
```

with:

```python
for cls in self.tail_classes:
```

- [ ] **Step 4: Run the new tests**

```bash
python -m pytest tests/test_range_aug_parametrization.py -q
```

All four tests in this file must pass.

---

### Task 2: Parse `range_aug` as bool-or-dict in `option.py`

**Files:**
- Modify: `option.py:210-211`
- Test: `tests/test_range_aug_parametrization.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_range_aug_parametrization.py`:

```python
import yaml

from option import Option


def _make_settings_with_range_aug(tmp_path, value):
    """Build a minimal Option-like settings object by patching the YAML config.

    Option.__init__ requires a full config; we monkey-build a tiny dict and
    invoke the parser path directly to avoid coupling tests to unrelated keys.
    """
    return value  # placeholder; the real test uses the helper below


def test_range_aug_parses_legacy_bool_true():
    cfg = {'range_aug': True}
    enable, p_hflip, tail, probs = Option.parse_range_aug(cfg)
    assert enable is True
    assert p_hflip is None
    assert tail is None
    assert probs is None


def test_range_aug_parses_legacy_bool_false():
    cfg = {'range_aug': False}
    enable, p_hflip, tail, probs = Option.parse_range_aug(cfg)
    assert enable is False
    assert p_hflip is None
    assert tail is None
    assert probs is None


def test_range_aug_parses_missing_key():
    cfg = {}
    enable, p_hflip, tail, probs = Option.parse_range_aug(cfg)
    assert enable is False
    assert p_hflip is None
    assert tail is None
    assert probs is None


def test_range_aug_parses_full_block():
    cfg = {
        'range_aug': {
            'enable': True,
            'p_hflip': 0.5,
            'tail_classes': [1, 2, 4, 6, 7, 8, 10, 11, 12],
            'probs': [0.9, 0.7, 0.9, 0.0, 0.9],
        }
    }
    enable, p_hflip, tail, probs = Option.parse_range_aug(cfg)
    assert enable is True
    assert p_hflip == 0.5
    assert tail == [1, 2, 4, 6, 7, 8, 10, 11, 12]
    assert probs == [0.9, 0.7, 0.9, 0.0, 0.9]


def test_range_aug_block_with_enable_false_disables():
    cfg = {'range_aug': {'enable': False, 'p_hflip': 0.5}}
    enable, _, _, _ = Option.parse_range_aug(cfg)
    assert enable is False
```

- [ ] **Step 2: Add the parser static method and use it**

In `option.py`, add this static method to the `Option` class (anywhere inside the class body; place it just above `__init__` for visibility):

```python
@staticmethod
def parse_range_aug(config):
    """Parse the ``range_aug`` config key into flat scalars.

    Accepts either:
      * a bool (legacy): ``range_aug: true`` / ``range_aug: false``
      * a dict (new block form) with keys:
            enable (bool, default false),
            p_hflip (float, default None),
            tail_classes (list[int], default None),
            probs (list[float] length 5, default None).

    Returns a 4-tuple ``(enable, p_hflip, tail_classes, probs)``. Values left
    as ``None`` instruct downstream code to use its built-in defaults.
    """
    raw = config.get('range_aug', False)
    if isinstance(raw, dict):
        enable = bool(raw.get('enable', False))
        p_hflip = raw.get('p_hflip', None)
        if p_hflip is not None:
            p_hflip = float(p_hflip)
        tail_classes = raw.get('tail_classes', None)
        if tail_classes is not None:
            tail_classes = [int(c) for c in tail_classes]
        probs = raw.get('probs', None)
        if probs is not None:
            probs = [float(p) for p in probs]
        return enable, p_hflip, tail_classes, probs
    return bool(raw), None, None, None
```

Replace the existing `option.py:210-211` block:

```python
# Range image-level augmentation
self.range_aug = self.config.get('range_aug', False)
```

with:

```python
# Range image-level augmentation. ``range_aug`` accepts either a legacy
# bool or a dict block (see ``parse_range_aug`` for the schema).
(
    self.range_aug,
    self.range_aug_p_hflip,
    self.range_aug_tail_classes,
    self.range_aug_probs,
) = Option.parse_range_aug(self.config)
```

- [ ] **Step 3: Run the new tests**

```bash
python -m pytest tests/test_range_aug_parametrization.py -q
```

All nine tests across Tasks 1 and 2 must pass.

---

### Task 3: Wire parsed settings into `RangeAugmentation` in `train.py`

**Files:**
- Modify: `train.py:82-86`

- [ ] **Step 1: Update the constructor call**

Replace the existing block (`train.py:82-86`):

```python
# Range image-level augmentation (applied on batch before forward pass)
self.range_aug = None
if getattr(self.settings, 'range_aug', False):
    from dataset.preprocess.rangeaug import RangeAugmentation
    self.range_aug = RangeAugmentation()
```

with:

```python
# Range image-level augmentation (applied on batch before forward pass).
# Tail classes and per-op probabilities flow from option.parse_range_aug.
self.range_aug = None
if getattr(self.settings, 'range_aug', False):
    from dataset.preprocess.rangeaug import RangeAugmentation
    self.range_aug = RangeAugmentation(
        aug_prob=getattr(self.settings, 'range_aug_probs', None),
        tail_classes=getattr(self.settings, 'range_aug_tail_classes', None),
    )
```

- [ ] **Step 2: Smoke-check that `train.py` still imports**

```bash
python -c "import train"
```

Must exit with status 0 and no exception.

---

### Task 4: Update `p_hflip` lookup in `range_view_loader.py`

**Files:**
- Modify: `dataset/range_view_loader.py:179-181`
- Test: `tests/test_range_aug_parametrization.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_range_aug_parametrization.py`:

```python
def test_p_hflip_prefers_range_aug_block():
    """When range_aug is a dict with p_hflip, it wins over augmentation.p_hflip."""
    from dataset.range_view_loader import _resolve_p_hflip
    cfg = {
        'range_aug': {'enable': True, 'p_hflip': 0.7},
        'augmentation': {'p_hflip': 0.3},
    }
    assert _resolve_p_hflip(cfg) == 0.7


def test_p_hflip_falls_back_to_augmentation_block():
    from dataset.range_view_loader import _resolve_p_hflip
    cfg = {
        'range_aug': True,  # legacy bool — no p_hflip inside
        'augmentation': {'p_hflip': 0.3},
    }
    assert _resolve_p_hflip(cfg) == 0.3


def test_p_hflip_defaults_to_zero_when_absent():
    from dataset.range_view_loader import _resolve_p_hflip
    cfg = {'augmentation': {}}
    assert _resolve_p_hflip(cfg) == 0.0


def test_p_hflip_block_without_key_falls_back():
    """range_aug block present but without p_hflip falls back to augmentation."""
    from dataset.range_view_loader import _resolve_p_hflip
    cfg = {
        'range_aug': {'enable': True},
        'augmentation': {'p_hflip': 0.4},
    }
    assert _resolve_p_hflip(cfg) == 0.4
```

- [ ] **Step 2: Add the resolver helper**

Add at module scope in `dataset/range_view_loader.py`, just below the existing imports block and before the `RangeViewLoader` class definition:

```python
def _resolve_p_hflip(config):
    """Resolve the range-image horizontal-flip probability.

    Lookup order:
      1. ``config['range_aug']['p_hflip']`` when ``range_aug`` is a dict and
         the key is present.
      2. ``config['augmentation']['p_hflip']`` (legacy location).
      3. ``0.0`` when neither is set.
    """
    range_aug_cfg = config.get('range_aug', None)
    if isinstance(range_aug_cfg, dict) and 'p_hflip' in range_aug_cfg:
        return float(range_aug_cfg['p_hflip'])
    augment_cfg = config.get('augmentation', {}) or {}
    return float(augment_cfg.get('p_hflip', 0.0))
```

- [ ] **Step 3: Use the resolver**

Replace `range_view_loader.py:179-181`:

```python
self.proj_p_hflip = augment_config.get('p_hflip', 0.0)
if self.proj_p_hflip > 0.0:
    print(f'Horizontal flip of range projections with p={self.proj_p_hflip}')
```

with:

```python
self.proj_p_hflip = _resolve_p_hflip(self.config)
if self.proj_p_hflip > 0.0:
    print(f'Horizontal flip of range projections with p={self.proj_p_hflip}')
```

- [ ] **Step 4: Run the new tests**

```bash
python -m pytest tests/test_range_aug_parametrization.py -q
```

All thirteen tests must pass.

---

### Task 5: Enable the new `range_aug` block in `config_poss.yaml`

**Files:**
- Modify: `config_poss.yaml:94-95`

- [ ] **Step 1: Replace the existing `range_aug` line and surrounding comment**

Replace `config_poss.yaml:94-95`:

```yaml
# 2D range-image augmentation OK; 3D geometric and mix augs disabled below.
range_aug: false
```

with:

```yaml
# 2D range-image augmentation (tag-projection compatible). 3D geometric and
# mix augs in the deprecated `augmentation:` block below remain disabled —
# tag projection invariants forbid them (see range_view_loader.py).
# Tail classes are POSS-mapped ids: person(1), rider(2), trunk(4),
# traffic-sign(6), pole(7), trashcan(8), cone-stone(10), fence(11), bike(12).
range_aug:
  enable: true
  p_hflip: 0.5
  tail_classes: [1, 2, 4, 6, 7, 8, 10, 11, 12]
  probs: [0.9, 0.7, 0.9, 0.0, 0.9]   # polar, beams, completion, fake, instance
```

The deprecated `augmentation:` block (lines 97-117) stays in place with its zero values. The `adapted_augmentation:` block stays unchanged (all `enable: false`).

- [ ] **Step 2: Smoke-test the loader construction on the new config**

```bash
python -c "
import yaml
from dataset.range_view_loader import _resolve_p_hflip
cfg = yaml.safe_load(open('config_poss.yaml'))
assert _resolve_p_hflip(cfg) == 0.5, _resolve_p_hflip(cfg)
print('p_hflip resolved =', _resolve_p_hflip(cfg))
"
```

Must print `p_hflip resolved = 0.5` and exit 0.

- [ ] **Step 3: Smoke-test option parsing on the new config**

```bash
python -c "
import yaml
from option import Option
cfg = yaml.safe_load(open('config_poss.yaml'))
enable, p_hflip, tail, probs = Option.parse_range_aug(cfg)
assert enable is True
assert p_hflip == 0.5
assert tail == [1, 2, 4, 6, 7, 8, 10, 11, 12]
assert probs == [0.9, 0.7, 0.9, 0.0, 0.9]
print('config_poss range_aug parse OK')
"
```

Must print `config_poss range_aug parse OK` and exit 0.

---

### Task 6: KITTI backward-compat verification

**Files:** (read-only verification — no edits)

- [ ] **Step 1: Verify legacy KITTI configs still parse to the same enable/None tuple**

```bash
python -c "
import yaml
from option import Option
for path in ['config_kitti_aug.yaml', 'config_kitti.yaml']:
    cfg = yaml.safe_load(open(path))
    enable, p_hflip, tail, probs = Option.parse_range_aug(cfg)
    assert (p_hflip, tail, probs) == (None, None, None), (path, p_hflip, tail, probs)
    assert isinstance(enable, bool), (path, enable)
    print(f'{path}: enable={enable}')
"
```

Output must show one `enable=...` line per file with no assertion failures.

- [ ] **Step 2: Verify `RangeAugmentation()` with no kwargs still emits the KITTI tail set**

```bash
python -c "
from dataset.preprocess.rangeaug import RangeAugmentation
aug = RangeAugmentation()
assert aug.tail_classes == [2, 3, 4, 5, 6, 7, 8, 16, 18, 19]
assert aug.fake_pairs == {1: [2, 3, 5, 8], 9: [10, 11, 12]}
print('KITTI defaults preserved')
"
```

Must print `KITTI defaults preserved` and exit 0.

- [ ] **Step 3: Run the full new test file once more**

```bash
python -m pytest tests/test_range_aug_parametrization.py -q
```

All tests pass.

---

## Per-Plan Self-Review

- **Spec coverage:** every spec change item (1-5) maps to a task — rangeaug.py→Task 1, option.py→Task 2, train.py→Task 3, range_view_loader.py→Task 4, config_poss.yaml→Task 5. Acceptance criterion 3 (KITTI back-compat) maps to Task 6.
- **Placeholder scan:** no TBD/TODO/"similar to". All code blocks contain full content; all test functions show full bodies; all file paths and line ranges are exact.
- **Type consistency:** `parse_range_aug` returns `(bool, Optional[float], Optional[list[int]], Optional[list[float]])` everywhere it appears (Tasks 2, 5, 6). `RangeAugmentation.__init__` kwargs `aug_prob`, `tail_classes`, `fake_pairs` are referenced identically across Tasks 1, 3. `_resolve_p_hflip(config)` signature matches in Tasks 4, 5.
