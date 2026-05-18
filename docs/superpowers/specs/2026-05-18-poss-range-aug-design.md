# SemanticPOSS — re-enable tag-compatible augmentation (Tier 1)

**Date:** 2026-05-18
**Status:** Design
**Scope:** Only Tier 1 (2D range-image augmentations + horizontal range-image flip). Tier 2 (switching POSS to spherical projection to unlock 3D point-cloud augs) is explicitly out of scope.

## Background

`config_poss.yaml` currently disables augmentation. Two independent disables are in play:

1. **Config-level:** `range_aug: false`, and every entry in the `augmentation:` block is zeroed.
2. **Code-level (forced):** `RangeViewLoader.__init__` (dataset/range_view_loader.py:202-210) sets `self.augmentor`, `point_sampler`, `polarmix`, `instance_cutmix`, `clustermix`, `instance_copy` to `None` whenever `projection_mode == 'tag'`. POSS uses tag projection, so even if the config probabilities were non-zero, all of those would still be inert.

The reason is structural: `TagProjection` reads a precomputed boolean H×W tag mask from disk (`*.tag`) that pins each point to a fixed (row,col), and asserts `tag.sum() == n_points`. Any 3D rotation/translation/flip, point-subset, or mix-augmentation invalidates that mapping.

A separate family of augmentations operates **after projection** on the (B,C,H,W) batched range image and has no dependency on the tag mask. That family is fully tag-compatible and is what this spec re-enables.

## Tag-compatible augmentations (in scope)

- **`RangeAugmentation`** (`dataset/preprocess/rangeaug.py`) — applied on GPU at `train.py:492` and `train.py:872`. Five ops: range_polar, range_beams, range_completion, range_fake, range_instance.
- **`p_hflip`** — 2D pixel flip of the projected range image inside `crop_inputs` (`range_view_loader.py:573`). Mirrors `px *= -1`. Currently keyed off `augmentation.p_hflip`.

## Known issues with `RangeAugmentation` for POSS

1. `TAIL_CLASSES = [2, 3, 4, 5, 6, 7, 8, 16, 18, 19]` (rangeaug.py:63) is hardcoded for SemanticKITTI's 20-class mapped taxonomy. POSS has 14 mapped classes (0..13), so 16/18/19 silently no-op; 2..8 partially overlap but include head classes (car=3, plants=5) and miss real POSS tail classes (person=1, bike=12, cone-stone=10, fence=11).
2. `_range_fake` (rangeaug.py:166) uses KITTI-specific class pairs (1→bicycle, 9→parking…). Defaulted off (`aug_prob[3] = 0.0`), so non-blocking — but it must remain off for POSS or get its own POSS pair map.

True POSS tail classes by content frequency (`dataset/semantic_poss/semantic-poss.yaml`): cone-stone (10), traffic-sign (6), trashcan (8), rider (2), pole (7), trunk (4), fence (11), bike (12), person (1). Recommended POSS tail set: `[1, 2, 4, 6, 7, 8, 10, 11, 12]`.

## Design

### Config schema (new `range_aug` block form)

Restructure `range_aug` from a bare bool into an optional block, with backward-compat:

```yaml
range_aug:
  enable: true
  p_hflip: 0.5
  tail_classes: [1, 2, 4, 6, 7, 8, 10, 11, 12]   # POSS tail
  probs: [0.9, 0.7, 0.9, 0.0, 0.9]               # polar, beams, completion, fake, instance
```

Back-compat: `range_aug: true` or `range_aug: false` continues to parse — when bool, defaults are taken (KITTI tail classes, default probs, `p_hflip` from the legacy `augmentation.p_hflip` if present).

`p_hflip` lookup order in the loader: `range_aug.p_hflip` → `augmentation.p_hflip` → `0.0`.

The deprecated `augmentation:` block is left as-is in `config_poss.yaml` (all zeros). The loader still reads it for the 3D `Augmentor`, but on POSS those values are also gated to `None` by the tag-projection branch, so they have no effect. Retiring the block is deferred follow-up.

### Code changes

1. **`dataset/preprocess/rangeaug.py`**
   - `RangeAugmentation.__init__` gains `tail_classes` and `fake_pairs` kwargs. Defaults preserve current KITTI behaviour:
     `tail_classes=[2,3,4,5,6,7,8,16,18,19]`,
     `fake_pairs={1: [2,3,5,8], 9: [10,11,12]}`.
   - Replace the class constant `TAIL_CLASSES` with `self.tail_classes`; replace the inline dict in `_range_fake` with `self.fake_pairs`.

2. **`option.py`**
   - Replace `self.range_aug = self.config.get('range_aug', False)` with logic that accepts bool or dict:
     - bool → `enable=<bool>`, defaults elsewhere.
     - dict → read `enable`, `p_hflip`, `tail_classes`, `probs`.
   - Expose on settings: `range_aug` (bool, kept as the on/off flag for back-compat with `train.py:84`), `range_aug_tail_classes`, `range_aug_probs`, `range_aug_p_hflip` (Optional[float]).

3. **`train.py:85`**
   - `self.range_aug = RangeAugmentation(tail_classes=settings.range_aug_tail_classes, aug_prob=settings.range_aug_probs)`.

4. **`dataset/range_view_loader.py:179`**
   - `p_hflip` lookup: prefer `self.config.get('range_aug', {}).get('p_hflip')` when `range_aug` is a dict; else fall back to `augment_config.get('p_hflip', 0.0)`.

5. **`config_poss.yaml`**
   - Replace the `range_aug: false` line with the new block (enable=true, POSS tail classes, `p_hflip: 0.5`, default probs with fake=0.0).
   - Leave the existing `augmentation:` block untouched (all zeros).

### Files touched

- `dataset/preprocess/rangeaug.py`
- `option.py`
- `train.py`
- `dataset/range_view_loader.py`
- `config_poss.yaml`

### Files NOT touched (intentional)

- KITTI configs (`config_kitti_aug.yaml`, etc.) — back-compat keeps `range_aug: true` working with unchanged defaults.
- `dataset/preprocess/projection.py`, `dataset/semantic_poss/parser.py` — no projection-mode change.
- `adapted_augmentation:` (polarmix / cutmix / clustermix / instance_copy) — incompatible with tag projection; out of scope.

## Acceptance

1. Training POSS with the new `range_aug` block runs without runtime errors and prints `"[INFO] Range image augmentation enabled with probabilities [...]"` once at startup.
2. Training POSS with the new block prints the `p_hflip` enablement line at loader construction.
3. KITTI training with `range_aug: true` (legacy bool form) still runs and behaves identically to the current code (same tail classes, same default probabilities).
4. A run of POSS with augmentation enabled shows mIoU recovery vs. the current augmentation-disabled baseline. (Measurement is post-implementation, not a code criterion.)

## Risks

- **Over-mixing on a small dataset.** POSS train set (sequences 00,01,02,04,05) is smaller than KITTI. Default per-op probs `[0.9, 0.7, 0.9, 0.0, 0.9]` may be aggressive. Mitigation: probs are config-exposed; can be retuned without code change.
- **Tail-class list correctness.** The proposed `[1, 2, 4, 6, 7, 8, 10, 11, 12]` is derived from content frequencies in `semantic-poss.yaml`; if a class is actually well-represented in practice (e.g., bike during dense scenes), inclusion may over-amplify. Adjustable via config.
- **`_range_instance` shape coupling.** `_range_instance` pastes columns indexed by `label_b == cls`. Works on any (H,W); 40×1800 vs. 64×2048 is fine.

## Follow-ups (out of scope for this spec)

- Retire the deprecated `augmentation:` block by making `RangeViewLoader.__init__` use `.get(...)` defaults instead of `augment_config['p_flipx']` etc. — then delete the block from `config_poss.yaml`.
- Tier 2: switch POSS to spherical projection (`projection_mode: "spherical"`) to unlock 3D augmentations (polarmix, instance_cutmix, etc.). Requires validating per-pixel layout equivalence with the precomputed tag mask and likely rebuilding the instance bank for POSS.
- Move POSS `_range_fake` to use POSS-specific pairs if anyone enables it (currently defaulted off).
