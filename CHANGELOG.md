# Changelog

## 2026-05-18
- feat(poss): re-enable tag-compatible augmentation for SemanticPOSS.
  - `RangeAugmentation` now takes `tail_classes` and `fake_pairs` kwargs (KITTI defaults preserved).
  - `range_aug` config key accepts either legacy bool or new dict block (`enable`, `p_hflip`, `tail_classes`, `probs`); parsing centralized in `Option.parse_range_aug`.
  - New `_resolve_p_hflip` helper in `range_view_loader.py` reads `p_hflip` from the new block with fallback to the legacy `augmentation:` block.
  - `config_poss.yaml` flipped on with POSS-mapped tail classes `[1, 2, 4, 6, 7, 8, 10, 11, 12]` and `p_hflip: 0.5`.
- Files: `dataset/preprocess/rangeaug.py`, `option.py`, `train.py`, `dataset/range_view_loader.py`, `config_poss.yaml`, `tests/test_range_aug_parametrization.py`.
- Follow-ups: retire deprecated `augmentation:` block in POSS once `RangeViewLoader.__init__` uses `.get(...)` defaults; consider Tier 2 (spherical projection) to unlock 3D point-cloud augs.
