# Changelog

## 2026-05-18
- feat(rangeaug): parametrize `RangeAugmentation` with `tail_classes` and `fake_pairs` kwargs so non-KITTI datasets (POSS) can configure tail-class behaviour. KITTI defaults preserved.
- Files: `dataset/preprocess/rangeaug.py`, `tests/test_range_aug_parametrization.py`
