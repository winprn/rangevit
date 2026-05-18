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


from option import Option


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
