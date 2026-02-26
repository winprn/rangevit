import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.optim.boundary_loss import BoundaryLoss


def _one_hot_prob(label: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(label.long(), num_classes=n_classes).permute(0, 3, 1, 2).float()


def _boundary_loss_module():
    return BoundaryLoss(ignore_index=0, horizontal_wrap=True)


def test_boundary_loss_zero_when_all_invalid():
    criterion = _boundary_loss_module()
    labels = torch.zeros((1, 3, 3), dtype=torch.long)
    mask = torch.zeros_like(labels, dtype=torch.float32)
    prob = _one_hot_prob(labels, n_classes=2)

    loss = criterion(prob, labels, mask)
    assert abs(loss.item()) < 1e-6, f"Expected zero loss for all-invalid sample, got {loss.item()}"


def test_boundary_loss_perfect_prediction_is_zero():
    criterion = _boundary_loss_module()
    labels = torch.tensor([[[1, 2], [2, 1]]], dtype=torch.long)
    mask = torch.ones_like(labels, dtype=torch.float32)
    prob = _one_hot_prob(labels, n_classes=3)

    loss = criterion(prob, labels, mask)
    assert abs(loss.item()) < 1e-6, f"Expected near-zero loss for perfect prediction, got {loss.item()}"


def test_boundary_loss_skips_empty_union_samples():
    criterion = _boundary_loss_module()
    labels = torch.tensor(
        [
            [[1, 1], [1, 1]],
            [[1, 2], [2, 1]],
        ],
        dtype=torch.long,
    )
    mask = torch.ones_like(labels, dtype=torch.float32)
    prob = _one_hot_prob(labels, n_classes=3)

    batch_loss = criterion(prob, labels, mask)
    single_loss = criterion(prob[1:2], labels[1:2], mask[1:2])
    assert torch.allclose(batch_loss, single_loss, atol=1e-6), "Batch loss should ignore samples with empty union"


def test_boundary_loss_mixed_valid_invalid_neighbor_regression():
    criterion = _boundary_loss_module()

    labels = torch.tensor(
        [[[0, 2, 0],
          [0, 1, 0],
          [0, 0, 0]]],
        dtype=torch.long,
    )
    mask = torch.ones_like(labels, dtype=torch.float32)

    good_prob = _one_hot_prob(labels.clamp_min(0), n_classes=3)

    pred_labels_bad = labels.clone()
    pred_labels_bad[0, 0, 1] = 1
    bad_prob = _one_hot_prob(pred_labels_bad.clamp_min(0), n_classes=3)

    good_loss = criterion(good_prob, labels, mask).item()
    bad_loss = criterion(bad_prob, labels, mask).item()
    assert bad_loss > good_loss + 1e-6, f"Expected bad_loss > good_loss, got {bad_loss} vs {good_loss}"


def run_all_tests():
    tests = [
        test_boundary_loss_zero_when_all_invalid,
        test_boundary_loss_perfect_prediction_is_zero,
        test_boundary_loss_skips_empty_union_samples,
        test_boundary_loss_mixed_valid_invalid_neighbor_regression,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
            raise
    print(f"Passed {passed}/{len(tests)} tests.")


if __name__ == "__main__":
    run_all_tests()
