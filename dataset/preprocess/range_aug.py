# RangeAug: Range Image Augmentation Functions for RangeFormer
# Reference: Kong et al. 2023 - RangeFormer paper
# These are 2D image-level augmentations applied after range projection

import numpy as np
import random
from typing import List


def range_mix(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, yb: np.ndarray,
              kmix: int):
    """
    RangeMix: exchange inclination bands between two scans.

    Args:
        xa, xb: rv arrays (C, H, W)
        ya, yb: label maps (H, W)
        kmix: number of inclination partitions
    """
    xa_ = xa.copy()
    ya_ = ya.copy()

    H, W = xa.shape[1], xa.shape[2]
    band_height = max(1, H // kmix)
    band_indices = list(range(kmix))
    num_exchange = random.randint(1, max(1, kmix - 1))
    exchange_groups = random.sample(band_indices, num_exchange)

    # Optional azimuthal subdivisions to avoid harsh seams
    azimuth_splits = random.choice([1, 2, 4])
    column_width = W // azimuth_splits

    for band in exchange_groups:
        r0 = band * band_height
        r1 = H if band == kmix - 1 else min(H, (band + 1) * band_height)
        if r1 <= r0:
            continue
        for split in range(azimuth_splits):
            c0 = split * column_width
            c1 = W if split == azimuth_splits - 1 else min(W, (split + 1) * column_width)
            if c1 <= c0:
                continue
            xa_[:, r0:r1, c0:c1] = xb[:, r0:r1, c0:c1]
            ya_[r0:r1, c0:c1] = yb[r0:r1, c0:c1]

    return xa_, ya_


def range_union(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, yb: np.ndarray,
                kunion: float = 0.5):
    """
    RangeUnion: fill void pixels (existence channel = 0) in A with valid pixels from B.
    """
    xa_ = xa.copy()
    ya_ = ya.copy()
    existence_a = xa_[-1]
    existence_b = xb[-1]
    void = existence_a == 0
    candidates = np.where(void & (existence_b > 0))
    if candidates[0].size == 0:
        return xa_, ya_

    total_candidates = candidates[0].size
    K = max(1, int(total_candidates * kunion))
    pick_idx = np.random.choice(total_candidates, size=K, replace=False)
    rows = candidates[0][pick_idx]
    cols = candidates[1][pick_idx]

    xa_[:, rows, cols] = xb[:, rows, cols]
    ya_[rows, cols] = yb[rows, cols]
    return xa_, ya_


def range_paste(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, yb: np.ndarray,
                tail_classes: List[int]):
    """
    RangePaste: paste rare semantic classes from xb into xa.
    """
    if not tail_classes:
        return xa, ya
    xa_ = xa.copy()
    ya_ = ya.copy()

    for sem_class in tail_classes:
        mask = (yb == sem_class)
        if not np.any(mask):
            continue
        xa_[:, mask] = xb[:, mask]
        ya_[mask] = sem_class
    return xa_, ya_


def range_shift(xa: np.ndarray, ya: np.ndarray):
    """
    RangeShift: shift image along width (azimuth) by random offset.
    """
    xa_ = xa.copy()
    ya_ = ya.copy()
    h, w = xa.shape[1], xa.shape[2]
    if w <= 1:
        return xa_, ya_
    p = random.randint(int(0.25 * w), int(0.75 * w))
    xa_ = np.concatenate([xa[:, :, p:], xa[:, :, :p]], axis=2)
    ya_ = np.concatenate([ya[:, p:], ya[:, :p]], axis=1)
    return xa_, ya_
