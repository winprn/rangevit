# RangeAug: Range Image Augmentation Functions for RangeFormer
# Reference: Kong et al. 2023 - RangeFormer paper
# These are 2D image-level augmentations applied after range projection

import numpy as np
import random
from typing import Tuple, List


def range_mix(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, yb: np.ndarray,
              mix_strategy: Tuple[int, int]):
    """
    RangeMix per paper pseudo-code (grid mixing).

    Args:
        xa, xb: rv arrays (C, H, W)
        ya, yb: label maps (H, W)
        mix_strategy: tuple (phi, theta) dividing H and W into blocks

    Returns:
        xa_: mixed range image
        ya_: mixed label map
    """
    xa_ = xa.copy()
    ya_ = ya.copy()
    phi, theta = mix_strategy
    mix_h = int(xa.shape[1] / phi)
    mix_w = int(xa.shape[2] / theta)
    for i in range(1, mix_h + 1):
        for j in range(1, mix_w + 1):
            r0 = (i - 1) * phi
            c0 = (j - 1) * theta
            r1 = min(i * phi, xa.shape[1])
            c1 = min(j * theta, xa.shape[2])
            xa_[:, r0:r1, c0:c1] = xb[:, r0:r1, c0:c1]
            ya_[r0:r1, c0:c1] = yb[r0:r1, c0:c1]
    return xa_, ya_


def range_union(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, yb: np.ndarray,
                kunion: float = 0.5):
    """
    RangeUnion: fill void pixels (existence channel = 0) in A with B

    Args:
        xa, xb: rv arrays (C, H, W)
        ya, yb: label maps (H, W)
        kunion: fraction of void pixels to fill

    Returns:
        xa_: augmented range image
        ya_: augmented label map
    """
    xa_ = xa.copy()
    ya_ = ya.copy()
    mask = xa_[-1, :, :]  # existence channel (last channel)
    void = mask == 0
    # choose random subset of voids (kunion fraction)
    void_coords = np.stack(np.where(void), axis=1)
    if void_coords.shape[0] == 0:
        return xa_, ya_
    K = int(void_coords.shape[0] * kunion)
    pick_idx = np.random.choice(void_coords.shape[0], size=K, replace=False)
    chosen = void_coords[pick_idx]
    for (r, c) in chosen:
        xa_[:, r, c] = xb[:, r, c]
        ya_[r, c] = yb[r, c]
    return xa_, ya_


def range_paste(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, yb: np.ndarray,
                sem_classes: List[int]):
    """
    RangePaste: paste pixels from xb that belong to rare semantic classes into xa

    Args:
        xa, xb: rv arrays (C, H, W)
        ya, yb: label maps (H, W)
        sem_classes: list of rare semantic class IDs to paste

    Returns:
        xa_: augmented range image
        ya_: augmented label map
    """
    xa_ = xa.copy()
    ya_ = ya.copy()
    for sem_class in sem_classes:
        pix = (yb == sem_class)
        if pix.sum() == 0:
            continue
        xa_[:, pix] = xb[:, pix]
        ya_[pix] = yb[pix]
    return xa_, ya_


def range_shift(xa: np.ndarray, ya: np.ndarray):
    """
    RangeShift: shift image along width (azimuth) by random kshift in [W/4, 3W/4]

    Args:
        xa: rv array (C, H, W)
        ya: label map (H, W)

    Returns:
        xa_: shifted range image
        ya_: shifted label map
    """
    xa_ = xa.copy()
    ya_ = ya.copy()
    h, w = xa.shape[1], xa.shape[2]
    p = random.randint(int(0.25 * w), int(0.75 * w))
    xa_ = np.concatenate([xa[:, :, p:], xa[:, :, :p]], axis=2)
    ya_ = np.concatenate([ya[:, p:], ya[:, :p]], axis=1)
    return xa_, ya_
