import numpy as np


class KNNI(object):
    """
    Range-dependent KNNI (row-wise).
    Fills invalid pixels by copying the nearest (smallest range) valid neighbor
    within a horizontal window on the same row.
    """
    def __init__(self, window_size=5, invalid_value=-1.0):
        if window_size < 3 or (window_size % 2) == 0:
            raise ValueError("window_size must be an odd integer >= 3")
        self.window_size = int(window_size)
        self.invalid_value = float(invalid_value)

    def __call__(self, proj_pointcloud, proj_range, proj_mask):
        if proj_range.ndim != 2:
            raise ValueError("proj_range must be HxW")
        if proj_pointcloud.ndim != 3:
            raise ValueError("proj_pointcloud must be HxWxC")
        if proj_mask.shape != proj_range.shape:
            raise ValueError("proj_mask must match proj_range shape")

        h, w = proj_range.shape
        half = self.window_size // 2

        # Use original projections as candidates (do not chain-fill).
        src_range = proj_range
        src_pointcloud = proj_pointcloud
        src_mask = proj_mask

        out_pointcloud = proj_pointcloud.copy()
        out_range = proj_range.copy()
        out_mask = proj_mask.copy()

        for v in range(h):
            for u in range(w):
                if out_mask[v, u]:
                    continue
                best_u = None
                best_range = None
                for s in range(-half, half + 1):
                    if s == 0:
                        continue
                    uu = u + s
                    if uu < 0 or uu >= w:
                        continue
                    if src_mask[v, uu] == 0:
                        continue
                    cand_range = src_range[v, uu]
                    if cand_range <= self.invalid_value:
                        continue
                    if best_range is None or cand_range < best_range:
                        best_range = cand_range
                        best_u = uu
                if best_u is not None:
                    out_pointcloud[v, u] = src_pointcloud[v, best_u]
                    out_range[v, u] = src_range[v, best_u]
                    out_mask[v, u] = 1

        return out_pointcloud, out_range, out_mask
