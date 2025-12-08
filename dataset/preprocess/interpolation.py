import numpy as np


class RangeInterpolator:
    """Interpolate missing pixels in a range view using a local window."""

    def __init__(self, window_size=(1, 3), min_neighbors=2, ignore_index=19):
        self.window_h = max(1, int(window_size[0]))
        self.window_w = max(1, int(window_size[1]))
        self.min_neighbors = max(1, int(min_neighbors))
        self.ignore_index = ignore_index

    def __call__(
        self,
        proj_pointcloud,
        proj_range,
        proj_sem_label=None,
        proj_mask=None,
        propagate_labels=True,
    ):
        """
        Args:
            proj_pointcloud (np.ndarray): [H, W, C] projected points with -1 for invalid.
            proj_range (np.ndarray): [H, W] range map with -1 for invalid.
            proj_sem_label (np.ndarray | None): [H, W] semantic labels.
            proj_mask (np.ndarray | None): [H, W] binary validity mask (optional).
            propagate_labels (bool): whether to assign labels to interpolated points.

        Returns:
            tuple: updated (proj_pointcloud, proj_range, proj_sem_label, proj_mask)
        """
        h, w = proj_pointcloud.shape[:2]

        base_valid = proj_mask.astype(bool) if proj_mask is not None else (proj_range > 0)

        interp_pointcloud = proj_pointcloud.copy()
        interp_range = proj_range.copy()
        interp_mask = base_valid.copy()
        interp_sem_label = proj_sem_label.copy() if proj_sem_label is not None else None

        half_h = self.window_h // 2
        half_w = self.window_w // 2

        for y in range(h):
            y0 = max(0, y - half_h)
            y1 = min(h, y + half_h + 1)
            for x in range(w):
                if base_valid[y, x]:
                    continue

                x0 = max(0, x - half_w)
                x1 = min(w, x + half_w + 1)

                window_mask = base_valid[y0:y1, x0:x1]
                if window_mask.sum() < self.min_neighbors:
                    continue

                ny, nx = np.nonzero(window_mask)
                ny = ny + y0
                nx = nx + x0

                neighbor_points = proj_pointcloud[ny, nx]
                mean_point = neighbor_points.mean(axis=0)

                interp_pointcloud[y, x] = mean_point
                interp_range[y, x] = np.linalg.norm(mean_point[:3], ord=2)
                interp_mask[y, x] = True

                if propagate_labels and (interp_sem_label is not None):
                    neighbor_labels = proj_sem_label[ny, nx]
                    uniq_labels = np.unique(neighbor_labels)
                    uniq_labels = uniq_labels[uniq_labels != self.ignore_index]
                    if uniq_labels.size == 1:
                        interp_sem_label[y, x] = uniq_labels[0]
                    elif uniq_labels.size > 1:
                        interp_sem_label[y, x] = self.ignore_index

        return (
            interp_pointcloud,
            interp_range,
            interp_sem_label,
            interp_mask.astype(np.int32),
        )
