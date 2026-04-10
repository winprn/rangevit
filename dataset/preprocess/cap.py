import numpy as np


def compute_centerness_scores(points_xyz: np.ndarray,
                              instance_ids: np.ndarray,
                              min_points: int = 3,
                              sigma_xyz=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Compute a per-point CAP score from instance-centered anisotropic Gaussian distance.

    Points with instance id 0 or tiny fragments fall back to score 0.
    Scores are normalized to [0, 1] independently for each instance.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    instance_ids = np.asarray(instance_ids).reshape(-1)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f'points_xyz must have shape [N, 3], got {points_xyz.shape}')
    if instance_ids.shape[0] != points_xyz.shape[0]:
        raise ValueError(
            f'instance_ids length ({instance_ids.shape[0]}) must match points ({points_xyz.shape[0]})')

    sigma = np.asarray(sigma_xyz, dtype=np.float32).reshape(1, 3)
    sigma = np.maximum(sigma, 1e-6)
    scores = np.zeros((points_xyz.shape[0],), dtype=np.float32)

    unique_instances = np.unique(instance_ids)
    for ins_id in unique_instances:
        if ins_id == 0:
            continue
        valid_idx = np.where(instance_ids == ins_id)[0]
        if valid_idx.shape[0] < min_points:
            continue

        pts = points_xyz[valid_idx]
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)

        diff = (pts - center[None, :]) / sigma
        sq_mahal = np.sum(diff * diff, axis=1)
        # exp(-0.5 * d^2) is proportional to a Gaussian pdf and avoids scipy dependency here.
        gaussian = np.exp(-0.5 * sq_mahal)
        gmin = float(gaussian.min())
        gmax = float(gaussian.max())
        if gmax > gmin:
            gaussian = (gaussian - gmin) / (gmax - gmin)
        else:
            gaussian = np.zeros_like(gaussian)
        scores[valid_idx] = gaussian.astype(np.float32)

    return scores
