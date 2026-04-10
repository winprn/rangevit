import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset.preprocess.cap import compute_centerness_scores
from dataset.preprocess.projection import RangeProjection, ScanProjection


def test_cap_scores_center_is_higher_than_edges():
    points = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)
    inst = np.array([5, 5, 5], dtype=np.int32)

    scores = compute_centerness_scores(points, inst)

    assert scores.shape == (3,)
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_range_projection_depth_default_unchanged():
    proj = RangeProjection(fov_up=45.0, fov_down=-45.0, proj_w=8, proj_h=8)
    points = np.array([
        [2.0, 0.0, 0.0, 0.1],
        [1.0, 0.0, 0.0, 0.9],
    ], dtype=np.float32)

    _, _, proj_idx, _ = proj.doProjection(points)

    # Both points collide to the same pixel; baseline keeps the closer point.
    assert int(proj_idx[4, 4]) == 1


def test_range_projection_cap_can_override_depth_order():
    proj = RangeProjection(fov_up=45.0, fov_down=-45.0, proj_w=8, proj_h=8)
    points = np.array([
        [2.0, 0.0, 0.0, 0.1],
        [1.0, 0.0, 0.0, 0.9],
    ], dtype=np.float32)
    # Give the farther point a much stronger CAP score.
    point_scores = np.array([1.0, 0.1], dtype=np.float32)

    _, _, proj_idx, _ = proj.doProjection(points, point_scores=point_scores, score_eps=0.01)

    assert int(proj_idx[4, 4]) == 0


def test_scan_projection_cap_can_override_depth_order():
    proj = ScanProjection(proj_w=8, proj_h=4)
    points = np.array([
        [2.0, 0.0, 0.0, 0.1],
        [1.0, 0.0, 0.0, 0.9],
    ], dtype=np.float32)
    point_scores = np.array([1.0, 0.1], dtype=np.float32)

    _, _, proj_idx, _ = proj.doProjection(points, point_scores=point_scores, score_eps=0.01)

    assert int(proj_idx[0, 3]) == 0
