import numpy as np

from dataset.preprocess.projection import TagProjection


H, W = 40, 1800


def test_tag_projection_scatters_features_into_correct_cells():
    pts = np.array(
        [[1.0, 2.0, 0.5, 0.1],
         [3.0, -1.0, 0.0, 0.7],
         [-2.0, 1.5, 0.2, 0.4]],
        dtype=np.float32,
    )
    cells = [0, 250, H * W - 1]
    tag = np.zeros(H * W, dtype=np.bool_)
    tag[cells] = True

    proj = TagProjection(proj_h=H, proj_w=W)
    proj_xyz_i, proj_range, proj_idx, proj_mask = proj.doProjection(pts, tag)

    assert proj_xyz_i.shape == (H, W, 4)
    assert proj_range.shape == (H, W)
    assert proj_idx.shape == (H, W)
    assert proj_mask.shape == (H, W)

    flat_mask = proj_mask.reshape(-1)
    assert flat_mask.dtype == bool or flat_mask.dtype == np.bool_
    assert int(flat_mask.sum()) == 3
    for c in cells:
        assert flat_mask[c]

    flat_xyz_i = proj_xyz_i.reshape(-1, 4)
    flat_range = proj_range.reshape(-1)
    flat_idx = proj_idx.reshape(-1)
    for i, c in enumerate(cells):
        np.testing.assert_allclose(flat_xyz_i[c], pts[i])
        np.testing.assert_allclose(flat_range[c], np.linalg.norm(pts[i, :3]))
        assert flat_idx[c] == i


def test_tag_projection_rejects_length_mismatch():
    pts = np.zeros((2, 4), dtype=np.float32)
    tag = np.zeros(H * W, dtype=np.bool_)
    tag[0] = True  # only 1 occupied cell but 2 points
    proj = TagProjection(proj_h=H, proj_w=W)
    import pytest
    with pytest.raises(ValueError):
        proj.doProjection(pts, tag)


def test_tag_projection_caches_uproj_indices():
    pts = np.array([[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.6]], dtype=np.float32)
    tag = np.zeros(H * W, dtype=np.bool_)
    tag[[3, 9]] = True
    proj = TagProjection(proj_h=H, proj_w=W)
    proj.doProjection(pts, tag)
    cached = proj.cached_data
    assert "uproj_x_idx" in cached and "uproj_y_idx" in cached and "uproj_depth" in cached
    # First point lives in cell 3 -> row 0, col 3; second in cell 9 -> row 0, col 9
    assert cached["uproj_y_idx"].tolist() == [0, 0]
    assert cached["uproj_x_idx"].tolist() == [3, 9]
