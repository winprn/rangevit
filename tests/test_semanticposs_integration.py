import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataset.semantic_poss.parser import SemanticPOSS
from dataset.range_view_loader import RangeViewLoader


H, W = 40, 1800


@pytest.fixture
def tiny_poss_root(tmp_path):
    seq = "03"
    seq_dir = tmp_path / "sequences" / seq
    (seq_dir / "velodyne").mkdir(parents=True)
    (seq_dir / "labels").mkdir()
    (seq_dir / "tag").mkdir()

    n = 6
    pts = np.array(
        [[1, 0, 0, 0.2], [0, 1, 0, 0.3], [-1, 0, 0, 0.4],
         [0, -1, 0, 0.5], [2, 2, 0.5, 0.6], [-2, -2, 0.1, 0.7]],
        dtype=np.float32,
    )
    pts.tofile(seq_dir / "velodyne" / "000000.bin")

    sem = np.array([22, 22, 9, 9, 7, 4], dtype=np.uint32)
    inst = np.zeros(n, dtype=np.uint32)
    ((inst << 16) | sem).astype(np.uint32).tofile(seq_dir / "labels" / "000000.label")

    tag = np.zeros(H * W, dtype=np.uint8)
    tag[[0, 1, 2, 3, 4, 5]] = 1
    tag.astype(np.bool_).tofile(seq_dir / "tag" / "000000.tag")

    return str(tmp_path / "sequences")


def _minimal_config():
    return {
        "model": {
            "image_size": [H, W],
            "original_image_size": [H, W],
            "train_full_image": True,
        },
        "sensor": {
            "name": "Pandora40",
            "type": "spherical",
            "projection_mode": "tag",
            "proj_h": H,
            "proj_w": W,
            "fov_up": 7.0,
            "fov_down": -16.0,
            "fov_left": -180,
            "fov_right": 180,
            "img_mean": [0.0, 0.0, 0.0, 0.0, 0.0],
            "img_stds": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        "augmentation": {
            "p_flipx": 0.0, "p_flipy": 0.0,
            "p_transx": 0.0, "trans_xmin": 0, "trans_xmax": 0,
            "p_transy": 0.0, "trans_ymin": 0, "trans_ymax": 0,
            "p_transz": 0.0, "trans_zmin": 0, "trans_zmax": 0,
            "p_rot_roll": 0.0, "rot_rollmin": 0, "rot_rollmax": 0,
            "p_rot_pitch": 0.0, "rot_pitchmin": 0, "rot_pitchmax": 0,
            "p_rot_yaw": 0.0, "rot_yawmin": 0, "rot_yawmax": 0,
        },
        "adapted_augmentation": {"use_mapped_labels": True},
        "knni": {"enable": False},
    }


def test_loader_returns_correct_shapes(tiny_poss_root):
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "semantic_poss", "semantic-poss.yaml"
    )
    parser = SemanticPOSS(root=tiny_poss_root, sequences=[3], config_path=config_path)
    loader = RangeViewLoader(dataset=parser, config=_minimal_config(), is_train=True)

    feat, label, mask = loader[0]
    assert feat.shape == (5, H, W)
    assert label.shape == (H, W)
    assert mask.shape == (H, W)
    # Mask should mark 6 occupied cells
    assert int(mask.sum().item()) == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TinyViM SSM kernels require CUDA")
def test_tinyvim_forward_pass_on_poss_shape():
    from models.tinyvim_adapter import TinyViMAdapter

    adapter = TinyViMAdapter(
        backbone_name="tinyvim_base",
        in_channels=5,
        stem_stride=(1, 1),
        stage_embedding_strides=[(1, 2), (1, 2), (1, 2)],
        use_fpn_decoder=True,
    ).cuda()
    feats, _ = adapter(torch.randn(1, 5, H, W).cuda(), return_features=True)
    assert [f.shape[2] for f in feats] == [H, H, H, H]
    assert [f.shape[3] for f in feats] == [1800, 900, 450, 225]
