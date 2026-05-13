import os
import numpy as np
import pytest

from dataset.semantic_poss.parser import SemanticPOSS


H, W = 40, 1800


@pytest.fixture
def tiny_poss_root(tmp_path):
    """Build a 1-sequence, 1-frame mock POSS directory."""
    seq = "00"
    seq_dir = tmp_path / "sequences" / seq
    (seq_dir / "velodyne").mkdir(parents=True)
    (seq_dir / "labels").mkdir()
    (seq_dir / "tag").mkdir()

    # 5 points, all "ground" (raw id 22) except one "person" (raw id 4)
    n_points = 5
    pts = np.array(
        [[1.0, 2.0, 0.1, 0.5],
         [2.0, 1.0, 0.0, 0.3],
         [-1.0, 0.5, 0.2, 0.7],
         [0.5, -1.5, 0.1, 0.4],
         [3.0, 0.0, 1.5, 0.9]],
        dtype=np.float32,
    )
    pts.tofile(seq_dir / "velodyne" / "000000.bin")

    sem_raw = np.array([22, 22, 22, 22, 4], dtype=np.uint32)
    inst = np.zeros(n_points, dtype=np.uint32)
    packed = (inst << 16) | sem_raw
    packed.astype(np.uint32).tofile(seq_dir / "labels" / "000000.label")

    # Tag mask: 5 cells set true, rest false
    tag = np.zeros(H * W, dtype=np.bool_)
    tag[[0, 5, 100, 1000, H * W - 1]] = True
    tag.tofile(seq_dir / "tag" / "000000.tag")

    return str(tmp_path / "sequences")


def test_parser_loads_frame(tiny_poss_root):
    config = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "semantic_poss", "semantic-poss.yaml"
    )
    parser = SemanticPOSS(root=tiny_poss_root, sequences=[0], config_path=config)
    assert len(parser) == 1
    pc, sem, inst = parser.loadDataByIndex(0)
    assert pc.shape == (5, 4)
    assert sem.shape == (5,)
    assert inst.shape == (5,)


def test_parser_returns_tag_mask(tiny_poss_root):
    config = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "semantic_poss", "semantic-poss.yaml"
    )
    parser = SemanticPOSS(root=tiny_poss_root, sequences=[0], config_path=config)
    tag = parser.loadTagByIndex(0)
    assert tag.dtype == np.bool_
    assert tag.shape == (H * W,)
    assert int(tag.sum()) == 5


def test_label_mapping_collapses_traffic_signs_and_persons(tiny_poss_root):
    config = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "semantic_poss", "semantic-poss.yaml"
    )
    parser = SemanticPOSS(root=tiny_poss_root, sequences=[0], config_path=config)
    raw = np.array([0, 4, 5, 10, 11, 12, 22], dtype=np.int32)
    mapped = parser.labelMapping(raw)
    # 4,5 -> 1; 10,11,12 -> 6; 22 -> 13; 0 -> 0
    assert mapped.tolist() == [0, 1, 1, 6, 6, 6, 13]


def test_parse_path_info(tiny_poss_root):
    config = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "semantic_poss", "semantic-poss.yaml"
    )
    parser = SemanticPOSS(root=tiny_poss_root, sequences=[0], config_path=config)
    seq_id, frame_id = parser.parsePathInfoByIndex(0)
    assert seq_id == "00"
    assert frame_id == "000000"
