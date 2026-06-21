import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from dataset.semantic_kitti import SemanticKitti
from dataset.range_view_loader import RangeViewLoader


DATA_CONFIG_PATH = REPO_ROOT / "dataset" / "semantic_kitti" / "semantic-kitti.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare valid pixel counts with KNNI on vs off."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "config" / "kitti" / "main" / "config.yaml"),
        help="Path to config YAML.",
    )
    parser.add_argument("--seq", type=str, required=True, help="Sequence id (e.g., 00, 08).")
    parser.add_argument("--frame", type=str, required=True, help="Frame id (e.g., 0 or 000123).")
    return parser.parse_args()


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def find_frame_index(dataset: SemanticKitti, seq_id: str, frame_id: str) -> int:
    seq_id = f"{int(seq_id):02d}"
    frame_id = f"{int(frame_id):06d}"
    for idx in range(len(dataset)):
        seq, frame = dataset.parsePathInfoByIndex(idx)
        if seq == seq_id and frame == frame_id:
            return idx
    raise ValueError(f"Frame not found for seq={seq_id} frame={frame_id}")


def count_valid_pixels(config: dict, data_root: Path, seq: str, frame: str) -> int:
    semkitti_ds = SemanticKitti(
        root=str(data_root),
        sequences=[int(seq)],
        config_path=str(DATA_CONFIG_PATH),
        has_label=config.get("has_label", True),
    )
    loader = RangeViewLoader(
        dataset=semkitti_ds,
        config=config,
        is_train=False,
        return_uproj=False,
        use_kpconv=False,
    )
    frame_idx = find_frame_index(semkitti_ds, seq, frame)
    _, _, proj_mask_tensor = loader[frame_idx]
    return int(proj_mask_tensor.sum().item())


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    base_config = yaml.safe_load(open(config_path, "r"))
    data_root = resolve_path(config_dir, base_config["data_root"])

    config_off = dict(base_config)
    knni_cfg = dict(config_off.get("knni", {}))
    knni_cfg["enable"] = False
    config_off["knni"] = knni_cfg

    config_on = dict(base_config)
    knni_cfg_on = dict(config_on.get("knni", {}))
    knni_cfg_on["enable"] = True
    config_on["knni"] = knni_cfg_on

    valid_off = count_valid_pixels(config_off, data_root, args.seq, args.frame)
    valid_on = count_valid_pixels(config_on, data_root, args.seq, args.frame)

    print(f"valid_pixels (knni OFF): {valid_off}")
    print(f"valid_pixels (knni ON) : {valid_on}")
    print(f"delta: {valid_on - valid_off}")


if __name__ == "__main__":
    main()
