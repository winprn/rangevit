"""
Plot SemanticKITTI class distribution per sequence for train+valid splits.

Example:
python tests/test_semantickitti_sequence_class_distribution.py --config config/kitti/main/config_tinyvim.yaml
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG = REPO_ROOT / "dataset" / "semantic_kitti" / "semantic-kitti.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifact" / "sequence_class_distribution"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot SemanticKITTI mapped-class distribution per sequence "
            "for train and valid splits."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "config" / "kitti" / "main" / "config_tinyvim.yaml"),
        help="Path to project config YAML (contains data.data_root).",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=str(DEFAULT_DATA_CONFIG),
        help="Path to semantic-kitti.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save per-sequence plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Plot DPI.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_learning_lut(learning_map: dict) -> np.ndarray:
    max_key = max(int(k) for k in learning_map.keys())
    lut = np.zeros(max_key + 1, dtype=np.int32)
    for k, v in learning_map.items():
        lut[int(k)] = int(v)
    return lut


def count_sequence_points(seq_dir: Path, class_map_lut: np.ndarray, n_classes: int) -> np.ndarray:
    label_dir = seq_dir / "labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")

    counts = np.zeros(n_classes, dtype=np.int64)
    label_files = sorted(label_dir.glob("*.label"))
    if not label_files:
        raise FileNotFoundError(f"No .label files in: {label_dir}")

    for label_path in label_files:
        raw = np.fromfile(label_path, dtype=np.uint32)
        sem = (raw & 0xFFFF).astype(np.int32)
        if sem.size == 0:
            continue
        sem = np.clip(sem, 0, class_map_lut.shape[0] - 1)
        mapped = class_map_lut[sem]
        binc = np.bincount(mapped, minlength=n_classes)
        counts += binc.astype(np.int64)
    return counts


def print_table(seq_id: int, class_names: list, counts: np.ndarray):
    total = int(counts.sum())
    print(f"\nSequence {seq_id:02d}")
    print("+---------------+------------------+------------+")
    print("|   Class Name  | Number of Points | Percentage |")
    print("+---------------+------------------+------------+")
    for i, name in enumerate(class_names):
        c = int(counts[i])
        pct = (100.0 * c / total) if total > 0 else 0.0
        print(f"| {name:^13} | {c:^16} | {pct:>8.4f}%  |")
    print("+---------------+------------------+------------+")
    print(f"Total points: {total}")


def plot_sequence(seq_id: int, class_names: list, counts: np.ndarray, output_dir: Path, dpi: int):
    total = counts.sum()
    pct = (counts / total * 100.0) if total > 0 else np.zeros_like(counts, dtype=np.float64)
    x = np.arange(len(class_names))

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()

    width = 0.42
    bars1 = ax1.bar(x - width / 2, counts, width=width, color="#1f77b4", label="Number of Points")
    bars2 = ax2.bar(x + width / 2, pct, width=width, color="#ff7f0e", label="Percentage (%)")

    ax1.set_ylabel("Number of Points")
    ax2.set_ylabel("Percentage (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha="right")
    ax1.set_title(f"SemanticKITTI Class Distribution - Sequence {seq_id:02d}")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)

    handles = [bars1, bars2]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sequence_{seq_id:02d}_class_distribution.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    args = parse_args()

    cfg_path = Path(args.config).resolve()
    data_cfg_path = Path(args.data_config).resolve()
    out_dir = Path(args.output_dir).resolve()

    cfg = load_yaml(cfg_path)
    data_cfg = load_yaml(data_cfg_path)

    data_root = resolve_path(cfg_path.parent, cfg["data"]["data_root"])
    split = data_cfg["split"]
    target_sequences = sorted(set(split["train"] + split["valid"]))

    learning_map = data_cfg["learning_map"]
    mapped_class_name = data_cfg["mapped_class_name"]
    n_classes = len(mapped_class_name)
    class_names = [mapped_class_name[i] for i in range(n_classes)]
    class_map_lut = build_learning_lut(learning_map)

    print(f"Data root: {data_root}")
    print(f"Data config: {data_cfg_path}")
    print(f"Output dir: {out_dir}")
    print(f"Sequences (train+valid): {[f'{s:02d}' for s in target_sequences]}")

    missing = []
    for seq_id in target_sequences:
        seq_dir = data_root / f"{seq_id:02d}"
        if not seq_dir.exists():
            missing.append(seq_id)
            continue

        counts = count_sequence_points(seq_dir, class_map_lut, n_classes)
        print_table(seq_id, class_names, counts)
        plot_sequence(seq_id, class_names, counts, out_dir, args.dpi)

    if missing:
        print(f"\nWarning: missing sequence folders: {[f'{s:02d}' for s in missing]}")
        print("Check your config data.data_root path.")


if __name__ == "__main__":
    main()
