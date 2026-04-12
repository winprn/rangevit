import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from dataset.semantic_kitti import SemanticKitti


DEFAULT_DATA_CONFIG = REPO_ROOT / "dataset" / "semantic_kitti" / "semantic-kitti.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifact" / "sideview_export"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export SemanticKITTI side-view point cloud images. "
            "Supports smooth height/depth coloring and semantic coloring."
        )
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "config_kitti_tinyvim.yaml"))
    parser.add_argument("--data-config", type=str, default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--frame", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--mode",
        type=str,
        default="scatter_height",
        choices=("scatter_height", "scatter_semantic"),
    )
    parser.add_argument("--bg", type=str, default="white", choices=("white", "black"))
    parser.add_argument("--color-by", type=str, default="z", choices=("z", "depth"))
    parser.add_argument(
        "--view-side",
        type=str,
        default="left",
        choices=("left", "right"),
        help="Camera placed on the left or right side of the vehicle.",
    )
    parser.add_argument("--x-min", type=float, default=-20.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--z-min", type=float, default=-4.0)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--y-min", type=float, default=-50.0)
    parser.add_argument("--y-max", type=float, default=50.0)
    parser.add_argument("--point-size", type=float, default=0.8)
    parser.add_argument("--canvas-width", type=float, default=10.0)
    parser.add_argument("--canvas-height", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def get_config_value(config: dict, key: str, default=None, section: str = None):
    if key in config:
        return config[key]
    if section is not None:
        section_cfg = config.get(section, {})
        if isinstance(section_cfg, dict) and key in section_cfg:
            return section_cfg[key]
    return default


def find_frame_index(dataset: SemanticKitti, seq_id: str, frame_id: str) -> int:
    seq_id = f"{int(seq_id):02d}"
    frame_id = f"{int(frame_id):06d}"
    for idx in range(len(dataset)):
        seq, frame = dataset.parsePathInfoByIndex(idx)
        if seq == seq_id and frame == frame_id:
            return idx
    raise ValueError(f"Frame not found for seq={seq_id} frame={frame_id}")


def load_points_and_labels(config_path: Path, data_cfg_path: Path, seq_id: str, frame_id: str):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = yaml.safe_load(data_cfg_path.read_text(encoding="utf-8"))
    data_root_str = get_config_value(config, "data_root", section="data")
    if data_root_str is None:
        raise KeyError("Missing required config key: data.data_root")
    data_root = resolve_path(config_path.parent, data_root_str)
    has_label = bool(get_config_value(config, "has_label", True, section="data"))

    dataset = SemanticKitti(
        root=str(data_root),
        sequences=[int(seq_id)],
        config_path=str(data_cfg_path),
        has_label=has_label,
    )
    frame_idx = find_frame_index(dataset, seq_id, frame_id)
    pointcloud, raw_labels, _ = dataset.loadDataByIndex(frame_idx)
    mapped_labels = dataset.labelMapping(raw_labels.astype(np.int32))
    return pointcloud[:, :3].astype(np.float32), mapped_labels.astype(np.int32), data_cfg


def build_semantic_color_lut(data_cfg: dict, background: str) -> np.ndarray:
    color_map_inv = data_cfg["color_map_inv"]
    max_key = max(int(k) for k in color_map_inv.keys())
    lut = np.zeros((max_key + 1, 3), dtype=np.float32)
    for k, bgr in color_map_inv.items():
        lut[int(k)] = np.asarray(bgr[::-1], dtype=np.float32) / 255.0
    lut[0] = np.array([1.0, 1.0, 1.0], dtype=np.float32) if background == "white" else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return lut


def robust_normalize(values: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo = np.percentile(values, low_pct)
    hi = np.percentile(values, high_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def compute_scalar_values(points_xyz: np.ndarray, color_by: str) -> np.ndarray:
    if color_by == "z":
        return points_xyz[:, 2]
    return np.sqrt(points_xyz[:, 0] ** 2 + points_xyz[:, 1] ** 2)


def filter_points(points_xyz: np.ndarray, mapped_labels: np.ndarray, args):
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    keep = (
        (x >= args.x_min) & (x <= args.x_max) &
        (y >= args.y_min) & (y <= args.y_max) &
        (z >= args.z_min) & (z <= args.z_max)
    )
    points_xyz = points_xyz[keep]
    mapped_labels = mapped_labels[keep]
    if points_xyz.shape[0] == 0:
        raise ValueError("No points left after applying the selected side-view bounds.")
    return points_xyz, mapped_labels


def transform_side_view(points_xyz: np.ndarray, view_side: str):
    # Side view: horizontal axis is forward x, vertical axis is z.
    # Left/right setting only changes drawing order so near-side points stay visible.
    horiz = points_xyz[:, 0]
    vert = points_xyz[:, 2]
    depth = points_xyz[:, 1]
    if view_side == "left":
        order = np.argsort(depth)
    else:
        order = np.argsort(-depth)
    return horiz, vert, depth, order


def export_scatter(horiz: np.ndarray, vert: np.ndarray, colors, out_path: Path, args):
    bg = args.bg
    fig, ax = plt.subplots(
        figsize=(args.canvas_width, args.canvas_height),
        facecolor=bg,
        constrained_layout=False,
    )
    ax.set_facecolor(bg)
    ax.scatter(
        horiz,
        vert,
        c=colors,
        s=args.point_size,
        marker="s",
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(args.z_min, args.z_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_output_path(output_dir: Path, mode: str, color_by: str, view_side: str, seq_id: str, frame_id: str) -> Path:
    if mode == "scatter_height":
        name = f"side_{mode}_{color_by}_{view_side}_seq{seq_id}_frame{frame_id}.png"
    else:
        name = f"side_{mode}_{view_side}_seq{seq_id}_frame{frame_id}.png"
    return output_dir / name


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    data_cfg_path = Path(args.data_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    seq_id = f"{int(args.seq):02d}"
    frame_id = f"{int(args.frame):06d}"

    points_xyz, mapped_labels, data_cfg = load_points_and_labels(config_path, data_cfg_path, seq_id, frame_id)
    points_xyz, mapped_labels = filter_points(points_xyz, mapped_labels, args)
    horiz, vert, _, order = transform_side_view(points_xyz, args.view_side)

    if args.mode == "scatter_height":
        scalar_values = compute_scalar_values(points_xyz, args.color_by)
        colors = plt.get_cmap("turbo")(robust_normalize(scalar_values[order]))[:, :3]
    else:
        color_lut = build_semantic_color_lut(data_cfg, args.bg)
        colors = color_lut[np.clip(mapped_labels[order], 0, color_lut.shape[0] - 1)]

    out_path = make_output_path(output_dir, args.mode, args.color_by, args.view_side, seq_id, frame_id)
    export_scatter(horiz[order], vert[order], colors, out_path, args)

    print(f"Loaded points: {points_xyz.shape[0]}")
    print(f"Mode: {args.mode}")
    print(f"View side: {args.view_side}")
    print(f"Output saved to {out_path}")


if __name__ == "__main__":
    main()
