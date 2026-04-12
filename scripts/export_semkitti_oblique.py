import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from dataset.semantic_kitti import SemanticKitti


DEFAULT_DATA_CONFIG = REPO_ROOT / "dataset" / "semantic_kitti" / "semantic-kitti.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifact" / "oblique_export"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export SemanticKITTI oblique top-down point-cloud views with a 3D camera."
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
        default="height",
        choices=("height", "semantic", "black"),
    )
    parser.add_argument("--bg", type=str, default="white", choices=("white", "black"))
    parser.add_argument("--color-by", type=str, default="z", choices=("z", "depth"))
    parser.add_argument("--x-min", type=float, default=-20.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-50.0)
    parser.add_argument("--y-max", type=float, default=50.0)
    parser.add_argument("--z-min", type=float, default=-4.0)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--azim", type=float, default=-115.0, help="Camera azimuth in degrees.")
    parser.add_argument("--elev", type=float, default=32.0, help="Camera elevation in degrees.")
    parser.add_argument("--roll", type=float, default=0.0, help="Camera roll in degrees if supported.")
    parser.add_argument("--point-size", type=float, default=0.9)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--canvas-width", type=float, default=8.0)
    parser.add_argument("--canvas-height", type=float, default=5.2)
    parser.add_argument("--zoom", type=float, default=1.35, help="Camera zoom multiplier. Larger means tighter framing.")
    parser.add_argument("--frame-pad", type=float, default=0.03, help="Fractional padding around occupied point extents.")
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
        raise ValueError("No points left after applying the selected crop bounds.")
    return points_xyz, mapped_labels


def prepare_colors(points_xyz: np.ndarray, mapped_labels: np.ndarray, data_cfg: dict, args):
    if args.mode == "black":
        return np.zeros((points_xyz.shape[0], 3), dtype=np.float32)
    if args.mode == "height":
        scalar_values = compute_scalar_values(points_xyz, args.color_by)
        norm = robust_normalize(scalar_values)
        return plt.get_cmap("viridis")(norm)[:, :3]
    color_lut = build_semantic_color_lut(data_cfg, args.bg)
    safe_labels = np.clip(mapped_labels, 0, color_lut.shape[0] - 1)
    return color_lut[safe_labels]


def compute_axis_limits(values: np.ndarray, crop_min: float, crop_max: float, pad_fraction: float, zoom: float):
    used_min = float(np.min(values))
    used_max = float(np.max(values))
    span = max(used_max - used_min, 1e-3)
    pad = span * max(pad_fraction, 0.0)
    center = 0.5 * (used_min + used_max)
    half_span = 0.5 * (span + 2.0 * pad) / max(zoom, 1e-3)
    lo = max(crop_min, center - half_span)
    hi = min(crop_max, center + half_span)
    if hi <= lo:
        return crop_min, crop_max
    return lo, hi


def draw_oblique(points_xyz: np.ndarray, colors: np.ndarray, out_path: Path, args):
    fig = plt.figure(figsize=(args.canvas_width, args.canvas_height), facecolor=args.bg)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(args.bg)

    # Draw farther points first to preserve scan structure from the chosen camera.
    cam_azim = np.deg2rad(args.azim)
    cam_xy = np.array([np.cos(cam_azim), np.sin(cam_azim)], dtype=np.float32)
    depth_order = np.argsort(points_xyz[:, 0] * cam_xy[0] + points_xyz[:, 1] * cam_xy[1])

    pts = points_xyz[depth_order]
    cols = colors[depth_order]
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=cols,
        s=args.point_size,
        marker="s",
        linewidths=0,
        edgecolors="none",
        depthshade=False,
    )

    ax.view_init(elev=args.elev, azim=args.azim, roll=args.roll)
    x_lo, x_hi = compute_axis_limits(pts[:, 0], args.x_min, args.x_max, args.frame_pad, args.zoom)
    y_lo, y_hi = compute_axis_limits(pts[:, 1], args.y_min, args.y_max, args.frame_pad, args.zoom)
    z_lo, z_hi = compute_axis_limits(pts[:, 2], args.z_min, args.z_max, args.frame_pad, args.zoom)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    try:
        ax.set_box_aspect((x_hi - x_lo, y_hi - y_lo, 0.4 * (z_hi - z_lo)))
    except Exception:
        pass
    try:
        ax.dist = max(4.0, 8.0 / max(args.zoom, 1e-3))
    except Exception:
        pass
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    data_cfg_path = Path(args.data_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    seq_id = f"{int(args.seq):02d}"
    frame_id = f"{int(args.frame):06d}"

    points_xyz, mapped_labels, data_cfg = load_points_and_labels(config_path, data_cfg_path, seq_id, frame_id)
    points_xyz, mapped_labels = filter_points(points_xyz, mapped_labels, args)
    colors = prepare_colors(points_xyz, mapped_labels, data_cfg, args)

    suffix = f"{args.mode}_{args.color_by}" if args.mode == "height" else args.mode
    out_path = output_dir / f"oblique_{suffix}_seq{seq_id}_frame{frame_id}.png"
    draw_oblique(points_xyz, colors, out_path, args)

    print(f"Loaded points: {points_xyz.shape[0]}")
    print(f"Mode: {args.mode}")
    print(f"Camera: azim={args.azim}, elev={args.elev}, roll={args.roll}")
    print(f"Output saved to {out_path}")


if __name__ == "__main__":
    main()
