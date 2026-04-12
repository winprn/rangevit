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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifact" / "bev_export"
DEFAULT_X_MIN = -20.0
DEFAULT_X_MAX = 80.0
DEFAULT_Y_MIN = -50.0
DEFAULT_Y_MAX = 50.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export SemanticKITTI BEV visualizations with driving-oriented axes. "
            "Supports scatter and rasterized BEV rendering for height/depth and semantic views."
        )
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "config_kitti_tinyvim.yaml"))
    parser.add_argument("--data-config", type=str, default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--seq", type=str, required=True, help="Sequence id, e.g. 08.")
    parser.add_argument("--frame", type=str, required=True, help="Frame id, e.g. 219 or 000219.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--mode",
        type=str,
        default="scatter_height",
        choices=("scatter_height", "scatter_semantic", "raster_height", "raster_semantic"),
        help="Output mode.",
    )
    parser.add_argument("--bg", type=str, default="white", choices=("white", "black"))
    parser.add_argument("--color-by", type=str, default="z", choices=("z", "depth"))
    parser.add_argument("--x-min", type=float, default=DEFAULT_X_MIN)
    parser.add_argument("--x-max", type=float, default=DEFAULT_X_MAX)
    parser.add_argument("--y-min", type=float, default=DEFAULT_Y_MIN)
    parser.add_argument("--y-max", type=float, default=DEFAULT_Y_MAX)
    parser.add_argument(
        "--centered-360",
        action="store_true",
        help="Use a centered square crop derived from the provided bounds.",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Flip the horizontal BEV axis after the orientation transform.",
    )
    parser.add_argument("--point-size", type=float, default=0.8)
    parser.add_argument("--canvas-width", type=float, default=8.0)
    parser.add_argument("--canvas-height", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.2,
        help="Meters per pixel for raster mode.",
    )
    parser.add_argument(
        "--density-log",
        action="store_true",
        help="Use log-scaled density when generating raster height maps.",
    )
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


def compute_crop_bounds(args):
    if not args.centered_360:
        return args.x_min, args.x_max, args.y_min, args.y_max
    radius = max(abs(args.x_min), abs(args.x_max), abs(args.y_min), abs(args.y_max))
    return -radius, radius, -radius, radius


def filter_and_crop_points(points_xyz: np.ndarray, mapped_labels: np.ndarray, bounds):
    x_min, x_max, y_min, y_max = bounds
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    keep = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    points_xyz = points_xyz[keep]
    mapped_labels = mapped_labels[keep]
    if points_xyz.shape[0] == 0:
        raise ValueError("No points left after applying the selected crop bounds.")
    return points_xyz, mapped_labels


def transform_orientation(points_xyz: np.ndarray, flip_y: bool):
    # BEV convention: horizontal axis is y, vertical axis is x, forward is up.
    bev_h = points_xyz[:, 1].copy()
    bev_v = points_xyz[:, 0].copy()
    if flip_y:
        bev_h = -bev_h
    return bev_h, bev_v


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


def make_output_path(output_dir: Path, mode: str, color_by: str, seq_id: str, frame_id: str) -> Path:
    if "height" in mode:
        name = f"{mode}_{color_by}_seq{seq_id}_frame{frame_id}.png"
    else:
        name = f"{mode}_seq{seq_id}_frame{frame_id}.png"
    return output_dir / name


def get_background_colors(background: str):
    if background == "white":
        return "white", np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return "black", np.array([0.0, 0.0, 0.0], dtype=np.float32)


def export_scatter_bev(bev_h: np.ndarray, bev_v: np.ndarray, colors, out_path: Path, bounds, args):
    bg_name, _ = get_background_colors(args.bg)
    fig, ax = plt.subplots(
        figsize=(args.canvas_width, args.canvas_height),
        facecolor=bg_name,
        constrained_layout=False,
    )
    ax.set_facecolor(bg_name)
    ax.scatter(
        bev_h,
        bev_v,
        c=colors,
        s=args.point_size,
        marker="s",
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )
    _, _, y_min, y_max = bounds
    x_min, x_max, _, _ = bounds
    h_min = -y_max if args.flip_y else y_min
    h_max = -y_min if args.flip_y else y_max
    ax.set_xlim(h_min, h_max)
    ax.set_ylim(x_min, x_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)


def compute_grid_dimensions(bounds, resolution: float):
    x_min, x_max, y_min, y_max = bounds
    width = int(np.ceil((y_max - y_min) / resolution))
    height = int(np.ceil((x_max - x_min) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid crop bounds or resolution produced an empty raster grid.")
    return height, width


def points_to_grid_indices(points_xyz: np.ndarray, bounds, resolution: float):
    x_min, x_max, y_min, y_max = bounds
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    rows = np.floor((x_max - x) / resolution).astype(np.int32)
    cols = np.floor((y - y_min) / resolution).astype(np.int32)
    height, width = compute_grid_dimensions(bounds, resolution)
    keep = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return rows[keep], cols[keep], keep, height, width


def rasterize_height(points_xyz: np.ndarray, bounds, resolution: float, color_by: str):
    rows, cols, keep, height, width = points_to_grid_indices(points_xyz, bounds, resolution)
    values = compute_scalar_values(points_xyz[keep], color_by)
    density = np.zeros((height, width), dtype=np.int32)
    max_values = np.full((height, width), -np.inf, dtype=np.float32)
    np.add.at(density, (rows, cols), 1)
    np.maximum.at(max_values, (rows, cols), values)
    valid = density > 0
    return max_values, density, valid


def rasterize_semantic(points_xyz: np.ndarray, mapped_labels: np.ndarray, bounds, resolution: float):
    rows, cols, keep, height, width = points_to_grid_indices(points_xyz, bounds, resolution)
    labels = mapped_labels[keep]
    flat_idx = rows * width + cols
    grid_labels = np.zeros((height, width), dtype=np.int32)
    unique_cells = np.unique(flat_idx)
    for cell in unique_cells:
        cell_mask = flat_idx == cell
        label_values = labels[cell_mask]
        majority = np.bincount(label_values).argmax()
        r = cell // width
        c = cell % width
        grid_labels[r, c] = majority
    valid = np.zeros((height, width), dtype=bool)
    valid[rows, cols] = True
    return grid_labels, valid


def colorize_raster_height(max_values: np.ndarray, density: np.ndarray, valid: np.ndarray, background: str,
                           color_by: str, density_log: bool):
    _, bg_rgb = get_background_colors(background)
    rgb = np.tile(bg_rgb[None, None, :], (max_values.shape[0], max_values.shape[1], 1))
    if not np.any(valid):
        return rgb
    value_norm = robust_normalize(max_values[valid])
    colors = plt.get_cmap("turbo")(value_norm)[:, :3]
    if density_log:
        alpha = robust_normalize(np.log1p(density[valid].astype(np.float32)), low_pct=0.0, high_pct=100.0)
    else:
        alpha = robust_normalize(density[valid].astype(np.float32), low_pct=0.0, high_pct=100.0)
    alpha = 0.35 + 0.65 * alpha
    blended = bg_rgb[None, :] * (1.0 - alpha[:, None]) + colors * alpha[:, None]
    rgb[valid] = blended
    return rgb


def colorize_raster_semantic(grid_labels: np.ndarray, valid: np.ndarray, color_lut: np.ndarray, background: str):
    _, bg_rgb = get_background_colors(background)
    rgb = np.tile(bg_rgb[None, None, :], (grid_labels.shape[0], grid_labels.shape[1], 1))
    safe_labels = np.clip(grid_labels[valid], 0, color_lut.shape[0] - 1)
    rgb[valid] = color_lut[safe_labels]
    return rgb


def export_raster_bev(image_rgb: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, image_rgb)


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    data_cfg_path = Path(args.data_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    seq_id = f"{int(args.seq):02d}"
    frame_id = f"{int(args.frame):06d}"

    points_xyz, mapped_labels, data_cfg = load_points_and_labels(config_path, data_cfg_path, seq_id, frame_id)
    bounds = compute_crop_bounds(args)
    points_xyz, mapped_labels = filter_and_crop_points(points_xyz, mapped_labels, bounds)
    bev_h, bev_v = transform_orientation(points_xyz, flip_y=args.flip_y)

    if args.mode == "scatter_height":
        scalar_values = compute_scalar_values(points_xyz, args.color_by)
        order = np.argsort(points_xyz[:, 0])[::-1]
        scalar_norm = robust_normalize(scalar_values[order])
        colors = plt.get_cmap("turbo")(scalar_norm)[:, :3]
        out_path = make_output_path(output_dir, args.mode, args.color_by, seq_id, frame_id)
        export_scatter_bev(bev_h[order], bev_v[order], colors, out_path, bounds, args)
    elif args.mode == "scatter_semantic":
        color_lut = build_semantic_color_lut(data_cfg, args.bg)
        order = np.argsort(np.sqrt(points_xyz[:, 0] ** 2 + points_xyz[:, 1] ** 2))[::-1]
        colors = color_lut[np.clip(mapped_labels[order], 0, color_lut.shape[0] - 1)]
        out_path = make_output_path(output_dir, args.mode, args.color_by, seq_id, frame_id)
        export_scatter_bev(bev_h[order], bev_v[order], colors, out_path, bounds, args)
    elif args.mode == "raster_height":
        max_values, density, valid = rasterize_height(points_xyz, bounds, args.resolution, args.color_by)
        image_rgb = colorize_raster_height(max_values, density, valid, args.bg, args.color_by, args.density_log)
        out_path = make_output_path(output_dir, args.mode, args.color_by, seq_id, frame_id)
        export_raster_bev(image_rgb, out_path)
    else:
        color_lut = build_semantic_color_lut(data_cfg, args.bg)
        grid_labels, valid = rasterize_semantic(points_xyz, mapped_labels, bounds, args.resolution)
        image_rgb = colorize_raster_semantic(grid_labels, valid, color_lut, args.bg)
        out_path = make_output_path(output_dir, args.mode, args.color_by, seq_id, frame_id)
        export_raster_bev(image_rgb, out_path)

    print(f"Loaded points: {points_xyz.shape[0]}")
    print(f"Bounds x:[{bounds[0]}, {bounds[1]}] y:[{bounds[2]}, {bounds[3]}]")
    print(f"Mode: {args.mode}")
    print(f"Output saved to {out_path}")


if __name__ == "__main__":
    main()
