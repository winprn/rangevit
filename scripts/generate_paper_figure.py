"""Generate qualitative comparison figures for the RangeViM paper.

Produces range-image views and top-down (BEV) views with ground truth,
RangeViT predictions, and RangeViM predictions side by side.

Usage
-----
python scripts/generate_paper_figure.py \
    --config_a config_kitti_tinyvim.yaml \
    --checkpoint_a /path/to/rangevim_best.pth \
    --label_a RangeViM \
    --config_b config_kitti.yaml \
    --checkpoint_b /path/to/rangevit_best.pth \
    --label_b RangeViT \
    --seq 08 --frame 0 \
    --knn \
    --output_dir artifact/paper_figures
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import imageio.v2 as imageio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from dataset.semantic_kitti import SemanticKitti
from dataset.range_view_loader import RangeViewLoader
from main import build_rangevit_model
from utils.inference.inference_utils import inference
from utils.postproc.knn import KNN

# ---------------------------------------------------------------------------
# SemanticKITTI colour definitions (raw label → RGB)
# ---------------------------------------------------------------------------
SEMANTIC_KITTI_RAW_COLORS = {
    0: [0, 0, 0], 1: [0, 0, 0], 10: [245, 150, 100], 11: [245, 230, 100],
    13: [250, 80, 100], 15: [150, 60, 30], 16: [255, 0, 0],
    18: [180, 30, 80], 20: [255, 0, 0], 30: [30, 30, 255],
    31: [200, 40, 255], 32: [90, 30, 150], 40: [255, 0, 255],
    44: [255, 150, 255], 48: [75, 0, 75], 49: [75, 0, 175],
    50: [0, 200, 255], 51: [50, 120, 255], 52: [0, 150, 255],
    60: [170, 255, 150], 70: [0, 175, 0], 71: [0, 60, 135],
    72: [80, 240, 150], 80: [150, 240, 255], 81: [0, 0, 255],
    99: [255, 255, 50],
}

# Mapped class (0-19) → human-readable name
CLASS_NAMES = {
    0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck",
    5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist",
    9: "road", 10: "parking", 11: "sidewalk", 12: "other-ground",
    13: "building", 14: "fence", 15: "vegetation", 16: "trunk",
    17: "terrain", 18: "pole", 19: "traffic-sign",
}

DATA_CONFIG_PATH = "./dataset/semantic_kitti/semantic-kitti.yaml"


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _build_learning_map_inv_lut(learning_map_inv: dict) -> np.ndarray:
    max_key = max(int(k) for k in learning_map_inv.keys())
    lut = np.zeros(max_key + 1, dtype=np.int32)
    for k, v in learning_map_inv.items():
        lut[int(k)] = int(v)
    return lut


def _build_color_lut(color_map: dict, min_size: int = 0) -> np.ndarray:
    max_key = max(max(int(k) for k in color_map.keys()), min_size)
    lut = np.zeros((max_key + 1, 3), dtype=np.uint8)
    for k, rgb in color_map.items():
        lut[int(k)] = np.array(rgb, dtype=np.uint8)
    return lut


def colorize_mapped_labels(
    mapped_labels: np.ndarray,
    learning_map_inv_lut: np.ndarray,
    color_lut: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert mapped labels (0-19) → RGB via raw label colours."""
    raw_labels = learning_map_inv_lut[mapped_labels]
    raw_labels = np.clip(raw_labels, 0, color_lut.shape[0] - 1)
    colors = color_lut[raw_labels]
    if mask is not None:
        colors[~mask] = [0, 0, 0]
    return colors


def build_mapped_color_lut(learning_map_inv_lut, color_lut):
    """Build a direct mapped-class-id → RGB LUT (20 entries)."""
    n = min(20, learning_map_inv_lut.shape[0])
    lut = np.zeros((20, 3), dtype=np.uint8)
    for c in range(n):
        raw = learning_map_inv_lut[c]
        if raw < color_lut.shape[0]:
            lut[c] = color_lut[raw]
    return lut


# ---------------------------------------------------------------------------
# Config / model helpers
# ---------------------------------------------------------------------------

def build_settings_from_config(config: dict):
    """Minimal settings namespace for build_rangevit_model + inference."""

    class _S:
        pass

    s = _S()
    s.config = config
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    s.in_channels = model_cfg.get("in_channels", config.get("in_channels", 5))
    s.n_classes = data_cfg.get("n_classes", config.get("n_classes", 20))
    s.vit_backbone = model_cfg.get("vit_backbone", config.get("vit_backbone", "vit_small_patch16_384"))
    s.image_size = tuple(model_cfg.get("image_size", config.get("image_size", [64, 512])))
    s.patch_size = tuple(model_cfg.get("patch_size", config.get("patch_size", [2, 8])))
    s.patch_stride = tuple(model_cfg.get("patch_stride", config.get("patch_stride", [2, 8])))
    s.reuse_pos_emb = config.get("reuse_pos_emb", False)
    s.reuse_patch_emb = config.get("reuse_patch_emb", False)
    s.pretrained_channel_adaptation = config.get("pretrained_channel_adaptation", "repeat")
    s.conv_stem = config.get("conv_stem", "ConvStem")
    s.stem_base_channels = config.get("stem_base_channels", 32)
    s.D_h = config.get("D_h", 256)
    s.skip_filters = config.get("skip_filters", 0)

    decoder_cfg = config.get("decoder", "up_conv")
    if isinstance(decoder_cfg, dict):
        s.decoder = decoder_cfg.get("name", "up_conv")
        tinyvim_fuse_cfg = decoder_cfg.get("tinyvim_fuse_aux", {})
    else:
        s.decoder = decoder_cfg
        tinyvim_fuse_cfg = {}

    s.fuse_proj_channels = int(tinyvim_fuse_cfg.get("proj_channels", config.get("fuse_proj_channels", 128)))
    s.fuse_mid_channels = int(tinyvim_fuse_cfg.get("mid_channels", config.get("fuse_mid_channels", 256)))
    s.fuse_out_channels = int(tinyvim_fuse_cfg.get("out_channels", config.get("fuse_out_channels", 128)))
    s.fuse_preproj = bool(tinyvim_fuse_cfg.get("preproj", config.get("fuse_preproj", True)))
    s.aux_enable = bool(tinyvim_fuse_cfg.get("aux_enable", config.get("aux_enable", False)))

    s.use_kpconv = str(config.get("point_postproc", "none")).lower() == "kpconv"
    s.window_size = tuple(model_cfg.get("window_size", config.get("window_size", config.get("original_image_size", [64, 2048]))))
    s.window_stride = tuple(model_cfg.get("window_stride", config.get("window_stride", s.window_size)))
    s.use_sliding_window = model_cfg.get("use_sliding_window", config.get("use_sliding_window", True))
    s.pretrained_model = config.get("pretrained_model", None)
    s.checkpoint = config.get("checkpoint", None)
    return s


def resolve_path(base_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p).resolve()


def find_frame_index(dataset: SemanticKitti, seq_id: str, frame_id: str) -> int:
    seq_id = f"{int(seq_id):02d}"
    frame_id = f"{int(frame_id):06d}"
    for idx in range(len(dataset)):
        seq, frame = dataset.parsePathInfoByIndex(idx)
        if seq == seq_id and frame == frame_id:
            return idx
    raise ValueError(f"Frame not found: seq={seq_id} frame={frame_id}")


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    config = yaml.safe_load(open(config_path))
    settings = build_settings_from_config(config)
    model = build_rangevit_model(settings, pretrained_path=None)
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"  missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)}")
    model.to(device).eval()
    return model, settings


def run_inference(model, settings, proj_feature: torch.Tensor, device: torch.device) -> np.ndarray:
    x = proj_feature.unsqueeze(0).to(device)
    with torch.no_grad():
        seg_map = inference(
            model.rangevit,
            [x],
            [{"flip": False}],
            ori_shape=x.shape[2:4],
            window_size=settings.window_size,
            window_stride=settings.window_stride,
            batch_size=1,
            use_kpconv=settings.use_kpconv,
            use_sliding_window=settings.use_sliding_window,
        )
    return seg_map.argmax(dim=0).cpu().numpy().astype(np.int32)


def apply_knn(pred_2d, proj_range, uproj_x, uproj_y, uproj_depth, knn_params, n_classes, device):
    """Apply KNN post-processing, mapping 2D preds to refined 3D labels."""
    knn = KNN(knn_params, n_classes)
    proj_range_t = torch.from_numpy(proj_range).float().to(device)
    uproj_depth_t = torch.from_numpy(uproj_depth).float().to(device)
    pred_t = torch.from_numpy(pred_2d).long().to(device)
    px = torch.from_numpy(uproj_x).long().to(device)
    py = torch.from_numpy(uproj_y).long().to(device)
    with torch.no_grad():
        refined = knn(proj_range_t, uproj_depth_t, pred_t, px, py)
    return refined.cpu().numpy().astype(np.int32)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_range_image(labels, mask, lmap_inv_lut, color_lut, path: Path):
    colors = colorize_mapped_labels(labels, lmap_inv_lut, color_lut, mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), colors.astype(np.uint8))


def save_bev_image(
    pointcloud: np.ndarray,
    labels: np.ndarray,
    mapped_color_lut: np.ndarray,
    path: Path,
    bev_range: float = 50.0,
    point_size: float = 0.15,
    dpi: int = 300,
    figsize: tuple = (8, 8),
    bg: str = "white",
):
    """Render bird's-eye view (top-down, x-forward, y-left)."""
    x, y = pointcloud[:, 0], pointcloud[:, 1]
    valid = (
        (x >= -bev_range) & (x <= bev_range) &
        (y >= -bev_range) & (y <= bev_range)
    )
    x, y, lbl = x[valid], y[valid], labels[valid]

    # Colour per point
    lbl_clipped = np.clip(lbl, 0, 19)
    colors = mapped_color_lut[lbl_clipped].astype(np.float32) / 255.0

    # Render far points first so near objects paint on top
    depth = np.sqrt(x ** 2 + y ** 2)
    order = np.argsort(-depth)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    # In BEV: horizontal = y (left-right), vertical = x (forward)
    ax.scatter(
        y[order], x[order],
        c=colors[order],
        s=point_size,
        marker=".",
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlim(bev_range, -bev_range)   # y: right to left
    ax.set_ylim(-bev_range, bev_range)   # x: behind to forward
    ax.set_aspect("equal")
    ax.axis("off")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0, facecolor=bg, dpi=dpi)
    plt.close(fig)


def save_bev_diff_image(
    pointcloud: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    path: Path,
    bev_range: float = 50.0,
    point_size: float = 0.15,
    dpi: int = 300,
    figsize: tuple = (8, 8),
    bg: str = "white",
):
    """Render BEV showing correct (green) vs misclassified (red) points."""
    x, y = pointcloud[:, 0], pointcloud[:, 1]
    valid = (
        (x >= -bev_range) & (x <= bev_range) &
        (y >= -bev_range) & (y <= bev_range)
    )
    x, y = x[valid], y[valid]
    gt_v, pred_v = gt_labels[valid], pred_labels[valid]

    # Skip unlabeled (class 0) for diff
    labeled = gt_v != 0
    correct = labeled & (gt_v == pred_v)
    wrong = labeled & (gt_v != pred_v)
    unlabeled_mask = ~labeled

    # Assign colours: green=correct, red=wrong, light gray=unlabeled
    colors = np.full((x.shape[0], 3), 0.85)  # light gray default
    colors[correct] = [0.4, 0.8, 0.4]   # green
    colors[wrong] = [0.9, 0.2, 0.2]     # red

    # Draw order: unlabeled first, correct, then wrong on top
    order = np.concatenate([
        np.where(unlabeled_mask)[0],
        np.where(correct)[0],
        np.where(wrong)[0],
    ])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    ax.scatter(
        y[order], x[order],
        c=colors[order],
        s=point_size,
        marker=".",
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlim(bev_range, -bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect("equal")
    ax.axis("off")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0, facecolor=bg, dpi=dpi)
    plt.close(fig)


def save_legend(mapped_color_lut: np.ndarray, path: Path, dpi: int = 300):
    """Save a standalone class colour legend."""
    handles = []
    for c in range(1, 20):  # skip unlabeled
        rgb = mapped_color_lut[c].astype(np.float32) / 255.0
        handles.append(Line2D([0], [0], marker="s", color="w",
                              markerfacecolor=rgb, markersize=10,
                              label=CLASS_NAMES[c], linewidth=0))

    fig, ax = plt.subplots(figsize=(3, 5), dpi=dpi)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.legend(handles=handles, loc="center", frameon=False,
              fontsize=8, ncol=1, handletextpad=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate paper comparison figures")

    p.add_argument("--config_a", required=True, help="Config YAML for model A (RangeViM)")
    p.add_argument("--checkpoint_a", required=True, help="Checkpoint for model A")
    p.add_argument("--label_a", default="RangeViM", help="Display label for model A")

    p.add_argument("--config_b", required=True, help="Config YAML for model B (RangeViT)")
    p.add_argument("--checkpoint_b", required=True, help="Checkpoint for model B")
    p.add_argument("--label_b", default="RangeViT", help="Display label for model B")

    p.add_argument("--data_root", default=None,
                   help="Override data root path (default: read from config_a)")
    p.add_argument("--seq", required=True, help="Sequence id (e.g. 08)")
    p.add_argument("--frame", required=True, help="Frame id (e.g. 0 or 000123)")
    p.add_argument("--knn", action="store_true", help="Apply KNN post-processing to 3D predictions")
    p.add_argument("--output_dir", default="artifact/paper_figures", help="Output directory")
    p.add_argument("--bev_range", type=float, default=50.0, help="BEV range in metres")
    p.add_argument("--point_size", type=float, default=0.15, help="BEV scatter point size")
    p.add_argument("--dpi", type=int, default=300, help="Output DPI")
    p.add_argument("--bev_bg", default="white", help="BEV background colour (e.g. white, black)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    seq_str = f"{int(args.seq):02d}"
    frame_str = f"{int(args.frame):06d}"
    tag = f"seq{seq_str}_frame{frame_str}"

    # ---- Colour LUTs ----
    dataset_cfg = yaml.safe_load(open(DATA_CONFIG_PATH))
    lmap_inv_lut = _build_learning_map_inv_lut(dataset_cfg["learning_map_inv"])
    color_lut = _build_color_lut(SEMANTIC_KITTI_RAW_COLORS, min_size=int(lmap_inv_lut.max()))
    mapped_color_lut = build_mapped_color_lut(lmap_inv_lut, color_lut)

    # ---- Load data (use config_a for data settings) ----
    config_a = yaml.safe_load(open(args.config_a))
    data_cfg = config_a.get("data", config_a)
    if args.data_root is not None:
        data_root = Path(args.data_root).resolve()
    else:
        data_root = resolve_path(Path(args.config_a).resolve().parent,
                                 data_cfg.get("data_root", config_a.get("data_root")))

    # Disable KNNI for clean visualization
    knni_cfg = dict(config_a.get("knni", {}))
    knni_cfg["enable"] = False
    config_a["knni"] = knni_cfg

    semkitti = SemanticKitti(
        root=str(data_root),
        sequences=[int(args.seq)],
        config_path=str(DATA_CONFIG_PATH),
        has_label=data_cfg.get("has_label", config_a.get("has_label", True)),
    )
    loader = RangeViewLoader(
        dataset=semkitti, config=config_a,
        is_train=False, return_uproj=True, use_kpconv=False,
    )

    frame_idx = find_frame_index(semkitti, args.seq, args.frame)
    print(f"Scene: seq={seq_str}  frame={frame_str}  (loader idx={frame_idx})")

    (proj_feature, proj_label, proj_mask,
     proj_range_t, uproj_x_t, uproj_y_t,
     uproj_depth_t, sem_label_points_t) = loader[frame_idx]

    gt_labels_2d = proj_label.numpy().astype(np.int32)
    mask_2d = proj_mask.numpy().astype(bool)
    proj_range = proj_range_t.numpy().astype(np.float32)
    uproj_x = uproj_x_t.numpy().astype(np.int64)
    uproj_y = uproj_y_t.numpy().astype(np.int64)
    uproj_depth = uproj_depth_t.numpy().astype(np.float32)
    gt_labels_3d = sem_label_points_t.numpy().astype(np.int32)

    # Raw point cloud for BEV
    pointcloud, sem_label_raw, _ = semkitti.loadDataByIndex(frame_idx)
    # pointcloud: [N, 4] (x, y, z, intensity)

    # ---- Load models ----
    print(f"\n--- Model A: {args.label_a} ---")
    model_a, settings_a = load_model(args.config_a, args.checkpoint_a, device)
    print(f"\n--- Model B: {args.label_b} ---")
    model_b, settings_b = load_model(args.config_b, args.checkpoint_b, device)

    # ---- Inference ----
    print("\nRunning inference...")
    pred_a_2d = run_inference(model_a, settings_a, proj_feature, device)
    pred_b_2d = run_inference(model_b, settings_b, proj_feature, device)

    # ---- Map 2D → 3D ----
    pred_a_3d = pred_a_2d[uproj_y, uproj_x]
    pred_b_3d = pred_b_2d[uproj_y, uproj_x]

    # ---- Optional KNN post-processing ----
    if args.knn:
        print("Applying KNN post-processing...")
        knn_params = {
            "knn": config_a.get("knn_k", 5),
            "search": config_a.get("knn_search", 13),
            "sigma": config_a.get("knn_sigma", 1.0),
            "cutoff": config_a.get("knn_cutoff", 1.0),
        }
        n_cls = data_cfg.get("n_classes", config_a.get("n_classes", 20))
        pred_a_3d = apply_knn(pred_a_2d, proj_range, uproj_x, uproj_y, uproj_depth, knn_params, n_cls, device)
        pred_b_3d = apply_knn(pred_b_2d, proj_range, uproj_x, uproj_y, uproj_depth, knn_params, n_cls, device)

    # ---- Save range-image views ----
    print("\nSaving range-image views...")
    save_range_image(gt_labels_2d, mask_2d, lmap_inv_lut, color_lut,
                     output_dir / f"range_gt_{tag}.png")
    save_range_image(pred_a_2d, mask_2d, lmap_inv_lut, color_lut,
                     output_dir / f"range_pred_{args.label_a}_{tag}.png")
    save_range_image(pred_b_2d, mask_2d, lmap_inv_lut, color_lut,
                     output_dir / f"range_pred_{args.label_b}_{tag}.png")

    # ---- Save BEV views ----
    print("Saving BEV views...")
    bev_kw = dict(
        mapped_color_lut=mapped_color_lut,
        bev_range=args.bev_range,
        point_size=args.point_size,
        dpi=args.dpi,
        bg=args.bev_bg,
    )
    save_bev_image(pointcloud, gt_labels_3d,
                   path=output_dir / f"bev_gt_{tag}.png", **bev_kw)
    save_bev_image(pointcloud, pred_a_3d,
                   path=output_dir / f"bev_pred_{args.label_a}_{tag}.png", **bev_kw)
    save_bev_image(pointcloud, pred_b_3d,
                   path=output_dir / f"bev_pred_{args.label_b}_{tag}.png", **bev_kw)

    # ---- Save BEV diff views (correct=green, wrong=red) ----
    print("Saving BEV diff views...")
    diff_kw = dict(
        bev_range=args.bev_range,
        point_size=args.point_size,
        dpi=args.dpi,
        bg=args.bev_bg,
    )
    save_bev_diff_image(pointcloud, gt_labels_3d, pred_a_3d,
                        path=output_dir / f"bev_diff_{args.label_a}_{tag}.png", **diff_kw)
    save_bev_diff_image(pointcloud, gt_labels_3d, pred_b_3d,
                        path=output_dir / f"bev_diff_{args.label_b}_{tag}.png", **diff_kw)

    # ---- Save legend ----
    save_legend(mapped_color_lut, output_dir / "class_legend.png", dpi=args.dpi)

    # ---- Print summary stats ----
    def miou(gt, pred, nc=20):
        valid = gt != 0
        g, p = gt[valid], pred[valid]
        if g.size == 0:
            return float("nan")
        ious = []
        for c in range(1, nc):
            inter = ((g == c) & (p == c)).sum()
            union = (g == c).sum() + (p == c).sum() - inter
            if union > 0:
                ious.append(inter / union)
        return float(np.mean(ious)) if ious else float("nan")

    print(f"\n{'='*50}")
    print(f"Range-image mIoU:")
    print(f"  {args.label_a}: {miou(gt_labels_2d[mask_2d], pred_a_2d[mask_2d]):.4f}")
    print(f"  {args.label_b}: {miou(gt_labels_2d[mask_2d], pred_b_2d[mask_2d]):.4f}")
    print(f"3D point mIoU{' (KNN)' if args.knn else ''}:")
    print(f"  {args.label_a}: {miou(gt_labels_3d, pred_a_3d):.4f}")
    print(f"  {args.label_b}: {miou(gt_labels_3d, pred_b_3d):.4f}")
    print(f"{'='*50}")

    print(f"\nOutputs saved to {output_dir}/")
    print(f"  range_gt_{tag}.png")
    print(f"  range_pred_{args.label_a}_{tag}.png")
    print(f"  range_pred_{args.label_b}_{tag}.png")
    print(f"  bev_gt_{tag}.png")
    print(f"  bev_pred_{args.label_a}_{tag}.png")
    print(f"  bev_pred_{args.label_b}_{tag}.png")
    print(f"  bev_diff_{args.label_a}_{tag}.png")
    print(f"  bev_diff_{args.label_b}_{tag}.png")
    print(f"  class_legend.png")


if __name__ == "__main__":
    main()
