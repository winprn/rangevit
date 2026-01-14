import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import imageio.v2 as imageio

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from dataset.semantic_kitti import SemanticKitti
from dataset.range_view_loader import RangeViewLoader
from main import build_rangevit_model
from utils.inference.inference_utils import inference


# Explicit (raw) SemanticKITTI label -> RGB mapping (0-255), provided by user.
SEMANTIC_KITTI_COLORS = {
    0: [0, 0, 0],
    1: [0, 0, 0],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
}

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifact" / "range_image"
DATA_CONFIG_PATH = REPO_ROOT / "dataset" / "semantic_kitti" / "semantic-kitti.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SemanticKITTI range image for a single sequence/frame. "
        "If the config specifies a checkpoint, also run inference and plot predictions."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "config_kitti.yaml"),
        help="Path to config YAML (e.g., config_kitti.yaml).",
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        help="Sequence id (e.g., 00, 08).",
    )
    parser.add_argument(
        "--frame",
        type=str,
        required=True,
        help="Frame id (e.g., 0 or 000123).",
    )
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


def build_learning_map_inv_lut(learning_map_inv: dict) -> np.ndarray:
    max_key = max(int(k) for k in learning_map_inv.keys())
    lut = np.zeros(max_key + 1, dtype=np.int32)
    for k, v in learning_map_inv.items():
        lut[int(k)] = int(v)
    return lut


def build_color_lut(color_map: dict, min_size: int = 0) -> np.ndarray:
    max_key = max(max(int(k) for k in color_map.keys()), min_size)
    lut = np.zeros((max_key + 1, 3), dtype=np.uint8)
    for k, rgb in color_map.items():
        lut[int(k)] = np.array(rgb, dtype=np.uint8)
    return lut


def colorize_labels(mapped_labels: np.ndarray, learning_map_inv_lut: np.ndarray, color_lut: np.ndarray) -> np.ndarray:
    raw_labels = learning_map_inv_lut[mapped_labels]
    raw_labels = np.clip(raw_labels, 0, color_lut.shape[0] - 1)
    colors = color_lut[raw_labels]
    return colors


def compute_miou(gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int = 0) -> float:
    """Compute mIoU over valid pixels (gt != ignore_index)."""
    if num_classes <= 0:
        return float("nan")
    valid_mask = gt != ignore_index
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]
    if gt_valid.size == 0:
        return float("nan")

    gt_valid = np.clip(gt_valid, 0, num_classes - 1)
    pred_valid = np.clip(pred_valid, 0, num_classes - 1)

    hist = np.bincount(
        gt_valid * num_classes + pred_valid, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    ious = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        tp = hist[c, c]
        denom = hist[c, :].sum() + hist[:, c].sum() - tp
        if denom > 0:
            ious.append(tp / denom)
    if not ious:
        return float("nan")
    return float(np.mean(ious))


def compute_per_class_iou(gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int = 0):
    """
    Return list of (class_id, iou) for classes present in gt or pred (excluding ignore_index).
    """
    valid_mask = gt != ignore_index
    gt = gt[valid_mask]
    pred = pred[valid_mask]
    if gt.size == 0:
        return []
    ious = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        gt_c = gt == c
        pred_c = pred == c
        inter = (gt_c & pred_c).sum()
        union = gt_c.sum() + pred_c.sum() - inter
        if union > 0:
            iou = float(inter / union)
            ious.append((c, iou))
    return ious


def compute_mislabel_iou(mis_mask: np.ndarray, valid_mask: np.ndarray) -> float:
    """
    Compute IoU of the mislabel class vs total valid pixels (void excluded).
    Equivalent to (#misclassified valid pixels) / (#valid pixels).
    """
    valid_count = valid_mask.sum()
    if valid_count == 0:
        return float("nan")
    mis_count = (mis_mask & valid_mask).sum()
    return float(mis_count / valid_count)


def save_image(array: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, array.astype(np.uint8))


def build_settings_from_config(config: dict):
    # Minimal settings namespace for build_rangevit_model and inference_utils.
    class Settings:
        pass

    settings = Settings()
    settings.config = config
    settings.in_channels = config.get("in_channels", 5)
    settings.n_classes = config.get("n_classes", 20)
    settings.vit_backbone = config.get("vit_backbone", "vit_small_patch16_384")
    settings.image_size = tuple(config.get("image_size", [32, 384]))
    settings.patch_size = tuple(config.get("patch_size", [2, 8]))
    settings.patch_stride = tuple(config.get("patch_stride", [2, 8]))
    settings.reuse_pos_emb = config.get("reuse_pos_emb", False)
    settings.reuse_patch_emb = config.get("reuse_patch_emb", False)
    settings.conv_stem = config.get("conv_stem", "ConvStem")
    settings.stem_base_channels = config.get("stem_base_channels", 32)
    settings.D_h = config.get("D_h", 256)
    settings.skip_filters = config.get("skip_filters", 0)
    settings.decoder = config.get("decoder", "up_conv")
    settings.use_kpconv = str(config.get("point_postproc", "none")).lower() == "kpconv"
    settings.window_size = tuple(config.get("window_size", config.get("original_image_size", [32, 384])))
    settings.window_stride = tuple(config.get("window_stride", settings.window_size))
    settings.use_sliding_window = config.get("use_sliding_window", True)
    settings.pretrained_model = config.get("pretrained_model", None)
    settings.checkpoint = config.get("checkpoint", None)
    return settings


def load_model_for_inference(settings, device: torch.device):
    model = build_rangevit_model(settings, pretrained_path=settings.pretrained_model)
    checkpoint_path = settings.checkpoint
    if checkpoint_path is None:
        return model.to(device)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint load status: {msg}")
    return model.to(device)


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = yaml.safe_load(open(config_path, "r"))

    data_root = resolve_path(config_dir, config["data_root"])
    dataset_cfg = yaml.safe_load(open(DATA_CONFIG_PATH, "r"))

    learning_map_inv_lut = build_learning_map_inv_lut(dataset_cfg["learning_map_inv"])
    color_lut = build_color_lut(SEMANTIC_KITTI_COLORS, min_size=int(learning_map_inv_lut.max()))

    semkitti_ds = SemanticKitti(
        root=str(data_root),
        sequences=[int(args.seq)],
        config_path=str(DATA_CONFIG_PATH),
        has_label=config.get("has_label", True),
    )
    loader = RangeViewLoader(
        dataset=semkitti_ds,
        config=config,
        is_train=False,
        return_uproj=True,
        use_kpconv=False,
    )

    frame_idx = find_frame_index(semkitti_ds, args.seq, args.frame)
    (
        proj_feature_tensor,
        proj_sem_label_tensor,
        proj_mask_tensor,
        _proj_range,
        _uproj_x,
        _uproj_y,
        _uproj_depth,
        _sem_label_points,
    ) = loader[frame_idx]

    mapped_labels = proj_sem_label_tensor.numpy().astype(np.int32)
    mask = proj_mask_tensor.numpy().astype(bool)
    gt_colors = colorize_labels(mapped_labels, learning_map_inv_lut, color_lut)
    # Keep invalid pixels black.
    gt_colors[~mask] = np.array([0, 0, 0], dtype=np.uint8)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seq_id = f"{int(args.seq):02d}"
    frame_id = f"{int(args.frame):06d}"
    gt_path = DEFAULT_OUTPUT_DIR / f"range_gt_seq{seq_id}_frame{frame_id}.png"
    save_image(gt_colors, gt_path)

    # Optional inference (only if checkpoint is set in the config)
    if config.get("checkpoint"):
        settings = build_settings_from_config(config)
        checkpoint_path = resolve_path(config_dir, settings.checkpoint)
        settings.checkpoint = str(checkpoint_path)
        model = load_model_for_inference(settings, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()

        device = next(model.parameters()).device
        input_feature = proj_feature_tensor.unsqueeze(0).to(device)
        im_meta = {"flip": False}
        with torch.no_grad():
            seg_map = inference(
                model.rangevit,
                [input_feature],
                [im_meta],
                ori_shape=input_feature.shape[2:4],
                window_size=settings.window_size,
                window_stride=settings.window_stride,
                batch_size=1,
                use_kpconv=settings.use_kpconv,
                use_sliding_window=settings.use_sliding_window,
            )
        pred_labels = seg_map.argmax(dim=0).cpu().numpy().astype(np.int32)

        pred_colors = colorize_labels(pred_labels, learning_map_inv_lut, color_lut)
        pred_colors[~mask] = np.array([0, 0, 0], dtype=np.uint8)
        pred_path = DEFAULT_OUTPUT_DIR / f"range_pred_seq{seq_id}_frame{frame_id}.png"
        save_image(pred_colors, pred_path)
        print(f"Inference image saved to {pred_path}")

        # Compute mIoU on valid pixels (mask) ignoring class 0 as void.
        gt_labels = mapped_labels.astype(np.int32)
        miou = compute_miou(
            gt_labels[mask],
            pred_labels[mask],
            num_classes=int(config.get("n_classes", 20)),
            ignore_index=0,
        )
        print(f"mIoU (valid pixels, ignore_index=0): {miou:.4f}")

        # Highlight misclassified pixels (valid, non-void only).
        mis_mask = mask & (gt_labels != 0) & (pred_labels != gt_labels)
        mis_highlight = np.zeros_like(pred_colors, dtype=np.uint8)
        mis_highlight[mis_mask] = np.array([255, 0, 0], dtype=np.uint8)  # red for misclass
        mis_path = DEFAULT_OUTPUT_DIR / f"range_misclass_seq{seq_id}_frame{frame_id}.png"
        save_image(mis_highlight, mis_path)
        print(f"Misclassification mask saved to {mis_path}")
        mis_iou = compute_mislabel_iou(mis_mask, valid_mask=mask & (gt_labels != 0))
        print(f"Mislabel IoU vs valid pixels (void excluded): {mis_iou:.4f}")

        valid_mask = mask & (gt_labels != 0)
        valid_pixels = int(valid_mask.sum())
        mis_pixels = int((mis_mask & valid_mask).sum())
        correct_pixels = valid_pixels - mis_pixels
        mis_pct = (mis_pixels / valid_pixels * 100.0) if valid_pixels > 0 else 0.0
        correct_pct = (correct_pixels / valid_pixels * 100.0) if valid_pixels > 0 else 0.0
        print(f"Valid pixels: {valid_pixels}")
        print(f"Correct predictions: {correct_pixels} ({correct_pct:.2f}%)")
        print(f"Misclassified predictions: {mis_pixels} ({mis_pct:.2f}%)")

        per_class_ious = compute_per_class_iou(gt_labels[mask], pred_labels[mask], num_classes=int(config.get("n_classes", 20)), ignore_index=0)
        per_class_ious_sorted = sorted(per_class_ious, key=lambda x: x[1])
        if per_class_ious_sorted:
            print("Per-class IoU (sorted low->high, excluding void=0):")
            for cid, iou in per_class_ious_sorted:
                print(f"  class {cid}: {iou:.4f}")
            mean_per_class_iou = sum(i for _, i in per_class_ious_sorted) / len(per_class_ious_sorted)
            print(f"Mean per-class IoU (excluding void=0): {mean_per_class_iou:.4f}")

        # FP: predicted non-void, wrong class. FN: GT non-void, predicted void (class 0).
        fp_mask = valid_mask & (pred_labels != gt_labels) & (pred_labels != 0)
        fn_mask = valid_mask & (pred_labels == 0)

        fp_highlight = np.zeros_like(pred_colors, dtype=np.uint8)
        fn_highlight = np.zeros_like(pred_colors, dtype=np.uint8)
        fp_highlight[fp_mask] = np.array([0, 0, 255], dtype=np.uint8)  # blue for FP
        fn_highlight[fn_mask] = np.array([0, 255, 0], dtype=np.uint8)  # green for FN

        fp_path = DEFAULT_OUTPUT_DIR / f"range_fp_seq{seq_id}_frame{frame_id}.png"
        fn_path = DEFAULT_OUTPUT_DIR / f"range_fn_seq{seq_id}_frame{frame_id}.png"
        save_image(fp_highlight, fp_path)
        save_image(fn_highlight, fn_path)
        print(f"FP mask saved to {fp_path}")
        print(f"FN mask saved to {fn_path}")
    else:
        print("No checkpoint in config; skipping inference, mIoU, and misclassification mask.")

    print(f"GT range image saved to {gt_path}")


if __name__ == "__main__":
    main()
