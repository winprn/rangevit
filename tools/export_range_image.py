import argparse
import os
import sys
import yaml
import numpy as np
import torch
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from option import Option
from main import build_rangevit_model
from dataset.semantic_kitti.parser import SemanticKitti
from dataset.range_view_loader import RangeViewLoader
from utils.inference.inference_utils import inference as sliding_inference


def load_settings(config_path, data_root, save_path=None, run_id=None, checkpoint=None):
    # Minimal Option wrapper to reuse model construction logic
    args = argparse.Namespace(
        config_path=config_path,
        save_path=save_path,
        id=run_id,
        pretrained_model=None,
        checkpoint=checkpoint,
        window_stride=None,
        mini=False,
        val_only=True,
        test_split=False,
        save_eval_results=False,
        full=False,
        log_frequency=100,
        save_frequent=None,
        seed=1,
        continue_training=False,
        data_root=data_root,
        num_workers=0,  # unused here
    )
    settings = Option(config_path, args)
    settings.data_root = data_root
    settings.id = run_id or settings.id
    settings.checkpoint = checkpoint
    settings.pretrained_model = None
    settings.val_only = True
    settings.test_split = False
    settings.save_eval_results = False
    settings.log_frequency = 0
    settings.num_workers = 0
    settings.use_kpconv = False
    return settings


def build_semantickitti_dataset(data_root, seq, has_label=False):
    cfg_path = os.path.join('dataset', 'semantic_kitti', 'semantic-kitti.yaml')
    return SemanticKitti(
        root=data_root,
        sequences=[seq],
        config_path=cfg_path,
        has_label=has_label)


def find_frame_index(dataset, seq, frame_id):
    target_seq = f"{int(seq):02d}"
    target_frame = f"{int(frame_id):06d}"
    for idx, path in enumerate(dataset.pointcloud_files):
        seq_id, frame = dataset.parsePathInfoByIndex(idx)
        if seq_id == target_seq and frame == target_frame:
            return idx
    raise ValueError(f"Frame {target_frame} in sequence {target_seq} not found.")


def apply_color_map(label_map, color_map_inv):
    h, w = label_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in color_map_inv.items():
        rgb = [v[2], v[1], v[0]]  # BGR -> RGB
        color_img[label_map == int(k)] = rgb
    return color_img


def save_range_image(range_image, output_dir, seq, frame_id, label_map=None, color_map_inv=None):
    os.makedirs(output_dir, exist_ok=True)
    out_npy = os.path.join(output_dir, f"{seq:02d}_{frame_id:06d}_range.npy")
    np.save(out_npy, range_image)

    # If labels provided, colorize by label map; otherwise normalize range.
    if label_map is not None and color_map_inv is not None:
        color_img = apply_color_map(label_map, color_map_inv)
    else:
        valid = range_image > 0
        if valid.any():
            r_min, r_max = range_image[valid].min(), range_image[valid].max()
            scaled = (255.0 * (range_image - r_min) / max(r_max - r_min, 1e-6)).clip(0, 255)
        else:
            scaled = np.zeros_like(range_image)
        color_img = scaled.astype(np.uint8)

    out_png = os.path.join(output_dir, f"{seq:02d}_{frame_id:06d}_range.png")
    Image.fromarray(color_img).save(out_png)

    print(f"Saved range image to {out_png} (PNG) and {out_npy} (raw)")
    return out_png


def save_labels(labels, class_map_lut_inv, output_dir, seq, frame_id):
    os.makedirs(output_dir, exist_ok=True)
    mapped = class_map_lut_inv[labels]
    out_path = os.path.join(output_dir, f"{seq:02d}_{frame_id:06d}.label")
    mapped.astype(np.uint32).tofile(out_path)
    print(f"Saved predicted labels to {out_path}")
    return out_path


def save_color_label_png(label_map, color_map_inv, output_dir, seq, frame_id, suffix="pred"):
    os.makedirs(output_dir, exist_ok=True)
    h, w = label_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in color_map_inv.items():
        # color_map_inv is BGR in config; convert to RGB for Pillow.
        rgb = [v[2], v[1], v[0]]
        color_img[label_map == int(k)] = rgb
    out_png = os.path.join(output_dir, f"{seq:02d}_{frame_id:06d}_{suffix}.png")
    Image.fromarray(color_img).save(out_png)
    print(f"Saved color label map to {out_png}")
    return out_png


def run_projection_only(config, data_root, seq, frame_id, output_dir, use_gt_labels=False):
    if config.get('use_kpconv', False):
        raise NotImplementedError("export_range_image currently supports use_kpconv=False.")
    dataset = build_semantickitti_dataset(data_root, seq, has_label=use_gt_labels)
    loader = RangeViewLoader(
        dataset=dataset,
        config=config,
        is_train=False,
        return_uproj=True,
        use_kpconv=config.get('use_kpconv', False))

    idx = find_frame_index(dataset, seq, frame_id)
    (
        proj_feature_tensor,
        proj_sem_label_tensor,
        proj_mask_tensor,
        proj_range_tensor,
        uproj_x_tensor,
        uproj_y_tensor,
        uproj_depth_tensor,
        sem_label_tensor,
        _,
    ) = loader[idx]

    # proj_range_tensor is already unnormalized range image
    save_range_image(
        proj_range_tensor.numpy(),
        output_dir,
        seq,
        frame_id,
        label_map=proj_sem_label_tensor.numpy() if use_gt_labels else None,
        color_map_inv=dataset.data_config.get('color_map_inv'),
    )

    return {
        'proj_feature': proj_feature_tensor,
        'proj_range': proj_range_tensor,
        'uproj_x': uproj_x_tensor,
        'uproj_y': uproj_y_tensor,
        'uproj_depth': uproj_depth_tensor,
        'proj_label': proj_sem_label_tensor,
        'dataset': dataset,
    }


def run_with_checkpoint(config_path, config, data_root, seq, frame_id, output_dir, checkpoint, device):
    settings = load_settings(config_path, data_root, save_path=config.get('save_path'), checkpoint=checkpoint)
    model = build_rangevit_model(settings, pretrained_path=None)
    model.eval()
    model.to(device)

    data = run_projection_only(config, data_root, seq, frame_id, output_dir)
    proj_feature = data['proj_feature'].unsqueeze(0).to(device)  # 1 x 5 x H x W

    im_meta = dict(flip=False)
    seg_map = sliding_inference(
        model.rangevit,
        [proj_feature],
        [im_meta],
        ori_shape=proj_feature.shape[2:4],
        window_size=config['window_size'],
        window_stride=config['window_stride'],
        batch_size=1,
        use_kpconv=config.get('use_kpconv', False),
    )  # [n_cls, H, W]

    pred_argmax = seg_map.argmax(dim=0)  # H x W

    uproj_x = data['uproj_x'].long()
    uproj_y = data['uproj_y'].long()
    pred_np = pred_argmax[uproj_y, uproj_x].cpu().numpy().astype(np.int32)

    color_map_inv = data['dataset'].data_config['color_map_inv']
    save_range_image(
        data['proj_range'].numpy(),
        output_dir,
        seq,
        frame_id,
        label_map=pred_argmax.cpu().numpy(),
        color_map_inv=color_map_inv,
    )
    save_labels(pred_np, data['dataset'].class_map_lut_inv, output_dir, seq, frame_id)
    save_color_label_png(pred_argmax.cpu().numpy(), color_map_inv, output_dir, seq, frame_id, suffix="pred_color")


def main():
    parser = argparse.ArgumentParser(description='Export range image (SemanticKITTI only).')
    parser.add_argument('--config', required=True, help='Path to config_kitti.yaml')
    parser.add_argument('--data_root', default=None, help='SemanticKITTI data root (optional, overrides config).')
    parser.add_argument('--sequence', type=int, required=True, help='Sequence id (e.g., 8).')
    parser.add_argument('--frame', type=int, required=True, help='Frame id (e.g., 0 or 1234).')
    parser.add_argument('--output_dir', default='outputs/range_exports', help='Directory to save outputs.')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path to also export predicted labels.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for inference.')
    parser.add_argument('--use_gt_labels', action='store_true', help='Colorize range image with ground-truth labels (SemanticKITTI).')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = config.copy()
    config['has_label'] = args.use_gt_labels  # allow GT labels when requested
    config['use_kpconv'] = False  # current export supports range-view path only

    data_root = args.data_root or config.get('data_root')
    if data_root is None:
        raise ValueError("data_root must be provided either in the config file or via --data_root.")

    # if config.get('model_type', 'rangevit').lower() != 'rangevit':
    #     raise NotImplementedError("export_range_image supports model_type=='rangevit' only.")

    # Ensure required config fields are present
    for key in ('sensor', 'augmentation', 'window_size', 'window_stride', 'image_size', 'original_image_size'):
        if key not in config:
            raise ValueError(f"Missing '{key}' in config: {args.config}")

    # Always output range image
    projection_data = run_projection_only(
        config,
        data_root,
        args.sequence,
        args.frame,
        args.output_dir,
        use_gt_labels=args.use_gt_labels,
    )

    # Optionally run prediction
    if args.checkpoint:
        run_with_checkpoint(args.config, config, data_root, args.sequence, args.frame, args.output_dir, args.checkpoint, args.device)


if __name__ == '__main__':
    main()
