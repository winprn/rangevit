"""Visualize range image augmentations on two SemanticKITTI scans.

Usage:
    python tests/test_range_augmentation.py \
        --data_root ../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences \
        --sequence 08 --scan_a 000000 --scan_b 000100
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset.preprocess.projection import RangeProjection
from dataset.preprocess.rangeaug import RangeAugmentation


# SemanticKITTI learning map (raw label -> mapped 0-19)
LEARNING_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
    51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
    99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}

# SemanticKITTI sensor params
SENSOR = dict(fov_up=3.0, fov_down=-25.0, proj_w=2048, proj_h=64)

# 20-class colormap (index 0 = unlabeled/black)
CLASS_COLORS = np.array([
    [0, 0, 0],        # 0  unlabeled
    [100, 150, 245],   # 1  car
    [100, 230, 245],   # 2  bicycle
    [30, 60, 150],     # 3  motorcycle
    [80, 30, 180],     # 4  truck
    [0, 0, 255],       # 5  other-vehicle
    [255, 30, 30],     # 6  person
    [255, 40, 200],    # 7  bicyclist
    [150, 30, 90],     # 8  motorcyclist
    [255, 0, 255],     # 9  road
    [255, 150, 255],   # 10 parking
    [75, 0, 75],       # 11 sidewalk
    [175, 0, 75],      # 12 other-ground
    [255, 200, 0],     # 13 building
    [255, 120, 50],    # 14 fence
    [0, 175, 0],       # 15 vegetation
    [135, 60, 0],      # 16 trunk
    [150, 240, 80],    # 17 terrain
    [255, 240, 150],   # 18 pole
    [255, 0, 0],       # 19 traffic-sign
], dtype=np.uint8)


def load_scan(data_root, sequence, scan_id):
    """Load a SemanticKITTI scan and its labels, project to range image."""
    bin_path = os.path.join(data_root, sequence, 'velodyne', f'{scan_id}.bin')
    label_path = os.path.join(data_root, sequence, 'labels', f'{scan_id}.label')

    if not os.path.exists(bin_path):
        raise FileNotFoundError(f'Scan not found: {bin_path}')
    if not os.path.exists(label_path):
        raise FileNotFoundError(f'Label not found: {label_path}')

    # Load point cloud (N, 4): x, y, z, intensity
    pcd = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    # Load labels and apply learning map
    raw_label = np.fromfile(label_path, dtype=np.int32)
    sem_label = raw_label & 0xFFFF
    max_key = max(LEARNING_MAP.keys()) + 1
    lut = np.zeros(max_key, dtype=np.int32)
    for k, v in LEARNING_MAP.items():
        lut[k] = v
    mapped_label = lut[sem_label]

    # Project to range image
    proj = RangeProjection(**SENSOR)
    proj_pointcloud, proj_range, proj_idx, proj_mask = proj.doProjection(pcd)

    # Build 5-channel feature: [range, x, y, z, intensity]
    proj_range_t = torch.from_numpy(proj_range).unsqueeze(0)  # (1, H, W)
    proj_xyz_t = torch.from_numpy(proj_pointcloud[:, :, :3]).permute(2, 0, 1)  # (3, H, W)
    proj_intensity_t = torch.from_numpy(proj_pointcloud[:, :, 3]).unsqueeze(0)  # (1, H, W)
    feature = torch.cat([proj_range_t, proj_xyz_t, proj_intensity_t], dim=0)  # (5, H, W)

    # Build label image
    label_img = np.zeros((SENSOR['proj_h'], SENSOR['proj_w']), dtype=np.int64)
    valid = proj_idx >= 0
    label_img[valid] = mapped_label[proj_idx[valid]]
    label_t = torch.from_numpy(label_img)

    mask_t = torch.from_numpy(proj_mask).float()

    return feature, label_t, mask_t


def label_to_color(label_np):
    """Convert a (H, W) label array to an (H, W, 3) RGB image."""
    h, w = label_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(CLASS_COLORS)):
        color[label_np == cls_id] = CLASS_COLORS[cls_id]
    return color


def save_comparison(original_feat, original_label, aug_feat, aug_label, title, output_path):
    """Save a 2x2 comparison grid: [orig range, aug range, orig labels, aug labels]."""
    fig, axes = plt.subplots(2, 2, figsize=(24, 6))

    orig_range = original_feat[0].numpy()
    aug_range = aug_feat[0].numpy()
    orig_label_color = label_to_color(original_label.numpy())
    aug_label_color = label_to_color(aug_label.numpy())

    axes[0, 0].imshow(orig_range, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Original Range')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(aug_range, cmap='viridis', aspect='auto')
    axes[0, 1].set_title(f'{title} Range')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(orig_label_color, aspect='auto')
    axes[1, 0].set_title('Original Labels')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(aug_label_color, aspect='auto')
    axes[1, 1].set_title(f'{title} Labels')
    axes[1, 1].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize range image augmentations')
    parser.add_argument('--data_root', required=True, help='Path to SemanticKITTI sequences directory')
    parser.add_argument('--sequence', required=True, help='Sequence number (e.g., 08)')
    parser.add_argument('--scan_a', required=True, help='First scan index (e.g., 000000)')
    parser.add_argument('--scan_b', required=True, help='Second scan index for mixing (e.g., 000100)')
    parser.add_argument('--output_dir', default='tests/range_aug_output', help='Output directory for PNGs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading scan A: sequence {args.sequence}, scan {args.scan_a}')
    feat_a, label_a, mask_a = load_scan(args.data_root, args.sequence, args.scan_a)

    print(f'Loading scan B: sequence {args.sequence}, scan {args.scan_b}')
    feat_b, label_b, mask_b = load_scan(args.data_root, args.sequence, args.scan_b)

    # Stack into batch of 2
    features = torch.stack([feat_a, feat_b])   # (2, 5, H, W)
    labels = torch.stack([label_a, label_b])    # (2, H, W)
    masks = torch.stack([mask_a, mask_b])       # (2, H, W)

    # Save original
    save_comparison(feat_a, label_a, feat_b, label_b,
                    'Original (scan_a vs scan_b)',
                    os.path.join(args.output_dir, 'original.png'))

    # Test each augmentation individually (prob=1.0 to force activation)
    augmentations = {
        'RangePolar':      [1.0, 0.0, 0.0, 0.0, 0.0],
        'RangeBeams':      [0.0, 1.0, 0.0, 0.0, 0.0],
        'RangeCompletion': [0.0, 0.0, 1.0, 0.0, 0.0],
        'RangeInstance':   [0.0, 0.0, 0.0, 0.0, 1.0],
    }

    for name, probs in augmentations.items():
        print(f'Applying {name}...')
        aug = RangeAugmentation(aug_prob=probs)
        aug_features, aug_labels = aug(features.clone(), labels.clone(), masks.clone())
        save_comparison(feat_a, label_a, aug_features[0], aug_labels[0],
                        name, os.path.join(args.output_dir, f'{name.lower()}.png'))

    # All augmentations combined (default probabilities)
    print('Applying all augmentations...')
    aug_all = RangeAugmentation()
    aug_features, aug_labels = aug_all(features.clone(), labels.clone(), masks.clone())
    save_comparison(feat_a, label_a, aug_features[0], aug_labels[0],
                    'All Augmentations', os.path.join(args.output_dir, 'all_augmentations.png'))

    print(f'\nDone! All images saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
