"""
Compute WPD statistics for SemanticKITTI dataset.
This script calculates class frequencies, alpha, beta, and normalized weights
for the Weighted Paste and Drop (WPD) augmentation.
"""

import numpy as np
import os
import yaml
from collections import defaultdict


def compute_wpd_statistics(data_root, kitti_config_path, output_path='wpd_stats.yaml'):
    """
    Compute WPD statistics for SemanticKITTI training set.

    Args:
        data_root: Path to SemanticKITTI dataset sequences
        kitti_config_path: Path to semantic-kitti.yaml
        output_path: Path to save computed statistics
    """
    # Load SemanticKITTI config
    with open(kitti_config_path, 'r') as f:
        kitti_config = yaml.safe_load(f)

    learning_map = kitti_config['learning_map']
    train_sequences = kitti_config['split']['train']

    # Initialize counters
    class_point_counts = defaultdict(int)  # semantic point counts
    class_instance_counts = defaultdict(int)  # instance counts (for panoptic)
    total_points = 0

    print("Computing class statistics from training set...")

    # Iterate through training sequences
    for seq in train_sequences:
        seq_str = f"{seq:02d}"
        label_path = os.path.join(data_root, seq_str, "labels")

        if not os.path.exists(label_path):
            print(f"Warning: {label_path} not found, skipping sequence {seq_str}")
            continue

        # Get all label files in sequence
        label_files = sorted([f for f in os.listdir(label_path) if f.endswith('.label')])

        print(f"Processing sequence {seq_str} ({len(label_files)} frames)...")

        for label_file in label_files:
            label_file_path = os.path.join(label_path, label_file)

            # Load labels (uint32 format: lower 16 bits = semantic, upper 16 bits = instance)
            labels = np.fromfile(label_file_path, dtype=np.uint32)

            # Extract semantic and instance labels
            sem_labels = labels & 0xFFFF  # Lower 16 bits
            inst_labels = labels >> 16     # Upper 16 bits

            # Map to learning labels
            mapped_labels = np.vectorize(learning_map.get)(sem_labels, 0)

            # Count points per class
            unique_classes, counts = np.unique(mapped_labels, return_counts=True)
            for cls, count in zip(unique_classes, counts):
                class_point_counts[int(cls)] += int(count)

            # Count instances per class (for panoptic)
            for cls in range(20):  # 0-19 learning labels
                cls_mask = mapped_labels == cls
                if cls_mask.sum() > 0:
                    # Count unique instances of this class
                    unique_instances = np.unique(inst_labels[cls_mask])
                    # Filter out instance ID 0 (typically means "stuff" class or no instance)
                    unique_instances = unique_instances[unique_instances != 0]
                    class_instance_counts[int(cls)] += len(unique_instances)

            total_points += len(labels)

    print(f"\nTotal points processed: {total_points}")
    print(f"Class distribution:")

    # Compute frequencies and weights
    epsilon = 1e-6
    n_classes = 20  # SemanticKITTI learning labels: 0-19

    # Initialize arrays
    class_frequencies = np.zeros(n_classes)
    alpha = np.zeros(n_classes)
    beta = np.zeros(n_classes)

    for cls in range(n_classes):
        point_count = class_point_counts.get(cls, 0)
        instance_count = class_instance_counts.get(cls, 0)

        # Frequency
        f_i = point_count / (total_points + epsilon)
        class_frequencies[cls] = f_i

        # Alpha (inverse frequency weight)
        alpha[cls] = 1.0 / (f_i + epsilon)

        # Beta (instance-to-semantic ratio)
        # For "stuff" classes, we treat beta as 1.0
        if instance_count > 0 and point_count > 0:
            beta[cls] = instance_count / point_count
        else:
            beta[cls] = 1.0

        # Print statistics
        class_name = kitti_config['mapped_class_name'].get(cls, f"class_{cls}")
        print(f"  {cls:2d} {class_name:20s}: points={point_count:12d} ({f_i:.6f}), "
              f"instances={instance_count:8d}, alpha={alpha[cls]:.2f}, beta={beta[cls]:.6f}")

    # Normalize weights for semantic segmentation
    w_semantic = alpha / (alpha.max() + epsilon)

    # Normalize weights for panoptic segmentation
    w_panoptic = (alpha * beta) / ((alpha * beta).max() + epsilon)

    # Define long-tail threshold
    threshold = 0.1

    # Identify long-tail classes
    long_tail_semantic = w_semantic > threshold
    long_tail_panoptic = w_panoptic > threshold

    # Compute paste/drop probabilities
    # Paste probability: p_i = max(0, w_i - t) for long-tail classes
    p_semantic = np.maximum(0, w_semantic - threshold)
    p_panoptic = np.maximum(0, w_panoptic - threshold)

    # Drop probability: d_i = max(0, t - w_i) for non-long-tail classes
    d_semantic = np.maximum(0, threshold - w_semantic)
    d_panoptic = np.maximum(0, threshold - w_panoptic)

    # Print summary table for verification
    header = (
        f"\n{'ID':>3}  {'Class':<20}  {'Points':>12}  {'Freq':>10}  "
        f"{'Alpha':>10}  {'Instances':>10}  {'w_norm':>10}  {'LongTail':>8}"
    )
    print(header)
    print("-" * len(header))
    for cls in range(n_classes):
        class_name = kitti_config['mapped_class_name'].get(cls, f"class_{cls}")
        point_count = class_point_counts.get(cls, 0)
        freq = class_frequencies[cls]
        alpha_val = alpha[cls]
        inst_count = class_instance_counts.get(cls, 0)
        w_val = w_semantic[cls]
        tail_flag = "Y" if long_tail_semantic[cls] else "N"
        print(
            f"{cls:3d}  {class_name:<20}  {point_count:12d}  {freq:10.6f}  "
            f"{alpha_val:10.2f}  {inst_count:10d}  {w_val:10.6f}  {tail_flag:>8}"
        )

    # Prepare output
    stats = {
        'threshold': float(threshold),
        'epsilon': epsilon,
        'total_points': int(total_points),
        'n_classes': n_classes,
        'semantic': {
            'class_frequencies': class_frequencies.tolist(),
            'alpha': alpha.tolist(),
            'w_norm': w_semantic.tolist(),
            'long_tail_classes': [int(i) for i in np.where(long_tail_semantic)[0]],
            'paste_prob': p_semantic.tolist(),
            'drop_prob': d_semantic.tolist(),
        },
        'panoptic': {
            'beta': beta.tolist(),
            'w_norm': w_panoptic.tolist(),
            'long_tail_classes': [int(i) for i in np.where(long_tail_panoptic)[0]],
            'paste_prob': p_panoptic.tolist(),
            'drop_prob': d_panoptic.tolist(),
        },
        'class_names': {i: kitti_config['mapped_class_name'].get(i, f"class_{i}")
                       for i in range(n_classes)}
    }

    # Save statistics
    with open(output_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

    print(f"\n=== Semantic Segmentation Weights ===")
    print(f"Long-tail classes (w > {threshold}):")
    for cls in stats['semantic']['long_tail_classes']:
        class_name = stats['class_names'][cls]
        print(f"  {cls:2d} {class_name:20s}: w={w_semantic[cls]:.4f}, "
              f"p={p_semantic[cls]:.4f}")

    print(f"\n=== Panoptic Segmentation Weights ===")
    print(f"Long-tail classes (w > {threshold}):")
    for cls in stats['panoptic']['long_tail_classes']:
        class_name = stats['class_names'][cls]
        print(f"  {cls:2d} {class_name:20s}: w={w_panoptic[cls]:.4f}, "
              f"p={p_panoptic[cls]:.4f}")

    print(f"\nStatistics saved to {output_path}")

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute WPD statistics for SemanticKITTI')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to SemanticKITTI dataset sequences (e.g., /path/to/dataset/sequences/)')
    parser.add_argument('--config', type=str,
                       default='dataset/semantic_kitti/semantic-kitti.yaml',
                       help='Path to semantic-kitti.yaml config file')
    parser.add_argument('--output', type=str,
                       default='dataset/preprocess/wpd_stats.yaml',
                       help='Output path for WPD statistics')

    args = parser.parse_args()

    compute_wpd_statistics(args.data_root, args.config, args.output)
