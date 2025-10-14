"""
Weighted Paste and Drop (WPD) Augmentation for Range-View LiDAR Semantic Segmentation

Implementation based on the paper addressing:
1. Overfitting (insufficient diverse data for transformer models)
2. Context reliance (model relying on priors like "bikes are on roads")
3. Class imbalance (road/car dominate, rare classes underperform)

WPD operates on range images after standard point-cloud augmentations.
"""

import numpy as np
import torch
import yaml
import os


class WPDConfig:
    """Configuration for WPD augmentation."""

    def __init__(self, stats_path, mode='semantic', threshold=0.1):
        """
        Args:
            stats_path: Path to WPD statistics YAML file
            mode: 'semantic' or 'panoptic'
            threshold: Threshold for long-tail classification (default: 0.1)
        """
        self.mode = mode
        self.threshold = threshold

        # Load statistics
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"WPD statistics file not found: {stats_path}")

        with open(stats_path, 'r') as f:
            stats = yaml.safe_load(f)

        self.n_classes = stats['n_classes']
        self.class_names = stats['class_names']

        # Load mode-specific statistics
        mode_stats = stats[mode]
        self.w_norm = np.array(mode_stats['w_norm'])
        self.long_tail_classes = set(mode_stats['long_tail_classes'])
        self.paste_prob = np.array(mode_stats['paste_prob'])
        self.drop_prob = np.array(mode_stats['drop_prob'])

        # For semantic mode, load alpha for focal loss
        if mode == 'semantic':
            self.alpha = np.array(stats['semantic']['alpha'])
        else:
            self.alpha = np.array(stats['semantic']['alpha'])
            self.beta = np.array(stats['panoptic']['beta'])

        print(f"WPD Config initialized in {mode} mode")
        print(f"Long-tail classes: {sorted(self.long_tail_classes)}")

    def is_long_tail(self, class_id):
        """Check if a class is long-tail."""
        return class_id in self.long_tail_classes

    def get_paste_prob(self, class_id):
        """Get paste probability for a class."""
        return self.paste_prob[class_id] if class_id < len(self.paste_prob) else 0.0

    def get_drop_prob(self, class_id):
        """Get drop probability for a class."""
        return self.drop_prob[class_id] if class_id < len(self.drop_prob) else 0.0


class WPDAugmentor:
    """
    Weighted Paste and Drop augmentation for range-view LiDAR segmentation.

    Operates on projected range images with channels [range, x, y, z, intensity].
    """

    def __init__(self, wpd_config, apply_rate=0.6, ignore_label=0):
        """
        Args:
            wpd_config: WPDConfig instance with class weights and probabilities
            apply_rate: Probability of applying WPD to a batch (default: 0.6)
            ignore_label: Label value for dropped pixels (default: 0)
        """
        self.config = wpd_config
        self.apply_rate = apply_rate
        self.ignore_label = ignore_label

    def __call__(self, frame_a_dict, frame_b_dict, apply_wpd=None):
        """
        Apply WPD augmentation to two frames.

        Args:
            frame_a_dict: Dictionary containing base frame data:
                - 'features': [C, H, W] torch.Tensor (range, x, y, z, intensity)
                - 'labels': [H, W] torch.Tensor (semantic labels)
                - 'instances': [H, W] torch.Tensor (instance IDs, optional)
                - 'mask': [H, W] torch.Tensor (validity mask)
                - 'range': [H, W] torch.Tensor (range values for z-buffer)
            frame_b_dict: Dictionary containing donor frame data (same structure as frame_a)
            apply_wpd: If None, randomly decide based on apply_rate. If bool, use that value.

        Returns:
            Dictionary with augmented frame_a data (same keys as input)
        """
        # Decide whether to apply WPD
        if apply_wpd is None:
            apply_wpd = np.random.rand() < self.apply_rate

        if not apply_wpd:
            return frame_a_dict

        # Extract data from dictionaries
        features_a = frame_a_dict['features'].clone()  # [C, H, W]
        labels_a = frame_a_dict['labels'].clone()      # [H, W]
        mask_a = frame_a_dict['mask'].clone()          # [H, W]
        range_a = frame_a_dict['range'].clone()        # [H, W]

        features_b = frame_b_dict['features']
        labels_b = frame_b_dict['labels']
        mask_b = frame_b_dict['mask']
        range_b = frame_b_dict['range']

        instances_a = frame_a_dict.get('instances', None)
        instances_b = frame_b_dict.get('instances', None)
        if instances_a is not None:
            instances_a = instances_a.clone()

        H, W = labels_a.shape

        # ===== PASTE: Add long-tail instances from frame_b =====
        if instances_b is not None:
            features_a, labels_a, mask_a, instances_a = self._paste_instances(
                features_a, labels_a, mask_a, range_a, instances_a,
                features_b, labels_b, mask_b, range_b, instances_b
            )
        else:
            # If no instance labels, paste by class regions
            features_a, labels_a, mask_a = self._paste_by_class(
                features_a, labels_a, mask_a, range_a,
                features_b, labels_b, mask_b, range_b
            )

        # ===== DROP: Remove non-long-tail points from frame_a =====
        features_a, labels_a, mask_a, instances_a = self._drop_frequent_classes(
            features_a, labels_a, mask_a, instances_a
        )

        # Return augmented frame
        output = {
            'features': features_a,
            'labels': labels_a,
            'mask': mask_a,
            'range': range_a,  # Keep original range for consistency
        }
        if instances_a is not None:
            output['instances'] = instances_a

        return output

    def _paste_instances(self, features_a, labels_a, mask_a, range_a, instances_a,
                        features_b, labels_b, mask_b, range_b, instances_b):
        """
        Paste long-tail instances from frame_b into frame_a.

        Uses z-buffer style collision resolution: keep nearer points or prefer long-tail.
        """
        # Get unique instances in frame_b
        unique_instances_b = torch.unique(instances_b)
        unique_instances_b = unique_instances_b[unique_instances_b != 0]  # Filter out 0 (background)

        # Track next available instance ID in frame_a
        max_inst_a = instances_a.max().item() if instances_a is not None else 0
        next_inst_id = max_inst_a + 1

        for inst_id in unique_instances_b:
            # Get mask for this instance
            inst_mask_b = (instances_b == inst_id)

            # Get majority class for this instance
            inst_labels = labels_b[inst_mask_b]
            if len(inst_labels) == 0:
                continue

            # Compute majority class (most frequent label in the instance)
            class_id = torch.mode(inst_labels).values.item()

            # Check if this class is long-tail
            if not self.config.is_long_tail(class_id):
                continue

            # Sample whether to paste this instance
            paste_prob = self.config.get_paste_prob(class_id)
            if np.random.rand() > paste_prob:
                continue

            # Collision resolution: prefer nearer points or long-tail
            # Create paste mask: where donor is valid and (base is invalid OR donor is nearer)
            base_invalid = ~mask_a.bool()
            donor_nearer = range_b < range_a
            paste_mask = inst_mask_b & (base_invalid | donor_nearer)

            # Paste features, labels, and instance IDs
            features_a[:, paste_mask] = features_b[:, paste_mask]
            labels_a[paste_mask] = labels_b[paste_mask]
            mask_a[paste_mask] = mask_b[paste_mask]

            if instances_a is not None:
                instances_a[paste_mask] = next_inst_id
                next_inst_id += 1

            # Update range buffer (for subsequent pastes)
            range_a[paste_mask] = range_b[paste_mask]

        return features_a, labels_a, mask_a, instances_a

    def _paste_by_class(self, features_a, labels_a, mask_a, range_a,
                       features_b, labels_b, mask_b, range_b):
        """
        Paste long-tail class regions from frame_b (when instance labels unavailable).

        This is a simplified version that operates on semantic classes rather than instances.
        """
        for class_id in self.config.long_tail_classes:
            # Get mask for this class in donor frame
            class_mask_b = (labels_b == class_id) & mask_b.bool()

            if class_mask_b.sum() == 0:
                continue

            # Sample whether to paste this class
            paste_prob = self.config.get_paste_prob(class_id)
            if np.random.rand() > paste_prob:
                continue

            # Collision resolution
            base_invalid = ~mask_a.bool()
            donor_nearer = range_b < range_a
            paste_mask = class_mask_b & (base_invalid | donor_nearer)

            # Paste features and labels
            features_a[:, paste_mask] = features_b[:, paste_mask]
            labels_a[paste_mask] = labels_b[paste_mask]
            mask_a[paste_mask] = mask_b[paste_mask]
            range_a[paste_mask] = range_b[paste_mask]

        return features_a, labels_a, mask_a

    def _drop_frequent_classes(self, features, labels, mask, instances):
        """
        Drop non-long-tail (frequent) class pixels from the frame.

        Creates "holes" to weaken background priors and context reliance.
        """
        H, W = labels.shape

        # Iterate over all classes
        for class_id in range(self.config.n_classes):
            # Skip long-tail classes (we want to keep those)
            if self.config.is_long_tail(class_id):
                continue

            # Get drop probability for this class
            drop_prob = self.config.get_drop_prob(class_id)
            if drop_prob <= 0:
                continue

            # Get mask for this class
            class_mask = (labels == class_id) & mask.bool()
            if class_mask.sum() == 0:
                continue

            # Generate random drop mask using Bernoulli sampling
            # Each pixel has probability drop_prob of being dropped
            drop_mask = class_mask & (torch.rand(H, W, device=labels.device) < drop_prob)

            # Drop pixels by setting them to ignore label
            labels[drop_mask] = self.ignore_label
            mask[drop_mask] = 0  # Mark as invalid

            # Optionally zero out features (or set to a sentinel value)
            features[:, drop_mask] = 0

            if instances is not None:
                instances[drop_mask] = 0

        return features, labels, mask, instances


def wpd_augment(frame_a_proj, frame_a_labels, frame_a_mask, frame_a_range,
                frame_b_proj, frame_b_labels, frame_b_mask, frame_b_range,
                wpd_augmentor, frame_a_instances=None, frame_b_instances=None):
    """
    Convenience function for WPD augmentation.

    Args:
        frame_a_proj: [C, H, W] projected features (base frame)
        frame_a_labels: [H, W] semantic labels (base frame)
        frame_a_mask: [H, W] validity mask (base frame)
        frame_a_range: [H, W] range values (base frame)
        frame_b_proj: [C, H, W] projected features (donor frame)
        frame_b_labels: [H, W] semantic labels (donor frame)
        frame_b_mask: [H, W] validity mask (donor frame)
        frame_b_range: [H, W] range values (donor frame)
        wpd_augmentor: WPDAugmentor instance
        frame_a_instances: [H, W] instance labels (optional)
        frame_b_instances: [H, W] instance labels (optional)

    Returns:
        Tuple of (features, labels, mask, instances) for augmented frame_a
    """
    frame_a_dict = {
        'features': frame_a_proj,
        'labels': frame_a_labels,
        'mask': frame_a_mask,
        'range': frame_a_range,
        'instances': frame_a_instances,
    }

    frame_b_dict = {
        'features': frame_b_proj,
        'labels': frame_b_labels,
        'mask': frame_b_mask,
        'range': frame_b_range,
        'instances': frame_b_instances,
    }

    result = wpd_augmentor(frame_a_dict, frame_b_dict)

    return (result['features'], result['labels'], result['mask'],
            result.get('instances', None))
