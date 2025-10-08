# RangeFormer Dataset Loader
# Extends RangeViewLoader with RangeAug support

import numpy as np
import torch
import random
from typing import Optional, Tuple
from torch.utils.data import Dataset

from .range_view_loader import RangeViewLoader
from .preprocess.range_aug import range_mix, range_union, range_paste, range_shift


class RangeFormerLoader(RangeViewLoader):
    """
    Dataset loader for RangeFormer with RangeAug support.

    Extends RangeViewLoader to add RangeAug & STR support per RangeFormer paper.
    """

    def __init__(self, dataset, config, data_len=-1, is_train=True, return_uproj=False, use_kpconv=False):
        super().__init__(dataset, config, data_len, is_train, return_uproj, use_kpconv)

        # RangeAug configuration
        self.use_range_aug = config.get('augmentation', {}).get('use_range_aug', False) and is_train
        if self.use_range_aug:
            aug_config = config['augmentation']
            self.range_aug_prob = aug_config.get('range_aug_prob', 1.0)
            self.p_range_shift = aug_config.get('p_range_shift', 0.9)
            self.p_range_mix = aug_config.get('p_range_mix', 0.2)
            self.p_range_union = aug_config.get('p_range_union', 0.9)
            self.p_range_paste = aug_config.get('p_range_paste', 1.0)
            tail_ratio = aug_config.get('range_paste_tail_ratio', None)
            if tail_ratio is not None and hasattr(self.dataset, 'cls_freq'):
                self.range_paste_classes = self._compute_tail_classes(tail_ratio)
            else:
                self.range_paste_classes = aug_config.get('range_paste_classes', [])
            self.range_paste_classes = [int(c) for c in self.range_paste_classes]

            print(f'RangeAug enabled: prob={self.range_aug_prob}')
            print(f'  RangeShift p={self.p_range_shift}')
            print(f'  RangeMix p={self.p_range_mix}')
            print(f'  RangeUnion p={self.p_range_union}')
            print(f'  RangePaste p={self.p_range_paste}, tail={self.range_paste_classes}')

        # STR configuration
        str_config = config.get('str', {})
        self.use_str = bool(str_config.get('enabled', False))
        self.str_num_views = int(str_config.get('num_views', 1))
        if self.use_str and self.str_num_views < 1:
            raise ValueError('STR enabled but num_views < 1')

        self.str_inference_views = int(str_config.get('inference_views', self.str_num_views))
        self.str_align_inference = bool(str_config.get('align_inference', True))
        # Training view width (expected to match config image_size[1])
        self.str_view_width = self.config['image_size'][1] if self.use_str else None
        # Full resolution width (used for inference/validation stitching)
        self.str_full_width = self.config['original_image_size'][1]

        if self.use_str:
            if self.str_full_width % self.str_num_views != 0:
                raise ValueError(
                    f'STR view partition mismatch: full width {self.str_full_width} '
                    f'is not divisible by num_views {self.str_num_views}')
            expected_width = self.str_full_width // self.str_num_views
            if self.str_view_width != expected_width:
                raise ValueError(
                    f'image_size width ({self.str_view_width}) must equal '
                    f'original_image_size width / num_views ({expected_width})')

    def _get_view_slice(self, view_idx: int, total_width: int) -> Tuple[int, int]:
        """
        Compute column slice [start, end) for a given view index.
        """
        if not self.use_str:
            return 0, total_width

        view_width = self.str_view_width or (total_width // self.str_num_views)
        start = view_idx * view_width
        end = start + view_width
        if end > total_width:
            end = total_width
        return start, end

    def _slice_view(self, rv: np.ndarray, label: np.ndarray, view_slice: Tuple[int, int]):
        start, end = view_slice
        rv = rv[:, :, start:end]
        label = label[:, start:end]
        return rv, label

    def _compute_tail_classes(self, tail_ratio: float):
        """
        Determine tail classes based on dataset class frequency histogram.
        """
        cls_freq = getattr(self.dataset, 'cls_freq', None)
        if cls_freq is None:
            return []
        ignore_idx = 0
        valid_indices = [i for i in range(len(cls_freq)) if i != ignore_idx and cls_freq[i] > 0]
        if not valid_indices:
            return []
        sorted_valid = sorted(valid_indices, key=lambda idx: cls_freq[idx])
        tail_count = max(1, int(np.ceil(len(sorted_valid) * tail_ratio)))
        tail_classes = sorted_valid[:tail_count]
        return tail_classes

    def apply_range_aug(self, rv_np, label_np, view_slice: Optional[Tuple[int, int]] = None):
        """
        Apply RangeAug augmentations to range image and labels.

        Args:
            rv_np: (6, H, W) range image numpy array
            label_np: (H, W) label map numpy array
            view_slice: optional (start, end) width slice for STR view

        Returns:
            rv_aug: augmented range image
            label_aug: augmented labels
        """
        if view_slice is not None:
            rv_np, label_np = self._slice_view(rv_np, label_np, view_slice)

        # Decide whether to apply RangeAug
        if random.random() > self.range_aug_prob:
            return rv_np, label_np

        # Get another random sample for mixing operations
        rand_idx = random.randint(0, len(self.dataset) - 1)
        pointcloud_b, sem_label_b, _ = self.dataset.loadDataByIndex(rand_idx)
        sem_label_b = self.dataset.labelMapping(sem_label_b)

        # Apply 3D augmentation if training
        if self.is_train and (self.scan_proj is False) and (self.augmentor is not None):
            pointcloud_b = self.augmentor.doAugmentation(pointcloud_b)

        # Project second point cloud
        proj_pointcloud_b, proj_range_b, proj_idx_b, proj_mask_b = self.projection.doProjection(pointcloud_b)

        # Determine slice window for second sample (match selected view if provided)
        if view_slice is not None:
            start, end = view_slice
        else:
            start, end = 0, proj_pointcloud_b.shape[1]

        H = rv_np.shape[1]
        slice_width = end - start

        # Create second range view cropped to the same slice
        rv_b = np.zeros((6, H, slice_width), dtype=np.float32)
        rv_b[0] = proj_pointcloud_b[:, start:end, 0]  # x
        rv_b[1] = proj_pointcloud_b[:, start:end, 1]  # y
        rv_b[2] = proj_pointcloud_b[:, start:end, 2]  # z
        rv_b[3] = proj_range_b[:, start:end]          # depth
        rv_b[4] = proj_pointcloud_b[:, start:end, 3]  # intensity
        rv_b[5] = proj_mask_b[:, start:end]           # existence

        # Create label map for second sample (same slice)
        label_b = np.zeros((H, slice_width), dtype=np.int32)
        mask_b = proj_idx_b[:, start:end] > 0
        label_b[mask_b] = sem_label_b[proj_idx_b[:, start:end][mask_b]]

        # Slice current sample if needed (ensures same width)
        if view_slice is not None:
            rv_np, label_np = self._slice_view(rv_np, label_np, (0, slice_width))

        rv_aug = rv_np.copy()
        label_aug = label_np.copy()

        # Apply augmentations with probabilities
        # RangeShift (independent)
        if random.random() < self.p_range_shift:
            rv_aug, label_aug = range_shift(rv_aug, label_aug)

        # RangeMix
        if random.random() < self.p_range_mix:
            kmix = random.choice([2, 3, 4, 5, 6])
            rv_aug, label_aug = range_mix(rv_aug, label_aug, rv_b, label_b, kmix)

        # RangeUnion (fill voids)
        if random.random() < self.p_range_union:
            kunion = random.uniform(0.3, 0.7)
            rv_aug, label_aug = range_union(rv_aug, label_aug, rv_b, label_b, kunion)

        # RangePaste (paste rare classes)
        if random.random() < self.p_range_paste and len(self.range_paste_classes) > 0:
            rv_aug, label_aug = range_paste(rv_aug, label_aug, rv_b, label_b, self.range_paste_classes)

        return rv_aug, label_aug

    def __getitem__(self, index):
        """
        Get item with RangeAug support.

        Returns:
            For RangeFormer (use_kpconv=False):
                proj_feature_tensor: (6, H, W) range image
                proj_sem_label_tensor: (H, W) semantic labels
                proj_mask_tensor: (H, W) valid pixel mask

            For RangeFormer+KPConv (use_kpconv=True):
                Same as RangeViewLoader with additional point-level data
        """
        if self.use_kpconv:
            # Use parent's KPConv loader (would need modification for RangeAug)
            return super().get_item_for_kpconv(index)
        else:
            return self.get_item_rangeformer(index)

    def get_item_rangeformer(self, index):
        """Get item specifically for RangeFormer training."""
        # Load point cloud and labels
        pointcloud, sem_label, inst_label = self.dataset.loadDataByIndex(index)
        sem_label = self.dataset.labelMapping(sem_label)

        # Apply 3D augmentation
        if self.is_train and (self.scan_proj is False) and (self.augmentor is not None):
            pointcloud = self.augmentor.doAugmentation(pointcloud)

        # Range projection
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)

        # Create 6-channel range image
        H, W = proj_range.shape
        rv_np = np.zeros((6, H, W), dtype=np.float32)
        rv_np[0] = proj_pointcloud[:, :, 0]  # x
        rv_np[1] = proj_pointcloud[:, :, 1]  # y
        rv_np[2] = proj_pointcloud[:, :, 2]  # z
        rv_np[3] = proj_range                 # depth
        rv_np[4] = proj_pointcloud[:, :, 3]   # intensity
        rv_np[5] = proj_mask                  # existence

        # Create label map
        proj_sem_label = np.zeros((H, W), dtype=np.int32)
        mask = proj_idx > 0
        proj_sem_label[mask] = sem_label[proj_idx[mask]]

        # STR view selection (training only)
        selected_view_slice = None
        if self.use_str and self.is_train:
            view_idx = random.randint(0, self.str_num_views - 1)
            selected_view_slice = self._get_view_slice(view_idx, rv_np.shape[2])

        # Apply RangeAug (2D augmentations)
        if self.use_range_aug and self.is_train:
            rv_np, proj_sem_label = self.apply_range_aug(rv_np, proj_sem_label, selected_view_slice)
        elif selected_view_slice is not None:
            rv_np, proj_sem_label = self._slice_view(rv_np, proj_sem_label, selected_view_slice)

        # Convert to tensors
        proj_mask_tensor = torch.from_numpy(rv_np[5].astype(np.float32))  # existence channel
        proj_sem_label_tensor = torch.from_numpy(proj_sem_label.astype(np.int64))

        # Create feature tensor (6 channels)
        proj_feature_tensor = torch.from_numpy(rv_np.astype(np.float32))

        # Normalize features
        # Note: For RangeFormer, we may want different normalization per channel
        # Here we apply the normalization from config
        if hasattr(self, 'proj_img_mean') and hasattr(self, 'proj_img_stds'):
            # Extend mean/std if needed (config might have only 5 channels)
            img_mean = self.proj_img_mean
            img_stds = self.proj_img_stds

            # Ensure we have 6 values
            if len(img_mean) == 5:
                img_mean = torch.cat([img_mean, torch.tensor([0.5])])
                img_stds = torch.cat([img_stds, torch.tensor([0.5])])

            proj_feature_tensor = (proj_feature_tensor - img_mean[:, None, None]) / img_stds[:, None, None]
            proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0)

        # Apply 2D cropping augmentation (horizontal flip handled in parent class logic)
        proj_tensor = torch.cat([
            proj_feature_tensor,
            proj_sem_label_tensor.unsqueeze(0).float(),
            proj_mask_tensor.unsqueeze(0)
        ], dim=0)  # (8, H, W): 6 features + label + mask

        # Apply standard 2D augmentations (crop, flip)
        if self.aug_ops is not None:
            proj_tensor = self.aug_ops(proj_tensor)

        # Optional horizontal flip
        if self.is_train and random.random() < self.proj_p_hflip:
            proj_tensor = torch.flip(proj_tensor, dims=[2])

        # Split back
        proj_feature_tensor = proj_tensor[:6]
        proj_sem_label_tensor = proj_tensor[6].long()
        proj_mask_tensor = proj_tensor[7]

        # Return data
        if self.return_uproj:
            # For inference, we might need projection indices
            px, py = self.projection.cached_data['px'], self.projection.cached_data['py']
            return proj_feature_tensor, proj_sem_label_tensor, proj_mask_tensor, proj_idx, px, py
        else:
            return proj_feature_tensor, proj_sem_label_tensor, proj_mask_tensor

    def __len__(self):
        if self.data_len > 0:
            return self.data_len
        else:
            return len(self.dataset)
