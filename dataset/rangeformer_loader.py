# RangeFormer Dataset Loader
# Extends RangeViewLoader with RangeAug support

import numpy as np
import torch
import random
from torch.utils.data import Dataset

from .range_view_loader import RangeViewLoader
from .preprocess.range_aug import range_mix, range_union, range_paste, range_shift


class RangeFormerLoader(RangeViewLoader):
    """
    Dataset loader for RangeFormer with RangeAug support.

    Extends RangeViewLoader to add 2D range image augmentations:
    - RangeMix: Grid-based mixing of two range images
    - RangeUnion: Fill voids with another image
    - RangePaste: Paste rare semantic classes
    - RangeShift: Azimuthal shift
    """

    def __init__(self, dataset, config, data_len=-1, is_train=True, return_uproj=False, use_kpconv=False):
        super().__init__(dataset, config, data_len, is_train, return_uproj, use_kpconv)

        # RangeAug configuration
        self.use_range_aug = config.get('augmentation', {}).get('use_range_aug', False) and is_train
        if self.use_range_aug:
            aug_config = config['augmentation']
            self.range_aug_prob = aug_config.get('range_aug_prob', 0.5)
            self.p_range_shift = aug_config.get('p_range_shift', 0.5)
            self.p_range_mix = aug_config.get('p_range_mix', 0.3)
            self.p_range_union = aug_config.get('p_range_union', 0.2)
            self.p_range_paste = aug_config.get('p_range_paste', 0.1)
            self.range_paste_classes = aug_config.get('range_paste_classes', [1, 2, 3, 4, 5, 6, 7, 8])

            print(f'RangeAug enabled with probability {self.range_aug_prob}')
            print(f'  RangeShift: {self.p_range_shift}')
            print(f'  RangeMix: {self.p_range_mix}')
            print(f'  RangeUnion: {self.p_range_union}')
            print(f'  RangePaste: {self.p_range_paste}')

    def apply_range_aug(self, rv_np, label_np):
        """
        Apply RangeAug augmentations to range image and labels.

        Args:
            rv_np: (6, H, W) range image numpy array
            label_np: (H, W) label map numpy array

        Returns:
            rv_aug: augmented range image
            label_aug: augmented labels
        """
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

        # Create second range view
        H, W = rv_np.shape[1], rv_np.shape[2]
        rv_b = np.zeros((6, H, W), dtype=np.float32)
        rv_b[0] = proj_pointcloud_b[:, :, 0]  # x
        rv_b[1] = proj_pointcloud_b[:, :, 1]  # y
        rv_b[2] = proj_pointcloud_b[:, :, 2]  # z
        rv_b[3] = proj_range_b                 # depth
        rv_b[4] = proj_pointcloud_b[:, :, 3]   # intensity
        rv_b[5] = proj_mask_b                  # existence

        # Create label map for second sample
        label_b = np.zeros((H, W), dtype=np.int32)
        mask_b = proj_idx_b > 0
        label_b[mask_b] = sem_label_b[proj_idx_b[mask_b]]

        rv_aug = rv_np.copy()
        label_aug = label_np.copy()

        # Apply augmentations with probabilities
        # RangeShift (independent)
        if random.random() < self.p_range_shift:
            rv_aug, label_aug = range_shift(rv_aug, label_aug)

        # RangeMix
        if random.random() < self.p_range_mix:
            # Random grid strategy
            phi = random.choice([4, 8, 16])
            theta = random.choice([8, 16, 32])
            rv_aug, label_aug = range_mix(rv_aug, label_aug, rv_b, label_b, (phi, theta))

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

        # Apply RangeAug (2D augmentations)
        if self.use_range_aug and self.is_train:
            rv_np, proj_sem_label = self.apply_range_aug(rv_np, proj_sem_label)

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
