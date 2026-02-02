# Copyright 2022 - Valeo Comfort and Driving Assistance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from scipy.spatial.ckdtree import cKDTree as kdtree

from .preprocess import augmentor, projection, ClusterMix, InstanceCopy, PolarMix, InstanceCutMix


class RangeViewLoader(Dataset):
    def __init__(self, dataset, config, data_len=-1, is_train=True, return_uproj=False, use_kpconv=False, return_point_data=False):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.data_len = data_len
        self.return_uproj = return_uproj
        self.use_kpconv = use_kpconv
        self.return_point_data = return_point_data

        augment_params = augmentor.AugmentParams()
        augment_config = self.config['augmentation']
        adapted_config = self.config.get('adapted_augmentation', {})
        self.adapted_use_mapped_labels = bool(adapted_config.get('use_mapped_labels', False))

        # Point cloud augmentations
        if self.is_train:
            augment_params.setFlipProb(
                p_flipx=augment_config['p_flipx'], p_flipy=augment_config['p_flipy'])
            augment_params.setTranslationParams(
                p_transx=augment_config['p_transx'], trans_xmin=augment_config[
                    'trans_xmin'], trans_xmax=augment_config['trans_xmax'],
                p_transy=augment_config['p_transy'], trans_ymin=augment_config[
                    'trans_ymin'], trans_ymax=augment_config['trans_ymax'],
                p_transz=augment_config['p_transz'], trans_zmin=augment_config[
                    'trans_zmin'], trans_zmax=augment_config['trans_zmax'])
            augment_params.setRotationParams(
                p_rot_roll=augment_config['p_rot_roll'], rot_rollmin=augment_config[
                    'rot_rollmin'], rot_rollmax=augment_config['rot_rollmax'],
                p_rot_pitch=augment_config['p_rot_pitch'], rot_pitchmin=augment_config[
                    'rot_pitchmin'], rot_pitchmax=augment_config['rot_pitchmax'],
                p_rot_yaw=augment_config['p_rot_yaw'], rot_yawmin=augment_config[
                    'rot_yawmin'], rot_yawmax=augment_config['rot_yawmax'])
            if 'p_scale' in augment_config:
                augment_params.sefScaleParams(
                    p_scale=augment_config['p_scale'],
                    scale_min=augment_config['scale_min'],
                    scale_max=augment_config['scale_max'])
                print(f'Adding scaling augmentation with range [{augment_params.scale_min}, {augment_params.scale_max}] and probability {augment_params.p_scale}')
            self.augmentor = augmentor.Augmentor(augment_params)
        else:
            self.augmentor = None

        pointsample_cfg = adapted_config.get('pointsample', {})
        self.point_sampler = None
        if self.is_train and pointsample_cfg.get('enable', False):
            self.point_sampler = augmentor.PointSampler(
                num_points=pointsample_cfg.get('num_points', 0),
                replace=pointsample_cfg.get('replace', False),
                inplace=True,
            )

        polarmix_cfg = adapted_config.get('polarmix', {})
        self.polarmix = None
        if self.is_train and polarmix_cfg.get('enable', False):
            self.polarmix = PolarMix(
                classes=polarmix_cfg.get('classes', []),
                prob=polarmix_cfg.get('prob', 0.5),
            )

        cutmix_cfg = adapted_config.get('instance_cutmix', {})
        self.instance_cutmix = None
        if self.is_train and cutmix_cfg.get('enable', False):
            bank_root = cutmix_cfg.get('instance_bank_root', 'cache/instance_bank')
            self.instance_cutmix = InstanceCutMix(
                rootdir=bank_root,
                classes=cutmix_cfg.get('classes', []),
                num_to_add=tuple(cutmix_cfg.get('num_to_add', [0, 2])),
                min_points=cutmix_cfg.get('min_points', 20),
                prob=cutmix_cfg.get('prob', 1.0),
            )

        clustermix_cfg = adapted_config.get('clustermix', {})
        self.clustermix = None
        if self.is_train and clustermix_cfg.get('enable', False):
            self.clustermix = ClusterMix(
                prob=clustermix_cfg.get('prob', 0.5),
                vertical_prob=clustermix_cfg.get('vertical_prob', 0.5),
                sector_width=clustermix_cfg.get('sector_width', np.pi / 2),
                height_ratio=clustermix_cfg.get('height_ratio', 0.5),
            )

        instcopy_cfg = adapted_config.get('instance_copy', {})
        self.instance_copy = None
        if self.is_train and instcopy_cfg.get('enable', False):
            self.instance_copy = InstanceCopy(
                prob=instcopy_cfg.get('prob', 0.5),
                classes=instcopy_cfg.get('classes', []),
                max_instances_per_class=instcopy_cfg.get('max_instances_per_class', 1),
            )

        self.proj_p_hflip = augment_config.get('p_hflip', 0.0)
        if self.proj_p_hflip > 0.0:
            print(f'Horizontal flip of range projections with p={self.proj_p_hflip}')

        projection_config = self.config['sensor']
        self.scan_proj = projection_config.get('scan_proj', False)
        if self.scan_proj:
            print('Use scan-based range projection.')
            self.projection = projection.ScanProjection(
                proj_h=projection_config['proj_h'], proj_w=projection_config['proj_w'],
            )
        else:
            self.projection = projection.RangeProjection(
                fov_up=projection_config['fov_up'], fov_down=projection_config['fov_down'],
                fov_left=projection_config['fov_left'], fov_right=projection_config['fov_right'],
                proj_h=projection_config['proj_h'], proj_w=projection_config['proj_w'],
            )
        self.proj_img_mean = torch.tensor(self.config['sensor']['img_mean'], dtype=torch.float)
        self.proj_img_stds = torch.tensor(self.config['sensor']['img_stds'], dtype=torch.float)

        # Image augmentations
        if self.is_train:
            self.crop_size = self.config['image_size']
            self.aug_ops = T.Compose([
                T.RandomCrop(
                    size=(self.config['image_size'][0],
                          self.config['image_size'][1])),
            ])
        else:
            self.crop_size = self.config['original_image_size']
            self.aug_ops = T.Compose([
                T.CenterCrop((self.config['original_image_size'][0],
                              self.config['original_image_size'][1]))
            ])

    def _maybe_sample_points(self, pointcloud, sem_label=None, inst_label=None):
        if self.point_sampler is None:
            return pointcloud, sem_label, inst_label
        return self.point_sampler(pointcloud, sem_label, inst_label)

    def _load_mix_sample(self):
        mix_index = np.random.randint(0, len(self.dataset))
        return self.dataset.loadDataByIndex(mix_index)

    def get_item_for_kpconv(self, index):
        '''
        proj_feature_tensor: CxHxW
        proj_sem_label_tensor: HxW
        proj_mask_tensor: HxW
        '''
        pointcloud, sem_label, inst_label = self.dataset.loadDataByIndex(index)
        points_xyz = pointcloud[:, :3]
        labels_are_mapped = False
        if self.is_train and self.adapted_use_mapped_labels:
            sem_label = self.dataset.labelMapping(sem_label)
            labels_are_mapped = True

        if self.is_train:
            if self.instance_cutmix is not None:
                pointcloud, sem_label = self.instance_cutmix(pointcloud, sem_label, inst_label)
                inst_label = None  # inst_label is invalid after cutmix changes point count
            if self.polarmix is not None:
                mix_pc, mix_sem, _ = self._load_mix_sample()
                if labels_are_mapped:
                    mix_sem = self.dataset.labelMapping(mix_sem)
                pointcloud, sem_label = self.polarmix(pointcloud, sem_label, mix_pc, mix_sem)
                inst_label = None  # inst_label is invalid after polarmix changes point count
            pointcloud, sem_label, inst_label = self._maybe_sample_points(pointcloud, sem_label, inst_label)
            if self.scan_proj is False and self.augmentor is not None:
                pointcloud = self.augmentor.doAugmentation(pointcloud)  # n, 4

            if (self.clustermix is not None) or (self.instance_copy is not None):
                mix_pc, mix_sem, mix_inst = self._load_mix_sample()
                if labels_are_mapped:
                    mix_sem = self.dataset.labelMapping(mix_sem)
                mix_pc, mix_sem, mix_inst = self._maybe_sample_points(mix_pc, mix_sem, mix_inst)
                if self.scan_proj is False and self.augmentor is not None:
                    mix_pc = self.augmentor.doAugmentation(mix_pc)
                if self.clustermix is not None:
                    pointcloud, sem_label = self.clustermix(pointcloud, sem_label, mix_pc, mix_sem)
                    inst_label = None  # inst_label is invalid after clustermix changes point count
                if self.instance_copy is not None:
                    pointcloud, sem_label = self.instance_copy(
                        pointcloud, sem_label, mix_pc, mix_sem, mix_inst)
                    inst_label = None  # inst_label is invalid after instance_copy changes point count
        # if self.use_rangeinterpolation:
        #     pointcloud, sem_label = self.range_interpolation(pointcloud, sem_label)
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)
        px, py = self.projection.cached_data['px'], self.projection.cached_data['py']

        proj_mask_tensor = torch.from_numpy(proj_mask)
        mask = proj_idx > 0
        proj_sem_label = np.zeros((proj_mask.shape[0], proj_mask.shape[1]), dtype=np.float32)
        if labels_are_mapped:
            proj_sem_label[mask] = sem_label[proj_idx[mask]]
        else:
            proj_sem_label[mask] = self.dataset.labelMapping(sem_label[proj_idx[mask]])
        proj_sem_label_tensor = torch.from_numpy(proj_sem_label)
        proj_sem_label_tensor = proj_sem_label_tensor * proj_mask_tensor.float()

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0)], 0)

        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        proj_tensor = torch.cat(
            (proj_feature_tensor,
            proj_sem_label_tensor.unsqueeze(0),
            proj_mask_tensor.float().unsqueeze(0)), dim=0)

        if self.is_train:
            proj_tensor, px, py, points_xyz, sem_label = crop_inputs(
                proj_tensor, px, py, points_xyz, sem_label,
                self.crop_size, center_crop=False, p_hflip=self.proj_p_hflip)
        else:
            _, h, w = proj_tensor.shape

            # Normalize them to be between -1 and 1.
            px = 2.0 * ((px / w) - 0.5)
            py = 2.0 * ((py / h) - 0.5)

        tree = kdtree(points_xyz)
        _, knns = tree.query(points_xyz, k=7)

        output = {
            'input2d': proj_tensor[:5],
            'label2d': proj_tensor[5],
            'mask2d': proj_tensor[6],
            'px': torch.from_numpy(px).float(),
            'py': torch.from_numpy(py).float(),
            'points_xyz': torch.from_numpy(points_xyz).float(),
            'knns': torch.from_numpy(knns).long(),
            'labels': torch.from_numpy(sem_label).long(),
            'num_points': points_xyz.shape[0],
            'index': index,
        }

        if self.return_uproj:
            assert self.is_train is False

            output['range'] = torch.from_numpy(proj_range)
            output['uproj_x'] = torch.from_numpy(self.projection.cached_data['uproj_x_idx']).long()
            output['uproj_y'] = torch.from_numpy(self.projection.cached_data['uproj_y_idx']).long()
            output['uproj_depth'] = torch.from_numpy(self.projection.cached_data['uproj_depth']).float()

        return output

    def get_item_for_fusion(self, index):
        '''
        Get data for fusion model, returning point attributes and pixel coordinates.

        Returns dict with:
            proj_image: (C, H, W) range image features
            proj_labels: (H, W) projected semantic labels
            proj_mask: (H, W) valid pixel mask
            point_attrs: (N, 5) raw point attributes [x, y, z, intensity, range]
            point_coords: (N, 2) pixel coordinates [y, x] for each point
            point_labels: (N,) per-point semantic labels
        '''
        pointcloud, sem_label, inst_label = self.dataset.loadDataByIndex(index)

        # Apply full augmentation pipeline from kpconv
        labels_are_mapped = False
        if self.is_train and self.adapted_use_mapped_labels:
            sem_label = self.dataset.labelMapping(sem_label)
            labels_are_mapped = True

        if self.is_train:
            # Mix-based augmentations before basic augmentation
            if self.instance_cutmix is not None:
                pointcloud, sem_label = self.instance_cutmix(pointcloud, sem_label, inst_label)
                inst_label = None  # inst_label is invalid after cutmix changes point count
            if self.polarmix is not None:
                mix_pc, mix_sem, _ = self._load_mix_sample()
                if labels_are_mapped:
                    mix_sem = self.dataset.labelMapping(mix_sem)
                pointcloud, sem_label = self.polarmix(pointcloud, sem_label, mix_pc, mix_sem)
                inst_label = None  # inst_label is invalid after polarmix changes point count

            # Point sampling
            pointcloud, sem_label, inst_label = self._maybe_sample_points(pointcloud, sem_label, inst_label)

            # Basic augmentation (rotation, translation, scale)
            if self.scan_proj is False and self.augmentor is not None:
                pointcloud = self.augmentor.doAugmentation(pointcloud)  # n, 4

            # ClusterMix and InstanceCopy require a second sample
            if (self.clustermix is not None) or (self.instance_copy is not None):
                mix_pc, mix_sem, mix_inst = self._load_mix_sample()
                if labels_are_mapped:
                    mix_sem = self.dataset.labelMapping(mix_sem)
                mix_pc, mix_sem, mix_inst = self._maybe_sample_points(mix_pc, mix_sem, mix_inst)
                if self.scan_proj is False and self.augmentor is not None:
                    mix_pc = self.augmentor.doAugmentation(mix_pc)
                if self.clustermix is not None:
                    pointcloud, sem_label = self.clustermix(pointcloud, sem_label, mix_pc, mix_sem)
                    inst_label = None  # inst_label is invalid after clustermix changes point count
                if self.instance_copy is not None:
                    pointcloud, sem_label = self.instance_copy(
                        pointcloud, sem_label, mix_pc, mix_sem, mix_inst)
                    inst_label = None  # inst_label is invalid after instance_copy changes point count

        # Perform projection
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)

        # Get cached pixel coordinates (before sorting by depth)
        # These are the continuous pixel coordinates for each point
        px = self.projection.cached_data['px'].copy()  # (N,) x pixel coord
        py = self.projection.cached_data['py'].copy()  # (N,) y pixel coord
        uproj_x = self.projection.cached_data['uproj_x_idx'].copy()  # (N,) discrete x
        uproj_y = self.projection.cached_data['uproj_y_idx'].copy()  # (N,) discrete y
        depth = self.projection.cached_data['uproj_depth'].copy()  # (N,) range/depth

        # Create projected tensors
        proj_mask_tensor = torch.from_numpy(proj_mask)
        mask = proj_idx > 0
        proj_sem_label = np.zeros((proj_mask.shape[0], proj_mask.shape[1]), dtype=np.float32)
        # Handle label mapping based on whether labels were already mapped during augmentation
        if labels_are_mapped:
            proj_sem_label[mask] = sem_label[proj_idx[mask]]
            sem_label_mapped = sem_label
        else:
            sem_label_mapped = self.dataset.labelMapping(sem_label)
            proj_sem_label[mask] = sem_label_mapped[proj_idx[mask]]
        proj_sem_label_tensor = torch.from_numpy(proj_sem_label)
        proj_sem_label_tensor = proj_sem_label_tensor * proj_mask_tensor.float()

        # Build range image features
        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0)], 0)

        # Normalize features
        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        # Combine for augmentation
        proj_tensor = torch.cat(
            (proj_feature_tensor,
             proj_sem_label_tensor.unsqueeze(0),
             proj_mask_tensor.float().unsqueeze(0)), dim=0)

        # Apply cropping/augmentation
        if self.is_train:
            proj_tensor, px, py, pointcloud, sem_label_mapped, depth = crop_inputs_for_fusion(
                proj_tensor, px, py, pointcloud, sem_label_mapped, depth,
                self.crop_size, center_crop=False, p_hflip=self.proj_p_hflip)
        else:
            proj_tensor = self.aug_ops(proj_tensor)
            # For validation, we need to adjust coordinates based on center crop
            _, h, w = proj_tensor.shape
            orig_h, orig_w = self.config['sensor']['proj_h'], self.config['sensor']['proj_w']
            offset_y = (orig_h - h) // 2
            offset_x = (orig_w - w) // 2

            # Adjust pixel coordinates for center crop
            px = px - offset_x
            py = py - offset_y

            # Filter valid points (within cropped region)
            valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            px = px[valid]
            py = py[valid]
            pointcloud = pointcloud[valid]
            sem_label_mapped = sem_label_mapped[valid]
            depth = depth[valid]

        # Build point attributes: [x, y, z, intensity, range]
        point_attrs = np.zeros((pointcloud.shape[0], 5), dtype=np.float32)
        point_attrs[:, :3] = pointcloud[:, :3]  # xyz
        point_attrs[:, 3] = pointcloud[:, 3]     # intensity
        point_attrs[:, 4] = depth                 # range

        # Point coordinates in pixel space: [y, x]
        point_coords = np.stack([py, px], axis=1).astype(np.float32)

        output = {
            'proj_image': proj_tensor[:5],           # (5, H, W)
            'proj_labels': proj_tensor[5].long(),    # (H, W)
            'proj_mask': proj_tensor[6],             # (H, W)
            'point_attrs': torch.from_numpy(point_attrs).float(),   # (N, 5)
            'point_coords': torch.from_numpy(point_coords).float(), # (N, 2)
            'point_labels': torch.from_numpy(sem_label_mapped).long(), # (N,)
            'num_points': pointcloud.shape[0],
            'index': index,
        }

        return output

    def __getitem__(self, index):
        '''
        proj_feature_tensor: CxHxW
        proj_sem_label_tensor: HxW
        proj_mask_tensor: HxW
        '''
        if self.use_kpconv:
            return self.get_item_for_kpconv(index)

        if self.return_point_data:
            return self.get_item_for_fusion(index)

        pointcloud, sem_label, inst_label = self.dataset.loadDataByIndex(index)
        labels_are_mapped = False
        if self.is_train and self.adapted_use_mapped_labels:
            sem_label = self.dataset.labelMapping(sem_label)
            labels_are_mapped = True
        if self.is_train:
            if self.instance_cutmix is not None:
                pointcloud, sem_label = self.instance_cutmix(pointcloud, sem_label, inst_label)
            if self.polarmix is not None:
                mix_pc, mix_sem, _ = self._load_mix_sample()
                if labels_are_mapped:
                    mix_sem = self.dataset.labelMapping(mix_sem)
                pointcloud, sem_label = self.polarmix(pointcloud, sem_label, mix_pc, mix_sem)
            pointcloud, sem_label, inst_label = self._maybe_sample_points(pointcloud, sem_label, inst_label)
            if self.augmentor is not None:
                pointcloud = self.augmentor.doAugmentation(pointcloud)  # n, 4

            if (self.clustermix is not None) or (self.instance_copy is not None):
                mix_pc, mix_sem, mix_inst = self._load_mix_sample()
                if labels_are_mapped:
                    mix_sem = self.dataset.labelMapping(mix_sem)
                mix_pc, mix_sem, mix_inst = self._maybe_sample_points(mix_pc, mix_sem, mix_inst)
                if self.augmentor is not None:
                    mix_pc = self.augmentor.doAugmentation(mix_pc)
                if self.clustermix is not None:
                    pointcloud, sem_label = self.clustermix(pointcloud, sem_label, mix_pc, mix_sem)
                if self.instance_copy is not None:
                    pointcloud, sem_label = self.instance_copy(
                        pointcloud, sem_label, mix_pc, mix_sem, mix_inst)
        # if self.use_rangeinterpolation:
        #     pointcloud, sem_label = self.range_interpolation(pointcloud, sem_label)
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)

        proj_mask_tensor = torch.from_numpy(proj_mask)
        mask = proj_idx > 0
        proj_sem_label = np.zeros((proj_mask.shape[0], proj_mask.shape[1]), dtype=np.float32)
        if labels_are_mapped:
            proj_sem_label[mask] = sem_label[proj_idx[mask]]
        else:
            proj_sem_label[mask] = self.dataset.labelMapping(sem_label[proj_idx[mask]])
        proj_sem_label_tensor = torch.from_numpy(proj_sem_label)
        proj_sem_label_tensor = proj_sem_label_tensor * proj_mask_tensor.float()

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0)], 0)

        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None,
                                                                                          None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        if self.return_uproj:
            if not labels_are_mapped:
                sem_label = self.dataset.labelMapping(sem_label)
            sem_label = torch.from_numpy(sem_label).long()

            uproj_x_tensor = torch.from_numpy(self.projection.cached_data['uproj_x_idx']).long()
            uproj_y_tensor = torch.from_numpy(self.projection.cached_data['uproj_y_idx']).long()
            uproj_depth_tensor = torch.from_numpy(self.projection.cached_data['uproj_depth']).float()

            return proj_feature_tensor, proj_sem_label_tensor, proj_mask_tensor, torch.from_numpy(
                proj_range), uproj_x_tensor, uproj_y_tensor, uproj_depth_tensor, sem_label
        else:
            proj_tensor = torch.cat(
                (proj_feature_tensor,
                proj_sem_label_tensor.unsqueeze(0),
                proj_mask_tensor.float().unsqueeze(0)), dim=0)

            # Data augmentation
            proj_tensor = self.aug_ops(proj_tensor)

            return proj_tensor[0:5], proj_tensor[5], proj_tensor[6]

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)


def count_num_of_valid_points(py, px, offset_y, offset_x, h, w):
    py = (py - offset_y) / h
    px = (px - offset_x) / w
    valid = (px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)
    return valid.astype('float64').sum()


def crop_inputs(proj_tensor, px, py, points_xyz, labels, crop_size, center_crop=False, p_hflip=0.0):
    if center_crop:
        _, h, w = proj_tensor.shape
        assert h == crop_size[0] and w == crop_size[1]
        offset_y, offset_x = 0, 0
    else:
        MIN_NUM_POINTS = 1
        NUM_ITERS = 10
        for _ in range(NUM_ITERS):
            offset_y, offset_x, h, w = T.RandomCrop.get_params(proj_tensor, crop_size)
            num_valid_points = count_num_of_valid_points(py, px, offset_y, offset_x, h, w)
            if num_valid_points > MIN_NUM_POINTS:
                break
            print(f'num_valid_points = {num_valid_points}')
        assert h == crop_size[0] and w == crop_size[1]
    proj_tensor = TF.crop(proj_tensor, offset_y, offset_x, h, w)

    py = (py - offset_y) / h
    px = (px - offset_x) / w
    valid = (px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)

    labels = labels[valid]
    px = px[valid]
    py = py[valid]
    points_xyz = points_xyz[valid, :]
    px = 2.0 * (px - 0.5)
    py = 2.0 * (py - 0.5)

    if np.random.uniform() < p_hflip:
        proj_tensor = TF.hflip(proj_tensor)
        px *= -1

    return proj_tensor, px, py, points_xyz, labels


def crop_inputs_for_fusion(proj_tensor, px, py, pointcloud, labels, depth, crop_size, center_crop=False, p_hflip=0.0):
    """
    Crop inputs for fusion model, keeping track of point coordinates in pixel space.

    Args:
        proj_tensor: (C, H, W) projected features + labels + mask
        px: (N,) x pixel coordinates for each point
        py: (N,) y pixel coordinates for each point
        pointcloud: (N, 4) point cloud [x, y, z, intensity]
        labels: (N,) per-point labels
        depth: (N,) range/depth for each point
        crop_size: (h, w) target crop size
        center_crop: whether to use center crop
        p_hflip: probability of horizontal flip

    Returns:
        proj_tensor: cropped projection tensor
        px: adjusted x pixel coordinates (in cropped image space)
        py: adjusted y pixel coordinates (in cropped image space)
        pointcloud: filtered point cloud
        labels: filtered labels
        depth: filtered depth values
    """
    if center_crop:
        _, h, w = proj_tensor.shape
        assert h == crop_size[0] and w == crop_size[1]
        offset_y, offset_x = 0, 0
    else:
        MIN_NUM_POINTS = 1
        NUM_ITERS = 10
        for _ in range(NUM_ITERS):
            offset_y, offset_x, h, w = T.RandomCrop.get_params(proj_tensor, crop_size)
            num_valid_points = count_num_of_valid_points(py, px, offset_y, offset_x, h, w)
            if num_valid_points > MIN_NUM_POINTS:
                break
            print(f'num_valid_points = {num_valid_points}')
        assert h == crop_size[0] and w == crop_size[1]

    proj_tensor = TF.crop(proj_tensor, offset_y, offset_x, h, w)

    # Adjust pixel coordinates to cropped image space
    px_adj = px - offset_x
    py_adj = py - offset_y

    # Filter valid points (within cropped region)
    valid = (px_adj >= 0) & (px_adj < w) & (py_adj >= 0) & (py_adj < h)

    labels = labels[valid]
    px_adj = px_adj[valid]
    py_adj = py_adj[valid]
    pointcloud = pointcloud[valid]
    depth = depth[valid]

    # Handle horizontal flip
    if np.random.uniform() < p_hflip:
        proj_tensor = TF.hflip(proj_tensor)
        px_adj = w - 1 - px_adj  # Flip x coordinates

    return proj_tensor, px_adj, py_adj, pointcloud, labels, depth


def custom_collate_kpconv_fn(list_data):
    output = {}
    for key in list_data[0].keys():
        if key in ('input2d', 'mask2d', 'label2d'):
            output[key] = torch.stack([v[key] for v in list_data], dim=0)
        elif key in ('px', 'py', 'points_xyz', 'knns', 'labels'):
            output[key] = torch.cat([v[key] for v in list_data], dim=0)
        elif key in ('num_points', 'index'):
            output[key] = torch.LongTensor([v[key] for v in list_data])
    return output


def collate_fn_fusion(list_data):
    """
    Custom collate function for fusion model.

    Stacks range images and labels as usual, but concatenates point data across
    the batch while adding batch indices to coordinates.

    Args:
        list_data: List of dicts from get_item_for_fusion

    Returns:
        dict with:
            proj_image: (B, C, H, W) batched range images
            proj_labels: (B, H, W) batched projected labels
            proj_mask: (B, H, W) batched masks
            point_attrs: (total_N, 5) concatenated point attributes
            point_coords: (total_N, 3) coordinates with batch idx [batch_idx, y, x]
            point_labels: (total_N,) concatenated per-point labels
            num_points: (B,) number of points per sample
            batch_indices: (total_N,) batch index for each point (alternative format)
    """
    batch_size = len(list_data)

    # Stack image-like tensors
    proj_images = torch.stack([v['proj_image'] for v in list_data], dim=0)
    proj_labels = torch.stack([v['proj_labels'] for v in list_data], dim=0)
    proj_masks = torch.stack([v['proj_mask'] for v in list_data], dim=0)

    # Concatenate point data with batch indices
    all_point_attrs = []
    all_point_coords = []
    all_point_labels = []
    all_batch_indices = []
    num_points = []

    for batch_idx, v in enumerate(list_data):
        n_points = v['point_attrs'].shape[0]
        num_points.append(n_points)

        all_point_attrs.append(v['point_attrs'])
        all_point_labels.append(v['point_labels'])

        # Add batch index to coordinates: [batch_idx, y, x]
        batch_idx_tensor = torch.full((n_points, 1), batch_idx, dtype=torch.float32)
        coords_with_batch = torch.cat([batch_idx_tensor, v['point_coords']], dim=1)
        all_point_coords.append(coords_with_batch)

        # Also store batch indices separately for convenience
        all_batch_indices.append(torch.full((n_points,), batch_idx, dtype=torch.long))

    output = {
        'proj_image': proj_images,                              # (B, C, H, W)
        'proj_labels': proj_labels,                             # (B, H, W)
        'proj_mask': proj_masks,                                # (B, H, W)
        'point_attrs': torch.cat(all_point_attrs, dim=0),       # (total_N, 5)
        'point_coords': torch.cat(all_point_coords, dim=0),     # (total_N, 3) [batch_idx, y, x]
        'point_labels': torch.cat(all_point_labels, dim=0),     # (total_N,)
        'num_points': torch.LongTensor(num_points),             # (B,)
        'batch_indices': torch.cat(all_batch_indices, dim=0),   # (total_N,)
    }

    return output
