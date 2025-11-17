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
import os
import cv2
import sys
import yaml

from .preprocess import augmentor, projection
from .preprocess.wpd import WPDConfig, WPDAugmentor


class RangeViewLoader(Dataset):
    def __init__(self, dataset, config, data_len=-1, is_train=True, return_uproj=False, use_kpconv=False):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.data_len = data_len
        self.return_uproj = return_uproj
        self.use_kpconv = use_kpconv

        augment_params = augmentor.AugmentParams()
        augment_config = self.config['augmentation']

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
            if 'p_polarmix' in augment_config:
                augment_params.setPolarMixParams(augment_config['p_polarmix'])
                print(f'Adding PolarMix with probability {augment_params.p_polarmix}')
            self.augmentor = augmentor.Augmentor(augment_params)

            # Initialize WPD augmentation
            if 'use_wpd' in augment_config and augment_config['use_wpd']:
                wpd_stats_path = augment_config.get('wpd_stats_path', 'dataset/preprocess/wpd_stats.yaml')
                wpd_mode = augment_config.get('wpd_mode', 'semantic')
                wpd_apply_rate = augment_config.get('wpd_apply_rate', 0.6)
                try:
                    wpd_config = WPDConfig(wpd_stats_path, mode=wpd_mode)
                    self.wpd_augmentor = WPDAugmentor(wpd_config, apply_rate=wpd_apply_rate)
                    print(f'WPD augmentation enabled (mode={wpd_mode}, apply_rate={wpd_apply_rate})')
                except FileNotFoundError as e:
                    print(f'Warning: {e}')
                    print('WPD augmentation disabled. Please run compute_wpd_stats.py first.')
                    self.wpd_augmentor = None
            else:
                self.wpd_augmentor = None
        else:
            self.augmentor = None
            self.wpd_augmentor = None

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

        self.get_item_for_kpconv(0)

    def get_item_for_kpconv(self, index):
        '''
        proj_feature_tensor: CxHxW
        proj_sem_label_tensor: HxW
        proj_mask_tensor: HxW
        '''
        pointcloud, sem_label, _ = self.dataset.loadDataByIndex(index)
        points_xyz = pointcloud[:, :3]
        sem_label = self.dataset.labelMapping(sem_label)

        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)
        px, py = self.projection.cached_data['px'], self.projection.cached_data['py']

        proj_mask_tensor = torch.from_numpy(proj_mask)
        mask = proj_idx > 0
        proj_sem_label = np.zeros((proj_mask.shape[0], proj_mask.shape[1]), dtype=np.float32)
        proj_sem_label[mask] = sem_label[proj_idx[mask]]
        proj_sem_label_tensor = torch.from_numpy(proj_sem_label)
        proj_sem_label_tensor = proj_sem_label_tensor * proj_mask_tensor.float()

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        proj_existence_tensor = proj_mask_tensor.float()
        # proj_feature_tensor = torch.cat(
        #     [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0), proj_existence_tensor.unsqueeze(0)], 0)
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0)], 0)

        # Apply WPD augmentation (before normalization)
        if self.is_train and self.wpd_augmentor is not None:
            proj_feature_tensor, proj_sem_label_tensor, proj_mask_tensor, proj_range_tensor = \
                self._apply_wpd_augmentation(
                    proj_feature_tensor,
                    proj_sem_label_tensor,
                    proj_mask_tensor,
                    proj_range_tensor,
                    include_existence_channel=False)

        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        proj_tensor = torch.cat(
            (proj_feature_tensor,
            proj_sem_label_tensor.unsqueeze(0),
            proj_mask_tensor.float().unsqueeze(0)), dim=0)

        if self.is_train:
            save_path = os.getenv('AUG_VIS_PATH', './aug_visualizations')
            if os.getenv('SAVE_AUG_VIS'):
                save_proj_tensor_as_images(proj_tensor, index, save_path, pointcloud=pointcloud, labels=sem_label)
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


    def __getitem__(self, index):
        '''
        proj_feature_tensor: CxHxW
        proj_sem_label_tensor: HxW
        proj_mask_tensor: HxW
        '''
        if self.use_kpconv:
            return self.get_item_for_kpconv(index)

        pointcloud, sem_label, _ = self.dataset.loadDataByIndex(index)
        if self.is_train:
            pointcloud = self.augmentor.doAugmentation(pointcloud)  # n, 4
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)

        proj_mask_tensor = torch.from_numpy(proj_mask)
        mask = proj_idx > 0
        proj_sem_label = np.zeros((proj_mask.shape[0], proj_mask.shape[1]), dtype=np.float32)
        proj_sem_label[mask] = self.dataset.labelMapping(sem_label[proj_idx[mask]])
        proj_sem_label_tensor = torch.from_numpy(proj_sem_label)
        proj_sem_label_tensor = proj_sem_label_tensor * proj_mask_tensor.float()

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        proj_existence_tensor = proj_mask_tensor.float()
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0), proj_existence_tensor.unsqueeze(0)], 0)

        if self.is_train and self.wpd_augmentor is not None:
            proj_feature_tensor, proj_sem_label_tensor, proj_mask_tensor, proj_range_tensor = \
                self._apply_wpd_augmentation(
                    proj_feature_tensor,
                    proj_sem_label_tensor,
                    proj_mask_tensor,
                    proj_range_tensor,
                    include_existence_channel=True)
            proj_range = proj_range_tensor.numpy()

        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        if self.return_uproj:
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

            # if self.is_train and os.getenv('SAVE_AUG_VIS'):
            #     save_path = os.getenv('AUG_VIS_PATH', './aug_visualizations')
            #     save_proj_tensor_as_images(proj_tensor, index, save_path)

            return proj_tensor[0:5], proj_tensor[6], proj_tensor[7]

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)

    def _apply_wpd_augmentation(self, proj_feature_tensor, proj_sem_label_tensor,
                                proj_mask_tensor, proj_range_tensor,
                                include_existence_channel):
        """
        Apply WPD augmentation by sampling a donor frame and mixing range-view tensors.
        """
        donor_index = np.random.randint(0, len(self.dataset))
        pointcloud_donor, sem_label_donor, _ = self.dataset.loadDataByIndex(donor_index)
        sem_label_donor = self.dataset.labelMapping(sem_label_donor)

        pointcloud_donor = self.augmentor.doAugmentation(pointcloud_donor)
        proj_pointcloud_donor, proj_range_donor, proj_idx_donor, proj_mask_donor = self.projection.doProjection(pointcloud_donor)

        proj_mask_tensor_donor = torch.from_numpy(proj_mask_donor)
        mask_donor = proj_idx_donor > 0
        proj_sem_label_donor = np.zeros((proj_mask_donor.shape[0], proj_mask_donor.shape[1]), dtype=np.float32)
        proj_sem_label_donor[mask_donor] = sem_label_donor[proj_idx_donor[mask_donor]]
        proj_sem_label_tensor_donor = torch.from_numpy(proj_sem_label_donor)
        proj_sem_label_tensor_donor = proj_sem_label_tensor_donor * proj_mask_tensor_donor.float()

        proj_range_tensor_donor = torch.from_numpy(proj_range_donor)
        proj_xyz_tensor_donor = torch.from_numpy(proj_pointcloud_donor[..., :3])
        proj_intensity_tensor_donor = torch.from_numpy(proj_pointcloud_donor[..., 3])
        proj_intensity_tensor_donor = proj_intensity_tensor_donor.ne(-1).float() * proj_intensity_tensor_donor

        donor_channels = [
            proj_range_tensor_donor.unsqueeze(0),
            proj_xyz_tensor_donor.permute(2, 0, 1),
            proj_intensity_tensor_donor.unsqueeze(0),
        ]
        if include_existence_channel:
            donor_channels.append(proj_mask_tensor_donor.float().unsqueeze(0))
        proj_feature_tensor_donor = torch.cat(donor_channels, 0)

        frame_a_dict = {
            'features': proj_feature_tensor,
            'labels': proj_sem_label_tensor.long(),
            'mask': proj_mask_tensor,
            'range': proj_range_tensor,
        }
        frame_b_dict = {
            'features': proj_feature_tensor_donor,
            'labels': proj_sem_label_tensor_donor.long(),
            'mask': proj_mask_tensor_donor,
            'range': proj_range_tensor_donor,
        }

        result = self.wpd_augmentor(frame_a_dict, frame_b_dict)

        return result['features'], result['labels'].float(), result['mask'], result['range']


def load_color_map():
    """Load color map from SemanticKITTI config or use default nuScenes colors."""
    color_map = None

    # Try to load SemanticKITTI color map
    kitti_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'semantic_kitti/semantic-kitti.yaml'
    )

    if os.path.exists(kitti_config_path):
        try:
            with open(kitti_config_path, 'r') as f:
                kitti_config = yaml.safe_load(f)
                if 'learning_map_inv' in kitti_config and 'color_map' in kitti_config:
                    # Create color map for learning labels (0-19)
                    learning_map_inv = kitti_config['learning_map_inv']
                    color_map_orig = kitti_config['color_map']
                    color_map = {}
                    for learning_id, orig_id in learning_map_inv.items():
                        if orig_id in color_map_orig:
                            color_map[learning_id] = color_map_orig[orig_id]
                    print(f'Loaded SemanticKITTI color map with {len(color_map)} classes')
        except Exception as e:
            print(f'Warning: Could not load SemanticKITTI color map: {e}')

    # Default nuScenes color map (if SemanticKITTI not available)
    if color_map is None:
        color_map = {
            0: [255, 255, 255],  # ignore - white
            1: [255, 120, 50],   # barrier - orange
            2: [255, 192, 203],  # bicycle - pink
            3: [255, 0, 0],      # bus - red
            4: [0, 150, 245],    # car - blue
            5: [0, 255, 255],    # construction_vehicle - cyan
            6: [255, 0, 255],    # motorcycle - magenta
            7: [255, 255, 0],    # pedestrian - yellow
            8: [255, 128, 0],    # traffic_cone - orange
            9: [255, 240, 150],  # trailer - light yellow
            10: [135, 60, 0],    # truck - brown
            11: [128, 64, 128],  # driveable_surface - purple
            12: [244, 35, 232],  # other_flat - magenta
            13: [107, 142, 35],  # sidewalk - olive
            14: [70, 70, 70],    # terrain - gray
            15: [102, 102, 156], # manmade - light purple
            16: [107, 142, 35],  # vegetation - green
        }
        print(f'Using default nuScenes color map with {len(color_map)} classes')

    return color_map


def save_proj_tensor_as_images(proj_tensor, index, save_path, pointcloud=None, labels=None, color_map=None):
    """
    Save proj_tensor channels as visualizable images and optionally save pointcloud and labels as .bin files.
    proj_tensor shape: [8, H, W] where channels are:
        0: range
        1-3: xyz
        4: intensity
        5: existence
        6: semantic label
        7: mask
    pointcloud: [N, 4] numpy array (x, y, z, intensity)
    labels: [N] numpy array of semantic labels
    """
    os.makedirs(save_path, exist_ok=True)

    # Load color map if not provided
    if color_map is None:
        color_map = load_color_map()

    # Convert to numpy and move channels to last dimension
    tensor_np = proj_tensor.numpy()  # [8, H, W]

    # Helper function to normalize to 0-255 range
    def normalize_to_uint8(data):
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min > 1e-6:
            normalized = (data - data_min) / (data_max - data_min) * 255
        else:
            normalized = np.zeros_like(data)
        return normalized.astype(np.uint8)

    # Helper function to apply color map to labels
    def apply_color_map(labels, color_map):
        h, w = labels.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for label_id, color in color_map.items():
            mask = labels == label_id
            # color_map is in BGR format (from YAML)
            colored[mask] = color
        return colored

    # Save each channel
    channel_names = ['range', 'x', 'y', 'z', 'intensity', 'semantic_label', 'mask']

    for ch_idx, ch_name in enumerate(channel_names):
        channel_data = tensor_np[ch_idx]

        if ch_name == 'semantic_label':
            # For semantic labels, use the dataset-specific color map
            label_img = channel_data.astype(np.int32)
            label_colored = apply_color_map(label_img, color_map)
            cv2.imwrite(f'{save_path}/sample_{index:06d}_{ch_name}.png', label_colored)
        elif ch_name == 'mask' or ch_name == 'existence':
            # Binary masks - save as black/white
            mask_img = (channel_data * 255).astype(np.uint8)
            cv2.imwrite(f'{save_path}/sample_{index:06d}_{ch_name}.png', mask_img)
        else:
            # For continuous values (range, xyz, intensity), normalize and apply colormap
            normalized = normalize_to_uint8(channel_data)
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(f'{save_path}/sample_{index:06d}_{ch_name}.png', colored)

    # Also save RGB composite of xyz
    xyz_data = tensor_np[1:4].transpose(1, 2, 0)  # [H, W, 3]
    xyz_normalized = np.zeros_like(xyz_data)
    for i in range(3):
        xyz_normalized[:, :, i] = normalize_to_uint8(xyz_data[:, :, i])
    cv2.imwrite(f'{save_path}/sample_{index:06d}_xyz_composite.png', xyz_normalized.astype(np.uint8))

    # Save pointcloud as .bin file (if provided)
    if pointcloud is not None:
        pointcloud_path = f'{save_path}/sample_{index:06d}.bin'
        pointcloud.astype(np.float32).tofile(pointcloud_path)
        print(f'Saved pointcloud to {pointcloud_path}')

    # Save labels as .label file (if provided)
    if labels is not None:
        label_path = f'{save_path}/sample_{index:06d}.label'
        # SemanticKITTI format: labels are saved as uint32
        labels.astype(np.uint32).tofile(label_path)
        print(f'Saved labels to {label_path}')

    print(f'Saved augmented visualizations for sample {index} to {save_path}/')


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

    # Clamp labels to valid range [0, 19] for SemanticKITTI
    labels = np.clip(labels, 0, 19)

    px = 2.0 * (px - 0.5)
    py = 2.0 * (py - 0.5)

    if np.random.uniform() < p_hflip:
        proj_tensor = TF.hflip(proj_tensor)
        px *= -1

    return proj_tensor, px, py, points_xyz, labels


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
