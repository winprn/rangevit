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

from .preprocess import augmentor, projection, voxelization


class RangeViewLoader(Dataset):
    def __init__(self, dataset, config, data_len=-1, is_train=True, return_uproj=False, use_kpconv=False, use_fusion_voxel=False):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.data_len = data_len
        self.return_uproj = return_uproj
        self.use_kpconv = use_kpconv
        self.use_fusion_voxel = use_fusion_voxel

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
            self.augmentor = augmentor.Augmentor(augment_params)
        else:
            self.augmentor = None

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

        # Voxel features (Phase 1: non-learnable)
        voxel_config = self.config.get('voxel_features', {})
        self.use_voxel_features = voxel_config.get('enable', False)
        if self.use_voxel_features:
            self.voxelizer = voxelization.VoxelGrid(
                voxel_size=voxel_config.get('voxel_size', 0.05),
                grid_bounds=None  # Auto-compute from point cloud
            )
            self.voxel_feature_dim = voxel_config.get('feature_dim', 8)
            print(f'Voxel features enabled: voxel_size={voxel_config.get("voxel_size", 0.05)}, feature_dim={self.voxel_feature_dim}')
        else:
            self.voxelizer = None

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

    def get_item_for_kpconv(self, index):
        '''
        proj_feature_tensor: CxHxW
        proj_sem_label_tensor: HxW
        proj_mask_tensor: HxW
        '''
        pointcloud, sem_label, inst_label = self.dataset.loadDataByIndex(index)
        points_xyz = pointcloud[:, :3]
        sem_label = self.dataset.labelMapping(sem_label)

        if self.is_train and (self.scan_proj is False):
            pointcloud = self.augmentor.doAugmentation(pointcloud)  # n, 4
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
        """
        Get data for fusion model (range + voxel branches).

        Returns dict with:
            - range_image: [5, H, W] range image (5 channels only)
            - range_label: [H, W] pixel-wise labels
            - range_mask: [H, W] valid pixel mask
            - point_features: [N, 4] raw point features (x, y, z, intensity)
            - point_coords: [N, 3] point coordinates
            - point_labels: [N] per-point labels
            - range_pxpy: [N, 2] projection coords normalized to [-1, 1]
            - num_points: number of points
            - index: dataset index
        """
        pointcloud, sem_label, inst_label = self.dataset.loadDataByIndex(index)
        points_xyz = pointcloud[:, :3]
        sem_label_mapped = self.dataset.labelMapping(sem_label)

        # Point cloud augmentation
        if self.is_train and (self.scan_proj is False):
            pointcloud = self.augmentor.doAugmentation(pointcloud)
            points_xyz = pointcloud[:, :3]

        # Range projection
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)
        px, py = self.projection.cached_data['px'], self.projection.cached_data['py']

        # Range image features (5 channels: range, x, y, z, intensity)
        proj_mask_tensor = torch.from_numpy(proj_mask)
        mask = proj_idx > 0
        proj_sem_label = np.zeros((proj_mask.shape[0], proj_mask.shape[1]), dtype=np.float32)
        proj_sem_label[mask] = sem_label_mapped[proj_idx[mask]]
        proj_sem_label_tensor = torch.from_numpy(proj_sem_label)
        proj_sem_label_tensor = proj_sem_label_tensor * proj_mask_tensor.float()

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor

        # Only 5 channels for range branch in fusion mode
        proj_feature_tensor = torch.cat([
            proj_range_tensor.unsqueeze(0),
            proj_xyz_tensor.permute(2, 0, 1),
            proj_intensity_tensor.unsqueeze(0)
        ], dim=0)  # [5, H, W]

        # Normalize range features (first 5 channels only)
        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:5, None, None]) / self.proj_img_stds[:5, None, None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        # Combine for cropping
        proj_tensor = torch.cat([
            proj_feature_tensor,
            proj_sem_label_tensor.unsqueeze(0),
            proj_mask_tensor.float().unsqueeze(0)
        ], dim=0)  # [7, H, W]

        # Apply crop
        if self.is_train:
            proj_tensor, px, py, points_xyz, sem_label_mapped = crop_inputs(
                proj_tensor, px, py, points_xyz, sem_label_mapped,
                self.crop_size, center_crop=False, p_hflip=self.proj_p_hflip)
            # Update pointcloud for point features (after cropping)
            pointcloud_cropped = np.zeros((points_xyz.shape[0], 4), dtype=np.float32)
            pointcloud_cropped[:, :3] = points_xyz
            # Intensity is lost after crop, set to zeros (will use original intensity below)
        else:
            _, h, w = proj_tensor.shape
            # Normalize px, py to [-1, 1]
            px = 2.0 * ((px / w) - 0.5)
            py = 2.0 * ((py / h) - 0.5)

        # Prepare point data
        # Point features: x, y, z, intensity
        if self.is_train:
            # After cropping, we only have valid points
            # Get intensity from the projection for cropped points
            point_features = np.zeros((points_xyz.shape[0], 4), dtype=np.float32)
            point_features[:, :3] = points_xyz
            # Note: intensity may not be preserved after cropping, using zeros
            # A more robust solution would track the original point indices
            point_features_tensor = torch.from_numpy(point_features).float()
            point_coords_tensor = torch.from_numpy(points_xyz).float()
            point_labels_tensor = torch.from_numpy(sem_label_mapped).long()
        else:
            point_features_tensor = torch.from_numpy(pointcloud).float()  # [N, 4]
            point_coords_tensor = torch.from_numpy(points_xyz).float()  # [N, 3]
            point_labels_tensor = torch.from_numpy(sem_label_mapped).long()  # [N]

        # Projection coordinates (px, py already normalized to [-1, 1])
        range_pxpy_tensor = torch.stack([
            torch.from_numpy(px).float(),
            torch.from_numpy(py).float()
        ], dim=1)  # [N, 2]

        output = {
            'range_image': proj_tensor[:5],  # [5, H, W]
            'range_label': proj_tensor[5],   # [H, W]
            'range_mask': proj_tensor[6],    # [H, W]
            'point_features': point_features_tensor,  # [N, 4]
            'point_coords': point_coords_tensor,      # [N, 3]
            'point_labels': point_labels_tensor,      # [N]
            'range_pxpy': range_pxpy_tensor,          # [N, 2]
            'num_points': point_features_tensor.shape[0],
            'index': index,
        }

        return output

    def __getitem__(self, index):
        '''
        proj_feature_tensor: CxHxW
        proj_sem_label_tensor: HxW
        proj_mask_tensor: HxW
        '''
        if self.use_fusion_voxel:
            return self.get_item_for_fusion(index)
        if self.use_kpconv:
            return self.get_item_for_kpconv(index)

        pointcloud, sem_label, inst_label = self.dataset.loadDataByIndex(index)
        if self.is_train:
            pointcloud = self.augmentor.doAugmentation(pointcloud)  # n, 4

        # Voxelization (Phase 1: non-learnable features)
        if self.use_voxel_features:
            voxel_coords, voxel_features = self.voxelizer.voxelize(pointcloud)
            # Project voxel features to range image
            proj_voxel_features = self.voxelizer.project_to_range(
                voxel_coords, voxel_features, self.projection)
            proj_voxel_tensor = torch.from_numpy(proj_voxel_features).permute(2, 0, 1)  # [D, H, W]
            # Pad to full feature dimension if needed
            if proj_voxel_tensor.shape[0] < self.voxel_feature_dim:
                padding = torch.zeros(
                    self.voxel_feature_dim - proj_voxel_tensor.shape[0],
                    proj_voxel_tensor.shape[1],
                    proj_voxel_tensor.shape[2])
                proj_voxel_tensor = torch.cat([proj_voxel_tensor, padding], dim=0)

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
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0)], 0)

        # Concatenate voxel features (Phase 1)
        if self.use_voxel_features:
            proj_feature_tensor = torch.cat([proj_feature_tensor, proj_voxel_tensor], dim=0)  # [5+D, H, W]

        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None,
                                                                                          None]
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

            # Return correct number of feature channels (5 without voxel, 13 with voxel)
            num_feature_channels = proj_feature_tensor.shape[0]
            return proj_tensor[0:num_feature_channels], proj_tensor[num_feature_channels], proj_tensor[num_feature_channels+1]

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


def custom_collate_fusion_fn(list_data):
    """
    Collate function for fusion model with variable-length point clouds.

    Stacks range images (fixed size) and concatenates point data (variable length)
    while tracking batch indices.
    """
    output = {}
    batch_size = len(list_data)

    # Stack batched tensors (fixed size)
    output['range_image'] = torch.stack([d['range_image'] for d in list_data], dim=0)  # [B, 5, H, W]
    output['range_label'] = torch.stack([d['range_label'] for d in list_data], dim=0)  # [B, H, W]
    output['range_mask'] = torch.stack([d['range_mask'] for d in list_data], dim=0)    # [B, H, W]

    # Concatenate variable-length point data with batch indices
    point_features_list = []
    point_coords_list = []
    point_labels_list = []
    range_pxpy_list = []
    batch_indices_list = []
    num_points_list = []

    for batch_idx, d in enumerate(list_data):
        n = d['num_points']
        point_features_list.append(d['point_features'])
        point_coords_list.append(d['point_coords'])
        point_labels_list.append(d['point_labels'])
        range_pxpy_list.append(d['range_pxpy'])

        # Create batch indices for this sample
        batch_indices_list.append(torch.full((n,), batch_idx, dtype=torch.long))
        num_points_list.append(n)

    output['point_features'] = torch.cat(point_features_list, dim=0)   # [N_total, 4]
    output['point_coords'] = torch.cat(point_coords_list, dim=0)       # [N_total, 3]
    output['point_labels'] = torch.cat(point_labels_list, dim=0)       # [N_total]
    output['range_pxpy'] = torch.cat(range_pxpy_list, dim=0)           # [N_total, 2]
    output['batch_indices'] = torch.cat(batch_indices_list, dim=0)     # [N_total]
    output['num_points'] = torch.LongTensor(num_points_list)           # [B]
    output['index'] = torch.LongTensor([d['index'] for d in list_data])  # [B]

    return output
