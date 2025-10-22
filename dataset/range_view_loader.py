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
            if 'p_rangemix' in augment_config:
                augment_params.setRangeMixParams(augment_config['p_rangemix'])
                self.p_rangemix = augment_config['p_rangemix']
                print(f'Adding RangeMix augmentation with probability {self.p_rangemix}')
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
            self.p_rangemix = 0.0

        mcf_config = self.config.get('mcf', {})
        self.use_mcf = self.is_train and mcf_config.get('enabled', False)
        self.mcf_num_subclouds = int(mcf_config.get('num_subclouds', 3))
        self.mcf_donor_order = mcf_config.get('donor_order', 'sequential')
        if self.mcf_num_subclouds < 2:
            self.use_mcf = False
        if self.use_mcf:
            print(f'Enabling Multi-Cloud Fusion (subclouds={self.mcf_num_subclouds}, donor_order={self.mcf_donor_order})')
        self.mcf_debug = bool(mcf_config.get('debug', False))

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

        self.get_item_for_kpconv(235)

    def _project_pointcloud(self, pointcloud, mapped_sem_label):

        proj_pointcloud, proj_range, proj_idx, _ = self.projection.doProjection(pointcloud)
        cached = self.projection.cached_data
        px = cached.get('px')
        py = cached.get('py')
        uproj_x = cached.get('uproj_x_idx')
        uproj_y = cached.get('uproj_y_idx')
        uproj_depth = cached.get('uproj_depth')

        mask = proj_idx >= 0
        proj_mask_tensor = torch.from_numpy(mask).bool()
        proj_sem_label = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)
        proj_sem_label[mask] = mapped_sem_label[proj_idx[mask]]
        proj_sem_label_tensor = torch.from_numpy(proj_sem_label)
        proj_sem_label_tensor = proj_sem_label_tensor * proj_mask_tensor.float()

        proj_range_tensor = torch.from_numpy(proj_range).float()
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3]).float()
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3]).float()
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor

        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0),
             proj_xyz_tensor.permute(2, 0, 1),
             proj_intensity_tensor.unsqueeze(0)], 0)

        return {
            'features': proj_feature_tensor,
            'labels': proj_sem_label_tensor,
            'mask': proj_mask_tensor,
            'range': proj_range_tensor,
            'proj_idx': proj_idx,
            'px': px.copy() if px is not None else None,
            'py': py.copy() if py is not None else None,
            'uproj_x_idx': uproj_x.copy() if uproj_x is not None else None,
            'uproj_y_idx': uproj_y.copy() if uproj_y is not None else None,
            'uproj_depth': uproj_depth.copy() if uproj_depth is not None else None,
        }

    def _build_subcloud_projections(self, pointcloud, mapped_sem_label):
        if not self.use_mcf or self.mcf_num_subclouds <= 1:
            return [self._project_pointcloud(pointcloud, mapped_sem_label)]

        x, y = pointcloud[:, 0], pointcloud[:, 1]
        yaw = np.mod(-np.arctan2(y, x), 2 * np.pi)
        bin_edges = np.linspace(0, 2 * np.pi, self.mcf_num_subclouds + 1)

        subclouds = []
        for idx in range(self.mcf_num_subclouds):
            start = bin_edges[idx]
            end = bin_edges[idx + 1]
            if idx == self.mcf_num_subclouds - 1:
                mask = (yaw >= start) & (yaw <= end)
            else:
                mask = (yaw >= start) & (yaw < end)

            if not np.any(mask):
                continue

            sub_pointcloud = pointcloud[mask]
            sub_labels = mapped_sem_label[mask]
            subclouds.append(self._project_pointcloud(sub_pointcloud, sub_labels))

        if len(subclouds) == 0:
            subclouds.append(self._project_pointcloud(pointcloud, mapped_sem_label))

        return subclouds

    def _apply_mcf(self, subclouds):
        debug_enabled = self.use_mcf and self.mcf_debug
        if debug_enabled:
            def occ(mask):
                return mask.float().mean().item() * 100.0 if mask.numel() > 0 else 0.0

            occ_str = ', '.join(
                f'{idx}:{occ(sc["mask"].bool() if sc["mask"].dtype != torch.bool else sc["mask"]):.2f}%'
                for idx, sc in enumerate(subclouds)
            )
            print(f'[MCF] Subcloud occupancies -> {occ_str}')

        valid_indices = [i for i, sc in enumerate(subclouds) if sc['mask'].any()]

        if len(valid_indices) == 0:
            # Fall back to the first subcloud (already zero-masked)
            sc = subclouds[0]
            mask_bool = sc['mask'].bool()
            mask_float = mask_bool.float()
            features = sc['features'] * mask_float.unsqueeze(0)
            labels = sc['labels'] * mask_float
            if debug_enabled:
                print('[MCF] No valid subclouds; falling back to primary projection.')
            return features, labels, mask_bool, 0

        if len(valid_indices) == 1 or not self.use_mcf:
            idx = valid_indices[0]
            sc = subclouds[idx]
            mask_bool = sc['mask'].bool()
            mask_float = mask_bool.float()
            features = sc['features'] * mask_float.unsqueeze(0)
            labels = sc['labels'] * mask_float
            if debug_enabled:
                single_occ = mask_float.mean().item() * 100.0
                print(f'[MCF] Single valid subcloud {idx}; occupancy {single_occ:.2f}%')
            return features, labels, mask_bool, idx

        selected_idx = int(np.random.choice(valid_indices))
        target = subclouds[selected_idx]
        fused_features = target['features'].clone()
        fused_labels = target['labels'].clone()
        fused_mask = target['mask'].clone().bool()

        if debug_enabled:
            base_occ = fused_mask.float().mean().item() * 100.0
            print(f'[MCF] Selected target subcloud {selected_idx} with occupancy {base_occ:.2f}%')

        donor_indices = [i for i in valid_indices if i != selected_idx]
        if self.mcf_donor_order == 'random':
            np.random.shuffle(donor_indices)

        for donor_idx in donor_indices:
            donor = subclouds[donor_idx]
            donor_mask = donor['mask'].bool()
            fill_mask = donor_mask & (~fused_mask)
            if debug_enabled:
                donor_occ = donor_mask.float().mean().item() * 100.0
                fillable = fill_mask.float().mean().item() * 100.0
                print(f'[MCF] Donor {donor_idx}: occ={donor_occ:.2f}% fillable={fillable:.2f}%')
            if fill_mask.any():
                fused_features[:, fill_mask] = donor['features'][:, fill_mask]
                fused_labels[fill_mask] = donor['labels'][fill_mask]
                fused_mask[fill_mask] = True

                if (~fused_mask).sum() == 0:
                    if debug_enabled:
                        print('[MCF] Target fully filled; early exit.')
                    break

        fused_mask_float = fused_mask.float()
        fused_features = fused_features * fused_mask_float.unsqueeze(0)
        fused_labels = fused_labels * fused_mask_float

        if debug_enabled:
            final_occ = fused_mask_float.mean().item() * 100.0
            print(f'[MCF] Final target occupancy {final_occ:.2f}% (gain {final_occ - base_occ:.2f}%)')

        return fused_features, fused_labels, fused_mask, selected_idx

    def _normalize_and_stack(self, features, labels, mask_bool):
        mask_float = mask_bool.float()
        norm_features = (features - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        norm_features = norm_features * mask_float.unsqueeze(0)
        masked_labels = labels * mask_float
        return torch.cat(
            (norm_features,
             masked_labels.unsqueeze(0),
             mask_float.unsqueeze(0)), dim=0)

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
            mix_index = np.random.randint(0, len(self.dataset))
            pointcloud_b, sem_label_b, _ = self.dataset.loadDataByIndex(mix_index)
            sem_label_b = self.dataset.labelMapping(sem_label_b)
            pointcloud, sem_label = self.augmentor.polarmix(pointcloud, sem_label, pointcloud_b, sem_label_b)
            points_xyz = pointcloud[:, :3]

        full_projection = self._project_pointcloud(pointcloud, sem_label)
        if self.use_mcf:
            subclouds = self._build_subcloud_projections(pointcloud, sem_label)
            features, labels, mask_bool, _ = self._apply_mcf(subclouds)
        else:
            features = full_projection['features']
            labels = full_projection['labels']
            mask_bool = full_projection['mask'].bool()

        range_tensor = features[0].clone()
        proj_range_np = range_tensor.cpu().numpy()

        if self.is_train and self.wpd_augmentor is not None:
            donor_index = np.random.randint(0, len(self.dataset))
            pointcloud_donor, sem_label_donor, _ = self.dataset.loadDataByIndex(donor_index)
            sem_label_donor = self.dataset.labelMapping(sem_label_donor)

            if self.use_mcf:
                donor_subclouds = self._build_subcloud_projections(pointcloud_donor, sem_label_donor)
                donor_features, donor_labels, donor_mask_bool, _ = self._apply_mcf(donor_subclouds)
            else:
                donor_proj = self._project_pointcloud(pointcloud_donor, sem_label_donor)
                donor_features = donor_proj['features']
                donor_labels = donor_proj['labels']
                donor_mask_bool = donor_proj['mask'].bool()

            frame_a_dict = {
                'features': features,
                'labels': labels.long(),
                'mask': mask_bool.float(),
                'range': range_tensor,
            }
            frame_b_dict = {
                'features': donor_features,
                'labels': donor_labels.long(),
                'mask': donor_mask_bool.float(),
                'range': donor_features[0].clone(),
            }

            result = self.wpd_augmentor(frame_a_dict, frame_b_dict)
            features = result['features']
            labels = result['labels'].float()
            mask_bool = result['mask'].bool()
            range_tensor = result['range']
            proj_range_np = range_tensor.cpu().numpy()

        if self.is_train:
            proj_tensor = self._normalize_and_stack(features, labels, mask_bool)
            proj_tensor, px, py, points_xyz, sem_label = crop_inputs(
                proj_tensor,
                full_projection['px'],
                full_projection['py'],
                points_xyz,
                sem_label,
                self.crop_size,
                center_crop=False,
                p_hflip=self.proj_p_hflip)
        else:
            mask_float = mask_bool.float()
            features = features * mask_float.unsqueeze(0)
            labels = labels * mask_float
            proj_tensor = torch.cat(
                (features,
                 labels.unsqueeze(0),
                 mask_float.unsqueeze(0)), dim=0)

            _, h, w = proj_tensor.shape
            px = full_projection['px']
            py = full_projection['py']
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
            output['range'] = torch.from_numpy(proj_range_np)
            if full_projection['uproj_x_idx'] is not None:
                output['uproj_x'] = torch.from_numpy(full_projection['uproj_x_idx']).long()
            if full_projection['uproj_y_idx'] is not None:
                output['uproj_y'] = torch.from_numpy(full_projection['uproj_y_idx']).long()
            if full_projection['uproj_depth'] is not None:
                output['uproj_depth'] = torch.from_numpy(full_projection['uproj_depth']).float()

        return output


    def __getitem__(self, index):
        '''
        proj_feature_tensor: CxHxW
        proj_sem_label_tensor: HxW
        proj_mask_tensor: HxW
        '''
        if self.use_kpconv:
            return self.get_item_for_kpconv(index)

        pointcloud, sem_label, inst_label = self.dataset.loadDataByIndex(index)
        if self.is_train:
            pointcloud = self.augmentor.doAugmentation(pointcloud)

        mapped_sem_label = self.dataset.labelMapping(sem_label)
        full_projection = self._project_pointcloud(pointcloud, mapped_sem_label)
        if self.use_mcf:
            subclouds = self._build_subcloud_projections(pointcloud, mapped_sem_label)
            features, labels, mask_bool, _ = self._apply_mcf(subclouds)
        else:
            features = full_projection['features']
            labels = full_projection['labels']
            mask_bool = full_projection['mask'].bool()

        range_tensor = features[0].clone()

        if self.return_uproj:
            mask_float = mask_bool.float()
            sem_label_tensor = torch.from_numpy(mapped_sem_label).long()
            uproj_x = full_projection['uproj_x_idx']
            uproj_y = full_projection['uproj_y_idx']
            uproj_depth = full_projection['uproj_depth']
            return (
                features,
                labels,
                mask_float,
                range_tensor,
                torch.from_numpy(uproj_x).long() if uproj_x is not None else None,
                torch.from_numpy(uproj_y).long() if uproj_y is not None else None,
                torch.from_numpy(uproj_depth).float() if uproj_depth is not None else None,
                sem_label_tensor
            )

        proj_tensor = self._normalize_and_stack(features, labels, mask_bool)

        if self.is_train and np.random.uniform() < self.p_rangemix:
            mix_index = torch.randint(0, len(self.dataset), (1,)).item()
            pointcloud_b, sem_label_b, _ = self.dataset.loadDataByIndex(mix_index)
            pointcloud_b = self.augmentor.doAugmentation(pointcloud_b)
            mapped_sem_label_b = self.dataset.labelMapping(sem_label_b)

            subclouds_b = self._build_subcloud_projections(pointcloud_b, mapped_sem_label_b)
            features_b, labels_b, mask_bool_b, _ = self._apply_mcf(subclouds_b)
            proj_tensor_b = self._normalize_and_stack(features_b, labels_b, mask_bool_b)

            proj_feature_mixed, proj_label_mixed = augmentor.Augmentor.rangemix(
                proj_tensor[:5], proj_tensor[5], proj_tensor_b[:5], proj_tensor_b[5])

            mask_channel = proj_tensor[6].unsqueeze(0)
            proj_tensor = torch.cat(
                (proj_feature_mixed,
                 proj_label_mixed.unsqueeze(0),
                 mask_channel), dim=0)

        proj_tensor = self.aug_ops(proj_tensor)

        return proj_tensor[0:5], proj_tensor[5], proj_tensor[6]

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)


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
    os.makedirs(save_path, exist_ok=True)
    if color_map is None:
        color_map = load_color_map()   # BGR

    tensor_np = proj_tensor.cpu().numpy()  # [7, H, W]  (C,H,W)
    H, W = tensor_np.shape[1], tensor_np.shape[2]
    range_ch   = tensor_np[0]
    x_ch, y_ch, z_ch = tensor_np[1], tensor_np[2], tensor_np[3]
    intensity  = tensor_np[4]
    sem_labels = tensor_np[5].astype(np.int32)      # already masked in your pipeline
    fused_mask = (tensor_np[6] > 0.5)               # fused, post-MCF mask

    def normalize_to_uint8(data):
        dmin, dmax = data.min(), data.max()
        if dmax - dmin > 1e-6:
            return ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
        return np.zeros_like(data, dtype=np.uint8)

    def apply_color_map_int(labels_2d, cmap):
        colored = np.zeros((H, W, 3), dtype=np.uint8)
        for lid, bgr in cmap.items():
            colored[labels_2d == lid] = bgr
        return colored

    # --- Save mask (should be ~0.71–0.76 white)
    cv2.imwrite(f'{save_path}/sample_{index:06d}_mask.png', (fused_mask * 255).astype(np.uint8))

    # --- Semantic label VIS *with mask applied*
    # 1) Background color for unmasked area (e.g., gray)
    bg = np.full((H, W, 3), 80, dtype=np.uint8)
    # 2) Colorized labels for masked area only
    sem_color = apply_color_map_int(sem_labels, color_map)
    sem_vis = bg.copy()
    sem_vis[fused_mask] = sem_color[fused_mask]
    cv2.imwrite(f'{save_path}/sample_{index:06d}_semantic_label.png', sem_vis)

    # (Optional) Make a transparent PNG instead:
    # alpha = (fused_mask.astype(np.uint8) * 255)
    # sem_rgba = np.dstack([sem_color[..., ::-1], alpha])  # if you want RGBA
    # cv2.imwrite(..., sem_rgba)

    # --- Continuous channels: they’re already masked upstream; just save
    for name, ch in [('range', range_ch), ('x', x_ch), ('y', y_ch), ('z', z_ch), ('intensity', intensity)]:
        colored = cv2.applyColorMap(normalize_to_uint8(ch), cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(f'{save_path}/sample_{index:06d}_{name}.png', colored)

    # --- XYZ composite (debug)
    xyz = np.stack([normalize_to_uint8(x_ch), normalize_to_uint8(y_ch), normalize_to_uint8(z_ch)], axis=-1)
    cv2.imwrite(f'{save_path}/sample_{index:06d}_xyz_composite.png', xyz)

    # (Point cloud / labels saving unchanged…)



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
