# RangeFormer Inference Script
# Supports RangePost strategy and KNN post-processing

import torch
import numpy as np
import os
import argparse
import yaml
from tqdm import tqdm

from models.rangeformer import create_rangeformer
from dataset.preprocess.projection import RangeProjection
from utils.postproc.knn import KNN


class RangeFormerInference:
    """
    Inference engine for RangeFormer with RangePost support.

    RangePost strategy:
    - Split point cloud into N subclouds
    - Rasterize and predict each independently
    - Merge predictions back to points
    - Optionally apply KNN post-processing
    """

    def __init__(self, config_path, checkpoint_path, device='cuda'):
        """
        Initialize inference engine.

        Args:
            config_path: Path to model config YAML
            checkpoint_path: Path to model checkpoint
            device: torch device
        """
        self.device = device

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Determine training view resolution (STR)
        image_size = self.config.get('image_size', [self.config['sensor']['proj_h'], self.config['sensor']['proj_w']])
        train_H, train_W = image_size

        # Create model
        model_config = {
            'H': train_H,
            'W': train_W,
            'num_classes': self.config['n_classes'],
            'depths': self.config['rangeformer']['depths'],
            'stage_channels': self.config['rangeformer']['stage_channels'],
            'heads': self.config['rangeformer']['heads'],
            'decoder_unify_ch': self.config['rangeformer'].get('decoder_unify_ch', 256),
            'mlp_ratio': self.config['rangeformer'].get('mlp_ratio', 4.0),
            'sr_ratios': self.config['rangeformer'].get('sr_ratios', [8, 4, 2, 1]),
        }

        self.model = create_rangeformer(model_config)
        self.model = self.model.to(device)

        # Load checkpoint
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print('Model loaded and ready for inference')

        # Create range projector
        sensor_config = self.config['sensor']
        self.rasterizer = RangeProjection(
            fov_up=sensor_config['fov_up'],
            fov_down=sensor_config['fov_down'],
            fov_left=sensor_config.get('fov_left', -180.0),
            fov_right=sensor_config.get('fov_right', 180.0),
            proj_w=sensor_config['proj_w'],
            proj_h=sensor_config['proj_h']
        )

        self.full_width = sensor_config['proj_w']
        self.full_height = sensor_config['proj_h']
        self.view_width = train_W

        # STR configuration
        str_config = self.config.get('str', {})
        self.str_enabled = bool(str_config.get('enabled', False))
        self.str_num_views = int(str_config.get('num_views', max(1, self.full_width // max(1, self.view_width))))
        self.str_inference_views = int(str_config.get('inference_views', self.str_num_views))

        # Inference settings
        infer_config = self.config.get('inference', {})
        self.use_range_post = infer_config.get('use_range_post', True)
        default_views = self.str_inference_views if self.str_enabled else max(1, self.full_width // max(1, self.view_width))
        self.num_views = default_views
        self.num_sub = infer_config.get('num_sub', default_views)
        self.use_knn = infer_config.get('use_knn', True)

        if self.use_knn:
            knn_params = infer_config.get('knn_params', {})
            self.knn = KNN(
                k=knn_params.get('k', 7),
                search=knn_params.get('search', 7),
                sigma=knn_params.get('sigma', 1.0),
                cutoff=knn_params.get('cutoff', 2.0)
            )
            print(f'KNN post-processing enabled')

        print(f'RangePost: {self.use_range_post} (num_sub={self.num_sub})')

        # Normalization params
        self.img_mean = torch.tensor(sensor_config['img_mean'], dtype=torch.float, device=device)
        self.img_stds = torch.tensor(sensor_config['img_stds'], dtype=torch.float, device=device)

        # Ensure we have 6 values for normalization
        if len(self.img_mean) == 5:
            self.img_mean = torch.cat([self.img_mean, torch.tensor([0.5], device=device)])
            self.img_stds = torch.cat([self.img_stds, torch.tensor([0.5], device=device)])

    def project_to_range_image(self, pointcloud):
        """
        Project point cloud to 6-channel range image.

        Args:
            pointcloud: (N, 4+) numpy array [x, y, z, intensity, ...]

        Returns:
            rv: (6, H, W) range image [x, y, z, depth, intensity, existence]
            proj_idx: (H, W) point indices
        """
        proj_pc, proj_range, proj_idx, proj_mask = self.rasterizer.doProjection(pointcloud)

        H, W = proj_range.shape
        rv = np.zeros((6, H, W), dtype=np.float32)
        rv[0] = proj_pc[:, :, 0]  # x
        rv[1] = proj_pc[:, :, 1]  # y
        rv[2] = proj_pc[:, :, 2]  # z
        rv[3] = proj_range         # depth
        rv[4] = proj_pc[:, :, 3] if proj_pc.shape[2] > 3 else 0.0  # intensity
        rv[5] = proj_mask          # existence

        return rv, proj_idx

    def predict_range_image(self, rv):
        """
        Run model on range image.

        Args:
            rv: (6, H, W) range image numpy array

        Returns:
            preds: (H, W) predicted class IDs
        """
        # Convert to tensor and normalize
        rv_tensor = torch.from_numpy(rv).float().to(self.device)
        rv_tensor = (rv_tensor - self.img_mean[:, None, None]) / self.img_stds[:, None, None]
        rv_tensor = rv_tensor * rv_tensor[5:6]  # Mask with existence channel

        # Add batch dimension
        rv_tensor = rv_tensor.unsqueeze(0)  # (1, 6, H, W)

        # Forward pass
        with torch.no_grad():
            logits, _ = self.model(rv_tensor)
            preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        return preds

    def map_to_points(self, pred_map, proj_idx, num_points, default_label=0):
        """
        Map 2D predictions back to 3D points.

        Args:
            pred_map: (H, W) predicted class IDs
            proj_idx: (H, W) point indices
            num_points: total number of points
            default_label: label for unmapped points

        Returns:
            point_labels: (num_points,) per-point predictions
        """
        point_labels = np.ones(num_points, dtype=np.int32) * default_label
        H, W = pred_map.shape

        for r in range(H):
            for c in range(W):
                idx = proj_idx[r, c]
                if idx >= 0:
                    point_labels[idx] = int(pred_map[r, c])

        return point_labels

    def _get_view_slice(self, view_idx: int, total_width: int):
        """
        Compute [start, end) for a given STR view index.
        """
        if self.view_width <= 0 or self.view_width >= total_width:
            return 0, total_width

        start = view_idx * self.view_width
        end = start + self.view_width
        if end > total_width:
            end = total_width
        return start, end

    def predict(self, pointcloud):
        """
        Standard single-pass prediction.

        Args:
            pointcloud: (N, 4+) numpy array

        Returns:
            predictions: (N,) per-point class IDs
        """
        N = pointcloud.shape[0]

        # Project to range image
        rv, proj_idx = self.project_to_range_image(pointcloud)

        total_width = rv.shape[2]

        # Predict (handle STR multi-view slicing if needed)
        if self.num_views <= 1 or self.view_width >= total_width:
            pred_map = self.predict_range_image(rv)
        else:
            pred_map = np.zeros((rv.shape[1], total_width), dtype=np.int32)
            for view_idx in range(self.num_views):
                start, end = self._get_view_slice(view_idx, total_width)
                if end <= start:
                    continue
                rv_view = rv[:, :, start:end]
                pred_view = self.predict_range_image(rv_view)
                pred_map[:, start:end] = pred_view

        # Map back to points
        predictions = self.map_to_points(pred_map, proj_idx, N)

        # KNN post-processing
        if self.use_knn:
            predictions = self.knn(pointcloud[:, :3], predictions)

        return predictions

    def predict_range_post(self, pointcloud):
        """
        RangePost prediction strategy.

        Args:
            pointcloud: (N, 4+) numpy array

        Returns:
            predictions: (N,) per-point class IDs
        """
        N = pointcloud.shape[0]

        final_pred = np.zeros(N, dtype=np.int32)
        if N == 0:
            return final_pred

        # Paper RangePost: split scan into equal-interval sub-clouds (every Z-th point)
        for view_idx in range(self.num_sub):
            idxs = np.arange(view_idx, N, self.num_sub, dtype=np.int32)
            if idxs.size == 0:
                continue

            subcloud = pointcloud[idxs]

            knn_state = self.use_knn
            self.use_knn = False
            sub_pred = self.predict(subcloud)
            self.use_knn = knn_state
            final_pred[idxs] = sub_pred

        # KNN post-processing
        if self.use_knn:
            final_pred = self.knn(pointcloud[:, :3], final_pred)

        return final_pred

    def infer(self, pointcloud):
        """
        Main inference method (chooses strategy based on config).

        Args:
            pointcloud: (N, 4+) numpy array

        Returns:
            predictions: (N,) per-point class IDs
        """
        if self.use_range_post:
            return self.predict_range_post(pointcloud)
        else:
            return self.predict(pointcloud)


def main():
    parser = argparse.ArgumentParser(description='RangeFormer Inference')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input point cloud (.npy or .bin)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions (default: input_pred.npy)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--no-range-post', action='store_true',
                        help='Disable RangePost strategy')
    parser.add_argument('--no-knn', action='store_true',
                        help='Disable KNN post-processing')

    args = parser.parse_args()

    # Create inference engine
    inference = RangeFormerInference(args.config, args.checkpoint, device=args.device)

    # Override config if specified
    if args.no_range_post:
        inference.use_range_post = False
    if args.no_knn:
        inference.use_knn = False

    # Load point cloud
    print(f'Loading point cloud from {args.input}')
    if args.input.endswith('.npy'):
        pointcloud = np.load(args.input)
    elif args.input.endswith('.bin'):
        pointcloud = np.fromfile(args.input, dtype=np.float32).reshape(-1, 4)
    else:
        raise ValueError('Input must be .npy or .bin file')

    print(f'Point cloud shape: {pointcloud.shape}')

    # Run inference
    print('Running inference...')
    predictions = inference.infer(pointcloud)

    # Save predictions
    if args.output is None:
        args.output = args.input.replace('.npy', '_pred.npy').replace('.bin', '_pred.npy')

    np.save(args.output, predictions)
    print(f'Predictions saved to {args.output}')
    print(f'Unique labels: {np.unique(predictions)}')


if __name__ == '__main__':
    main()
