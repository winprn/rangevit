# Copyright 2023 - Valeo Comfort and Driving Assistance
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


class BEVProjection:
    """
    Projects 3D point clouds to Bird's Eye View (BEV) raster representation.

    This creates a top-down view by discretizing points into a 2D grid
    and encoding features like height, intensity, and density.
    """

    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), grid_size=(256, 256),
                 feature_channels=8, height_slices=5):
        """
        Args:
            x_range: (min, max) range in meters for x-axis (forward)
            y_range: (min, max) range in meters for y-axis (left-right)
            grid_size: (H, W) output grid resolution
            feature_channels: number of output channels
            height_slices: number of vertical slices for multi-height encoding
        """
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.grid_h, self.grid_w = grid_size
        self.feature_channels = feature_channels
        self.height_slices = height_slices

        self.x_resolution = (self.x_max - self.x_min) / self.grid_h
        self.y_resolution = (self.y_max - self.y_min) / self.grid_w

    def doProjection(self, pointcloud):
        """
        Project point cloud to BEV grid.

        Args:
            pointcloud: np.array of shape (N, 4+) with columns [x, y, z, intensity, ...]

        Returns:
            bev_features: np.array of shape (feature_channels, grid_h, grid_w)
                Channels encode:
                - [0]: max height
                - [1]: mean height
                - [2]: min height
                - [3]: height variance
                - [4]: intensity mean
                - [5]: intensity max
                - [6]: point density (log scale)
                - [7]: occupancy mask
        """
        points_xyz = pointcloud[:, :3]
        intensity = pointcloud[:, 3] if pointcloud.shape[1] > 3 else np.ones(len(pointcloud))

        # Convert to grid indices
        x_img = np.floor((points_xyz[:, 0] - self.x_min) / self.x_resolution).astype(np.int32)
        y_img = np.floor((points_xyz[:, 1] - self.y_min) / self.y_resolution).astype(np.int32)

        # Filter points outside the grid
        valid_mask = (
            (x_img >= 0) & (x_img < self.grid_h) &
            (y_img >= 0) & (y_img < self.grid_w)
        )

        x_img = x_img[valid_mask]
        y_img = y_img[valid_mask]
        z_vals = points_xyz[valid_mask, 2]
        intensity_vals = intensity[valid_mask]

        # Initialize BEV grid
        bev_features = np.zeros((self.feature_channels, self.grid_h, self.grid_w), dtype=np.float32)

        if len(x_img) == 0:
            return bev_features

        # Aggregate features per grid cell
        # We'll use a dictionary to accumulate per-cell statistics
        from collections import defaultdict
        cell_data = defaultdict(lambda: {'z': [], 'intensity': []})

        for i in range(len(x_img)):
            key = (x_img[i], y_img[i])
            cell_data[key]['z'].append(z_vals[i])
            cell_data[key]['intensity'].append(intensity_vals[i])

        # Fill BEV grid with aggregated features
        for (xi, yi), data in cell_data.items():
            z_list = np.array(data['z'])
            i_list = np.array(data['intensity'])

            # Height statistics
            bev_features[0, xi, yi] = np.max(z_list)  # max height
            bev_features[1, xi, yi] = np.mean(z_list)  # mean height
            bev_features[2, xi, yi] = np.min(z_list)  # min height
            bev_features[3, xi, yi] = np.var(z_list) if len(z_list) > 1 else 0.0  # height variance

            # Intensity statistics
            bev_features[4, xi, yi] = np.mean(i_list)  # mean intensity
            bev_features[5, xi, yi] = np.max(i_list)  # max intensity

            # Density (log scale for better numeric range)
            bev_features[6, xi, yi] = np.log1p(len(z_list))  # log(1 + count)

            # Occupancy
            bev_features[7, xi, yi] = 1.0  # occupied

        return bev_features

    def doProjectionWithLabels(self, pointcloud, labels):
        """
        Project point cloud with semantic labels to BEV grid.

        Args:
            pointcloud: np.array of shape (N, 4+)
            labels: np.array of shape (N,) with semantic class labels

        Returns:
            bev_features: BEV feature grid
            bev_labels: np.array of shape (grid_h, grid_w) with majority-vote labels per cell
        """
        bev_features = self.doProjection(pointcloud)

        points_xyz = pointcloud[:, :3]

        # Convert to grid indices
        x_img = np.floor((points_xyz[:, 0] - self.x_min) / self.x_resolution).astype(np.int32)
        y_img = np.floor((points_xyz[:, 1] - self.y_min) / self.y_resolution).astype(np.int32)

        # Filter points outside the grid
        valid_mask = (
            (x_img >= 0) & (x_img < self.grid_h) &
            (y_img >= 0) & (y_img < self.grid_w)
        )

        x_img = x_img[valid_mask]
        y_img = y_img[valid_mask]
        labels_valid = labels[valid_mask]

        # Initialize label grid
        bev_labels = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)

        if len(x_img) == 0:
            return bev_features, bev_labels

        # Majority vote for labels per cell
        from collections import defaultdict, Counter
        cell_labels = defaultdict(list)

        for i in range(len(x_img)):
            key = (x_img[i], y_img[i])
            cell_labels[key].append(labels_valid[i])

        for (xi, yi), label_list in cell_labels.items():
            # Get most common label
            label_counter = Counter(label_list)
            majority_label = label_counter.most_common(1)[0][0]
            bev_labels[xi, yi] = majority_label

        return bev_features, bev_labels
