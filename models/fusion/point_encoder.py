# Copyright 2024 - Fusion Extension for RangeViT
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

"""
Point MLP Encoder for PointFusion architecture.

Encodes raw point features (xyz, intensity, cluster offset) into learned representations.
"""

import torch
import torch.nn as nn

__all__ = ['PointMLPEncoder']


class PointMLPEncoder(nn.Module):
    """
    Encode raw point features into learned representations.

    Input features (7 channels):
        - xyz (3): 3D coordinates
        - intensity (1): LiDAR reflectance
        - cluster_offset (3): relative position to voxel center

    Args:
        in_channels: Number of input channels (default: 7)
        hidden_dim: Output feature dimension (default: 256)
        if_dist: Whether to use SyncBatchNorm for distributed training
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 256,
        if_dist: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        BatchNorm = nn.SyncBatchNorm if if_dist else nn.BatchNorm1d

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            BatchNorm(64),
            nn.ReLU(inplace=False),

            nn.Linear(64, 128),
            BatchNorm(128),
            nn.ReLU(inplace=False),

            nn.Linear(128, hidden_dim),
            BatchNorm(hidden_dim),
            nn.ReLU(inplace=False),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """
        Encode point features.

        Args:
            point_features: [N, in_channels] raw point features

        Returns:
            [N, hidden_dim] encoded point features
        """
        return self.mlp(point_features)
