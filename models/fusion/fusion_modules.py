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
Fusion modules for multi-view feature fusion.

Implements:
- FusionMLP: Concatenation + MLP fusion at each fusion point
- PointTransform: Point feature transformation to match channel dimensions
"""

import torch
import torch.nn as nn

__all__ = ['FusionMLP', 'PointTransform']


class FusionMLP(nn.Module):
    """
    Concatenation + MLP fusion module.

    Takes features from range branch, voxel branch, and point transformation,
    concatenates them, and processes with MLP to produce fused point features.

    Args:
        in_channels_range: Number of input channels from range branch (R2P)
        in_channels_voxel: Number of input channels from voxel branch (V2P)
        in_channels_point: Number of input channels from point transform
        out_channels: Number of output channels for fused features
        hidden_ratio: Ratio for hidden layer dimension (hidden = out_channels * hidden_ratio)
        if_dist: Whether to use SyncBatchNorm for distributed training
        dropout_p: Dropout probability (0 for no dropout)
    """

    def __init__(
        self,
        in_channels_range: int,
        in_channels_voxel: int,
        in_channels_point: int,
        out_channels: int,
        hidden_ratio: float = 2.0,
        if_dist: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        self.in_channels_range = in_channels_range
        self.in_channels_voxel = in_channels_voxel
        self.in_channels_point = in_channels_point
        self.out_channels = out_channels

        total_in = in_channels_range + in_channels_voxel + in_channels_point
        hidden_dim = int(out_channels * hidden_ratio)

        layers = [
            nn.Linear(total_in, hidden_dim),
            nn.SyncBatchNorm(hidden_dim) if if_dist else nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
        ]

        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        layers.extend([
            nn.Linear(hidden_dim, out_channels),
            nn.SyncBatchNorm(out_channels) if if_dist else nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=False),
        ])

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        range_feats: torch.Tensor,
        voxel_feats: torch.Tensor,
        point_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features from range, voxel, and point representations.

        Args:
            range_feats: [N, C_r] features from range branch (via R2P)
            voxel_feats: [N, C_v] features from voxel branch (via V2P)
            point_feats: [N, C_p] features from point transform

        Returns:
            [N, C_out] fused point features
        """
        concat = torch.cat([range_feats, voxel_feats, point_feats], dim=1)
        return self.mlp(concat)


class PointTransform(nn.Module):
    """
    Point feature transformation module.

    Transforms point features to match the channel dimension at each fusion stage.
    Used to project raw point features or previous fused features to the target dimension.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        if_dist: Whether to use SyncBatchNorm for distributed training
        use_residual: Whether to use residual connection (requires in_channels == out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        if_dist: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = use_residual and (in_channels == out_channels)

        self.transform = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SyncBatchNorm(out_channels) if if_dist else nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=False),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform point features.

        Args:
            x: [N, C_in] input point features

        Returns:
            [N, C_out] transformed point features
        """
        out = self.transform(x)
        if self.use_residual:
            out = out + x
        return out


class FusionMLPWithResidual(nn.Module):
    """
    Fusion MLP with optional residual connection from point features.

    Similar to FusionMLP but adds the input point features as a residual
    to the output if dimensions match.

    Args:
        in_channels_range: Number of input channels from range branch
        in_channels_voxel: Number of input channels from voxel branch
        in_channels_point: Number of input channels from point transform
        out_channels: Number of output channels
        hidden_ratio: Ratio for hidden layer dimension
        if_dist: Whether to use SyncBatchNorm
        dropout_p: Dropout probability
    """

    def __init__(
        self,
        in_channels_range: int,
        in_channels_voxel: int,
        in_channels_point: int,
        out_channels: int,
        hidden_ratio: float = 2.0,
        if_dist: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        self.use_residual = (in_channels_point == out_channels)

        total_in = in_channels_range + in_channels_voxel + in_channels_point
        hidden_dim = int(out_channels * hidden_ratio)

        layers = [
            nn.Linear(total_in, hidden_dim),
            nn.SyncBatchNorm(hidden_dim) if if_dist else nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
        ]

        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        layers.extend([
            nn.Linear(hidden_dim, out_channels),
            nn.SyncBatchNorm(out_channels) if if_dist else nn.BatchNorm1d(out_channels),
        ])

        self.mlp = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        range_feats: torch.Tensor,
        voxel_feats: torch.Tensor,
        point_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features with optional residual connection.

        Args:
            range_feats: [N, C_r] features from range branch
            voxel_feats: [N, C_v] features from voxel branch
            point_feats: [N, C_p] features from point transform

        Returns:
            [N, C_out] fused point features
        """
        concat = torch.cat([range_feats, voxel_feats, point_feats], dim=1)
        out = self.mlp(concat)

        if self.use_residual:
            out = out + point_feats

        return self.relu(out)
