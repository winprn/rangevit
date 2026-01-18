# Copyright 2026 - RangeViT-Fusion
# Adapted from HARP-NeXt FeaturesEncoder

import torch
import torch.nn as nn


class FeaturesEncoder(nn.Module):
    """
    Encodes raw point attributes (xyz, intensity, range) into feature vectors.

    Architecture: Linear(in, 64) -> BN -> ReLU -> Linear(64, 128) -> BN -> ReLU -> Linear(128, d_model) -> BN -> ReLU

    Args:
        in_channels: Number of input channels per point (default: 5 for xyz + intensity + range)
        d_model: Output feature dimension (should match ViT d_model)
    """

    def __init__(self, in_channels: int = 5, d_model: int = 384):
        super().__init__()

        self.in_channels = in_channels
        self.d_model = d_model

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, point_attrs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_attrs: (N, in_channels) raw point attributes

        Returns:
            point_feats: (N, d_model) encoded point features
        """
        return self.mlp(point_attrs)
