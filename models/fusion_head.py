# Copyright 2026 - RangeViT-Fusion
# HARP-NeXt style prediction head

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """
    Prediction head that combines pixel and point features for per-point classification.

    Architecture:
        concat(mapped_pixel, point) -> Linear(d_pixel+d_point, hidden) -> BN -> ReLU
                                    -> Linear(hidden, hidden//2) -> BN -> ReLU
                                    -> Linear(hidden//2, n_classes)

    Args:
        d_pixel: Pixel feature dimension (from decoder)
        d_point: Point feature dimension (from ViT)
        n_classes: Number of semantic classes
        hidden_dim: Hidden dimension (defaults to d_point)
    """

    def __init__(
        self,
        d_pixel: int,
        d_point: int,
        n_classes: int,
        hidden_dim: int = None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_point

        concat_dim = d_pixel + d_point

        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, n_classes),
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

    def forward(self, mapped_pixel_feats: torch.Tensor, point_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mapped_pixel_feats: (N, d_pixel) pixel features mapped to points
            point_feats: (N, d_point) final point features

        Returns:
            logits: (N, n_classes) per-point class logits
        """
        combined = torch.cat([mapped_pixel_feats, point_feats], dim=1)
        return self.mlp(combined)
