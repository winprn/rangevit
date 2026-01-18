# Copyright 2026 - RangeViT-Fusion
# HARP-NeXt style prediction head

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """
    Prediction head that combines pixel and point features for per-point classification.

    Architecture:
        concat(mapped_pixel, point) -> Linear(2D, D) -> BN -> ReLU
                                    -> Linear(D, D//2) -> BN -> ReLU
                                    -> Linear(D//2, n_classes)

    Args:
        d_model: Feature dimension
        n_classes: Number of semantic classes
    """

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, n_classes),
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
            mapped_pixel_feats: (N, D) pixel features mapped to points
            point_feats: (N, D) final point features

        Returns:
            logits: (N, n_classes) per-point class logits
        """
        combined = torch.cat([mapped_pixel_feats, point_feats], dim=1)
        return self.mlp(combined)
