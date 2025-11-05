"""
Implementation of the Nearest Neighbors Range Interpolation (NNRI) post-processing
algorithm described in the FLARE paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NNRI(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        alpha: float,
        range_mean: float,
        range_std: float,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("NNRI kernel size must be odd.")
        if range_std <= 0:
            raise ValueError("Range standard deviation must be positive.")

        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.alpha = float(alpha)

        self.register_buffer("range_mean", torch.tensor(float(range_mean)))
        self.register_buffer("range_std", torch.tensor(float(range_std)))

    def forward(
        self,
        proj_range: torch.Tensor,
        unproj_range: torch.Tensor,
        proj_scores: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            proj_range: Tensor with shape (H, W) or (N, H, W) containing range image(s).
            unproj_range: Tensor with shape (P,) containing range values for all 3D points.
            proj_scores: Softmax scores with shape (C, H, W) or (N, C, H, W).
            px: Tensor with shape (P,) containing the x image coordinates of the 3D points.
            py: Tensor with shape (P,) containing the y image coordinates of the 3D points.

        Returns:
            Tensor with shape (P,) containing predicted labels for the 3D points.
        """
        if proj_scores.dim() == 3:
            proj_scores = proj_scores.unsqueeze(0)
        if proj_range.dim() == 2:
            proj_range = proj_range.unsqueeze(0)

        if proj_scores.size(0) != proj_range.size(0):
            raise ValueError(
                "proj_scores and proj_range must have the same number of stacked projections."
            )

        if proj_scores.shape[-2:] != proj_range.shape[-2:]:
            raise ValueError("proj_scores and proj_range must share the same spatial size.")

        device = proj_scores.device
        batch = proj_scores.size(0)
        n_classes = proj_scores.size(1)
        height, width = proj_scores.shape[-2:]

        px = px.long()
        py = py.long()

        idx = (py * width + px).long()
        idx = idx.to(device=device)

        proj_range = proj_range.to(device=device)
        proj_scores = proj_scores.to(device=device)
        unproj_range = unproj_range.to(device=device)

        # Prepare unfold views for scores and ranges.
        kernel_elem = self.kernel_size * self.kernel_size
        scores_unfold = F.unfold(
            proj_scores.view(batch * n_classes, 1, height, width),
            kernel_size=self.kernel_size,
            padding=self.pad,
        )
        scores_unfold = scores_unfold.view(batch, n_classes * kernel_elem, height * width)

        range_unfold = F.unfold(
            proj_range.unsqueeze(1),
            kernel_size=self.kernel_size,
            padding=self.pad,
        )

        gather_idx = idx.view(1, 1, -1).expand(batch, scores_unfold.size(1), -1)
        neighbor_scores = torch.gather(scores_unfold, 2, gather_idx)
        neighbor_scores = neighbor_scores.view(batch, n_classes, kernel_elem, -1)

        gather_idx_r = idx.view(1, 1, -1).expand(batch, range_unfold.size(1), -1)
        neighbor_ranges = torch.gather(range_unfold, 2, gather_idx_r)
        neighbor_ranges = neighbor_ranges.view(batch, kernel_elem, -1)

        center_offset = kernel_elem // 2
        center_scores = neighbor_scores[:, :, center_offset, :]

        unproj_range = unproj_range.view(1, 1, -1)
        rel_depth = torch.abs(neighbor_ranges - unproj_range)

        cutoff = torch.exp(
            -(unproj_range - self.range_mean) / (self.range_std + 1e-6)
        ) * self.alpha

        # Clamp relative depth with adaptive threshold and normalize to [0, 1].
        clamped_rel = torch.minimum(rel_depth, cutoff)
        normalized_rel = clamped_rel / (cutoff + 1e-6)

        # Invalid neighbors (negative range) should not contribute.
        invalid_mask = neighbor_ranges < 0
        weights = 1.0 - normalized_rel
        weights = torch.where(invalid_mask, torch.zeros_like(weights), weights)

        # Weighted aggregation of neighborhood softmax scores.
        weights = weights.unsqueeze(1)  # (batch, 1, K, P)
        weighted_scores = (neighbor_scores * weights).sum(dim=2)  # (batch, C, P)
        total_weights = weights.sum(dim=2).squeeze(1)  # (batch, P)

        # Aggregate across stacked projections.
        weighted_scores = weighted_scores.sum(dim=0)
        total_weights = total_weights.sum(dim=0)

        valid_mask = total_weights > 1e-6
        normalized_scores = torch.zeros_like(weighted_scores)
        normalized_scores[:, valid_mask] = (
            weighted_scores[:, valid_mask] / total_weights[valid_mask].unsqueeze(0)
        )

        # Fallback to center scores whenever no valid neighbors are available.
        fallback_scores = center_scores.mean(dim=0)
        normalized_scores[:, ~valid_mask] = fallback_scores[:, ~valid_mask]

        labels = torch.argmax(normalized_scores, dim=0).long()
        return labels
