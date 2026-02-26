import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    def __init__(self, ignore_index=0, horizontal_wrap=True, eps=1e-6):
        super(BoundaryLoss, self).__init__()
        self.ignore_index = ignore_index
        self.horizontal_wrap = horizontal_wrap
        self.eps = eps

    def __shift_label_and_valid(self, label: torch.Tensor, valid_mask: torch.Tensor, dy: int, dx: int):
        shifted_label = label
        shifted_valid = valid_mask

        # Horizontal shift (cyclic wrap for LiDAR azimuth).
        if dx != 0 and self.horizontal_wrap:
            shifted_label = torch.roll(shifted_label, shifts=dx, dims=2)
            shifted_valid = torch.roll(shifted_valid, shifts=dx, dims=2)
        elif dx != 0:
            if dx > 0:
                shifted_label = F.pad(shifted_label[:, :, :-dx], (dx, 0), value=0)
                shifted_valid = F.pad(shifted_valid[:, :, :-dx], (dx, 0), value=False)
            else:
                k = -dx
                shifted_label = F.pad(shifted_label[:, :, k:], (0, k), value=0)
                shifted_valid = F.pad(shifted_valid[:, :, k:], (0, k), value=False)

        # Vertical shift (padding only, no wrap).
        if dy > 0:
            shifted_label = F.pad(shifted_label[:, :-dy, :], (0, 0, dy, 0), value=0)
            shifted_valid = F.pad(shifted_valid[:, :-dy, :], (0, 0, dy, 0), value=False)
        elif dy < 0:
            k = -dy
            shifted_label = F.pad(shifted_label[:, k:, :], (0, 0, 0, k), value=0)
            shifted_valid = F.pad(shifted_valid[:, k:, :], (0, 0, 0, k), value=False)

        return shifted_label, shifted_valid

    def __shift_prob_and_valid(self, prob: torch.Tensor, valid_mask: torch.Tensor, dy: int, dx: int):
        shifted_prob = prob
        shifted_valid = valid_mask

        # Horizontal shift.
        if dx != 0 and self.horizontal_wrap:
            shifted_prob = torch.roll(shifted_prob, shifts=dx, dims=3)
            shifted_valid = torch.roll(shifted_valid, shifts=dx, dims=2)
        elif dx != 0:
            if dx > 0:
                shifted_prob = F.pad(shifted_prob[:, :, :, :-dx], (dx, 0), value=0.0)
                shifted_valid = F.pad(shifted_valid[:, :, :-dx], (dx, 0), value=False)
            else:
                k = -dx
                shifted_prob = F.pad(shifted_prob[:, :, :, k:], (0, k), value=0.0)
                shifted_valid = F.pad(shifted_valid[:, :, k:], (0, k), value=False)

        # Vertical shift.
        if dy > 0:
            shifted_prob = F.pad(shifted_prob[:, :, :-dy, :], (0, 0, dy, 0), value=0.0)
            shifted_valid = F.pad(shifted_valid[:, :-dy, :], (0, 0, dy, 0), value=False)
        elif dy < 0:
            k = -dy
            shifted_prob = F.pad(shifted_prob[:, :, k:, :], (0, 0, 0, k), value=0.0)
            shifted_valid = F.pad(shifted_valid[:, k:, :], (0, 0, 0, k), value=False)

        return shifted_prob, shifted_valid

    def __compute_boundary_mask(self, label: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        boundary = torch.zeros_like(label, dtype=torch.bool)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted_label, shifted_valid = self.__shift_label_and_valid(label, valid_mask, dy, dx)
            diff = (label != shifted_label) & valid_mask & shifted_valid
            boundary = boundary | diff
        return boundary.float()

    def __compute_pred_boundary_prob(self, prob: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        sims = []
        valid_pairs = []

        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted_prob, shifted_valid = self.__shift_prob_and_valid(prob, valid_mask, dy, dx)
            valid_pair = valid_mask & shifted_valid
            sim = (prob * shifted_prob).sum(dim=1)
            sim = torch.where(valid_pair, sim, torch.ones_like(sim))
            sims.append(sim)
            valid_pairs.append(valid_pair)

        sims = torch.stack(sims, dim=0)
        valid_pairs = torch.stack(valid_pairs, dim=0)

        min_sim, _ = sims.min(dim=0)
        has_valid_neighbor = valid_pairs.any(dim=0)
        boundary_prob = torch.where(has_valid_neighbor, 1.0 - min_sim, torch.zeros_like(min_sim))
        return (boundary_prob * valid_mask.float()).clamp(0.0, 1.0)

    def forward(self, prob: torch.Tensor, label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Public API: compute boundary mIoU-style loss.
        Inputs:
        - prob: class probabilities, shape (B, C, H, W)
        - label: integer labels, shape (B, H, W)
        - mask: validity mask, shape (B, H, W)
        """
        valid_mask = (mask > 0)
        if self.ignore_index is not None:
            valid_mask = valid_mask & (label != self.ignore_index)

        if not valid_mask.any():
            return torch.zeros([], device=prob.device, dtype=prob.dtype)

        target_boundary = self.__compute_boundary_mask(label, valid_mask)
        pred_boundary = self.__compute_pred_boundary_prob(prob, valid_mask)

        dims = (1, 2)
        intersection = (pred_boundary * target_boundary).sum(dim=dims)
        union = (pred_boundary + target_boundary - pred_boundary * target_boundary).sum(dim=dims)

        non_empty = union > 0
        if not non_empty.any():
            return torch.zeros([], device=prob.device, dtype=prob.dtype)

        iou = intersection[non_empty] / (union[non_empty] + self.eps)
        return 1.0 - iou.mean()
