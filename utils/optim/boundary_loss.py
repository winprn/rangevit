import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss using Laplacian edge maps.
    """

    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index
        kernel = torch.tensor(
            [[[-1.0, -1.0, -1.0],
              [-1.0, 8.0, -1.0],
              [-1.0, -1.0, -1.0]]],
            dtype=torch.float32
        )
        self.register_buffer('laplacian_kernel', kernel)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        device = probs.device
        kernel = self.laplacian_kernel.to(device).repeat(num_classes, 1, 1, 1)

        targets_clamped = targets.clone()
        if self.ignore_index is not None:
            targets_clamped = targets_clamped.clamp(min=0)

        target_one_hot = F.one_hot(
            targets_clamped.long(),
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float().to(device)

        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).unsqueeze(1).float().to(device)
            probs = probs * valid_mask
            target_one_hot = target_one_hot * valid_mask

        pred_boundary = F.conv2d(probs, kernel, padding=1, groups=num_classes).abs()
        target_boundary = F.conv2d(target_one_hot, kernel, padding=1, groups=num_classes).abs()

        pred_boundary = torch.sigmoid(pred_boundary)
        target_boundary = torch.clamp(target_boundary, 0.0, 1.0)

        loss = F.binary_cross_entropy(pred_boundary, target_boundary)
        return loss
