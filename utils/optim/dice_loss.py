import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multi-class soft Dice loss with ignore index support.
    """

    def __init__(self, ignore_index: int = 0, smooth: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        probs = torch.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        targets_clamped = targets.clone()
        if self.ignore_index is not None:
            targets_clamped = targets_clamped.clamp(min=0)

        target_one_hot = F.one_hot(
            targets_clamped.long(),
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * valid_mask
            target_one_hot = target_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        union = torch.sum(probs, dims) + torch.sum(target_one_hot, dims)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)

        if self.ignore_index is not None and self.ignore_index < num_classes:
            mask = torch.ones_like(dice_score, dtype=torch.bool)
            mask[self.ignore_index] = False
            dice_score = dice_score[mask]

        loss = 1.0 - dice_score
        return loss.mean()
