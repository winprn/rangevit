"""
Loss functions for semantic segmentation, including Weighted Focal Loss for WPD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for handling class imbalance.

    Used with WPD augmentation to reflect the new class balance created by
    weighted paste and drop operations.

    Formula: L = -α_i * (1 - p_t)^γ * log(p_t)
    where:
        - α_i: per-class weight (from WPD statistics)
        - γ: focusing parameter (typically 2.0)
        - p_t: predicted probability for the true class
    """

    def __init__(self, alpha=None, gamma=2.0, ignore_index=0, reduction='mean'):
        """
        Args:
            alpha: Class weights as tensor [num_classes] or None for uniform weights
            gamma: Focusing parameter (default: 2.0)
            ignore_index: Label value to ignore in loss computation (default: 0)
            reduction: 'mean', 'sum', or 'none'
        """
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C, H, W] or [N, C] logits
            targets: [N, H, W] or [N] ground truth labels

        Returns:
            Scalar loss value
        """
        # Flatten spatial dimensions if present
        if inputs.dim() > 2:
            N, C, H, W = inputs.shape
            inputs = inputs.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
            inputs = inputs.view(-1, C)  # [N*H*W, C]
            targets = targets.view(-1)  # [N*H*W]

        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)  # [N*H*W, C]

        # Get probabilities for true class
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))  # [N*H*W, C]
        targets_one_hot = targets_one_hot.float()
        pt = (probs * targets_one_hot).sum(dim=1)  # [N*H*W]

        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Compute cross entropy
        log_pt = F.log_softmax(inputs, dim=1)  # [N*H*W, C]
        ce_loss = -(log_pt * targets_one_hot).sum(dim=1)  # [N*H*W]

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # [N*H*W]
            focal_loss = alpha_t * focal_loss

        # Handle ignore_index
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            focal_loss = focal_loss * mask

            if self.reduction == 'mean':
                return focal_loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return focal_loss.sum()

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for semantic segmentation with WPD augmentation.

    Combines:
        1. Cross-entropy loss
        2. Weighted focal loss
        3. Optional Lovász-Softmax loss (for IoU optimization)
        4. Optional boundary loss
    """

    def __init__(self, alpha=None, gamma=2.0, ignore_index=0,
                 ce_weight=1.0, focal_weight=1.0, lovasz_weight=0.0, boundary_weight=0.0):
        """
        Args:
            alpha: Class weights for focal loss
            gamma: Focusing parameter for focal loss
            ignore_index: Label to ignore
            ce_weight: Weight for cross-entropy loss (λ1)
            focal_weight: Weight for focal loss (λ2)
            lovasz_weight: Weight for Lovász loss (λ3)
            boundary_weight: Weight for boundary loss (λ4)
        """
        super(CombinedLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.focal_loss = WeightedFocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)

        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C, H, W] logits
            targets: [N, H, W] ground truth labels

        Returns:
            Total loss and dict of component losses
        """
        losses = {}
        total_loss = 0

        # Cross-entropy loss
        if self.ce_weight > 0:
            ce = self.ce_loss(inputs, targets)
            losses['ce_loss'] = ce
            total_loss += self.ce_weight * ce

        # Focal loss
        if self.focal_weight > 0:
            focal = self.focal_loss(inputs, targets)
            losses['focal_loss'] = focal
            total_loss += self.focal_weight * focal

        # TODO: Add Lovász-Softmax loss if needed
        # TODO: Add boundary loss if needed

        losses['total_loss'] = total_loss
        return total_loss, losses


def load_wpd_loss_weights(wpd_stats_path, mode='semantic', device='cuda'):
    """
    Load class weights (alpha) from WPD statistics for use in focal loss.

    Args:
        wpd_stats_path: Path to wpd_stats.yaml
        mode: 'semantic' or 'panoptic'
        device: Device to place weights on

    Returns:
        Tensor of class weights [num_classes]
    """
    with open(wpd_stats_path, 'r') as f:
        stats = yaml.safe_load(f)

    alpha = np.array(stats['semantic']['alpha'])

    if mode == 'panoptic':
        beta = np.array(stats['panoptic']['beta'])
        alpha = alpha * beta

    # Normalize to [0, 1]
    alpha = alpha / (alpha.max() + 1e-6)

    return torch.tensor(alpha, dtype=torch.float32, device=device)


if __name__ == '__main__':
    # Test focal loss
    print("Testing Weighted Focal Loss...")

    # Create dummy data
    batch_size, num_classes, h, w = 2, 20, 64, 384
    inputs = torch.randn(batch_size, num_classes, h, w)
    targets = torch.randint(0, num_classes, (batch_size, h, w))

    # Test without weights
    loss_fn = WeightedFocalLoss(alpha=None, gamma=2.0, ignore_index=0)
    loss = loss_fn(inputs, targets)
    print(f"Loss without weights: {loss.item():.4f}")

    # Test with weights
    alpha = torch.rand(num_classes)
    loss_fn = WeightedFocalLoss(alpha=alpha, gamma=2.0, ignore_index=0)
    loss = loss_fn(inputs, targets)
    print(f"Loss with weights: {loss.item():.4f}")

    # Test combined loss
    print("\nTesting Combined Loss...")
    combined_loss_fn = CombinedLoss(alpha=alpha, gamma=2.0, ce_weight=1.0, focal_weight=1.0)
    total_loss, losses = combined_loss_fn(inputs, targets)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"  CE loss: {losses['ce_loss'].item():.4f}")
    print(f"  Focal loss: {losses['focal_loss'].item():.4f}")

    print("\nAll tests passed!")
