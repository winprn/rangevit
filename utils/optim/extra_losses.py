import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=n_classes).permute(0, 3, 1, 2).float()


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, n_classes: int, ignore_index: int = 0, eps: float = 1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if inputs.dim() != 4:
            raise ValueError(f"Invalid input shape, expected BxCxHxW, got {inputs.shape}")
        if inputs.shape[-2:] != targets.shape[-2:]:
            raise ValueError(f"Input and target spatial dims must match, got {inputs.shape} vs {targets.shape}")

        if mask is not None:
            mask = mask.unsqueeze(1)
            inputs = inputs * mask

        target_one_hot = _one_hot(targets, self.n_classes)
        if mask is not None:
            target_one_hot = target_one_hot * mask

        if self.ignore_index is not None and 0 <= self.ignore_index < inputs.shape[1]:
            keep = [i for i in range(inputs.shape[1]) if i != self.ignore_index]
            inputs = inputs[:, keep, :, :]
            target_one_hot = target_one_hot[:, keep, :, :]

        dims = (1, 2, 3)
        intersection = torch.sum(inputs * target_one_hot, dims)
        cardinality = torch.sum(inputs + target_one_hot, dims)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return torch.mean(1.0 - dice_score)


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """
    def __init__(self, theta0: int = 3):
        super().__init__()
        self.theta0 = theta0

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if pred.dim() != 4:
            raise ValueError(f"Invalid pred shape, expected BxCxHxW, got {pred.shape}")

        n, c, _, _ = pred.shape
        one_hot_gt = _one_hot(gt, c)

        if mask is not None:
            mask = mask.unsqueeze(1)
            pred = pred * mask
            one_hot_gt = one_hot_gt * mask

        gt_b = F.max_pool2d(
            1 - one_hot_gt,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        )
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        )
        pred_b -= 1 - pred

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)

        if mask is not None:
            mask = mask.view(n, 1, -1)
            gt_b = gt_b * mask
            pred_b = pred_b * mask

        eps = 1e-7
        P = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(pred_b, dim=2) + eps)
        R = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + eps)
        bf1 = 2 * P * R / (P + R + eps)
        return torch.mean(1 - bf1)
