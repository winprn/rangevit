import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_logits_and_targets(logits, targets, ignore_index):
    """
    logits: [B, C, H, W] or [B, C, N]
    targets: [B, H, W] or [B, N] with int labels
    returns:
        logits_flat: [P, C]
        targets_flat: [P]
    where P = number of valid (non-ignored) points.
    """
    if logits.dim() == 4:  # [B, C, H, W]
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
    elif logits.dim() == 3:  # [B, C, N]
        B, C, N = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
    else:
        raise ValueError(f"Unsupported logits shape: {logits.shape}")

    if ignore_index is not None:
        valid_mask = targets_flat != ignore_index
        logits_flat = logits_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

    return logits_flat, targets_flat


class FocalLossMultiClass(nn.Module):
    """
    Standard multi-class focal loss on logits.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,  # None or tensor/list of class weights
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [P, C]
        targets: [P]
        """
        if logits.numel() == 0:
            return logits.new_tensor(0.0)

        # base CE, per-point
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", weight=self.alpha
        )  # [P]

        # pt = probability assigned to true class
        pt = torch.exp(-ce_loss)  # since CE = -log(pt)

        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss  # [P]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


def lovasz_grad(gt_sorted):
    """
    Compute gradient of the Lovasz extension w.r.t sorted errors.
    gt_sorted: [P] binary ground-truth (1 for pos, 0 for neg) after sorting.
    """
    p = gt_sorted.numel()
    if p == 0:
        return gt_sorted

    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union

    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes="present"):
    """
    Multi-class Lovasz-Softmax loss (flat version).

    probas: [P, C] softmax probabilities
    labels: [P] int labels
    classes: 'present' or list of class indices
    """
    if probas.numel() == 0:
        return probas.new_tensor(0.0)

    C = probas.size(1)
    losses = []

    if classes == "present":
        class_indices = torch.unique(labels)
        class_indices = class_indices[class_indices != -1]
    else:
        class_indices = list(classes)

    for c in class_indices:
        fg = (labels == c).float()  # foreground mask for class c
        if fg.sum() == 0:
            continue

        pc = probas[:, int(c)]  # [P]
        # absolute error: 1 - p for positives, p for negatives
        errors = (fg - pc).abs()

        # sort by decreasing error
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]

        grad = lovasz_grad(fg_sorted)
        loss_c = torch.dot(errors_sorted, grad)
        losses.append(loss_c)

    if not losses:
        return probas.new_tensor(0.0)

    return torch.mean(torch.stack(losses))


class RangeViTLoss(nn.Module):
    """
    Combined loss used in RangeViT-style training:
    L_total = focal_loss + lovasz_softmax_loss
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.focal = FocalLossMultiClass(gamma=gamma, alpha=alpha, reduction="mean")
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] or [B, C, N]
        targets: [B, H, W] or [B, N] int labels
        """
        # flatten and drop ignored
        logits_flat, targets_flat = _flatten_logits_and_targets(
            logits, targets, self.ignore_index
        )

        if logits_flat.numel() == 0:
            return logits.new_tensor(0.0), logits.new_tensor(0.0), logits.new_tensor(0.0)

        # focal loss on logits
        focal_loss = self.focal(logits_flat, targets_flat)

        # Lovasz-Softmax on probabilities
        prob_flat = F.softmax(logits_flat, dim=1)
        lovasz_loss = lovasz_softmax_flat(prob_flat, targets_flat, classes="present")

        total_loss = focal_loss + lovasz_loss
        return total_loss, focal_loss, lovasz_loss


def make_class_weights_from_freq(freq, eps=1e-6, device="cpu"):
    """
    freq: 1D array-like of class frequencies (e.g. counts or probabilities)
    Returns weights ~ 1/sqrt(freq).
    """
    freq = torch.as_tensor(freq, dtype=torch.float32, device=device)
    w = 1.0 / torch.sqrt(freq + eps)
    # optional: normalize to keep loss scale reasonable
    w = w * (freq.numel() / w.sum())
    return w


def boundary_from_mask(mask, theta=3):
    """
    mask: [B, 1, H, W], values in [0, 1]
    Implements y_b = maxpool(1 - y) - (1 - y).
    """
    inv = 1.0 - mask
    dilated = F.max_pool2d(inv, kernel_size=theta, stride=1, padding=theta // 2)
    boundary = dilated - inv
    return boundary


def boundary_loss(logits, targets, num_classes, ignore_index=255, theta=3, eps=1e-6):
    """
    Differentiable approximation of the boundary loss from LENet.
    logits: [B, C, H, W]
    targets: [B, H, W]
    """
    B, C, H, W = logits.shape
    device = logits.device

    # softmax probabilities
    probs = F.softmax(logits, dim=1)  # [B, C, H, W]

    # build one-hot GT (with ignore masked out)
    valid_mask = (targets != ignore_index).unsqueeze(1).float()  # [B, 1, H, W]
    targets_clamped = targets.clone()
    targets_clamped[targets_clamped == ignore_index] = 0

    one_hot = F.one_hot(targets_clamped, num_classes=num_classes)  # [B, H, W, C]
    one_hot = one_hot.permute(0, 3, 1, 2).float()                  # [B, C, H, W]
    one_hot = one_hot * valid_mask

    total_num = logits.new_tensor(0.0)
    total_den = logits.new_tensor(0.0)

    for c in range(num_classes):
        gt_c = one_hot[:, c:c+1, :, :]           # [B,1,H,W]
        if gt_c.sum() <= 0:
            continue

        prob_c = probs[:, c:c+1, :, :] * valid_mask

        gt_b = boundary_from_mask(gt_c, theta=theta)
        pred_b = boundary_from_mask(prob_c, theta=theta)

        # boundary "intersection"
        inter = (pred_b * gt_b).sum()
        pred_sum = pred_b.sum()
        gt_sum = gt_b.sum()

        if gt_sum <= 0:
            continue

        precision_c = inter / (pred_sum + eps)
        recall_c = inter / (gt_sum + eps)

        total_num = total_num + precision_c * recall_c
        total_den = total_den + precision_c + recall_c

    if total_den <= 0:
        return logits.new_tensor(0.0)

    f1 = 2.0 * total_num / (total_den + eps)
    return 1.0 - f1  # L_bd


class LENetLoss(nn.Module):
    """
    Implements L_head = w1 * L_wce + w2 * L_ls + w3 * L_bd
    """

    def __init__(
        self,
        num_classes,
        class_freq,
        ignore_index=255,
        theta=3,
        w_ce=1.0,
        w_lovasz=1.5,
        w_boundary=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.theta = theta
        self.w_ce = w_ce
        self.w_lovasz = w_lovasz
        self.w_boundary = w_boundary

        class_weights = make_class_weights_from_freq(
            class_freq, device="cpu"
        )  # move later to device
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]
        targets: [B, H, W]
        """
        # move weights to correct device if needed
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)

        # Weighted cross-entropy (handles ignore_index internally)
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )

        # Lovasz-Softmax on valid pixels
        logits_flat, targets_flat = _flatten_logits_and_targets(
            logits, targets, self.ignore_index
        )
        
        if logits_flat.numel() == 0:
            lovasz_loss = logits.new_tensor(0.0)
        else:
            prob_flat = F.softmax(logits_flat, dim=1)
            lovasz_loss = lovasz_softmax_flat(prob_flat, targets_flat, classes="present")

        # Boundary loss
        # Only apply boundary loss if we have spatial dimensions [B, C, H, W]
        if logits.dim() == 4:
            bd_loss = boundary_loss(
                logits,
                targets,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                theta=self.theta,
            )
        else:
            # Fallback for point clouds or flat inputs if boundary loss isn't applicable
            bd_loss = logits.new_tensor(0.0)

        total = self.w_ce * ce_loss + self.w_lovasz * lovasz_loss + self.w_boundary * bd_loss
        return total, ce_loss, lovasz_loss, bd_loss
