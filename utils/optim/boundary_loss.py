
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

def one_hot(label, n_classes, device):
    """
    Convert a label tensor (B, H, W) to one-hot (B, C, H, W).
    """
    shape = label.shape
    one_hot = torch.zeros((shape[0], n_classes) + shape[1:], device=device)
    one_hot.scatter_(1, label.unsqueeze(1), 1.0)
    return one_hot

def compute_sdf(img_gt, out_shape):
    """
    Compute the normalized signed distance map of binary mask.
    img_gt: numpy array of shape (H, W) or (D, H, W), binary (0/1)
    """
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    posmask = img_gt.astype(bool)
    if posmask.any():
        negmask = ~posmask
        posdis = distance(posmask)
        negdis = distance(negmask)
        boundary = np.where(negmask, 0, posdis) + np.where(posmask, 0, negdis)
        
        # Signed distance: positive outside, negative inside
        # We want to minimize product of prob * sdf.
        # If prob=1 (pred foreground) and sdf>0 (outside), penalty.
        # If prob=1 (pred foreground) and sdf<0 (inside), reward (negative loss).
        
        # Standard definition: phi = dist(outside) - dist(inside)
        # outside: negdis
        # inside: posdis
        # phi = negdis - (posdis - 1)
        # But commonly used:
        sdf = negdis - posdis + 1
        normalized_sdf = sdf # / np.max(sdf) if we want normalization
    
    return normalized_sdf

class BoundaryLoss(nn.Module):
    def __init__(self, n_classes, ignore_index=0):
        super(BoundaryLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, probs, label):
        """
        probs: (B, C, H, W) - softmax probabilities
        label: (B, H, W) - ground truth labels
        """
        B, C, H, W = probs.shape
        
        # Convert label to one-hot (B, C, H, W)
        # We need to handle ignore_index. 
        # Usually we just ignore it in the loop or mask it out.
        
        # Pre-compute SDF on CPU
        # This is the bottleneck.
        with torch.no_grad():
            gt_sdf_np = np.zeros((B, C, H, W))
            label_np = label.cpu().numpy()
            
            for b in range(B):
                for c in range(self.n_classes):
                    if c == self.ignore_index:
                        continue
                    
                    # Create binary mask for class c
                    mask = (label_np[b] == c)
                    if mask.sum() == 0:
                        continue
                        
                    gt_sdf_np[b, c] = compute_sdf(mask, (H, W))
            
            gt_sdf = torch.from_numpy(gt_sdf_np).float().to(probs.device)

        # Compute loss: sum(probs * sdf)
        # We only care about the probability of the class c
        loss = torch.sum(probs * gt_sdf) / (B * H * W) # Average over pixels
        
        return loss
