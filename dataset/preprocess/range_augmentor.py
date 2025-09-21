import numpy as np
import torch
import random


class RangeImageAugmentor:
    """Augmentor for range image data including mixup and cutmix operations"""
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, p_mixup=0.0, p_cutmix=0.0):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.p_mixup = p_mixup
        self.p_cutmix = p_cutmix
        
    def mixup_data(self, x, y, alpha=1.0):
        """Apply mixup augmentation to range image data"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(self, x, y, alpha=1.0):
        """Apply cutmix augmentation to range image data"""
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        # Get image dimensions
        _, h, w = x.shape[-3:]
        
        # Generate random bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    
    def apply_augmentation(self, x, y):
        """Apply mixup or cutmix augmentation with given probabilities"""
        # Choose augmentation type
        rand = random.uniform(0, 1)
        
        if rand < self.p_mixup:
            return self.mixup_data(x, y, self.mixup_alpha)
        elif rand < self.p_mixup + self.p_cutmix:
            return self.cutmix_data(x, y, self.cutmix_alpha)
        else:
            # No augmentation
            return x, y, y, 1.0


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)