
import torch
import numpy as np
import random

class RangeAug:
    def __init__(self, config):
        self.config = config
        self.p_mix = config.get('p_mix', 0.0)
        self.p_union = config.get('p_union', 0.0)
        self.p_paste = config.get('p_paste', 0.0)
        self.p_shift = config.get('p_shift', 0.0)
        
        # Rare classes for RangePaste (SemanticKITTI defaults)
        # Typically: bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist
        # IDs: 11, 15, 18, 20, 30, 31, 32
        self.rare_classes = config.get('rare_classes', [11, 15, 18, 20, 30, 31, 32])

    def range_mix(self, tensor_a, tensor_b):
        """
        Swap horizontal strips (elevation ranges) between A and B.
        tensor: (C, H, W)
        """
        if random.random() > self.p_mix:
            return tensor_a
            
        C, H, W = tensor_a.shape
        # Randomly choose number of splits (2 to 6)
        k_mix = random.randint(2, 6)
        strip_h = H // k_mix
        
        # Create a mask or just swap slices
        # We want to mix A and B. 
        # Strategy: iterate strips, randomly pick from A or B.
        
        out = tensor_a.clone()
        for i in range(k_mix):
            if random.random() < 0.5:
                start = i * strip_h
                end = (i + 1) * strip_h if i < k_mix - 1 else H
                out[:, start:end, :] = tensor_b[:, start:end, :]
        
        return out

    def range_union(self, tensor_a, tensor_b):
        """
        Fill empty pixels in A with pixels from B.
        Assumes the last channel is the mask (1=valid, 0=empty) or check range channel (0).
        Here we assume the input tensor has the mask at index 6 (from loader).
        But the loader returns [feature(5), label(1), mask(1)].
        So mask is at index -1.
        """
        if random.random() > self.p_union:
            return tensor_a
            
        mask_a = tensor_a[-1, :, :] > 0
        void_a = ~mask_a
        
        out = tensor_a.clone()
        # Fill voids in A with B
        # We need to copy all channels
        out[:, void_a] = tensor_b[:, void_a]
        
        return out

    def range_paste(self, tensor_a, tensor_b):
        """
        Paste rare classes from B into A.
        Label is at index 5 (if 7 channels total).
        """
        if random.random() > self.p_paste:
            return tensor_a
            
        label_b = tensor_b[5, :, :] # Assumes label is at index 5
        
        mask_paste = torch.zeros_like(label_b, dtype=torch.bool)
        for cls in self.rare_classes:
            mask_paste |= (label_b == cls)
            
        out = tensor_a.clone()
        out[:, mask_paste] = tensor_b[:, mask_paste]
        
        return out

    def range_shift(self, tensor_a):
        """
        Circular shift along width (azimuth).
        """
        if random.random() > self.p_shift:
            return tensor_a
            
        C, H, W = tensor_a.shape
        # Shift amount: random between W/4 and 3W/4
        shift = random.randint(W // 4, 3 * W // 4)
        
        out = torch.roll(tensor_a, shifts=shift, dims=-1)
        return out

    def __call__(self, tensor_a, tensor_b=None):
        """
        Apply augmentations.
        tensor_a: (C, H, W) - Main sample
        tensor_b: (C, H, W) - Auxiliary sample (optional)
        """
        out = tensor_a
        
        if tensor_b is not None:
            out = self.range_mix(out, tensor_b)
            out = self.range_paste(out, tensor_b)
            out = self.range_union(out, tensor_b)
            
        out = self.range_shift(out)
        
        return out
