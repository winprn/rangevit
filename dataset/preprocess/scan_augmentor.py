import numpy as np
import torch

class ScanAugmentor:
    @staticmethod
    def range_mix(pcdA, lblA, pcdB, lblB, h = 64, w = 192):
        pcdA_, lblA_ = pcdA.clone(), lblA.clone()
        x, y, z = pcdA_
        print(x, y, z)
        theta = torch.atan2(y, x)
        print(theta)
        phi = torch.atan2(z, torch.sqrt(x*x + y*y))
        print(phi)
        # Use mean values for mixing dimensions since phi and theta are tensors
        mix_h, mix_w = int(h / phi.mean().item()), int(w / theta.mean().item())
        for i in range(1, mix_h):
            for j in range(1, mix_w):
                pcdA_[:, i-1:i, j-1:j] = pcdB[:, i-1:i, j-1:j]
                lblA_[:, i-1:i, j-1:j] = lblB[:, i-1:i, j-1:j]
        return pcdA_, lblA_