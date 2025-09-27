import numpy as np

class ScanAugmentor:
    @staticmethod
    def range_mix(pcdA, lblA, pcdB, lblB, h = 64, w = 192):
        pcdA_, lblA_ = pcdA.copy(), lblA.copy()
        x, y, z = pcdA_[1:3]
        theta = np.arctan2(y, x) 
        phi = np.arctan2(z/np.sqrt(x**2 + y**2))
        mix_h, mix_w = int(h / phi), int(w / theta)
        for i in range(1, mix_h):
            for j in range(1, mix_w):
                pcdA_[:, i-1:i, j-1:j] = pcdB[:, i-1:i, j-1:j]
                lblA_[:, i-1:i, j-1:j] = lblB[:, i-1:i, j-1:j]
        return pcdA_, lblA_