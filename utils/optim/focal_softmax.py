'''
From Z. Zhuang et al.
https://github.com/ICEORY/PMF
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalSoftmaxLoss(nn.Module):
    def __init__(self, n_classes, gamma=1, alpha=0.8, softmax=True, ignore_index=None):
        super(FocalSoftmaxLoss, self).__init__()
        self.gamma = gamma
        self.n_classes = n_classes
        self.ignore_index = ignore_index

        if isinstance(alpha, list):
            assert len(alpha) == n_classes, 'len(alpha)!=n_classes: {} vs. {}'.format(
                len(alpha), n_classes)
            self.alpha = torch.Tensor(alpha)
        elif isinstance(alpha, np.ndarray):
            assert alpha.shape[0] == n_classes, 'len(alpha)!=n_classes: {} vs. {}'.format(
                len(alpha), n_classes)
            self.alpha = torch.from_numpy(alpha)
        else:
            assert alpha < 1 and alpha > 0, 'invalid alpha: {}'.format(alpha)
            self.alpha = torch.zeros(n_classes)
            self.alpha[0] = alpha
            self.alpha[1:] += (1-alpha)
        self.softmax = softmax

    def forward(self, x, target, mask=None):
        '''compute focal loss
        x: N C or NCHW
        target: N, or NHW

        Args:
            x ([type]): [description]
            target ([type]): [description]
        '''

        if x.dim() > 2:
            pred = x.view(x.size(0), x.size(1), -1)
            pred = pred.transpose(1, 2)
            pred = pred.contiguous().view(-1, x.size(1))
        else:
            pred = x

        target = target.view(-1, 1).long()

        valid_mask = torch.ones_like(target.squeeze(1), dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask = valid_mask & (target.squeeze(1) != self.ignore_index)
        if mask is not None:
            if len(mask.size()) > 1:
                mask = mask.view(-1)
            valid_mask = valid_mask & (mask > 0)

        if valid_mask.sum() == 0:
            return torch.zeros([], device=x.device, dtype=x.dtype)

        if self.softmax:
            pred_softmax = F.softmax(pred, 1)
        else:
            pred_softmax = pred
        pred_softmax = pred_softmax.gather(1, target).view(-1)
        pred_softmax = pred_softmax[valid_mask]
        pred_logsoft = pred_softmax.clamp(1e-6).log()
        self.alpha = self.alpha.to(x.device)
        alpha = self.alpha.gather(0, target.squeeze())[valid_mask]
        loss = - (1-pred_softmax).pow(self.gamma)
        loss = loss * pred_logsoft * alpha
        return loss.mean()

