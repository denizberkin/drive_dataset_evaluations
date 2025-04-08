import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self,
                 bce_weight=0.5, dice_weight=0.5, eps=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, preds, targets):
        bce = F.binary_cross_entropy(preds, targets)
        dice = 1 - self.dice_coefficient(preds, targets)  # convert to loss = 1 - dice
        return self.bce_weight * bce + self.dice_weight * dice

    def dice_coefficient(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        return (2. * intersection + self.eps) / (union + self.eps)
        