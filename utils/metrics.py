import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import jaccard_score, f1_score, roc_auc_score


def mean_iou(preds, targets, threshold=0.5):
    preds_binary = (preds > threshold).cpu().numpy().flatten()
    targets_binary = targets.cpu().numpy().flatten()
    return torch.tensor(jaccard_score(targets_binary, preds_binary))


def f1(preds, targets, threshold=0.5):
    preds_binary = (preds > threshold).cpu().numpy().flatten()
    targets_binary = targets.cpu().numpy().flatten()
    return torch.tensor(f1_score(targets_binary, preds_binary))


def auc(preds, targets):
    preds_flat = preds.cpu().numpy().flatten()
    targets_flat = targets.cpu().numpy().flatten()
    if len(np.unique(targets_flat)) < 2:  # single class
        return torch.tensor(0.5)
    return torch.tensor(roc_auc_score(targets_flat, preds_flat))


def dice_coefficient(preds, targets, smooth=1.e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def bce_dice_loss(preds, targets, bce_weight=0.5, dice_weight=0.5):
    bce = F.binary_cross_entropy(preds, targets)
    dice = 1 - dice_coefficient(preds, targets)
    return bce_weight * bce + dice_weight * dice


def accuracy(preds, targets, threshold=0.5):
    preds_binary = (preds > threshold).float()
    correct = (preds_binary == targets).float().sum()
    total = torch.numel(targets)
    return correct / total
    

METRICS_DICT = {
    'mean_iou': mean_iou,
    'f1': f1,
    'auc': auc,
    'bce_dice_loss': bce_dice_loss,
    'accuracy': accuracy,
}