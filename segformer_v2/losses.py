# -*- coding: utf-8 -*-
"""SegFormer v2 损失: Focal + Dice + Lovász (聚焦 silie 小目标)。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha  # tensor (C,)
        self.gamma = gamma
        self.ignore_index = ignore_index
    def forward(self, logits, target):
        # logits (B,C,H,W), target (B,H,W)
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction='none', ignore_index=self.ignore_index)
        p = torch.exp(-ce)  # 真实类概率
        loss = ((1 - p) ** self.gamma) * ce
        return loss.mean()

class DiceLoss(nn.Module):
    """多类 Dice，只在前景类计算(忽略 bg)，避免背景主导。"""
    def __init__(self, num_classes=3, fg_classes=(1, 2), smooth=1.0, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.fg_classes = fg_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
    def forward(self, logits, target):
        pred = F.softmax(logits, dim=1)
        valid = (target != self.ignore_index)
        tgt = target.clone()
        tgt[~valid] = 0
        losses = []
        for c in self.fg_classes:
            pc = pred[:, c]
            tc = (tgt == c).float()
            inter = (pc * tc * valid.float()).sum(dim=(1, 2))
            union = (pc * valid.float() + tc).sum(dim=(1, 2))
            dice = (2 * inter + self.smooth) / (union + self.smooth)
            losses.append(1 - dice.mean())
        return sum(losses) / len(losses)

# ---- Lovász-Softmax (Berman et al. 2018) ----
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(logits, labels, classes='present'):
    """logits (N,C), labels (N,)."""
    if logits.numel() == 0: return 0.0 * logits.sum()
    C = logits.shape[1]
    losses = []
    for c in range(C):
        if classes != 'all' and c == 0:
            continue  # 跳过背景，聚焦缺陷
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            fg_class = logits[:, 0]
        else:
            fg_class = logits[:, c]
        errors = (fg - fg_class).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted).to(logits.device)
        losses.append(torch.dot(errors_sorted, grad))
    if not losses:
        return 0.0 * logits.sum()
    return sum(losses) / len(losses)

class LovaszSoftmax(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
    def forward(self, logits, target):
        # logits (B,C,H,W), target (B,H,W)
        B, C, H, W = logits.shape
        logits = F.softmax(logits, dim=1)
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
        target = target.reshape(-1)
        valid = (target != self.ignore_index)
        if valid.sum() == 0:
            return 0.0 * logits.sum()
        return lovasz_softmax_flat(logits[valid], target[valid])

class V2Loss(nn.Module):
    """组合: w_focal·Focal + w_dice·Dice + w_lovasz·Lovász"""
    def __init__(self, cfg, device):
        super().__init__()
        alpha = torch.tensor(cfg['focal_alpha'], dtype=torch.float32, device=device)
        self.focal = FocalLoss(alpha=alpha, gamma=cfg['focal_gamma'])
        self.dice = DiceLoss(num_classes=cfg['num_classes'])
        self.lovasz = LovaszSoftmax()
        self.wf, self.wd, self.wl = cfg['loss_focal'], cfg['loss_dice'], cfg['loss_lovasz']
    def forward(self, logits, target):
        return self.wf * self.focal(logits, target) + self.wd * self.dice(logits, target) + self.wl * self.lovasz(logits, target)
