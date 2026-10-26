# -*- coding: utf-8 -*-
"""SegFormer tile 数据集 + 训练增强 + 加权采样。"""
import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')  # 关闭版本检查刷屏
import glob, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path

# SegFormer 用 ImageNet 归一化
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_aug(train: bool, tile: int):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomScale(scale_limit=(1.0, 1.3), p=0.4),   # 只放大，保证 >= tile
            A.RandomCrop(height=tile, width=tile, p=1.0),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

class SegTileDataset(Dataset):
    def __init__(self, split_dir, train=True, tile=512):
        self.img_dir = Path(split_dir) / 'images'
        self.msk_dir = Path(split_dir) / 'masks'
        self.files = sorted(glob.glob(str(self.img_dir / '*.jpg')))
        self.aug = build_aug(train, tile)
        self.train = train
        self.tile = tile
        # 预计算每块各类像素数，用于加权采样
        self.class_pixels = []  # (silie_px, fanzhe_px)
        for f in self.files:
            m = cv2.imread(f.replace('.jpg', '.png').replace(str(self.img_dir), str(self.msk_dir)), cv2.IMREAD_UNCHANGED)
            if m is None:
                self.class_pixels.append((0, 0))
            else:
                self.class_pixels.append((int((m == 1).sum()), int((m == 2).sum())))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ip = self.files[idx]
        mp = ip.replace('.jpg', '.png').replace(str(self.img_dir), str(self.msk_dir))
        img = cv2.imread(ip)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        # 少数块可能因 RandomScale 边缘不等 tile，resize 回 tile
        if img.shape[0] != self.tile or img.shape[1] != self.tile:
            img = cv2.resize(img, (self.tile, self.tile), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.tile, self.tile), interpolation=cv2.INTER_NEAREST)
        augmented = self.aug(image=img, mask=mask)
        img = augmented['image']  # float32 HWC normalized
        mask = augmented['mask'].astype(np.int64)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask)
        return img, mask

def make_sample_weights(class_pixels, silie_w=3.0, fanzhe_w=1.5, bg_w=1.0):
    """加权采样权重：含 silie 的块最高，含 fanzhe 次之，纯背景最低。"""
    w = []
    for silie_px, fanzhe_px in class_pixels:
        if silie_px > 0:
            w.append(silie_w)
        elif fanzhe_px > 0:
            w.append(fanzhe_w)
        else:
            w.append(bg_w)
    return w
