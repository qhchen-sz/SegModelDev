# -*- coding: utf-8 -*-
"""Mask2Former 实例 Dataset（缓存版）：加载预计算 384 实例 mask + 图像，手动增强，绕开 processor。"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import glob, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
OUT = 384

class M2FTileDataset(Dataset):
    def __init__(self, split_dir, train=True, tile=512):
        self.img_dir = Path(split_dir) / 'images'
        self.cache_dir = Path(split_dir) / 'cache'
        self.files = sorted(glob.glob(str(self.img_dir / '*.jpg')))
        self.train = train
        self.tile = tile

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ip = self.files[idx]
        stem = Path(ip).stem
        img = cv2.imread(ip)  # BGR 512
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (OUT, OUT), interpolation=cv2.INTER_LINEAR)
        # 缓存的 mask
        c = torch.load(self.cache_dir / f'{stem}.pt', weights_only=False)
        masks = np.asarray(c['masks'])  # (N,384,384) float32
        classes = np.asarray(c['classes'])
        if self.train:
            # 随机翻转/旋转（image 与 masks 同步）
            if random.random() < 0.5:
                img = img[:, ::-1].copy(); masks = masks[:, :, ::-1].copy()
            if random.random() < 0.3:
                img = img[::-1].copy(); masks = masks[:, ::-1, :].copy()
            if random.random() < 0.3:
                img = np.rot90(img, 1).copy(); masks = np.rot90(masks, 1, axes=(1, 2)).copy()
            # 颜色抖动（仅图像）
            if random.random() < 0.4:
                img = self._jitter(img)
        # 归一化图像
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3,384,384)
        pixel_mask = torch.ones(OUT, OUT, dtype=torch.float32)
        mask_labels = torch.from_numpy(masks)  # (N,384,384)
        class_labels = torch.from_numpy(classes)  # (N,)
        return {'pixel_values': img, 'pixel_mask': pixel_mask,
                'mask_labels': mask_labels, 'class_labels': class_labels}

    def _jitter(self, img):
        # 简单亮度/对比度抖动
        b = 1.0 + (random.random() - 0.5) * 0.4
        c = 1.0 + (random.random() - 0.5) * 0.4
        img = img.astype(np.float32)
        img = (img - 128) * c + 128 * b
        return np.clip(img, 0, 255).astype(np.uint8)

def collate(batch):
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'pixel_mask': torch.stack([b['pixel_mask'] for b in batch]),
        'mask_labels': [b['mask_labels'] for b in batch],
        'class_labels': [b['class_labels'] for b in batch],
    }
