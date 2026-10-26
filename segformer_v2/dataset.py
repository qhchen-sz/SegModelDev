# -*- coding: utf-8 -*-
"""SegFormer v2 Dataset: 640 语义 tile + copy-paste silie + albumentations。"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import glob, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_aug(train, tile):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomScale(scale_limit=(1.0, 1.3), p=0.4),
            A.RandomCrop(height=tile, width=tile, p=1.0),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

class SegV2Dataset(Dataset):
    def __init__(self, split_dir, silie_crops_dir, train=True, tile=640, cp_prob=0.5):
        self.img_dir = Path(split_dir) / 'images'
        self.msk_dir = Path(split_dir) / 'masks'
        self.files = sorted(glob.glob(str(self.img_dir / '*.jpg')))
        self.train = train
        self.tile = tile
        self.cp_prob = cp_prob
        self.aug = build_aug(train, tile)
        # 加载 silie 抠图池
        self.crops = []
        if train and Path(silie_crops_dir).exists():
            for f in glob.glob(str(Path(silie_crops_dir) / '*_img.jpg')):
                m = f.replace('_img.jpg', '_mask.png')
                if os.path.exists(m):
                    self.crops.append((f, m))

    def __len__(self): return len(self.files)

    def _paste_silie(self, img, mask):
        if not self.crops or random.random() > self.cp_prob:
            return img, mask
        cf, mf = random.choice(self.crops)
        cimg = cv2.imread(cf)
        cmask = cv2.imread(mf, cv2.IMREAD_UNCHANGED)
        if cimg is None or cmask is None: return img, mask
        ch, cw = cimg.shape[:2]
        # 随机缩放
        s = random.uniform(0.7, 1.3)
        cimg = cv2.resize(cimg, (int(cw*s), int(ch*s)), interpolation=cv2.INTER_LINEAR)
        cmask = cv2.resize(cmask, (int(cw*s), int(ch*s)), interpolation=cv2.INTER_NEAREST)
        ch, cw = cimg.shape[:2]
        TH, TW = img.shape[:2]
        if ch >= TH or cw >= TW: return img, mask
        py = random.randint(0, TH - ch); px = random.randint(0, TW - cw)
        m2 = (cmask == 1)  # (ch, cw)
        roi = img[py:py+ch, px:px+cw].copy()
        roi[m2] = cimg[m2]
        img[py:py+ch, px:px+cw] = roi
        roi_m = mask[py:py+ch, px:px+cw].copy()
        roi_m[m2] = 1
        mask[py:py+ch, px:px+cw] = roi_m
        return img, mask

    def __getitem__(self, idx):
        ip = self.files[idx]
        mp = ip.replace('/images/', '/masks/').replace('\\images\\', '\\masks\\').replace('.jpg', '.png')
        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if img.shape[0] != self.tile or img.shape[1] != self.tile:
            img = cv2.resize(img, (self.tile, self.tile), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.tile, self.tile), interpolation=cv2.INTER_NEAREST)
        if self.train:
            img, mask = self._paste_silie(img.copy(), mask.copy())
        r = self.aug(image=img, mask=mask)
        img = torch.from_numpy(r['image']).permute(2, 0, 1).float()
        mask = torch.from_numpy(r['mask'].astype(np.int64))
        return img, mask
