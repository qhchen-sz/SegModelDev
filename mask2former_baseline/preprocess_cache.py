# -*- coding: utf-8 -*-
"""
预计算每 tile 的实例 mask(384x384 float32)+classes，缓存为 .pt，
训练时 dataset 直接加载，绕开昂贵的 Mask2FormerImageProcessor 每样本调用。
"""
import os, glob, yaml
import numpy as np
import cv2, torch
from pathlib import Path

def run(cfg):
    base = Path(cfg['project_root']) / 'mask2former_baseline' / 'dataset'
    out_size = 384  # processor 默认 resize 到 384
    for split in ['train', 'val']:
        ins_dir = base / split / 'instances'
        cache_dir = base / split / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(glob.glob(str(ins_dir / '*.npz')))
        for i, f in enumerate(files):
            stem = Path(f).stem
            out = cache_dir / f'{stem}.pt'
            if out.exists():
                continue
            d = np.load(f)
            masks = d['masks']; classes = d['classes']
            # 每个 mask resize 到 384
            if masks.shape[0] > 0:
                rm = np.stack([cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST) for m in masks], 0).astype(np.float32)
                rc = classes.astype(np.int64)
            else:
                rm = np.zeros((0, out_size, out_size), dtype=np.float32)
                rc = np.zeros((0,), dtype=np.int64)
            torch.save({'masks': rm, 'classes': rc}, out)
        print(f'{split}: cached {len(files)} tiles -> {cache_dir}')

if __name__ == '__main__':
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config.yaml', encoding='utf-8'))
    run(cfg)
