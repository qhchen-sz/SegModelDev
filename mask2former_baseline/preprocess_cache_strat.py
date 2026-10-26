# -*- coding: utf-8 -*-
"""Mask2Former 预计算缓存（分层重切版）: dataset_strat npz -> 384 .pt cache。
读 config_strat.yaml。绝不改动 preprocess_cache.py。"""
import os, glob, yaml
import numpy as np
import cv2, torch
from pathlib import Path

def run(cfg):
    base = Path(cfg['dataset_dir'])
    out_size = 384
    for split in ['train', 'val']:
        ins_dir = base / split / 'instances'
        cache_dir = base / split / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(glob.glob(str(ins_dir / '*.npz')))
        for i, f in enumerate(files):
            stem = Path(f).stem
            outp = cache_dir / f'{stem}.pt'
            if outp.exists(): continue
            d = np.load(f)
            masks = d['masks']; classes = d['classes']
            if masks.shape[0] > 0:
                rm = np.stack([cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST) for m in masks], 0).astype(np.float32)
                rc = classes.astype(np.int64)
            else:
                rm = np.zeros((0, out_size, out_size), dtype=np.float32)
                rc = np.zeros((0,), dtype=np.int64)
            torch.save({'masks': rm, 'classes': rc}, outp)
        print(f'{split}: cached {len(files)} tiles -> {cache_dir}')

if __name__ == '__main__':
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config_strat.yaml', encoding='utf-8'))
    run(cfg)
