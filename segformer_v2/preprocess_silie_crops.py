# -*- coding: utf-8 -*-
"""抠取 silie 缺陷裁剪(image + binary silie mask)，供 dataset copy-paste。"""
import os, json, glob, yaml
import numpy as np
import cv2
from pathlib import Path

def run(cfg):
    root = Path(cfg['project_root'])
    out = Path(cfg['silie_crops_dir']); out.mkdir(parents=True, exist_ok=True)
    train_imgs = [l.strip().replace('\\','/') for l in open(cfg['train_list'], encoding='utf-8') if l.strip()]
    n = 0
    for ip in train_imgs:
        jp = Path(ip).with_suffix('.json').as_posix()
        if not os.path.exists(jp): continue
        d = json.load(open(jp, encoding='utf-8'))
        img = cv2.imread(ip)
        if img is None: continue
        H, W = img.shape[:2]
        stem = Path(ip).stem
        for i, o in enumerate(d.get('objects', [])):
            cat = o.get('category','').strip().lower()
            if cat != 'silie': continue
            seg = o.get('segmentation', [])
            if not seg: continue
            poly = np.array(seg, dtype=np.int32)
            x0, y0 = poly[:,0].min(), poly[:,1].min()
            x1, y1 = poly[:,0].max(), poly[:,1].max()
            pad = 15
            x0, y0 = max(0, x0-pad), max(0, y0-pad)
            x1, y1 = min(W, x1+pad), min(H, y1+pad)
            if x1-x0 < 5 or y1-y0 < 5: continue
            crop = img[y0:y1, x0:x1].copy()
            cmask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
            cv2.fillPoly(cmask, [poly - np.array([x0, y0])], 1)
            cv2.imwrite(str(out / f'{stem}_{i}_img.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / f'{stem}_{i}_mask.png'), cmask)
            n += 1
    print(f'silie crops: {n} -> {out}')

if __name__ == '__main__':
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config.yaml', encoding='utf-8'))
    run(cfg)
