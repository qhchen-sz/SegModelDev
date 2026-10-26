# -*- coding: utf-8 -*-
"""SegFormer baseline 切块（分层重切版）: ISAT JSON -> 语义 mask -> 512 滑窗切块。
仅生成 train/val tiles 到 dataset_strat/（val_full 由 v2 convert 统一生成）。
读 config_strat.yaml。绝不改动 convert.py。"""
import os, json, glob, random, yaml
import numpy as np
import cv2
from pathlib import Path

CAT2ID = {'background': 0, 'silie': 1, 'fanzhe': 2}

def load_split(txt_path):
    paths = []
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: paths.append(line.replace('\\', '/'))
    return paths

def json_for(img_path):
    return Path(img_path).with_suffix('.json').as_posix()

def rasterize_mask(info, objects, W, H):
    mask = np.zeros((H, W), dtype=np.uint8)
    items = []
    for o in objects:
        cat = o.get('category', '').strip().lower()
        if cat in ('background', '__background__'): continue
        if cat not in CAT2ID:
            raise ValueError(f'未知类别 {o.get("category")!r} in {info.get("name")}')
        seg = o.get('segmentation', [])
        if not seg: continue
        poly = np.array(seg, dtype=np.int32)
        area = float(o.get('area', cv2.contourArea(poly)))
        items.append((area, CAT2ID[cat], poly))
    items.sort(key=lambda x: x[0])
    for area, cid, poly in items:
        cv2.fillPoly(mask, [poly], cid)
    return mask

def gen_tiles(img, mask, tile, stride):
    H, W = mask.shape
    for y in range(0, H, stride):
        y2 = min(y + tile, H); y0 = max(0, y2 - tile)
        for x in range(0, W, stride):
            x2 = min(x + tile, W); x0 = max(0, x2 - tile)
            yield (y0, x0, img[y0:y2, x0:x2], mask[y0:y2, x0:x2])

def run(cfg):
    out = Path(cfg['dataset_dir'])
    tile, stride, bg_ratio = cfg['tile_size'], cfg['stride'], cfg['bg_keep_ratio']
    rng = random.Random(cfg['seed'])
    splits = {'train': cfg['train_list'], 'val': cfg['val_list']}
    stats = {}
    for split, txt in splits.items():
        img_paths = load_split(txt)
        img_dir = out / split / 'images'; img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir = out / split / 'masks'; msk_dir.mkdir(parents=True, exist_ok=True)
        n_tiles = n_def = n_bg = 0
        for ip in img_paths:
            jp = json_for(ip)
            if not os.path.exists(ip) or not os.path.exists(jp):
                print(f'  [skip] 缺文件: {ip}'); continue
            d = json.load(open(jp, encoding='utf-8'))
            info = d['info']; W, H = info['width'], info['height']
            img = cv2.imread(ip)
            if img is None: continue
            mask = rasterize_mask(info, d.get('objects', []), W, H)
            stem = Path(ip).stem
            for (y0, x0, it, mt) in gen_tiles(img, mask, tile, stride):
                has_def = (mt > 0).any()
                if not has_def and rng.random() > bg_ratio: continue
                if has_def: n_def += 1
                else: n_bg += 1
                n_tiles += 1
                cv2.imwrite(str(img_dir / f'{stem}_{y0}_{x0}.jpg'), it, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(msk_dir / f'{stem}_{y0}_{x0}.png'), mt)
        stats[split] = (len(img_paths), n_tiles, n_def, n_bg)
    print('=== baseline 切块(分层重切) ===')
    for s, (ni, nt, nd, nb) in stats.items():
        print(f'{s}: 原图 {ni} 张, 总块 {nt} (含缺陷 {nd}, 纯背景 {nb})')
    print(f'输出目录: {out}')

if __name__ == '__main__':
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config_strat.yaml', encoding='utf-8'))
    run(cfg)
