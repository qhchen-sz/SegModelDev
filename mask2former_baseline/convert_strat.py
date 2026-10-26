# -*- coding: utf-8 -*-
"""Mask2Former 切块（分层重切版）: ISAT JSON -> 每 tile 实例标注 (masks+classes npz)。
输出到 dataset_strat/{train,val}/{images,instances}。读 config_strat.yaml。绝不改动 convert.py。"""
import os, json, glob, random, yaml
import numpy as np
import cv2
from pathlib import Path

CAT2ID = {'silie': 0, 'fanzhe': 1}

def load_split(txt):
    out = []
    for line in open(txt, encoding='utf-8'):
        line = line.strip()
        if line: out.append(line.replace('\\', '/'))
    return out

def json_for(img_path):
    return Path(img_path).with_suffix('.json').as_posix()

def rasterize_instances(info, objects, W, H):
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
    masks, classes = [], []
    for area, cid, poly in items:
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(m, [poly], 1)
        masks.append(m); classes.append(cid)
    return masks, classes

def gen_tiles(img, masks, classes, tile, stride):
    H, W = img.shape[:2]
    for y in range(0, H, stride):
        y2 = min(y + tile, H); y0 = max(0, y2 - tile)
        for x in range(0, W, stride):
            x2 = min(x + tile, W); x0 = max(0, x2 - tile)
            it = img[y0:y2, x0:x2]
            tile_masks = [m[y0:y2, x0:x2] for m in masks]
            yield (y0, x0, it, tile_masks, list(classes))

def run(cfg):
    out = Path(cfg['dataset_dir'])
    tile, stride, bg_ratio = cfg['tile_size'], cfg['stride'], cfg['bg_keep_ratio']
    rng = random.Random(cfg['seed'])
    splits = {'train': cfg['train_list'], 'val': cfg['val_list']}
    stats = {}
    for split, txt in splits.items():
        img_paths = load_split(txt)
        img_dir = out / split / 'images'; img_dir.mkdir(parents=True, exist_ok=True)
        ins_dir = out / split / 'instances'; ins_dir.mkdir(parents=True, exist_ok=True)
        n_tiles = n_def = n_bg = n_inst = 0
        for ip in img_paths:
            jp = json_for(ip)
            if not os.path.exists(ip) or not os.path.exists(jp):
                print('skip', ip); continue
            d = json.load(open(jp, encoding='utf-8'))
            info = d['info']; W, H = info['width'], info['height']
            img = cv2.imread(ip)
            if img is None: continue
            masks, classes = rasterize_instances(info, d.get('objects', []), W, H)
            stem = Path(ip).stem
            for (y0, x0, it, tm, tc) in gen_tiles(img, masks, classes, tile, stride):
                has_def = any(m.any() for m in tm)
                if not has_def and rng.random() > bg_ratio: continue
                if has_def: n_def += 1
                else: n_bg += 1
                n_tiles += 1
                keep = [(m, c) for m, c in zip(tm, tc) if m.any()]
                tm2 = [m for m, _ in keep]; tc2 = [c for _, c in keep]
                n_inst += len(tc2)
                cv2.imwrite(str(img_dir / f'{stem}_{y0}_{x0}.jpg'), it, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if tm2:
                    masks_arr = np.stack(tm2, 0).astype(np.uint8)
                    classes_arr = np.array(tc2, dtype=np.int64)
                else:
                    masks_arr = np.zeros((0, it.shape[0], it.shape[1]), dtype=np.uint8)
                    classes_arr = np.zeros((0,), dtype=np.int64)
                np.savez_compressed(str(ins_dir / f'{stem}_{y0}_{x0}.npz'),
                                     masks=masks_arr, classes=classes_arr)
        stats[split] = (len(img_paths), n_tiles, n_def, n_bg, n_inst)
    print('=== m2f 切块(分层重切) ===')
    for s, (ni, nt, nd, nb, nin) in stats.items():
        print(f'{s}: 原图 {ni}, 总块 {nt} (含缺陷 {nd}, 背景 {nb}), 实例数 {nin}')
    print('输出', out)

if __name__ == '__main__':
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config_strat.yaml', encoding='utf-8'))
    run(cfg)
