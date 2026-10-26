# -*- coding: utf-8 -*-
"""SegFormer v2 切块: ISAT JSON -> 语义 mask(0/1/2) -> 640 滑窗切块。"""
import os, json, glob, random, yaml
import numpy as np
import cv2
from pathlib import Path

CAT2ID = {'silie': 1, 'fanzhe': 2}

def load_split(txt):
    return [l.strip().replace('\\','/') for l in open(txt, encoding='utf-8') if l.strip()]

def json_for(ip): return Path(ip).with_suffix('.json').as_posix()

def rasterize(objects, W, H):
    mask = np.zeros((H, W), dtype=np.uint8)
    items = []
    for o in objects:
        cat = o.get('category','').strip().lower()
        if cat in ('background','__background__'): continue
        if cat not in CAT2ID: raise ValueError(f'未知类别 {o.get("category")!r}')
        seg = o.get('segmentation',[])
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
        y2 = min(y+tile, H); y0 = max(0, y2-tile)
        for x in range(0, W, stride):
            x2 = min(x+tile, W); x0 = max(0, x2-tile)
            yield (y0, x0, img[y0:y2, x0:x2], mask[y0:y2, x0:x2])

def run(cfg):
    root = cfg['project_root']
    out = Path(root) / 'segformer_v2' / 'dataset'
    tile, stride, bg = cfg['tile_size'], cfg['stride'], cfg['bg_keep_ratio']
    rng = random.Random(cfg['seed'])
    splits = {'train': cfg['train_list'], 'val': cfg['val_list']}
    stats = {}
    for split, txt in splits.items():
        img_dir = out / split / 'images'; img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir = out / split / 'masks'; msk_dir.mkdir(parents=True, exist_ok=True)
        full_dir = out / 'val_full' if split == 'val' else None
        if full_dir: full_dir.mkdir(parents=True, exist_ok=True)
        nt = nd = nb = 0; val_lines = []
        for ip in load_split(txt):
            jp = json_for(ip)
            if not os.path.exists(ip) or not os.path.exists(jp): continue
            d = json.load(open(jp, encoding='utf-8'))
            W, H = d['info']['width'], d['info']['height']
            img = cv2.imread(ip)
            if img is None: continue
            mask = rasterize(d.get('objects',[]), W, H)
            if split == 'val':
                fp = full_dir / (Path(ip).stem + '.png')
                cv2.imwrite(str(fp), mask)
                val_lines.append(f'{ip}\t{fp.as_posix()}')
            stem = Path(ip).stem
            for (y0, x0, it, mt) in gen_tiles(img, mask, tile, stride):
                has = (mt > 0).any()
                if not has and rng.random() > bg: continue
                if has: nd += 1
                else: nb += 1
                nt += 1
                cv2.imwrite(str(img_dir / f'{stem}_{y0}_{x0}.jpg'), it, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(msk_dir / f'{stem}_{y0}_{x0}.png'), mt)
        stats[split] = (len(load_split(txt)), nt, nd, nb)
        if split == 'val':
            (out / 'val_full.txt').write_text('\n'.join(val_lines), encoding='utf-8')
    print('=== v2 切块 ===')
    for s,(ni,nt,nd,nb) in stats.items():
        print(f'{s}: 原图 {ni}, 总块 {nt} (含缺陷 {nd}, 背景 {nb})')

if __name__ == '__main__':
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config.yaml', encoding='utf-8'))
    run(cfg)
