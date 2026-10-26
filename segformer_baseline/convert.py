# -*- coding: utf-8 -*-
"""
ISAT JSON -> class mask -> 滑窗 512x512 切块
按工程已有的 train.txt / val.txt 划分，不随机切。
"""
import os, sys, json, glob, random, yaml
import numpy as np
import cv2
from pathlib import Path

# 类别名 -> id（标注里 'fanzhe ' 带尾随空格，统一 strip + lower）
CAT2ID = {
    'background': 0,
    'silie': 1,
    'fanzhe': 2,
}

def load_split(txt_path):
    paths = []
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paths.append(line.replace('\\', '/'))
    return paths

def json_for(img_path):
    p = Path(img_path)
    return p.with_suffix('.json').as_posix()

def rasterize_mask(info, objects, W, H):
    """多边形 -> 单通道 uint8 mask (H,W), 值 0/1/2.
    按面积升序绘制：小缺陷最后画，不被大区域覆盖。"""
    mask = np.zeros((H, W), dtype=np.uint8)
    # 过滤 + 解析
    items = []
    for o in objects:
        cat = o.get('category', '').strip().lower()
        if cat in ('background', '__background__'):
            continue  # 背景类对象跳过，mask 默认全 0 即背景
        if cat not in CAT2ID:
            raise ValueError(f'未知类别 {o.get("category")!r} in {info.get("name")}')
        seg = o.get('segmentation', [])
        if not seg:
            continue
        poly = np.array(seg, dtype=np.int32)  # (N,2)
        area = float(o.get('area', cv2.contourArea(poly)))
        items.append((area, CAT2ID[cat], poly))
    items.sort(key=lambda x: x[0])  # 小面积最后画
    for area, cid, poly in items:
        cv2.fillPoly(mask, [poly], cid)
    return mask

def gen_tiles(img, mask, tile, stride):
    H, W = mask.shape
    for y in range(0, H, stride):
        y2 = min(y + tile, H)
        y0 = max(0, y2 - tile)  # 保证块高=tile（右下边界回退）
        for x in range(0, W, stride):
            x2 = min(x + tile, W)
            x0 = max(0, x2 - tile)
            it = img[y0:y2, x0:x2]
            mt = mask[y0:y2, x0:x2]
            yield (y0, x0, it, mt)

def run(cfg):
    root = cfg['project_root']
    out = Path(root) / 'segformer_baseline' / 'dataset'
    tile, stride, bg_ratio = cfg['tile_size'], cfg['stride'], cfg['bg_keep_ratio']
    rng = random.Random(cfg['seed'])

    splits = {'train': cfg['train_list'], 'val': cfg['val_list']}
    stats = {}
    for split, txt in splits.items():
        img_paths = load_split(txt)
        img_dir = out / split / 'images'
        msk_dir = out / split / 'masks'
        full_dir = out / 'val_full' if split == 'val' else None
        if full_dir:
            full_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        n_tiles, n_defect_tiles, n_bg_tiles = 0, 0, 0
        val_full_lines = []
        for ip in img_paths:
            jp = json_for(ip)
            if not os.path.exists(ip) or not os.path.exists(jp):
                print(f'  [skip] 缺文件: {ip}')
                continue
            d = json.load(open(jp, encoding='utf-8'))
            info = d['info']
            W, H = info['width'], info['height']
            img = cv2.imread(ip)  # BGR
            if img is None:
                print(f'  [skip] 读图失败: {ip}'); continue
            mask = rasterize_mask(info, d.get('objects', []), W, H)

            # val 保留全图 mask 用于全图评估
            if split == 'val':
                full_mask_path = full_dir / (Path(ip).stem + '.png')
                cv2.imwrite(str(full_mask_path), mask)
                val_full_lines.append(f'{ip}\t{full_mask_path.as_posix()}')

            stem = Path(ip).stem
            for (y0, x0, it, mt) in gen_tiles(img, mask, tile, stride):
                has_def = (mt > 0).any()
                if not has_def:
                    # 纯背景块按比例保留
                    if rng.random() > bg_ratio:
                        continue
                    n_bg_tiles += 1
                else:
                    n_defect_tiles += 1
                n_tiles += 1
                cv2.imwrite(str(img_dir / f'{stem}_{y0}_{x0}.jpg'), it, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(msk_dir / f'{stem}_{y0}_{x0}.png'), mt)
        stats[split] = (len(img_paths), n_tiles, n_defect_tiles, n_bg_tiles)
        if split == 'val':
            (out / 'val_full.txt').write_text('\n'.join(val_full_lines), encoding='utf-8')

    print('\n=== 切块统计 ===')
    for s, (ni, nt, nd, nb) in stats.items():
        print(f'{s}: 原图 {ni} 张, 总块 {nt} (含缺陷 {nd}, 纯背景 {nb})')
    print(f'输出目录: {out}')

if __name__ == '__main__':
    cfg_path = Path(__file__).parent / 'config.yaml'
    cfg = yaml.safe_load(open(cfg_path, encoding='utf-8'))
    run(cfg)
