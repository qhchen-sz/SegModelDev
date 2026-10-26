# -*- coding: utf-8 -*-
"""
SegFormer 全图滑窗推理 + 评估 (per-class IoU/mIoU) + overlay 可视化
读 val_full.txt (img_path<TAB>full_mask_path)，在原始大图上滑窗推理。
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import math, yaml
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
from dataset import IMAGENET_MEAN, IMAGENET_STD

COLORS = [  # BGR overlay 颜色
    (60, 60, 60),     # 0 background 灰
    (0, 0, 255),      # 1 silie 红
    (255, 0, 0),      # 2 fanzhe 蓝
]

def load_val_full(txt):
    lines = []
    for line in open(txt, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line: continue
        a, b = line.split('\t')
        lines.append((a, b))
    return lines

def sliding_predict(model, img_bgr, tile, stride, device):
    """img: HxW BGR uint8. 返回 HxW int 概念图。"""
    H, W = img_bgr.shape[:2]
    mean = np.array(IMAGENET_MEAN, dtype=np.float32) * 255
    std = np.array(IMAGENET_STD, dtype=np.float32) * 255
    acc = np.zeros((3, H, W), dtype=np.float32)
    cov = np.zeros((H, W), dtype=np.float32)
    # 规划网格，保证覆盖右下角
    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] + tile < H: ys.append(max(0, H - tile))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] + tile < W: xs.append(max(0, W - tile))
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                patch = img_bgr[y0:y0+tile, x0:x0+tile]
                ph, pw = patch.shape[:2]
                inp = (patch[:, :, ::-1].astype(np.float32) - mean) / std  # RGB norm
                t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device).float()
                logits = model(pixel_values=t).logits  # (1,3,h/4,w/4)
                logits = F.interpolate(logits, size=(ph, pw), mode='bilinear', align_corners=False)[0]
                acc[:, y0:y0+ph, x0:x0+pw] += logits.cpu().numpy()
                cov[y0:y0+ph, x0:x0+pw] += 1.0
    cov = np.maximum(cov, 1e-6)
    pred = (acc / cov[None]).argmax(0).astype(np.uint8)
    return pred

def overlay(img_bgr, pred, alpha=0.5):
    ov = img_bgr.copy()
    for c, color in enumerate(COLORS):
        m = pred == c
        ov[m] = (img_bgr[m].astype(np.float32) * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return ov

def iou(pred, gt, nc):
    inter = np.zeros(nc, dtype=np.float64); union = np.zeros(nc, dtype=np.float64)
    for c in range(nc):
        p = pred == c; t = gt == c
        inter[c] = (p & t).sum(); union[c] = (p | t).sum()
    return inter, union

def main():
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='full', help='runs/<run>/best 权重目录名')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config.yaml', encoding='utf-8'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = Path(cfg['project_root']) / 'segformer_baseline'
    out = Path(cfg['out_dir'])
    run_dir = out / a.run
    ov_dir = run_dir / 'overlays'; ov_dir.mkdir(parents=True, exist_ok=True)
    nc = cfg['num_classes']

    model = SegformerForSemanticSegmentation.from_pretrained(run_dir / 'best').to(device).eval()

    lines = load_val_full(str(base / 'dataset' / 'val_full.txt'))
    inter = np.zeros(nc, dtype=np.float64); union = np.zeros(nc, dtype=np.float64)
    times = []
    print(f'infer {len(lines)} val images (run={a.run}) ...')
    # warmup 1 张（不计入计时）
    wimg = cv2.imread(lines[0][0])
    _ = sliding_predict(model, wimg, cfg['tile_size'], cfg['infer_stride'], device)
    torch.cuda.synchronize() if device == 'cuda' else None
    for i, (ip, mp) in enumerate(lines):
        img = cv2.imread(ip)
        gt = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if device == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        pred = sliding_predict(model, img, cfg['tile_size'], cfg['infer_stride'], device)
        if device == 'cuda': torch.cuda.synchronize()
        times.append(time.time() - t0)
        a_i, b_i = iou(pred, gt, nc); inter += a_i; union += b_i
        cv2.imwrite(str(ov_dir / (Path(ip).stem + '.png')), overlay(img, pred))
        if i < 3 or i % 10 == 0:
            print(f'  [{i+1}/{len(lines)}] {Path(ip).name} {times[-1]*1000:.0f}ms')
    ious = inter / (union + 1e-9)
    avg_ms = float(np.mean(times)) * 1000
    print('\n=== SegFormer 全图评估 ===')
    print(f'background IoU={ious[0]:.4f}')
    print(f'silie IoU={ious[1]:.4f}')
    print(f'fanzhe IoU={ious[2]:.4f}')
    print(f'mIoU={ious.mean():.4f}')
    print(f'avg inference time = {avg_ms:.1f} ms/image  (n={len(times)})')
    with open(run_dir / 'eval_result.txt', 'w', encoding='utf-8') as f:
        f.write(f'background_IoU={ious[0]:.4f}\nsilie_IoU={ious[1]:.4f}\nfanzhe_IoU={ious[2]:.4f}\n'
                f'mIoU={ious.mean():.4f}\navg_inference_ms={avg_ms:.1f}\nnum_images={len(times)}\n')

if __name__ == '__main__':
    main()
