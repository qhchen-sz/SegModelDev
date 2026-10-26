# -*- coding: utf-8 -*-
"""YOLOv8n-seg 全图评估（分层重切版）: 用 newsiliec9_strat.yaml 训练后的 best.pt，
在 val_full_strat 上推理，实例 mask 涂成语义图，与 GT 算 per-class IoU/mIoU + 计时。
绝不改动 eval_yolo.py。"""
import os, time, yaml
import numpy as np
import cv2
import torch
from pathlib import Path

def paint_semantic(res, H, W):
    sem = np.zeros((H, W), dtype=np.uint8)
    if res.masks is None:
        return sem
    clses = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    polys = res.masks.xy
    order = np.argsort(confs)
    for k in order:
        poly = polys[k]
        if poly is None or len(poly) < 3:
            continue
        cv2.fillPoly(sem, [np.round(poly).astype(np.int32)], int(clses[k]))
    return sem

def iou(pred, gt, nc):
    inter = np.zeros(nc, dtype=np.float64); union = np.zeros(nc, dtype=np.float64)
    for c in range(nc):
        p = pred == c; t = gt == c
        inter[c] = (p & t).sum(); union[c] = (p | t).sum()
    return inter, union

def main():
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config_strat.yaml', encoding='utf-8'))
    nc = cfg['num_classes']
    root = Path(cfg['project_root'])
    out = Path(cfg['out_dir']) / 'yolo_eval'; out.mkdir(parents=True, exist_ok=True)
    ov_dir = out / 'overlays'; ov_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    yolo_ckpt = root / 'runs_strat' / 'train_strat' / 'weights' / 'best.pt'
    model = YOLO(str(yolo_ckpt))

    lines = []
    for line in open(cfg['val_full_txt'], encoding='utf-8'):
        line = line.rstrip('\n')
        if line: lines.append(line.split('\t'))
    inter = np.zeros(nc, dtype=np.float64); union = np.zeros(nc, dtype=np.float64)
    times = []
    print(f'YOLO infer {len(lines)} val images (分层重切) ...')
    _ = model(cv2.imread(lines[0][0]), imgsz=1200, verbose=False)
    torch.cuda.synchronize()
    for i, (ip, mp) in enumerate(lines):
        img = cv2.imread(ip)
        gt = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        H, W = img.shape[:2]
        torch.cuda.synchronize(); t0 = time.time()
        res = model(img, imgsz=1200, conf=0.25, iou=0.5, verbose=False)[0]
        torch.cuda.synchronize(); times.append(time.time() - t0)
        pred = paint_semantic(res, H, W)
        a, b = iou(pred, gt, nc); inter += a; union += b
        ov = img.copy()
        for c, color in enumerate([(60,60,60),(0,0,255),(255,0,0)]):
            m = pred == c
            ov[m] = (img[m].astype(np.float32)*0.5 + np.array(color,dtype=np.float32)*0.5).astype(np.uint8)
        cv2.imwrite(str(ov_dir / (Path(ip).stem + '.png')), ov)
        if i < 3 or i % 10 == 0:
            print(f'  [{i+1}/{len(lines)}] {Path(ip).name} {times[-1]*1000:.0f}ms  n_inst={len(res.boxes.cls) if res.boxes is not None else 0}')
    ious = inter / (union + 1e-9)
    avg_ms = float(np.mean(times)) * 1000
    print('\n=== YOLOv8n-seg 全图评估(分层重切) ===')
    print(f'background IoU={ious[0]:.4f}\nsilie IoU={ious[1]:.4f}\nfanzhe IoU={ious[2]:.4f}\nmIoU={ious.mean():.4f}')
    print(f'avg inference time = {avg_ms:.1f} ms/image  (n={len(times)})')
    with open(out / 'eval_result.txt', 'w', encoding='utf-8') as f:
        f.write(f'background_IoU={ious[0]:.4f}\nsilie_IoU={ious[1]:.4f}\nfanzhe_IoU={ious[2]:.4f}\n'
                f'mIoU={ious.mean():.4f}\navg_inference_ms={avg_ms:.1f}\nnum_images={len(times)}\n')

if __name__ == '__main__':
    main()
