# -*- coding: utf-8 -*-
"""SegFormer v2 全图滑窗推理: 640 tile + stride 320 + TTA(翻转) + mIoU + 计时。"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import argparse, time, yaml
import numpy as np
import cv2, torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
from postprocess import logits_to_labels, resolve_confidence_threshold

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255

def load_val_full(txt):
    out = []
    for line in open(txt, encoding='utf-8'):
        line = line.rstrip('\n')
        if line: out.append(line.split('\t'))
    return out

def _forward_tile(model, patch, device):
    inp = patch[:, :, ::-1].astype(np.float32)  # BGR -> RGB
    inp = (inp - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        lg = model(pixel_values=t).logits
    return lg[0].cpu().numpy()  # (C, h, w)

def sliding_predict(model, img_bgr, tile, stride, device, tta,
                    confidence_threshold=0.5):
    H, W = img_bgr.shape[:2]
    acc = None; cov = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] + tile < H: ys.append(max(0, H - tile))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] + tile < W: xs.append(max(0, W - tile))
    for y0 in ys:
        for x0 in xs:
            patch = img_bgr[y0:y0+tile, x0:x0+tile]
            ph, pw = patch.shape[:2]
            lg = _forward_tile(model, patch, device)
            # TTA: 翻转
            if tta:
                lg_f = _forward_tile(model, patch[:, ::-1], device)[:, :, ::-1]
                lg_v = _forward_tile(model, patch[::-1], device)[:, ::-1, :]
                lg = (lg + lg_f + lg_v) / 3.0
            lg = F.interpolate(torch.from_numpy(lg).unsqueeze(0), size=(ph, pw),
                               mode='bilinear', align_corners=False)[0].numpy()
            if acc is None:
                acc = np.zeros((lg.shape[0], H, W), dtype=np.float32)
            acc[:, y0:y0+ph, x0:x0+pw] += lg
            cov[y0:y0+ph, x0:x0+pw] += 1.0
    cov = np.maximum(cov, 1e-6)
    return logits_to_labels(acc / cov[None], confidence_threshold)

def overlay(img, pred, alpha=0.5):
    ov = img.copy()
    for c, col in enumerate([(60,60,60),(0,0,255),(255,0,0)]):
        m = pred == c
        ov[m] = (img[m].astype(np.float32)*(1-alpha) + np.array(col,dtype=np.float32)*alpha).astype(np.uint8)
    return ov

def iou(pred, gt, nc=3):
    inter = np.zeros(nc, dtype=np.float64); union = np.zeros(nc, dtype=np.float64)
    for c in range(nc):
        p = pred==c; t = gt==c
        inter[c] = (p&t).sum(); union[c] = (p|t).sum()
    return inter, union

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v2')
    ap.add_argument('--no_tta', action='store_true')
    ap.add_argument('--confidence-threshold', type=float, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config.yaml', encoding='utf-8'))
    threshold = resolve_confidence_threshold(cfg, a.confidence_threshold)
    device = 'cuda'
    base = Path(cfg['project_root']) / 'segformer_v2'
    out = Path(cfg['out_dir']) / a.run
    ov_dir = out / 'overlays'; ov_dir.mkdir(parents=True, exist_ok=True)
    tta = cfg['tta'] and not a.no_tta
    model = SegformerForSemanticSegmentation.from_pretrained(out / 'best').to(device).eval()
    lines = load_val_full(str(base / 'dataset' / 'val_full.txt'))
    inter = np.zeros(3, dtype=np.float64); union = np.zeros(3, dtype=np.float64)
    times = []
    _ = sliding_predict(model, cv2.imread(lines[0][0]), cfg['tile_size'], cfg['infer_stride'],
                        device, tta, threshold)
    torch.cuda.synchronize()
    print(f'infer {len(lines)} val images (TTA={tta}, confidence_threshold={threshold}) ...')
    for i, (ip, mp) in enumerate(lines):
        img = cv2.imread(ip); gt = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        torch.cuda.synchronize(); t0 = time.time()
        pred = sliding_predict(model, img, cfg['tile_size'], cfg['infer_stride'],
                               device, tta, threshold)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        a_i, b_i = iou(pred, gt); inter += a_i; union += b_i
        cv2.imwrite(str(ov_dir / (Path(ip).stem + '.png')), overlay(img, pred))
        if i < 3 or i % 10 == 0:
            print(f'  [{i+1}/{len(lines)}] {Path(ip).name} {times[-1]*1000:.0f}ms')
    ious = inter / (union + 1e-9); avg_ms = float(np.mean(times))*1000
    print(f'\n=== SegFormer v2 全图评估 (TTA={tta}) ===')
    print(f'background IoU={ious[0]:.4f}\nsilie IoU={ious[1]:.4f}\nfanzhe IoU={ious[2]:.4f}\nmIoU={ious.mean():.4f}')
    print(f'avg inference = {avg_ms:.1f} ms/image (n={len(times)})')
    with open(out / 'eval_result.txt', 'w', encoding='utf-8') as f:
        f.write(f'background_IoU={ious[0]:.4f}\nsilie_IoU={ious[1]:.4f}\nfanzhe_IoU={ious[2]:.4f}\n'
                f'mIoU={ious.mean():.4f}\navg_inference_ms={avg_ms:.1f}\nnum_images={len(times)}\n'
                f'use_tta={tta}\nconfidence_threshold={threshold}\n')

if __name__ == '__main__':
    main()
