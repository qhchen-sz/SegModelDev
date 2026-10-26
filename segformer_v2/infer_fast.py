# -*- coding: utf-8 -*-
"""
SegFormer v2 加速推理测试: 批量前向(bs16) + FP16 autocast + stride 448 + 无 TTA。
精度与原 v2 几乎一致(只去 TTA、提 stride overlap), 速度大幅提升。
输出 runs/v2_fast/eval_result.txt + overlays。
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import time, yaml
import numpy as np
import cv2, torch
import torch.nn.functional as F
from torch.amp import autocast
from transformers import SegformerForSemanticSegmentation
from pathlib import Path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TILE = 640
STRIDE = 448          # overlap ~30%
BATCH = 16
FP16 = True

def load_val_full(txt):
    out = []
    for line in open(txt, encoding='utf-8'):
        line = line.rstrip('\n')
        if line: out.append(line.split('\t'))
    return out

def collect_tiles(img_bgr, tile, stride):
    """返回 [(y0,x0,patch_padded_640, ph, pw)]"""
    H, W = img_bgr.shape[:2]
    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] + tile < H: ys.append(max(0, H - tile))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] + tile < W: xs.append(max(0, W - tile))
    tiles = []
    for y0 in ys:
        for x0 in xs:
            patch = img_bgr[y0:y0+tile, x0:x0+tile]
            ph, pw = patch.shape[:2]
            if ph < tile or pw < tile:
                patch = cv2.copyMakeBorder(patch, 0, tile-ph, 0, tile-pw, cv2.BORDER_REFLECT)
            tiles.append((y0, x0, patch, ph, pw))
    return tiles, H, W

def batch_forward(model, patches_bgr, device, fp16):
    """patches: list of (tile,tile,3) BGR uint8 -> logits tensor (B, C, tile, tile) on cpu"""
    arr = [p[:, :, ::-1].astype(np.float32) / 255.0 for p in patches_bgr]  # BGR->RGB, 0-1
    arr = [(a - IMAGENET_MEAN) / IMAGENET_STD for a in arr]
    t = torch.from_numpy(np.stack(arr, 0)).permute(0, 3, 1, 2).to(device).float()
    with torch.no_grad():
        with autocast('cuda', dtype=torch.float16) if fp16 else torch.no_grad():
            lg = model(pixel_values=t).logits  # (B, C, h/4, w/4)
    lg = F.interpolate(lg.float(), size=(TILE, TILE), mode='bilinear', align_corners=False)
    return lg.cpu().numpy()

def sliding_predict_fast(model, img_bgr, device, fp16):
    tiles, H, W = collect_tiles(img_bgr, TILE, STRIDE)
    nc = None
    acc = None; cov = np.zeros((H, W), dtype=np.float32)
    # 分批
    for i in range(0, len(tiles), BATCH):
        batch = tiles[i:i+BATCH]
        patches = [b[2] for b in batch]
        logits = batch_forward(model, patches, device, fp16)  # (B,C,640,640)
        if nc is None:
            nc = logits.shape[1]
            acc = np.zeros((nc, H, W), dtype=np.float32)
        for j, (y0, x0, _, ph, pw) in enumerate(batch):
            lg = logits[j][:, :ph, :pw]  # (C, ph, pw) 取有效区域
            acc[:, y0:y0+ph, x0:x0+pw] += lg
            cov[y0:y0+ph, x0:x0+pw] += 1.0
    cov = np.maximum(cov, 1e-6)
    return (acc / cov[None]).argmax(0).astype(np.uint8)

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
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config.yaml', encoding='utf-8'))
    device = 'cuda'
    base = Path(cfg['project_root']) / 'segformer_v2'
    out = Path(cfg['out_dir']) / 'v2_fast'
    ov_dir = out / 'overlays'; ov_dir.mkdir(parents=True, exist_ok=True)
    model = SegformerForSemanticSegmentation.from_pretrained(out.parent / 'v2' / 'best').to(device).eval()
    lines = load_val_full(str(base / 'dataset' / 'val_full.txt'))
    inter = np.zeros(3, dtype=np.float64); union = np.zeros(3, dtype=np.float64)
    times = []
    # warmup
    _ = sliding_predict_fast(model, cv2.imread(lines[0][0]), device, FP16)
    torch.cuda.synchronize()
    print(f'fast infer {len(lines)} images (bs{BATCH} fp16={FP16} stride{STRIDE} noTTA) ...')
    for i, (ip, mp) in enumerate(lines):
        img = cv2.imread(ip); gt = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        torch.cuda.synchronize(); t0 = time.time()
        pred = sliding_predict_fast(model, img, device, FP16)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        a_i, b_i = iou(pred, gt); inter += a_i; union += b_i
        cv2.imwrite(str(ov_dir / (Path(ip).stem + '.png')), overlay(img, pred))
        if i < 3 or i % 10 == 0:
            print(f'  [{i+1}/{len(lines)}] {Path(ip).name} {times[-1]*1000:.0f}ms')
    ious = inter / (union + 1e-9); avg_ms = float(np.mean(times))*1000
    print(f'\n=== SegFormer v2-fast (bs{BATCH} fp16 stride{STRIDE} noTTA) ===')
    print(f'background IoU={ious[0]:.4f}\nsilie IoU={ious[1]:.4f}\nfanzhe IoU={ious[2]:.4f}\nmIoU={ious.mean():.4f}')
    print(f'avg inference = {avg_ms:.1f} ms/image (n={len(times)})')
    with open(out / 'eval_result.txt', 'w', encoding='utf-8') as f:
        f.write(f'background_IoU={ious[0]:.4f}\nsilie_IoU={ious[1]:.4f}\nfanzhe_IoU={ious[2]:.4f}\n'
                f'mIoU={ious.mean():.4f}\navg_inference_ms={avg_ms:.1f}\nnum_images={len(times)}\n'
                f'config=bs{BATCH}_fp16{FP16}_stride{STRIDE}_noTTA\n')

if __name__ == '__main__':
    main()
