# -*- coding: utf-8 -*-
"""Mask2Former 全图滑窗推理（分层重切版）: 读 config_strat.yaml，val_full 来自 v2 dataset_strat，
输出到 runs_strat/<run>/eval_result.txt。绝不改动 eval.py。"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import time, yaml
import numpy as np
import cv2, torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from pathlib import Path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
OUT = 384

def load_val_full(txt):
    out = []
    for line in open(txt, encoding='utf-8'):
        line = line.rstrip('\n')
        if line: out.append(line.split('\t'))
    return out

def sliding_predict(model, proc, img_bgr, tile, stride, device):
    H, W = img_bgr.shape[:2]
    votes = np.zeros((3, H, W), dtype=np.float32)
    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] + tile < H: ys.append(max(0, H - tile))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] + tile < W: xs.append(max(0, W - tile))
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                patch = img_bgr[y0:y0+tile, x0:x0+tile]
                ph, pw = patch.shape[:2]
                inp = cv2.resize(patch, (OUT, OUT), interpolation=cv2.INTER_LINEAR)
                inp = inp[:, :, ::-1].astype(np.float32)
                inp = (inp / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device).float()
                pm = torch.ones(1, OUT, OUT, device=device)
                out = model(pixel_values=t, pixel_mask=pm)
                res = proc.post_process_instance_segmentation(out, target_sizes=[(OUT, OUT)],
                    threshold=0.5, mask_threshold=0.5)[0]
                seg = res['segmentation'].cpu().numpy()
                sem = np.zeros((OUT, OUT), dtype=np.uint8)
                for info in res['segments_info']:
                    sem[seg == info['id']] = int(info['label_id']) + 1
                sem = cv2.resize(sem, (pw, ph), interpolation=cv2.INTER_NEAREST)
                for c in range(3):
                    votes[c, y0:y0+ph, x0:x0+pw] += (sem == c)
    return votes.argmax(0).astype(np.uint8)

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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='full_strat')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config_strat.yaml', encoding='utf-8'))
    device = 'cuda'
    out = Path(cfg['out_dir']) / a.run
    ov_dir = out / 'overlays'; ov_dir.mkdir(parents=True, exist_ok=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(out / 'best').to(device).eval()
    proc = Mask2FormerImageProcessor.from_pretrained(cfg['model_ckpt'])
    lines = load_val_full(str(cfg['val_full_txt']))
    inter = np.zeros(3, dtype=np.float64); union = np.zeros(3, dtype=np.float64)
    times = []
    _ = sliding_predict(model, proc, cv2.imread(lines[0][0]), cfg['tile_size'], cfg['infer_stride'], device)
    torch.cuda.synchronize()
    print(f'infer {len(lines)} val images ...')
    for i, (ip, mp) in enumerate(lines):
        img = cv2.imread(ip); gt = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        torch.cuda.synchronize(); t0 = time.time()
        pred = sliding_predict(model, proc, img, cfg['tile_size'], cfg['infer_stride'], device)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        a_i, b_i = iou(pred, gt); inter += a_i; union += b_i
        cv2.imwrite(str(ov_dir / (Path(ip).stem + '.png')), overlay(img, pred))
        if i < 3 or i % 10 == 0:
            print(f'  [{i+1}/{len(lines)}] {Path(ip).name} {times[-1]*1000:.0f}ms')
    ious = inter / (union + 1e-9)
    avg_ms = float(np.mean(times)) * 1000
    print('\n=== Mask2Former 全图评估(分层重切) ===')
    print(f'background IoU={ious[0]:.4f}\nsilie IoU={ious[1]:.4f}\nfanzhe IoU={ious[2]:.4f}\nmIoU={ious.mean():.4f}')
    print(f'avg inference = {avg_ms:.1f} ms/image (n={len(times)})')
    with open(out / 'eval_result.txt', 'w', encoding='utf-8') as f:
        f.write(f'background_IoU={ious[0]:.4f}\nsilie_IoU={ious[1]:.4f}\nfanzhe_IoU={ious[2]:.4f}\n'
                f'mIoU={ious.mean():.4f}\navg_inference_ms={avg_ms:.1f}\nnum_images={len(times)}\n')

if __name__ == '__main__':
    main()
