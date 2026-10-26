# -*- coding: utf-8 -*-
"""交叉校验：用 segformer_v2/sweep_eval.py 的**参考实现**跑一遍 val_strat，
导出 label map 并算 IoU，用于和 C++/TensorRT 的输出逐像素对比。

刻意 import sweep_eval 复用它的 predict()/load_val_full()，而不是重新实现一遍——
重新实现会引入口径漂移，那样对比就失去意义了。

用途：当 C++ 的 IoU 与 sweep_eval.json 里的基线对不齐时，回答「是 fp16 内核差异
(可接受)，还是部署管线有 bug (不可接受)」。

用法（必须用 yolov10 这个 env 的 python）:
  F:/HymsonSoftWare/MiniForge/envs/yolov10/python.exe tools/crosscheck_eval.py \
      --best ../segformer_v2/runs_strat/sw_s4/best \
      --val-full ../segformer_v2/dataset_strat/val_full.txt \
      --out out_py
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                       # 工程根目录

sys.path.insert(0, str(ROOT / 'segformer_v2'))
import sweep_eval                               # noqa: E402  参考实现


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--best', required=True, help='best 目录（含 config.json + model.safetensors）')
    ap.add_argument('--val-full', required=True, help='val_full.txt（图<TAB>GT）')
    ap.add_argument('--out', required=True, help='label map 输出目录')
    ap.add_argument('--limit', type=int, default=-1)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    model = SegformerForSemanticSegmentation.from_pretrained(a.best).to('cuda').eval()
    lines = sweep_eval.load_val_full(a.val_full)
    if a.limit > 0:
        lines = lines[:a.limit]
    print(f'crosscheck: {len(lines)} images, best={a.best}')

    inter = np.zeros(3, np.float64)
    union = np.zeros(3, np.float64)
    times = []
    for ip, mp in lines:
        img = cv2.imread(ip)
        gt = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        torch.cuda.synchronize()
        t0 = time.time()
        pred = sweep_eval.predict(model, img)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

        cv2.imwrite(os.path.join(a.out, Path(ip).stem + '.png'), pred)
        for c in range(3):
            inter[c] += ((pred == c) & (gt == c)).sum()
            union[c] += ((pred == c) | (gt == c)).sum()

    ious = inter / (union + 1e-9)
    print(f'background IoU={ious[0]:.4f}')
    print(f'silie IoU={ious[1]:.4f}')
    print(f'fanzhe IoU={ious[2]:.4f}')
    print(f'mIoU={ious.mean():.4f}')
    print(f'avg inference = {np.mean(times) * 1000:.1f} ms/image (n={len(times)})')


if __name__ == '__main__':
    main()
