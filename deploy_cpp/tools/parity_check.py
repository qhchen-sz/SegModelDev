# -*- coding: utf-8 -*-
"""单 tile 数值对齐：把 PyTorch 模型的原始 logits 导出成 .bin，用于和
C++ TensorRT 的 `--dump-logits` 输出逐元素对比。

这一步只验证「引擎对不对」——刻意停在模型原始输出（3x160x160），不做上采样、
不做滑窗累加，把「后处理对不对」留给端到端 IoU 检查。

预处理与 infer_fast_strat.py:45-47 逐条一致。

用法（必须用 yolov10 这个 env 的 python）:
  F:/HymsonSoftWare/MiniForge/envs/yolov10/python.exe tools/parity_check.py \
      --img <任意一张图> --out <logits_torch.bin>
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TILE = 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', required=True, help='输入图；返回其左上角 TILE x TILE 的 logits')
    ap.add_argument('--best', default=None,
                    help='best 目录，默认 segformer_v2/runs_strat/v2_strat/best')
    ap.add_argument('--out', required=True, help='输出 float32 .bin')
    ap.add_argument('--fp16', action='store_true', default=True)
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent      # 工程根目录
    best = Path(a.best) if a.best else root / 'segformer_v2' / 'runs_strat' / 'v2_strat' / 'best'

    device = 'cuda'
    print(f'loading {best}')
    model = SegformerForSemanticSegmentation.from_pretrained(str(best)).to(device).eval()

    img = cv2.imread(a.img, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f'cannot read {a.img}')
    patch = img[0:TILE, 0:TILE]
    ph, pw = patch.shape[:2]
    if ph < TILE or pw < TILE:
        patch = cv2.copyMakeBorder(patch, 0, TILE - ph, 0, TILE - pw, cv2.BORDER_REFLECT)

    x = patch[:, :, ::-1].astype(np.float32) / 255.0     # BGR -> RGB, /255
    x = (x - MEAN) / STD
    t = torch.from_numpy(x).permute(2, 0, 1)[None].contiguous().to(device)

    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.float16, enabled=a.fp16):
            logits = model(pixel_values=t).logits

    arr = logits.float().cpu().numpy()[0]                # (3, 160, 160) —— 原始输出，不是插值后
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(a.out)
    print(f'wrote {a.out}  shape={arr.shape}  dtype={arr.dtype}  fp16={a.fp16}')
    print(f'  range=[{arr.min():.4f}, {arr.max():.4f}]')

    # 顺便报告一份插值到 640 后的 argmax 分布，肉眼确认模型不是全预测背景
    up = F.interpolate(logits.float(), size=(TILE, TILE), mode='bilinear', align_corners=False)
    pred = up.argmax(1)[0].cpu().numpy()
    for c in range(3):
        print(f'  tile argmax class {c}: {(pred == c).sum()} px')


if __name__ == '__main__':
    main()
