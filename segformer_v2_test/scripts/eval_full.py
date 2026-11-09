# -*- coding: utf-8 -*-
"""Full-image validation for one dataset/model.

Reuses the canonical 640/448/16 fp16 no-TTA predictor from pipeline/sweep_eval.py,
but evaluates only this dataset's val_full.txt (no hard-coded FIELD folders).

Usage:
    python eval_full.py --config <DS>/config_sw_s4.yaml --run sw_s4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from transformers import SegformerForSemanticSegmentation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pipeline"))
from sweep_eval import load_val_full, predict, BATCH, STRIDE, TILE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", default="sw_s4")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = Path(cfg["out_dir"]) / args.run
    best = out / "best"
    if not (best / "config.json").exists():
        print(f"[ERROR] best model not found: {best}")
        return 1

    model = SegformerForSemanticSegmentation.from_pretrained(best).to("cuda").eval()
    lines = load_val_full(str(cfg["val_full_txt"]))
    if not lines:
        print(f"[ERROR] empty val_full: {cfg['val_full_txt']}")
        return 1

    # warmup
    warm = cv2.imread(lines[0][0])
    if warm is None:
        print(f"[ERROR] unreadable warmup image: {lines[0][0]}")
        return 1
    _ = predict(model, warm)
    torch.cuda.synchronize()

    inter = np.zeros(3, dtype=np.float64)
    union = np.zeros(3, dtype=np.float64)
    gt_pix = np.zeros(3, dtype=np.int64)
    pred_pix = np.zeros(3, dtype=np.int64)   # TP+FP，用于 per-class Precision
    neg_fp = []
    n_pos = 0
    psi = psu = pfi = pfu = 0.0
    times = []

    for ip, mp in lines:
        img = cv2.imread(ip)
        gt = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if img is None or gt is None:
            print(f"[ERROR] unreadable pair: {ip} / {mp}")
            return 1
        if gt.ndim != 2:
            print(f"[ERROR] mask is not single-channel: {mp} shape={gt.shape}")
            return 1
        if gt.shape[:2] != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = predict(model, img)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        for c in range(3):
            inter[c] += int(((pred == c) & (gt == c)).sum())
            union[c] += int(((pred == c) | (gt == c)).sum())
        for c in range(3):
            gt_pix[c] += int((gt == c).sum())
            pred_pix[c] += int((pred == c).sum())

        if int((gt > 0).sum()) == 0:
            neg_fp.append(int((pred > 0).sum()))
        else:
            n_pos += 1
            psi += int(((pred == 1) & (gt == 1)).sum())
            psu += int(((pred == 1) | (gt == 1)).sum())
            pfi += int(((pred == 2) & (gt == 2)).sum())
            pfu += int(((pred == 2) | (gt == 2)).sum())

    ious = inter / (union + 1e-9)
    recalls = inter / (gt_pix + 1e-9)          # TP/(TP+FN)
    precisions = inter / (pred_pix + 1e-9)     # TP/(TP+FP)
    neg_fp = np.asarray(neg_fp, dtype=np.int64)
    times = np.asarray(times, dtype=np.float64) * 1000.0

    meta = {}
    meta_path = out / "best_meta.pt"
    if meta_path.exists():
        try:
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[WARN] cannot read {meta_path}: {e}")

    metrics = {
        "run": args.run,
        "config": str(args.config),
        "val_full_txt": str(cfg["val_full_txt"]),
        "bg_keep": cfg["bg_keep_ratio"],
        "alpha_bg": cfg["focal_alpha"][0],
        "n_val": len(lines),
        "n_pos": n_pos,
        "n_neg": int(len(neg_fp)),
        "mIoU": round(float(ious.mean()), 4),
        "bg": round(float(ious[0]), 4),
        "silie": round(float(ious[1]), 4),
        "fanzhe": round(float(ious[2]), 4),
        "mRecall": round(float(recalls.mean()), 4),
        "recall_bg": round(float(recalls[0]), 4),
        "recall_silie": round(float(recalls[1]), 4),
        "recall_fanzhe": round(float(recalls[2]), 4),
        "mPrecision": round(float(precisions.mean()), 4),
        "precision_bg": round(float(precisions[0]), 4),
        "precision_silie": round(float(precisions[1]), 4),
        "precision_fanzhe": round(float(precisions[2]), 4),
        "pos_silie": round(float(psi / (psu + 1e-9)), 4),
        "pos_fanzhe": round(float(pfi / (pfu + 1e-9)), 4),
        "neg_avg_fp": round(float(neg_fp.mean()), 1) if len(neg_fp) else 0.0,
        "neg_max_fp": int(neg_fp.max()) if len(neg_fp) else 0,
        "neg_gt100": int((neg_fp > 100).sum()),
        "neg_gt1000": int((neg_fp > 1000).sum()),
        "neg_clean": int((neg_fp == 0).sum()),
        "avg_ms": round(float(times.mean()), 1),
        "p50_ms": round(float(np.percentile(times, 50)), 1),
        "p90_ms": round(float(np.percentile(times, 90)), 1),
        "best_epoch": int(meta.get("epoch", -1)) if meta else -1,
        "best_tile_miou": round(float(meta.get("miou", 0.0)), 4) if meta else 0.0,
        "gt_pixels": {"bg": int(gt_pix[0]), "silie": int(gt_pix[1]), "fanzhe": int(gt_pix[2])},
    }
    (out / "eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# {args.run} 全图像素级指标（cemian1103）

- 验证集: `{cfg['val_full_txt']}`（{len(lines)} 张全图，正样本 {n_pos} / 负样本 {len(neg_fp)}）
- 口径: bs{BATCH} fp16 tile{TILE} stride{STRIDE} noTTA 全图滑窗（同 sweep_eval.predict）
- 指标定义: IoU=TP/(TP+FP+FN)，Recall=TP/(TP+FN)，Precision=TP/(TP+FP)

| 类别 | IoU | Recall | Precision |
|---|---|---|
| background | {ious[0]:.4f} | {recalls[0]:.4f} | {precisions[0]:.4f} |
| silie | {ious[1]:.4f} | {recalls[1]:.4f} | {precisions[1]:.4f} |
| fanzhe | {ious[2]:.4f} | {recalls[2]:.4f} | {precisions[2]:.4f} |
| **均值** | **{metrics['mIoU']:.4f}** | **{metrics['mRecall']:.4f}** | **{metrics['mPrecision']:.4f}** |

## 推理速度（全图）

| 平均 ms/图 | p50 | p90 |
|---|---|---|
| {metrics['avg_ms']} | {metrics['p50_ms']} | {metrics['p90_ms']} |

> 负样本 {metrics['n_neg']} 张：假阳像素均值 {metrics['neg_avg_fp']} / 最大 {metrics['neg_max_fp']}，>100px 共 {metrics['neg_gt100']} 张，全干净 {metrics['neg_clean']} 张
"""
    (out / "eval_report.md").write_text(report, encoding="utf-8")

    print(f"[{args.run}] val mIoU={metrics['mIoU']} "
          f"bg={metrics['bg']} silie={metrics['silie']} fanzhe={metrics['fanzhe']} | "
          f"mRecall={metrics['mRecall']} mPrecision={metrics['mPrecision']} | "
          f"pos_silie={metrics['pos_silie']} pos_fanzhe={metrics['pos_fanzhe']} | "
          f"neg avg_fp={metrics['neg_avg_fp']} max={metrics['neg_max_fp']} "
          f"gt100={metrics['neg_gt100']}/{metrics['n_neg']} clean={metrics['neg_clean']} | "
          f"avg_ms={metrics['avg_ms']}")
    print(f"-> {out / 'eval_metrics.json'}")
    print(f"-> {out / 'eval_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
