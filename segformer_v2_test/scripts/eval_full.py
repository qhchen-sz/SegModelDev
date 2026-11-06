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
from sweep_eval import load_val_full, predict  # noqa: E402


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

        if int((gt > 0).sum()) == 0:
            neg_fp.append(int((pred > 0).sum()))
        else:
            n_pos += 1
            psi += int(((pred == 1) & (gt == 1)).sum())
            psu += int(((pred == 1) | (gt == 1)).sum())
            pfi += int(((pred == 2) & (gt == 2)).sum())
            pfu += int(((pred == 2) | (gt == 2)).sum())

    ious = inter / (union + 1e-9)
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

    print(f"[{args.run}] val mIoU={metrics['mIoU']} "
          f"bg={metrics['bg']} silie={metrics['silie']} fanzhe={metrics['fanzhe']} | "
          f"pos_silie={metrics['pos_silie']} pos_fanzhe={metrics['pos_fanzhe']} | "
          f"neg avg_fp={metrics['neg_avg_fp']} max={metrics['neg_max_fp']} "
          f"gt100={metrics['neg_gt100']}/{metrics['n_neg']} clean={metrics['neg_clean']} | "
          f"avg_ms={metrics['avg_ms']}")
    print(f"-> {out / 'eval_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
