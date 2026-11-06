# -*- coding: utf-8 -*-
"""Score a directory of predicted masks against val_full.txt, and optionally
compare it pixel-by-pixel with a reference mask directory (e.g. C++ vs C#).

Supports both layouts:
    <pred-dir>/<stem>.png            (SegTest --eval)
    <pred-dir>/mask/<stem>.png       (run.bat --out ... --mask)

Usage:
    python score_masks.py --val-full <DS>/dataset_strat/val_full.txt \
        --pred-dir <DS>/runs_strat/sw_s4/cs_eval --out ...json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_mask(path):
    """Read an 8-bit mask as class indices.

    C++ writes grayscale PNG; GDI+ (SegTest) writes a palette PNG, which
    cv2.imread(IMREAD_UNCHANGED) expands to BGR. PIL preserves the palette
    indices, so use PIL for both.
    """
    try:
        im = Image.open(path)
        if im.mode in ("P", "L"):
            return np.array(im)
        return np.array(im.convert("L"))
    except Exception:
        return None


def find_mask(pred_dir, stem):
    # Prefer mask/<stem>.png (run.bat --eval --mask writes overlay <stem>.png
    # and the 0/1/2 mask under mask/). Fall back to flat <stem>.png (SegTest).
    p = pred_dir / "mask" / (stem + ".png")
    if p.exists():
        return p
    p = pred_dir / (stem + ".png")
    if p.exists():
        return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-full", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--ref-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = [l.strip() for l in open(args.val_full, encoding="utf-8") if l.strip()]
    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir) if args.ref_dir else None

    inter = np.zeros(3, dtype=np.int64)
    union = np.zeros(3, dtype=np.int64)
    pix = np.zeros(3, dtype=np.int64)
    missing = []
    ref_missing = []
    ref_pairs = 0
    ref_mismatch_px = 0
    ref_mismatch_imgs = 0

    for line in lines:
        parts = line.split("\t")
        if len(parts) != 2:
            missing.append({"line": line, "reason": "not image<TAB>mask"})
            continue
        ip, gt_path = Path(parts[0]), Path(parts[1])
        pm = find_mask(pred_dir, ip.stem)
        if pm is None:
            missing.append({"image": str(ip), "reason": "no predicted mask"})
            continue
        gt = read_mask(gt_path)
        pr = read_mask(pm)
        if gt is None or pr is None:
            missing.append({"image": str(ip), "pred": str(pm), "reason": "unreadable"})
            continue
        if pr.shape[:2] != gt.shape[:2]:
            pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        for c in range(3):
            inter[c] += int(((pr == c) & (gt == c)).sum())
            union[c] += int(((pr == c) | (gt == c)).sum())
            pix[c] += int((gt == c).sum())

        if ref_dir is not None:
            rm = find_mask(ref_dir, ip.stem)
            if rm is None:
                ref_missing.append(str(ip))
            else:
                ref = read_mask(rm)
                if ref is None:
                    ref_missing.append(str(ip))
                else:
                    if ref.shape[:2] != pr.shape[:2]:
                        ref = cv2.resize(ref, (pr.shape[1], pr.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                    diff = int((ref != pr).sum())
                    ref_pairs += 1
                    ref_mismatch_px += diff
                    if diff:
                        ref_mismatch_imgs += 1

    ious = inter / (union + 1e-9)
    result = {
        "val_full": args.val_full,
        "pred_dir": args.pred_dir,
        "ref_dir": args.ref_dir,
        "n_val": len(lines),
        "missing": missing,
        "bg_iou": round(float(ious[0]), 6),
        "silie_iou": round(float(ious[1]), 6),
        "fanzhe_iou": round(float(ious[2]), 6),
        "mIoU": round(float(ious.mean()), 6),
        "gt_pixels": {"bg": int(pix[0]), "silie": int(pix[1]), "fanzhe": int(pix[2])},
        "ref_pairs": ref_pairs,
        "ref_missing": ref_missing,
        "ref_mismatch_pixels": ref_mismatch_px,
        "ref_mismatch_images": ref_mismatch_imgs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mIoU={result['mIoU']} bg={result['bg_iou']} "
          f"silie={result['silie_iou']} fanzhe={result['fanzhe_iou']} "
          f"missing={len(missing)}")
    if ref_dir is not None:
        print(f"ref: pairs={ref_pairs} mismatch_px={ref_mismatch_px} "
              f"mismatch_imgs={ref_mismatch_imgs} missing={len(ref_missing)}")
    print(f"-> {out}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
