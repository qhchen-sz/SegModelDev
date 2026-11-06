# -*- coding: utf-8 -*-
"""抠取 train split 里的 silie 缺陷裁剪，供 SegV2Dataset 的 copy-paste 使用。

与 newsiliec9/segformer_v2/preprocess_silie_crops.py 逻辑一致，只把入口改成
--config，便于每个数据集使用自己的 config_sw_s4.yaml。

用法:
    python preprocess_silie_crops.py --config <DS>/config_sw_s4.yaml
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml


def run(cfg):
    root = Path(cfg["project_root"])
    out = Path(cfg["silie_crops_dir"])
    out.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    # 清理旧 crops，避免残留
    for f in list(out.glob("*_img.jpg")) + list(out.glob("*_mask.png")):
        f.unlink()

    train_imgs = [line.strip().replace("\\", "/")
                  for line in open(cfg["train_list"], encoding="utf-8")
                  if line.strip()]
    n = 0
    skipped = []
    for ip in train_imgs:
        jp = Path(ip).with_suffix(".json")
        if not jp.exists():
            skipped.append({"image": ip, "reason": "missing json"})
            continue
        d = json.load(open(jp, encoding="utf-8"))
        img = cv2.imread(ip)
        if img is None:
            skipped.append({"image": ip, "reason": "unreadable image"})
            continue
        h, w = img.shape[:2]
        stem = Path(ip).stem
        for i, obj in enumerate(d.get("objects", [])):
            cat = str(obj.get("category", "")).strip().lower()
            if cat != "silie":
                continue
            seg = obj.get("segmentation", [])
            if not seg:
                skipped.append({"image": ip, "object": i, "reason": "no segmentation"})
                continue
            poly = np.array(seg, dtype=np.int32)
            x0, y0 = int(poly[:, 0].min()), int(poly[:, 1].min())
            x1, y1 = int(poly[:, 0].max()), int(poly[:, 1].max())
            pad = 15
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
            if x1 - x0 < 5 or y1 - y0 < 5:
                skipped.append({"image": ip, "object": i, "reason": "degenerate bbox"})
                continue
            crop = img[y0:y1, x0:x1].copy()
            cmask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillPoly(cmask, [poly - np.array([x0, y0])], 1)
            if int((cmask > 0).sum()) == 0:
                skipped.append({"image": ip, "object": i, "reason": "empty mask"})
                continue
            cv2.imwrite(str(out / f"{stem}_{i}_img.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / f"{stem}_{i}_mask.png"), cmask)
            n += 1

    report = {
        "train_images": len(train_imgs),
        "crops": n,
        "skipped": skipped,
        "config": str(cfg.get("project_root", "")),
    }
    (logs / "crops_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"silie crops: {n} -> {out}")
    if skipped:
        print(f"skipped: {len(skipped)} (see {logs / 'crops_report.json'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    run(cfg)


if __name__ == "__main__":
    main()
