# -*- coding: utf-8 -*-
"""校验 labels / split / silie_crops / dataset_strat 的完整性和一致性。

用法:
    python validate_dataset.py --config <DS>/config_sw_s4.yaml --stage all
失败时退出码为 1，并把详细结果写到 <project_root>/logs/validate_report.json。
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_split(path):
    return [line.strip() for line in open(path, encoding="utf-8") if line.strip()]


def check_labels(cfg, rep):
    root = Path(cfg["project_root"])
    images = sorted((root / "images").glob("*.jpg"))
    labels = sorted((root / "labels").glob("*.txt"))
    empty = 0
    lines = 0
    bad = []
    classes = Counter()
    for lp in labels:
        rows = [r.split() for r in lp.read_text(encoding="utf-8").splitlines() if r.strip()]
        if not rows:
            empty += 1
        lines += len(rows)
        for r in rows:
            if len(r) != 5:
                bad.append({"label": lp.name, "reason": "not 5 fields", "row": r})
                continue
            try:
                c = int(r[0])
                vals = [float(v) for v in r[1:]]
            except ValueError:
                bad.append({"label": lp.name, "reason": "non-numeric", "row": r})
                continue
            if c not in (1, 2):
                bad.append({"label": lp.name, "reason": f"class {c}", "row": r})
            if any(v < 0 or v > 1 for v in vals):
                bad.append({"label": lp.name, "reason": "normalized value out of range", "row": r})
            classes[c] += 1
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    rep["labels"] = {
        "images": len(images),
        "label_files": len(labels),
        "empty_labels": empty,
        "label_lines": lines,
        "class_lines": dict(classes),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "bad_rows": bad,
    }
    if missing_labels or orphan_labels or bad:
        rep["errors"].append("label problems")


def check_split(cfg, rep):
    root = Path(cfg["project_root"])
    train = load_split(cfg["train_list"])
    val = load_split(cfg["val_list"])
    missing = [p for p in train + val if not Path(p).exists()]
    overlap = sorted(set(train) & set(val))
    no_label = [p for p in train + val
                if not (root / "labels" / (Path(p).stem + ".txt")).exists()]
    rep["split"] = {
        "train": len(train),
        "val": len(val),
        "missing_images": missing,
        "overlap": overlap,
        "missing_labels": no_label,
    }
    if missing or overlap or no_label:
        rep["errors"].append("split problems")


def check_crops(cfg, rep):
    out = Path(cfg["silie_crops_dir"])
    imgs = sorted(out.glob("*_img.jpg"))
    masks = sorted(out.glob("*_mask.png"))
    img_stems = {p.name.replace("_img.jpg", "") for p in imgs}
    mask_stems = {p.name.replace("_mask.png", "") for p in masks}
    bad = []
    values = Counter()
    fg_pixels = 0
    for mp in masks:
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if m is None:
            bad.append({"mask": mp.name, "reason": "unreadable"})
            continue
        if m.ndim != 2:
            bad.append({"mask": mp.name, "reason": f"shape {m.shape}"})
            continue
        uniq = [int(v) for v in np.unique(m)]
        values.update(uniq)
        if any(v not in (0, 1) for v in uniq):
            bad.append({"mask": mp.name, "reason": f"values {uniq}"})
        fg_pixels += int((m > 0).sum())
    rep["crops"] = {
        "images": len(imgs),
        "masks": len(masks),
        "missing_masks": sorted(img_stems - mask_stems),
        "orphan_masks": sorted(mask_stems - img_stems),
        "mask_values": dict(values),
        "foreground_pixels": fg_pixels,
        "bad": bad,
    }
    if not imgs or img_stems != mask_stems or bad or fg_pixels == 0:
        rep["errors"].append("silie crop problems")


def check_dataset(cfg, rep):
    data = Path(cfg["dataset_dir"])
    summary = {}
    expected_tile = int(cfg["tile_size"])
    for split in ("train", "val"):
        idir = data / split / "images"
        mdir = data / split / "masks"
        imgs = sorted(idir.glob("*.jpg"))
        masks = sorted(mdir.glob("*.png"))
        img_stems = {p.stem for p in imgs}
        mask_stems = {p.stem for p in masks}
        bad = []
        values = Counter()
        defect_tiles = 0
        for mp in masks:
            m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if m is None:
                bad.append({"mask": mp.name, "reason": "unreadable"})
                continue
            if m.ndim != 2:
                bad.append({"mask": mp.name, "reason": f"shape {m.shape}"})
                continue
            if m.shape != (expected_tile, expected_tile):
                bad.append({"mask": mp.name, "reason": f"size {m.shape}"})
            uniq = [int(v) for v in np.unique(m)]
            values.update(uniq)
            if any(v not in (0, 1, 2) for v in uniq):
                bad.append({"mask": mp.name, "reason": f"values {uniq}"})
            if int((m > 0).sum()) > 0:
                defect_tiles += 1
        # 抽查 tile 图片尺寸（全部检查成本可控）
        for ip in imgs:
            im = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
            if im is None:
                bad.append({"image": ip.name, "reason": "unreadable"})
                continue
            if im.shape[:2] != (expected_tile, expected_tile):
                bad.append({"image": ip.name, "reason": f"size {im.shape[:2]}"})
        summary[split] = {
            "images": len(imgs),
            "masks": len(masks),
            "missing_masks": sorted(img_stems - mask_stems),
            "orphan_masks": sorted(mask_stems - img_stems),
            "mask_values": dict(values),
            "defect_tiles": defect_tiles,
            "background_tiles": len(masks) - defect_tiles,
            "bad": bad,
        }
        if img_stems != mask_stems or bad:
            rep["errors"].append(f"{split} tile problems")

    # val_full
    val_full_txt = Path(cfg["val_full_txt"])
    val_full_dir = data / "val_full"
    lines = load_split(val_full_txt)
    pairs = []
    missing = []
    dim_mismatch = []
    bad_masks = []
    values = Counter()
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 2:
            missing.append({"line": line, "reason": "not image<TAB>mask"})
            continue
        ip, mp = Path(parts[0]), Path(parts[1])
        pairs.append((ip, mp))
        if not ip.exists() or not mp.exists():
            missing.append({"image": str(ip), "mask": str(mp), "reason": "missing file"})
            continue
        im = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if im is None or m is None:
            missing.append({"image": str(ip), "mask": str(mp), "reason": "unreadable"})
            continue
        if im.shape[:2] != m.shape[:2]:
            dim_mismatch.append({"image": str(ip), "image_shape": im.shape[:2],
                                 "mask_shape": m.shape[:2]})
        uniq = [int(v) for v in np.unique(m)]
        values.update(uniq)
        if any(v not in (0, 1, 2) for v in uniq):
            bad_masks.append({"mask": str(mp), "values": uniq})
    rep["val_full"] = {
        "txt": str(val_full_txt),
        "lines": len(lines),
        "png_files": len(list(val_full_dir.glob("*.png"))),
        "mask_values": dict(values),
        "missing": missing,
        "dim_mismatch": dim_mismatch,
        "bad_masks": bad_masks,
    }
    if missing or dim_mismatch or bad_masks or len(lines) != len(list(val_full_dir.glob("*.png"))):
        rep["errors"].append("val_full problems")
    rep["dataset"] = summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    rep = {"config": str(args.config), "errors": []}
    check_labels(cfg, rep)
    check_split(cfg, rep)
    check_crops(cfg, rep)
    check_dataset(cfg, rep)

    out = Path(cfg["project_root"]) / "logs" / "validate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== validate ===")
    print(f"labels: {rep['labels']['label_files']} files, {rep['labels']['empty_labels']} empty, "
          f"{rep['labels']['label_lines']} lines")
    print(f"split : train={rep['split']['train']} val={rep['split']['val']} "
          f"overlap={len(rep['split']['overlap'])}")
    print(f"crops : {rep['crops']['images']} img / {rep['crops']['masks']} mask")
    for split in ("train", "val"):
        s = rep["dataset"][split]
        print(f"{split:5s}: {s['images']} img / {s['masks']} mask, "
              f"defect={s['defect_tiles']} bg={s['background_tiles']}")
    print(f"val_full: {rep['val_full']['lines']} lines, {rep['val_full']['png_files']} png")
    if rep["errors"]:
        print("[FAIL]", rep["errors"])
        print(f"report: {out}")
        return 1
    print("[OK]", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
