# -*- coding: utf-8 -*-
"""ISAT JSON -> YOLO bbox labels.

用于给 split_stratified.py 提供 `labels/<stem>.txt`（只用到首列类别）。
不修改 images/ 下的 JPG / JSON / isat.yaml。

用法:
    python isat_to_yolo_labels.py --images <DS>/images --labels <DS>/labels \
        --report <DS>/logs/labels_report.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

CAT2ID = {"silie": 1, "fanzhe": 2}
BACKGROUND = {"", "background", "__background__"}


def obj_bbox(obj, width, height):
    """返回像素坐标 (x1,y1,x2,y2)，没有有效框时返回 None。"""
    bbox = obj.get("bbox")
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    else:
        seg = obj.get("segmentation") or []
        if len(seg) < 3:
            return None
        xs = [float(p[0]) for p in seg]
        ys = [float(p[1]) for p in seg]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

    x1 = max(0.0, min(width - 1.0, x1))
    x2 = max(0.0, min(width - 1.0, x2))
    y1 = max(0.0, min(height - 1.0, y1))
    y2 = max(0.0, min(height - 1.0, y2))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--allow-missing-json", action="store_true",
                    help="缺 JSON 时生成空 label 并继续（默认报错）")
    args = ap.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 干净重建，避免旧文件残留
    for old in labels_dir.glob("*.txt"):
        old.unlink()

    jpgs = sorted(images_dir.glob("*.jpg"))
    if not jpgs:
        print(f"[ERROR] no .jpg under {images_dir}")
        return 1

    missing = []
    unknown = []
    size_mismatch = []
    empty = 0
    lines_total = 0
    obj_count = {"silie": 0, "fanzhe": 0}
    img_count = {"silie": 0, "fanzhe": 0}
    jpg_with_json = 0

    for ip in jpgs:
        jp = ip.with_suffix(".json")
        if not jp.exists():
            missing.append(ip.name)
            if args.allow_missing_json:
                (labels_dir / (ip.stem + ".txt")).write_text("", encoding="utf-8")
                empty += 1
            continue

        jpg_with_json += 1
        d = json.load(open(jp, encoding="utf-8"))
        img = cv2.imread(str(ip))
        if img is None:
            print(f"[ERROR] cannot read {ip}")
            return 1
        h, w = img.shape[:2]
        info = d.get("info", {})
        if info.get("width") and info.get("height"):
            if int(info["width"]) != w or int(info["height"]) != h:
                size_mismatch.append({
                    "image": ip.name,
                    "json": [int(info["width"]), int(info["height"])],
                    "actual": [w, h],
                })

        lines = []
        present = set()
        for obj in d.get("objects", []):
            cat = str(obj.get("category", "")).strip().lower()
            if cat in BACKGROUND:
                continue
            if cat not in CAT2ID:
                unknown.append({"image": ip.name, "category": cat})
                continue
            box = obj_bbox(obj, w, h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cx = (x1 + x2) * 0.5 / w
            cy = (y1 + y2) * 0.5 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{CAT2ID[cat]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            obj_count[cat] += 1
            present.add(cat)

        (labels_dir / (ip.stem + ".txt")).write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8")
        if not lines:
            empty += 1
        lines_total += len(lines)
        for cat in present:
            img_count[cat] += 1

    label_files = list(labels_dir.glob("*.txt"))
    report = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "jpg": len(jpgs),
        "jpg_with_json": jpg_with_json,
        "missing_json": missing,
        "unknown_categories": unknown,
        "size_mismatch": size_mismatch,
        "label_files": len(label_files),
        "empty_labels": empty,
        "label_lines": lines_total,
        "objects": obj_count,
        "images_with": img_count,
    }

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"jpg={len(jpgs)} json={jpg_with_json} labels={len(label_files)} "
          f"empty={empty} lines={lines_total} "
          f"silie={obj_count['silie']} fanzhe={obj_count['fanzhe']}")
    if size_mismatch:
        print(f"[WARN] {len(size_mismatch)} json/image size mismatches (used actual image size)")
    if missing:
        print(f"[ERROR] missing JSON for {len(missing)} image(s): {missing}")
        if not args.allow_missing_json:
            return 1
    if unknown:
        print(f"[ERROR] unknown categories: {unknown}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
