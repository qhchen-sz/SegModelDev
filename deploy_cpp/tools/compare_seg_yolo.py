# -*- coding: utf-8 -*-
"""GT-less comparison of SegFormer (SegTest/TensorRT) and YOLO (ONNX).

There is no ground truth in the target test folders, so every metric here is a
model-vs-model agreement / descriptive statistic, NOT accuracy.

Usage:
    python compare_seg_yolo.py --config <test_data>/compare_seg_yolo.json
    python compare_seg_yolo.py --config <test_data>/compare_seg_yolo.json --preflight --limit 2
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TRT_ROOT = Path(r"E:\Tensorrt\TensorRT-8.6.1.6")
CUDA_BIN = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin")
OCV_BIN = Path(r"E:\opencv_exe\opencv\build\x64\vc16\bin")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
COLORS = {0: (60, 60, 60), 1: (0, 0, 255), 2: (255, 0, 0)}  # BGR
PALETTE = [0, 0, 0, 255, 0, 0, 0, 0, 255] + [0, 0, 0] * 253

SEG_RE = re.compile(
    r"^\s*(?P<name>\S+\.(?:jpg|jpeg|png|bmp))\s+"
    r"(?P<w>\d+)x(?P<h>\d+)\s+(?P<ms>[\d.]+)ms\s+"
    r"ng=(?P<ng>\d+)\s+"
    r"px\[bg=(?P<bg>\d+)\s+silie=(?P<silie>\d+)\s+fanzhe=(?P<fanzhe>\d+)\]\s+"
    r"comp\[bg=(?P<cbg>\d+)\s+silie=(?P<cs>\d+)\s+fanzhe=(?P<cf>\d+)\]\s+"
    r"box=(?P<nbox>\d+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- helpers
def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def imread_u(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_COLOR | getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 128)
    img = cv2.imdecode(buf, flag)
    if img is None:
        raise IOError(f"decode failed: {path}")
    return img


def imwrite_u(path, img, ext=None, quality=None):
    path = Path(path)
    ext = ext or path.suffix or ".png"
    params = [] if quality is None else [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"encode failed: {path}")
    buf.tofile(str(path))


def read_mask_u(path):
    with Image.open(path) as im:
        if im.mode in ("P", "L"):
            return np.array(im)
        return np.array(im.convert("L"))


def write_mask_u(path, sem):
    sem = np.ascontiguousarray(sem.astype(np.uint8))
    h, w = sem.shape
    im = Image.new("P", (w, h))
    im.putpalette(PALETTE)
    im.frombytes(sem.tobytes())
    im.save(str(path))


def iou_pair(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter, union, (inter / union if union else None)


def connected_count(mask_c):
    n, _, _, _ = cv2.connectedComponentsWithStats(
        mask_c.astype(np.uint8), connectivity=8)
    return int(n - 1)


def rel_to(path, base):
    return os.path.relpath(str(path), str(base)).replace("\\", "/")


def fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def load_config(config_path):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (base / p).resolve()

    cfg["_config_path"] = config_path
    cfg["_base"] = base
    cfg["input_dir"] = resolve(cfg["input_dir"])
    cfg["yolo_onnx"] = resolve(cfg["yolo_onnx"])
    cfg["seg_engine"] = resolve(cfg["seg_engine"])
    cfg["segtest_exe"] = resolve(cfg["segtest_exe"])
    cfg["results_dir"] = resolve(cfg.get("results_dir", "_results"))
    cfg.setdefault("yolo", {})
    cfg["yolo"].setdefault("imgsz", 1024)
    cfg["yolo"].setdefault("conf", 0.25)
    cfg["yolo"].setdefault("iou", 0.5)
    cfg["yolo"].setdefault("device", "cpu")
    cfg["yolo"].setdefault("max_det", 300)
    cfg.setdefault("thumb_height", 720)
    cfg.setdefault("top_n", 10)
    return cfg


def list_images(input_dir):
    files = sorted(p for p in Path(input_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    stems = [p.stem for p in files]
    dup = [s for s, n in Counter(stems).items() if n > 1]
    if dup:
        raise RuntimeError(f"duplicate stems in {input_dir}: {dup[:5]}")
    return files


def versions():
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "cv2", "PIL", "torch", "ultralytics", "onnxruntime"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception as e:
            out[mod] = f"missing: {e}"
    return out


def seg_env():
    env = os.environ.copy()
    parts = [str(TRT_ROOT / "lib"), str(TRT_ROOT / "bin"),
             str(CUDA_BIN), str(OCV_BIN)]
    env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
    return env


def decode_line(raw):
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------- SegFormer
def seg_mask_path(seg_out, stem):
    return Path(seg_out) / "mask" / (stem + ".png")


def seg_overlay_path(seg_out, stem):
    return Path(seg_out) / (stem + ".png")


def run_segtest_once(exe, engine, input_path, out_dir, env, log_path, append=False):
    cmd = [str(exe), str(engine), str(input_path), str(out_dir), "--mask"]
    lines = []
    with open(log_path, "ab" if append else "wb") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(Path(exe).parent), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for raw in proc.stdout:
            log.write(raw)
            lines.append(decode_line(raw).rstrip("\r\n"))
        rc = proc.wait()
    return rc, lines


def parse_seg_lines(lines):
    parsed = {}
    order = []
    for line in lines:
        m = SEG_RE.match(line)
        if not m:
            continue
        d = m.groupdict()
        stem = Path(d["name"]).stem
        parsed[stem] = {
            "ms": float(d["ms"]),
            "w": int(d["w"]), "h": int(d["h"]),
            "ng": int(d["ng"]),
            "px": [int(d["bg"]), int(d["silie"]), int(d["fanzhe"])],
            "ncomp": [int(d["cbg"]), int(d["cs"]), int(d["cf"])],
            "nbox": int(d["nbox"]),
        }
        order.append(stem)
    if order:
        parsed[order[0]]["first"] = True
    return parsed


def run_segformer(cfg, images, seg_out, log_path, individual=False,
                  retry_missing=True):
    exe, engine = cfg["segtest_exe"], cfg["seg_engine"]
    env = seg_env()
    ensure_dir(seg_out)
    ensure_dir(Path(seg_out) / "mask")
    info = {}
    t0 = time.perf_counter()
    if individual:
        for idx, img in enumerate(images):
            _, lines = run_segtest_once(
                exe, engine, img, seg_out, env, log_path, append=True)
            p = parse_seg_lines(lines)
            for stem, d in p.items():
                d["first"] = (idx == 0)
                d["retry"] = False
                info[stem] = d
    else:
        _, lines = run_segtest_once(
            exe, engine, cfg["input_dir"], seg_out, env, log_path, append=False)
        info.update(parse_seg_lines(lines))
    elapsed = time.perf_counter() - t0

    missing = [img for img in images
               if not seg_mask_path(seg_out, img.stem).exists()]
    retried = []
    if retry_missing:
        for img in missing:
            _, lines = run_segtest_once(
                exe, engine, img, seg_out, env, log_path, append=True)
            p = parse_seg_lines(lines)
            if img.stem in p:
                p[img.stem]["retry"] = True
                p[img.stem]["first"] = False
                info[img.stem] = p[img.stem]
            retried.append(img.stem)

    mismatches = []
    for img in images:
        stem = img.stem
        mp = seg_mask_path(seg_out, stem)
        rec = info.setdefault(stem, {})
        if not mp.exists():
            rec.update({"ok": False, "error": "missing mask"})
            continue
        m = read_mask_u(mp)
        px = [int((m == c).sum()) for c in (0, 1, 2)]
        ncomp = [0, connected_count(m == 1), connected_count(m == 2)]
        rec.update({"ok": True, "px_mask": px, "ncomp_mask": ncomp,
                    "shape": list(m.shape)})
        if "px" in rec and "ncomp" in rec:
            match = (rec["px"] == px and rec["ncomp"] == ncomp)
            rec["stat_match"] = bool(match)
            if not match:
                mismatches.append({
                    "image": img.name,
                    "engine_px": rec["px"], "mask_px": px,
                    "engine_ncomp": rec["ncomp"], "mask_ncomp": ncomp,
                })
        else:
            rec["stat_match"] = None
    return info, {"elapsed_s": round(elapsed, 2), "retried": retried,
                  "mismatches": mismatches,
                  "returncode_missing": len([1 for img in images
                                             if not seg_mask_path(seg_out, img.stem).exists()])}


# ---------------------------------------------------------------- YOLO
def yolo_names(model):
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def yolo_load(onnx_path):
    from ultralytics import YOLO
    model = YOLO(str(onnx_path), task="segment")
    names = yolo_names(model)
    sid = next((k for k, v in names.items() if v.strip().lower() == "silie"), None)
    fid = next((k for k, v in names.items() if v.strip().lower() == "fanzhe"), None)
    if sid is None or fid is None:
        raise RuntimeError(f"cannot find silie/fanzhe in YOLO names: {names}")
    return model, names, sid, fid


def yolo_mask_from_result(res, shape, sid, fid):
    h, w = shape[:2]
    sem = np.zeros((h, w), dtype=np.uint8)
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return sem, [], 0, {}
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy().astype(float)
    xyxy = boxes.xyxy.cpu().numpy().astype(float)
    bg_ignored = int(np.sum(~np.isin(cls, [sid, fid])))
    dets = []
    md = None
    masks = getattr(res, "masks", None)
    if masks is not None and getattr(masks, "data", None) is not None:
        try:
            md = masks.data.cpu().numpy()
        except Exception:
            md = None
    xy = None if masks is None or getattr(masks, "xy", None) is None else masks.xy

    maxconf = {}
    for k in np.argsort(conf):  # low confidence first, high wins overlaps
        c = int(cls[k])
        if c not in (sid, fid):
            continue
        mk_full = None
        if md is not None and k < len(md):
            mk = md[k] > 0.5
            if mk.shape != (h, w):
                mk_full = cv2.resize(mk.astype(np.uint8), (w, h),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mk_full = mk
            sem[mk_full] = c
        elif xy is not None and k < len(xy):
            poly = xy[k]
            if poly is not None and len(poly) >= 3:
                cv2.fillPoly(sem, [np.round(poly).astype(np.int32)], c)
        x1, y1, x2, y2 = xyxy[k]
        x1 = max(0.0, min(w - 1.0, x1)); x2 = max(0.0, min(w - 1.0, x2))
        y1 = max(0.0, min(h - 1.0, y1)); y2 = max(0.0, min(h - 1.0, y2))
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
        dets.append({
            "det_index": int(k),
            "class_id": c,
            "class_name": "silie" if c == sid else "fanzhe",
            "conf": float(conf[k]),
            "x1": round(x1, 2), "y1": round(y1, 2),
            "x2": round(x2, 2), "y2": round(y2, 2),
            "bbox_w": round(bw, 2), "bbox_h": round(bh, 2),
            "bbox_area": round(bw * bh, 2),
            "mask_px": int(mk_full.sum()) if mk_full is not None else 0,
        })
        maxconf[c] = max(maxconf.get(c, 0.0), float(conf[k]))
    return sem, dets, bg_ignored, maxconf


def run_yolo(cfg, images, yolo_out, limit=None):
    model, names, sid, fid = yolo_load(cfg["yolo_onnx"])
    y = cfg["yolo"]
    overlay_dir = ensure_dir(Path(yolo_out) / "overlays")
    mask_dir = ensure_dir(Path(yolo_out) / "masks")
    label_dir = ensure_dir(Path(yolo_out) / "labels")

    warm = np.zeros((y["imgsz"], y["imgsz"], 3), dtype=np.uint8)
    model(warm, imgsz=y["imgsz"], conf=y["conf"], iou=y["iou"],
          device=y["device"], verbose=False, max_det=y["max_det"])

    info, det_rows = {}, []
    t0 = time.perf_counter()
    for img_path in (images[:limit] if limit else images):
        stem = img_path.stem
        img = imread_u(img_path)
        t1 = time.perf_counter()
        res = model(img, imgsz=y["imgsz"], conf=y["conf"], iou=y["iou"],
                    device=y["device"], verbose=False, max_det=y["max_det"])[0]
        ms = (time.perf_counter() - t1) * 1000.0
        sem, dets, bg_ignored, maxconf = yolo_mask_from_result(
            res, img.shape, sid, fid)
        overlay = img.copy()
        for c, col in COLORS.items():
            m = sem == c
            if m.any():
                overlay[m] = (img[m].astype(np.float32) * 0.5 +
                              np.array(col, np.float32) * 0.5).astype(np.uint8)
        imwrite_u(overlay_dir / (stem + ".png"), overlay, ext=".png")
        write_mask_u(mask_dir / (stem + ".png"), sem)
        for d in dets:
            row = {"image": img_path.name, "stem": stem}
            row.update(d)
            det_rows.append(row)
        if dets:
            lines = [f"{d['class_id']} {d['conf']:.6f} "
                     f"{d['x1']:.2f} {d['y1']:.2f} {d['x2']:.2f} {d['y2']:.2f} "
                     f"{d['mask_px']}" for d in dets]
            (label_dir / (stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
        px = [int((sem == c).sum()) for c in (0, 1, 2)]
        info[stem] = {
            "ok": True, "ms": ms, "dets": dets,
            "det_silie": sum(1 for d in dets if d["class_id"] == sid),
            "det_fanzhe": sum(1 for d in dets if d["class_id"] == fid),
            "det_bg_ignored": bg_ignored,
            "maxconf_silie": maxconf.get(sid),
            "maxconf_fanzhe": maxconf.get(fid),
            "px": px,
            "ncomp": [0, connected_count(sem == 1), connected_count(sem == 2)],
        }
    elapsed = time.perf_counter() - t0
    return info, det_rows, {
        "elapsed_s": round(elapsed, 2), "names": names,
        "silie_id": sid, "fanzhe_id": fid,
    }


# ---------------------------------------------------------------- thumbnails
def make_thumb(orig, seg_ov, yolo_ov, out_path, row, thumb_h=720):
    def load(p):
        try:
            return imread_u(p)
        except Exception:
            return None

    panels = [("Original", load(orig)), ("SegFormer", load(seg_ov)),
              ("YOLO-ONNX", load(yolo_ov))]
    resized = []
    for label, img in panels:
        if img is None:
            img = np.full((thumb_h, thumb_h, 3), 128, np.uint8)
        h, w = img.shape[:2]
        if h > thumb_h:
            nw = max(1, int(round(w * thumb_h / h)))
            img = cv2.resize(img, (nw, thumb_h), interpolation=cv2.INTER_AREA)
        resized.append((label, img))
    gap, head, foot = 8, 40, 34
    total_w = sum(im.shape[1] for _, im in resized) + gap * (len(resized) - 1)
    canvas = np.full((head + thumb_h + foot, total_w, 3), 245, np.uint8)
    x = 0
    for label, im in resized:
        canvas[head:head + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(canvas, label, (x + 8, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.line(canvas, (x, head), (x, head + im.shape[0]), (200, 200, 200), 1)
        x += im.shape[1] + gap
    di = row.get("defect_iou")
    di_txt = "n/a" if di is None else f"{di:.3f}"
    area = max(1, (row.get("width") or 1) * (row.get("height") or 1))
    foot_txt = (f"flag={row.get('flag')} defectIoU={di_txt} "
                f"mismatch={100.0 * (row.get('mismatch_px') or 0) / area:.2f}%  "
                f"red=silie blue=fanzhe")
    cv2.putText(canvas, foot_txt, (8, head + thumb_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    ensure_dir(Path(out_path).parent)
    imwrite_u(out_path, canvas, ext=".jpg", quality=88)


# ---------------------------------------------------------------- metrics
PER_IMAGE_FIELDS = [
    "image", "width", "height", "flag", "error",
    "seg_ok", "seg_ms", "seg_first", "seg_retry", "seg_ng", "seg_stat_match",
    "seg_silie_px", "seg_fanzhe_px", "seg_silie_comp", "seg_fanzhe_comp",
    "yolo_ok", "yolo_ms", "yolo_det_silie", "yolo_det_fanzhe",
    "yolo_det_bg_ignored", "yolo_maxconf_silie", "yolo_maxconf_fanzhe",
    "yolo_silie_px", "yolo_fanzhe_px", "yolo_silie_comp", "yolo_fanzhe_comp",
    "seg_defect", "yolo_defect",
    "defect_iou", "silie_iou", "fanzhe_iou",
    "defect_inter_px", "defect_union_px",
    "silie_inter_px", "silie_union_px", "fanzhe_inter_px", "fanzhe_union_px",
    "mismatch_px", "defect_mismatch_px", "class_conflict_px",
    "seg_only_px", "yolo_only_px", "mismatch_rate", "thumb",
]

DET_FIELDS = [
    "image", "stem", "det_index", "class_id", "class_name", "conf",
    "x1", "y1", "x2", "y2", "bbox_w", "bbox_h", "bbox_area", "mask_px",
]


def build_rows(cfg, images, seg_info, yolo_info, results_dir):
    rows = []
    for img in images:
        stem = img.stem
        s, y = seg_info.get(stem, {}), yolo_info.get(stem, {})
        row = {"image": img.name, "width": None, "height": None,
               "flag": "error", "error": "", "thumb": ""}
        try:
            probe = imread_u(img)
            row["height"], row["width"] = probe.shape[:2]
        except Exception as e:
            row["error"] = f"image: {e}"
        sm_path = seg_mask_path(results_dir / "segformer", stem)
        ym_path = Path(results_dir) / "yolo" / "masks" / (stem + ".png")
        if not sm_path.exists() or not ym_path.exists():
            row["error"] = (row["error"] + "; " if row["error"] else "") + \
                f"missing mask seg={sm_path.exists()} yolo={ym_path.exists()}"
            rows.append(row)
            continue
        try:
            sm = read_mask_u(sm_path)
            ym = read_mask_u(ym_path)
            if sm.shape != ym.shape:
                ym = cv2.resize(ym, (sm.shape[1], sm.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            h, w = sm.shape[:2]
            s_px = [int((sm == c).sum()) for c in (0, 1, 2)]
            y_px = [int((ym == c).sum()) for c in (0, 1, 2)]
            s_comp = [0, connected_count(sm == 1), connected_count(sm == 2)]
            y_comp = [0, connected_count(ym == 1), connected_count(ym == 2)]
            seg_defect = (s_px[1] + s_px[2]) > 0
            yolo_defect = (y.get("det_silie", 0) + y.get("det_fanzhe", 0)) > 0
            flag = ("both" if seg_defect and yolo_defect else
                    "seg-only" if seg_defect else
                    "yolo-only" if yolo_defect else "clean")
            d_int, d_uni, d_iou = iou_pair(sm > 0, ym > 0)
            si_int, si_uni, si_iou = iou_pair(sm == 1, ym == 1)
            fa_int, fa_uni, fa_iou = iou_pair(sm == 2, ym == 2)
            mismatch = int((sm != ym).sum())
            defect_mismatch = int(((sm > 0) != (ym > 0)).sum())
            class_conflict = int(((sm > 0) & (ym > 0) & (sm != ym)).sum())
            seg_only = int(((sm > 0) & (ym == 0)).sum())
            yolo_only = int(((ym > 0) & (sm == 0)).sum())
            row.update({
                "width": w, "height": h, "flag": flag,
                "seg_ok": bool(s.get("ok", False)),
                "seg_ms": s.get("ms"),
                "seg_first": bool(s.get("first", False)),
                "seg_retry": bool(s.get("retry", False)),
                "seg_ng": s.get("ng", 1 if seg_defect else 0),
                "seg_stat_match": s.get("stat_match"),
                "seg_silie_px": s_px[1], "seg_fanzhe_px": s_px[2],
                "seg_silie_comp": s_comp[1], "seg_fanzhe_comp": s_comp[2],
                "yolo_ok": bool(y.get("ok", False)),
                "yolo_ms": y.get("ms"),
                "yolo_det_silie": y.get("det_silie", 0),
                "yolo_det_fanzhe": y.get("det_fanzhe", 0),
                "yolo_det_bg_ignored": y.get("det_bg_ignored", 0),
                "yolo_maxconf_silie": y.get("maxconf_silie"),
                "yolo_maxconf_fanzhe": y.get("maxconf_fanzhe"),
                "yolo_silie_px": y_px[1], "yolo_fanzhe_px": y_px[2],
                "yolo_silie_comp": y_comp[1], "yolo_fanzhe_comp": y_comp[2],
                "seg_defect": bool(seg_defect), "yolo_defect": bool(yolo_defect),
                "defect_iou": d_iou, "silie_iou": si_iou, "fanzhe_iou": fa_iou,
                "defect_inter_px": d_int, "defect_union_px": d_uni,
                "silie_inter_px": si_int, "silie_union_px": si_uni,
                "fanzhe_inter_px": fa_int, "fanzhe_union_px": fa_uni,
                "mismatch_px": mismatch,
                "defect_mismatch_px": defect_mismatch,
                "class_conflict_px": class_conflict,
                "seg_only_px": seg_only, "yolo_only_px": yolo_only,
                "mismatch_rate": round(mismatch / max(1, w * h), 8),
                "thumb": f"compare/side_by_side/{stem}_compare.jpg",
            })
        except Exception as e:
            row["error"] = f"metrics: {e}"
        rows.append(row)
    return rows


def timing_stats(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p95": None,
                "min": None, "max": None}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": len(a),
        "mean": round(float(a.mean()), 2),
        "median": round(float(np.median(a)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "min": round(float(a.min()), 2),
        "max": round(float(a.max()), 2),
    }


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_summary(cfg, rows, det_rows, results_dir, seg_meta, yolo_meta):
    n = len(rows)
    counts = Counter(r.get("flag", "error") for r in rows)
    errors = [r for r in rows if r.get("error")]
    seg_px = [sum(int(r.get(f"seg_{k}_px", 0) or 0) for r in rows)
              for k in ("silie", "fanzhe")]
    yolo_px = [sum(int(r.get(f"yolo_{k}_px", 0) or 0) for r in rows)
               for k in ("silie", "fanzhe")]
    d_int = sum(int(r.get("defect_inter_px", 0) or 0) for r in rows)
    d_uni = sum(int(r.get("defect_union_px", 0) or 0) for r in rows)
    si_int = sum(int(r.get("silie_inter_px", 0) or 0) for r in rows)
    si_uni = sum(int(r.get("silie_union_px", 0) or 0) for r in rows)
    fa_int = sum(int(r.get("fanzhe_inter_px", 0) or 0) for r in rows)
    fa_uni = sum(int(r.get("fanzhe_union_px", 0) or 0) for r in rows)
    macro = [r["defect_iou"] for r in rows if r.get("defect_iou") is not None]
    mismatch_rates = [r["mismatch_rate"] for r in rows
                      if r.get("mismatch_rate") is not None]
    det_silie = sum(1 for d in det_rows if d.get("class_name") == "silie")
    det_fanzhe = sum(1 for d in det_rows if d.get("class_name") == "fanzhe")
    empty_yolo = sum(1 for r in rows
                     if (r.get("yolo_det_silie", 0) + r.get("yolo_det_fanzhe", 0)) == 0)
    maxconf_s = [r.get("yolo_maxconf_silie") for r in rows
                 if r.get("yolo_maxconf_silie") is not None]
    maxconf_f = [r.get("yolo_maxconf_fanzhe") for r in rows
                 if r.get("yolo_maxconf_fanzhe") is not None]
    seg_times = [r.get("seg_ms") for r in rows
                 if r.get("seg_ms") is not None and not r.get("seg_first")
                 and not r.get("seg_retry")]
    yolo_times = [r.get("yolo_ms") for r in rows if r.get("yolo_ms") is not None]

    def top_list(key):
        items = [r for r in rows if (r.get(key) or 0) > 0 and not r.get("error")]
        items.sort(key=lambda r: (-(r.get(key) or 0),
                                  -(r.get("mismatch_px") or 0),
                                  r.get("image", "")))
        out = []
        for i, r in enumerate(items[:cfg["top_n"]], 1):
            out.append({
                "rank": i, "image": r["image"], "flag": r.get("flag"),
                "defect_iou": r.get("defect_iou"),
                "defect_mismatch_px": r.get("defect_mismatch_px"),
                "class_conflict_px": r.get("class_conflict_px"),
                "mismatch_px": r.get("mismatch_px"),
                "seg_silie_px": r.get("seg_silie_px"),
                "seg_fanzhe_px": r.get("seg_fanzhe_px"),
                "yolo_silie_px": r.get("yolo_silie_px"),
                "yolo_fanzhe_px": r.get("yolo_fanzhe_px"),
                "thumb": r.get("thumb"),
            })
        return out

    summary = {
        "schema_version": 1,
        "view": cfg["view"],
        "generated_at": now_iso(),
        "input_dir": str(cfg["input_dir"]),
        "results_dir": str(results_dir),
        "n_images": n,
        "models": {
            "segformer": {
                "engine": str(cfg["seg_engine"]),
                "exe": str(cfg["segtest_exe"]),
                "precision": "fp16",
                "tile": 640, "stride": 448, "batch": 16, "tta": False,
            },
            "yolo": {
                "onnx": str(cfg["yolo_onnx"]),
                "names": {str(k): v for k, v in yolo_meta.get("names", {}).items()},
                "silie_id": yolo_meta.get("silie_id"),
                "fanzhe_id": yolo_meta.get("fanzhe_id"),
                **cfg["yolo"],
            },
        },
        "definitions": {
            "no_gt": "No ground truth in this test set. All metrics are model-vs-model agreement, not accuracy.",
            "flag": {"both": "seg_defect and yolo_defect",
                     "seg-only": "seg_defect and not yolo_defect",
                     "yolo-only": "yolo_defect and not seg_defect",
                     "clean": "both masks/defect outputs empty"},
            "defect_iou": "|(seg>0)&(yolo>0)| / |(seg>0)|(yolo>0)|; null if union==0",
            "mismatch_px": "count(seg != yolo)",
        },
        "counts": {
            "both": counts.get("both", 0),
            "seg_only": counts.get("seg-only", 0),
            "yolo_only": counts.get("yolo-only", 0),
            "clean": counts.get("clean", 0),
            "errors": len(errors),
            "seg_defect_images": sum(1 for r in rows if r.get("seg_defect")),
            "yolo_defect_images": sum(1 for r in rows if r.get("yolo_defect")),
        },
        "pixels": {
            "seg": {"silie": seg_px[0], "fanzhe": seg_px[1],
                    "mean_silie": round(seg_px[0] / n, 1) if n else 0,
                    "mean_fanzhe": round(seg_px[1] / n, 1) if n else 0},
            "yolo": {"silie": yolo_px[0], "fanzhe": yolo_px[1],
                     "mean_silie": round(yolo_px[0] / n, 1) if n else 0,
                     "mean_fanzhe": round(yolo_px[1] / n, 1) if n else 0},
            "defect_union": {
                "inter": d_int, "union": d_uni,
                "iou": round(d_int / d_uni, 6) if d_uni else None,
            },
        },
        "agreement": {
            "defect_iou_micro": round(d_int / d_uni, 6) if d_uni else None,
            "defect_iou_macro_nonempty": round(float(np.mean(macro)), 6) if macro else None,
            "silie_iou_micro": round(si_int / si_uni, 6) if si_uni else None,
            "fanzhe_iou_micro": round(fa_int / fa_uni, 6) if fa_uni else None,
            "mean_mismatch_rate": round(float(np.mean(mismatch_rates)), 8) if mismatch_rates else None,
            "mean_defect_mismatch_px": round(float(np.mean(
                [r.get("defect_mismatch_px", 0) for r in rows])), 1) if n else 0,
            "mean_class_conflict_px": round(float(np.mean(
                [r.get("class_conflict_px", 0) for r in rows])), 1) if n else 0,
        },
        "detections": {
            "total": len(det_rows), "silie": det_silie, "fanzhe": det_fanzhe,
            "empty_images": empty_yolo,
            "max_conf_silie": max(maxconf_s) if maxconf_s else None,
            "max_conf_fanzhe": max(maxconf_f) if maxconf_f else None,
        },
        "timing_ms": {
            "segformer": timing_stats(seg_times),
            "yolo": timing_stats(yolo_times),
            "note": "SegFormer is TensorRT FP16 on GPU; YOLO is ONNX Runtime CPU. Not comparable.",
        },
        "top_defect_mismatch": top_list("defect_mismatch_px"),
        "top_class_conflict": top_list("class_conflict_px"),
        "segformer_retried": seg_meta.get("retried", []),
        "segformer_stat_mismatches": seg_meta.get("mismatches", []),
        "errors": errors,
        "outputs": {
            "per_image_csv": "per_image.csv",
            "detections_csv": "detections.csv",
            "summary_json": "summary.json",
            "summary_md": "summary.md",
            "segformer_dir": "segformer",
            "yolo_dir": "yolo",
            "compare_dir": "compare",
        },
    }
    return summary


def write_summary_md(summary, path):
    s = summary
    c = s["counts"]
    a = s["agreement"]
    px = s["pixels"]
    tm = s["timing_ms"]
    n = s["n_images"]

    def pct(v):
        return f"{100.0 * v / n:.1f}%" if n else "0.0%"

    lines = [
        f"# {s['view']}：SegFormer vs YOLO(ONNX) 无 GT 对比",
        "",
        f"> 生成时间：{s['generated_at']}  ",
        f"> 图片目录：`{s['input_dir']}`  ",
        f"> 图片数：{n}",
        "",
        "> **重要：测试集没有 GT。下面的 IoU/一致率是“两个模型之间的一致程度”，",
        "> 不是 accuracy / recall / precision，不能据此宣称谁更准。**",
        "> YOLO 是 ONNX Runtime **CPU**，SegFormer 是 TensorRT **GPU**，耗时不可横向比较。",
        "",
        "## 1. 参数",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| SegFormer engine | `{s['models']['segformer']['engine']}` |",
        "| SegFormer 运行 | 640 tile / stride 448 / batch 16 / fp16 / no TTA |",
        f"| YOLO ONNX | `{s['models']['yolo']['onnx']}` |",
        f"| YOLO 参数 | imgsz={s['models']['yolo']['imgsz']}, conf={s['models']['yolo']['conf']}, "
        f"iou={s['models']['yolo']['iou']}, device={s['models']['yolo']['device']} |",
        "| 类别 | 0 background, 1 silie(撕裂), 2 fanzhe(翻折) |",
        "",
        "## 2. 标记计数",
        "",
        "| 标记 | 数量 | 占比 | 含义 |",
        "|---|---:|---:|---|",
        f"| both | {c['both']} | {pct(c['both'])} | 两边都判有缺陷 |",
        f"| seg-only | {c['seg_only']} | {pct(c['seg_only'])} | 只有 SegFormer 判有缺陷 |",
        f"| yolo-only | {c['yolo_only']} | {pct(c['yolo_only'])} | 只有 YOLO 判有缺陷 |",
        f"| clean | {c['clean']} | {pct(c['clean'])} | 两边都判无缺陷 |",
        f"| error | {c['errors']} | {pct(c['errors'])} | 推理/结果缺失 |",
        "",
        "## 3. 缺陷像素",
        "",
        "| 模型 | silie 总像素 | fanzhe 总像素 | 每图均值(silie/fanzhe) |",
        "|---|---:|---:|---|",
        f"| SegFormer | {px['seg']['silie']} | {px['seg']['fanzhe']} | "
        f"{px['seg']['mean_silie']} / {px['seg']['mean_fanzhe']} |",
        f"| YOLO | {px['yolo']['silie']} | {px['yolo']['fanzhe']} | "
        f"{px['yolo']['mean_silie']} / {px['yolo']['mean_fanzhe']} |",
        "",
        "## 4. 模型间一致率（不是精度）",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 缺陷并集 micro IoU | {fmt(a['defect_iou_micro'])} |",
        f"| 缺陷并集 macro IoU（union>0 的图） | {fmt(a['defect_iou_macro_nonempty'])} |",
        f"| silie IoU | {fmt(a['silie_iou_micro'])} |",
        f"| fanzhe IoU | {fmt(a['fanzhe_iou_micro'])} |",
        f"| 平均像素不一致率 | {fmt(a['mean_mismatch_rate'], 6)} |",
        f"| 平均缺陷区域不一致像素 | {a['mean_defect_mismatch_px']} |",
        f"| 平均同类冲突像素（1 vs 2） | {a['mean_class_conflict_px']} |",
        "",
        "## 5. YOLO 检测",
        "",
        f"- 检测总数：{s['detections']['total']}（silie {s['detections']['silie']}，"
        f"fanzhe {s['detections']['fanzhe']}）",
        f"- 无检测图：{s['detections']['empty_images']}",
        f"- 最大置信度：silie {fmt(s['detections']['max_conf_silie'])}，"
        f"fanzhe {fmt(s['detections']['max_conf_fanzhe'])}",
        "",
        "## 6. 耗时",
        "",
        "| 模型 | n | mean | median | p95 |",
        "|---|---:|---:|---:|---:|",
        f"| SegFormer (GPU/TRT) | {tm['segformer']['n']} | {fmt(tm['segformer']['mean'], 1)} | "
        f"{fmt(tm['segformer']['median'], 1)} | {fmt(tm['segformer']['p95'], 1)} |",
        f"| YOLO (CPU/ONNX) | {tm['yolo']['n']} | {fmt(tm['yolo']['mean'], 1)} | "
        f"{fmt(tm['yolo']['median'], 1)} | {fmt(tm['yolo']['p95'], 1)} |",
        "",
        "> YOLO 是 CPU 推理；SegFormer 是 GPU TensorRT。耗时不可直接比较。",
        "",
        "## 7. Top 分歧（按缺陷区域不一致像素）",
        "",
        "| # | 图片 | flag | defectIoU | defect_mismatch | mismatch | seg(s/f) | yolo(s/f) | 缩略图 |",
        "|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for t in s["top_defect_mismatch"]:
        lines.append(
            f"| {t['rank']} | `{t['image']}` | {t['flag']} | {fmt(t['defect_iou'])} | "
            f"{t['defect_mismatch_px']} | {t['mismatch_px']} | "
            f"{t['seg_silie_px']}/{t['seg_fanzhe_px']} | "
            f"{t['yolo_silie_px']}/{t['yolo_fanzhe_px']} | "
            f"[jpg]({t['thumb']}) |")
    lines += [
        "",
        "## 8. Top 类别冲突（1 vs 2）",
        "",
        "| # | 图片 | flag | class_conflict | defectIoU | 缩略图 |",
        "|---:|---|---|---:|---:|---|",
    ]
    for t in s["top_class_conflict"]:
        lines.append(
            f"| {t['rank']} | `{t['image']}` | {t['flag']} | {t['class_conflict_px']} | "
            f"{fmt(t['defect_iou'])} | [jpg]({t['thumb']}) |")
    lines += [
        "",
        "## 9. 错误 / 重试",
        "",
        f"- SegFormer 单图重试：{len(s['segformer_retried'])}",
        f"- SegFormer 统计不一致：{len(s['segformer_stat_mismatches'])}",
        f"- 逐图错误：{len(s['errors'])}",
        "",
    ]
    if s["errors"]:
        lines += ["| 图片 | 错误 |", "|---|---|"]
        for e in s["errors"][:50]:
            lines.append(f"| `{e.get('image')}` | {e.get('error')} |")
        lines.append("")
    lines += [
        "## 10. 输出",
        "",
        "- `per_image.csv`：每图统计",
        "- `detections.csv`：YOLO 每个检测",
        "- `segformer\\`：SegFormer overlay + mask",
        "- `yolo\\`：YOLO overlay + mask + labels",
        "- `compare\\side_by_side\\`：原图 / SegFormer / YOLO 缩略并排图",
        "- `compare\\top_disagreements\\`：Top 分歧缩略图",
        "",
        "> 再次强调：没有 GT，本报告只描述两个模型的一致/分歧，不评价绝对准确率。",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- preflight
def run_preflight(cfg, images, preflight_dir, limit):
    ensure_dir(preflight_dir)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    v = versions()
    check("python imports", all(not str(v[k]).startswith("missing")
                                for k in ("numpy", "cv2", "PIL", "torch",
                                          "ultralytics", "onnxruntime")), v)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception as e:
        providers = [f"error: {e}"]
    check("onnxruntime CPU provider", "CPUExecutionProvider" in providers, providers)
    check("input dir", Path(cfg["input_dir"]).is_dir(), cfg["input_dir"])
    check("yolo onnx", Path(cfg["yolo_onnx"]).is_file(), cfg["yolo_onnx"])
    check("segformer engine", Path(cfg["seg_engine"]).is_file(), cfg["seg_engine"])
    check("segtest exe", Path(cfg["segtest_exe"]).is_file(), cfg["segtest_exe"])
    for p in (TRT_ROOT / "lib", TRT_ROOT / "bin", CUDA_BIN, OCV_BIN):
        check(f"runtime path {p.name}", p.is_dir(), p)

    # Unicode read/write
    try:
        img = imread_u(images[0])
        test_png = preflight_dir / "unicode_test_测试.png"
        imwrite_u(test_png, img, ext=".png")
        back = imread_u(test_png)
        check("unicode read/write", back.shape == img.shape, f"{img.shape} -> {back.shape}")
    except Exception as e:
        img = None
        check("unicode read/write", False, e)

    # YOLO load/names + inference
    try:
        model, names, sid, fid = yolo_load(cfg["yolo_onnx"])
        check("yolo task/names", True, {"names": names, "silie": sid, "fanzhe": fid})
        y = cfg["yolo"]
        warm = np.zeros((y["imgsz"], y["imgsz"], 3), np.uint8)
        model(warm, imgsz=y["imgsz"], conf=y["conf"], iou=y["iou"],
              device=y["device"], verbose=False, max_det=y["max_det"])
        for im in images[:limit]:
            arr = imread_u(im)
            res = model(arr, imgsz=y["imgsz"], conf=y["conf"], iou=y["iou"],
                        device=y["device"], verbose=False, max_det=y["max_det"])[0]
            sem, dets, bg, _ = yolo_mask_from_result(res, arr.shape, sid, fid)
            check(f"yolo {im.name}", sem.shape == arr.shape[:2],
                  f"det={len(dets)} bg_ignored={bg} shape={sem.shape}")
        # empty-result path
        class Empty:
            boxes = None
            masks = None
        sem0, det0, bg0, _ = yolo_mask_from_result(Empty(), (64, 64, 3), sid, fid)
        check("yolo empty result", sem0.shape == (64, 64) and sem0.sum() == 0, "zero mask")
    except Exception as e:
        check("yolo load/inference", False, e)

    # SegFormer individual inference + masks/stats
    seg_dir = preflight_dir / "segformer"
    log = preflight_dir / "segformer.log"
    try:
        info, meta = run_segformer(cfg, images[:limit], seg_dir, log,
                                   individual=True, retry_missing=True)
        for im in images[:limit]:
            rec = info.get(im.stem, {})
            mp = seg_mask_path(seg_dir, im.stem)
            ok = mp.exists()
            vals = None
            shape = None
            if ok:
                m = read_mask_u(mp)
                shape = m.shape
                vals = sorted(int(x) for x in np.unique(m))
                ok = set(vals) <= {0, 1, 2}
            check(f"segformer {im.name}", ok,
                  f"mask={mp.exists()} shape={shape} values={vals} "
                  f"stat_match={rec.get('stat_match')}")
        check("segformer stat match", not meta.get("mismatches"), meta.get("mismatches"))
    except Exception as e:
        check("segformer inference", False, e)

    # One thumbnail
    try:
        im = images[0]
        arr = imread_u(im)
        seg_ov = seg_overlay_path(seg_dir, im.stem)
        yolo_ov = Path(preflight_dir) / "yolo" / "overlays" / (im.stem + ".png")
        # yolo preflight was not written to disk above; generate one now
        if not yolo_ov.exists():
            ymodel, ynames, ys, yf = yolo_load(cfg["yolo_onnx"])
            ycfg = cfg["yolo"]
            res = ymodel(arr, imgsz=ycfg["imgsz"], conf=ycfg["conf"],
                         iou=ycfg["iou"], device=ycfg["device"],
                         verbose=False, max_det=ycfg["max_det"])[0]
            sem, _, _, _ = yolo_mask_from_result(res, arr.shape, ys, yf)
            ov = arr.copy()
            for c, col in COLORS.items():
                m = sem == c
                if m.any():
                    ov[m] = (arr[m].astype(np.float32) * 0.5 +
                             np.array(col, np.float32) * 0.5).astype(np.uint8)
            ensure_dir(yolo_ov.parent)
            imwrite_u(yolo_ov, ov, ext=".png")
        thumb = preflight_dir / "compare" / (im.stem + "_compare.jpg")
        make_thumb(im, seg_ov, yolo_ov, thumb,
                   {"flag": "preflight", "defect_iou": None, "mismatch_px": 0,
                    "width": arr.shape[1], "height": arr.shape[0]},
                   thumb_h=cfg["thumb_height"])
        check("thumbnail", thumb.exists() and imread_u(thumb).size > 0, thumb)
    except Exception as e:
        check("thumbnail", False, e)

    report = {"generated_at": now_iso(), "config": str(cfg["_config_path"]),
              "limit": limit, "checks": checks,
              "all_pass": all(c["ok"] for c in checks)}
    (preflight_dir / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== PREFLIGHT ===")
    for c in checks:
        print(("PASS " if c["ok"] else "FAIL ") + c["name"] + " | " + c["detail"])
    print("ALL PASS" if report["all_pass"] else "PREFLIGHT FAILED")
    return 0 if report["all_pass"] else 1


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-seg", action="store_true")
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--no-retry-missing", action="store_true")
    ap.add_argument("--top-n", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.top_n is not None:
        cfg["top_n"] = args.top_n
    images = list_images(cfg["input_dir"])
    if args.limit:
        images = images[:args.limit]
    if not images:
        print("[ERROR] no images")
        return 1

    if args.preflight:
        return run_preflight(cfg, images, cfg["results_dir"] / "_preflight",
                             args.limit or 2)

    results = cfg["results_dir"]
    if (results / "summary.json").exists() and not args.overwrite:
        print(f"[ERROR] {results / 'summary.json'} exists; use --overwrite to replace")
        return 1
    ensure_dir(results)
    for sub in ("segformer", "yolo", "compare"):
        if (results / sub).exists():
            shutil.rmtree(results / sub)
    for name in ("per_image.csv", "detections.csv", "summary.json", "summary.md",
                 "run_meta.json", "engine_stats.csv", "stat_mismatches.csv"):
        p = results / name
        if p.exists():
            p.unlink()

    seg_info, seg_meta = {}, {"elapsed_s": 0, "retried": [], "mismatches": []}
    if not args.skip_seg:
        print("[SegFormer] running SegTest.exe ...")
        seg_info, seg_meta = run_segformer(
            cfg, images, results / "segformer", results / "segformer" / "engine.log",
            individual=False, retry_missing=not args.no_retry_missing)
    else:
        print("[SegFormer] skipped, reusing existing masks")
        for img in images:
            mp = seg_mask_path(results / "segformer", img.stem)
            seg_info[img.stem] = {"ok": mp.exists(), "first": False, "retry": False}

    yolo_info, det_rows, yolo_meta = {}, [], {}
    if not args.skip_yolo:
        print("[YOLO] running ONNX backend ...")
        yolo_info, det_rows, yolo_meta = run_yolo(
            cfg, images, results / "yolo", limit=None)
    else:
        print("[YOLO] skipped, reusing existing masks")

    print("[metrics] building per-image rows ...")
    rows = build_rows(cfg, images, seg_info, yolo_info, results)

    print("[thumbnails] ...")
    for r in rows:
        stem = Path(r["image"]).stem
        thumb = results / "compare" / "side_by_side" / (stem + "_compare.jpg")
        make_thumb(cfg["input_dir"] / r["image"],
                   seg_overlay_path(results / "segformer", stem),
                   results / "yolo" / "overlays" / (stem + ".png"),
                   thumb, r, cfg["thumb_height"])
        r["thumb"] = rel_to(thumb, results)

    summary = build_summary(cfg, rows, det_rows, results, seg_meta, yolo_meta)
    # copy top thumbnails
    top_dir = ensure_dir(results / "compare" / "top_disagreements")
    for i, t in enumerate(summary["top_defect_mismatch"], 1):
        src = results / t["thumb"]
        if src.exists():
            dst = top_dir / f"top_{i:02d}_{Path(t['image']).stem}_compare.jpg"
            shutil.copy2(src, dst)

    write_csv(results / "per_image.csv", rows, PER_IMAGE_FIELDS)
    write_csv(results / "detections.csv", det_rows, DET_FIELDS)
    engine_rows = []
    for img in images:
        rec = seg_info.get(img.stem, {})
        engine_rows.append({
            "image": img.name, "stem": img.stem,
            "ms": rec.get("ms"), "ng": rec.get("ng"),
            "engine_px": rec.get("px"), "mask_px": rec.get("px_mask"),
            "engine_ncomp": rec.get("ncomp"), "mask_ncomp": rec.get("ncomp_mask"),
            "nbox": rec.get("nbox"), "stat_match": rec.get("stat_match"),
            "ok": rec.get("ok"), "error": rec.get("error", ""),
        })
    write_csv(results / "engine_stats.csv", engine_rows,
              ["image", "stem", "ms", "ng", "engine_px", "mask_px",
               "engine_ncomp", "mask_ncomp", "nbox", "stat_match", "ok", "error"])
    write_csv(results / "stat_mismatches.csv", seg_meta.get("mismatches", []),
              ["image", "engine_px", "mask_px", "engine_ncomp", "mask_ncomp"])

    meta = {
        "generated_at": now_iso(),
        "view": cfg["view"],
        "config_path": str(cfg["_config_path"]),
        "input_dir": str(cfg["input_dir"]),
        "results_dir": str(results),
        "n_images": len(images),
        "args": vars(args),
        "versions": versions(),
        "models": {
            "segformer": {
                "engine": str(cfg["seg_engine"]),
                "engine_bytes": Path(cfg["seg_engine"]).stat().st_size
                if Path(cfg["seg_engine"]).exists() else None,
            },
            "yolo": {
                "onnx": str(cfg["yolo_onnx"]),
                "onnx_bytes": Path(cfg["yolo_onnx"]).stat().st_size
                if Path(cfg["yolo_onnx"]).exists() else None,
                "names": {str(k): v for k, v in yolo_meta.get("names", {}).items()},
            },
        },
        "durations_s": {"segformer": seg_meta.get("elapsed_s"),
                        "yolo": yolo_meta.get("elapsed_s")},
    }
    (results / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (results / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(summary, results / "summary.md")

    print("\n=== DONE ===")
    print(f"images={len(images)} counts={summary['counts']}")
    print(f"defect_iou_micro={summary['agreement']['defect_iou_micro']} "
          f"silie_iou={summary['agreement']['silie_iou_micro']} "
          f"fanzhe_iou={summary['agreement']['fanzhe_iou_micro']}")
    print(f"summary: {results / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
