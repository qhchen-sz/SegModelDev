# -*- coding: utf-8 -*-
"""在测试文件夹（无 GT）上对比三个模型：SegFormer v2-fast(无TTA) / v2b(B方案) / YOLOv8n-seg。

口径与 compare_report_strat.md 一致：
  - v2-fast / v2b : bs16 fp16 stride448 tile640 noTTA（同一推理策略，仅权重不同）
  - yolo          : imgsz1200 conf0.25 iou0.5

输出（均写入 --out_dir，默认 <img_dir>/_results，文件名带 _3models/_v2b 后缀，不覆盖既有产物）：
  report_3models.md / stats_3models.csv / timing_3models.csv / overlays_v2b/*.png

用法:
  python batch_compare_3models.py --img_dir "C:/Users/hymson/Desktop/222"
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import argparse
import time
import yaml
import csv
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.amp import autocast
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
from postprocess import logits_to_labels, resolve_confidence_threshold

# ---- 权重 ----
V2_BEST = r"D:/qhchen/NewModel/ADevFolder/segformer_v2/runs_strat/v2_strat/best"
V2B_BEST = r"D:/qhchen/NewModel/ADevFolder/segformer_v2/runs_strat/v2b_strat/best"
YOLO_BEST = r"D:/qhchen/NewModel/ADevFolder/runs_strat/train_strat/weights/best.pt"

# ---- v2-fast / v2b 推理常量 ----
TILE, STRIDE, BATCH, FP16 = 640, 448, 16, True
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
COLORS = [(60, 60, 60), (0, 0, 255), (255, 0, 0)]  # BGR: bg灰 / silie红 / fanzhe蓝

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def collect_tiles(img_bgr, tile, stride):
    H, W = img_bgr.shape[:2]
    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] + tile < W:
        xs.append(max(0, W - tile))
    tiles = []
    for y0 in ys:
        for x0 in xs:
            patch = img_bgr[y0:y0 + tile, x0:x0 + tile]
            ph, pw = patch.shape[:2]
            if ph < tile or pw < tile:
                patch = cv2.copyMakeBorder(patch, 0, tile - ph, 0, tile - pw, cv2.BORDER_REFLECT)
            tiles.append((y0, x0, patch, ph, pw))
    return tiles, H, W


def batch_forward(model, patches_bgr):
    arr = [p[:, :, ::-1].astype(np.float32) / 255.0 for p in patches_bgr]  # BGR->RGB
    arr = [(a - IMAGENET_MEAN) / IMAGENET_STD for a in arr]
    t = torch.from_numpy(np.stack(arr, 0)).permute(0, 3, 1, 2).to(device).float()
    with torch.no_grad():
        with autocast('cuda', dtype=torch.float16):
            lg = model(pixel_values=t).logits
    lg = F.interpolate(lg.float(), size=(TILE, TILE), mode='bilinear', align_corners=False)
    return lg.cpu().numpy()


def seg_predict(model, img_bgr, confidence_threshold=0.5):
    tiles, H, W = collect_tiles(img_bgr, TILE, STRIDE)
    acc = None
    cov = np.zeros((H, W), dtype=np.float32)
    for i in range(0, len(tiles), BATCH):
        batch = tiles[i:i + BATCH]
        logits = batch_forward(model, [b[2] for b in batch])
        if acc is None:
            acc = np.zeros((logits.shape[1], H, W), dtype=np.float32)
        for j, (y0, x0, _, ph, pw) in enumerate(batch):
            acc[:, y0:y0 + ph, x0:x0 + pw] += logits[j][:, :ph, :pw]
            cov[y0:y0 + ph, x0:x0 + pw] += 1.0
    cov = np.maximum(cov, 1e-6)
    return logits_to_labels(acc / cov[None], confidence_threshold)


def yolo_predict(yolo_model, img_bgr):
    H, W = img_bgr.shape[:2]
    res = yolo_model(img_bgr, imgsz=1200, conf=0.25, iou=0.5, verbose=False)[0]
    sem = np.zeros((H, W), dtype=np.uint8)
    n_inst = 0
    if res.masks is not None:
        clses = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        polys = res.masks.xy
        for k in np.argsort(confs):  # 高 conf 后涂
            poly = polys[k]
            if poly is None or len(poly) < 3:
                continue
            cv2.fillPoly(sem, [np.round(poly).astype(np.int32)], int(clses[k]))
        n_inst = len(clses)
    return sem, n_inst


def overlay(img, pred, alpha=0.5):
    ov = img.copy()
    for c, col in enumerate(COLORS):
        m = pred == c
        ov[m] = (img[m].astype(np.float32) * (1 - alpha) + np.array(col, dtype=np.float32) * alpha).astype(np.uint8)
    return ov


def iou_between(a, b, nc=3):
    inter = np.zeros(nc, dtype=np.int64)
    union = np.zeros(nc, dtype=np.int64)
    for c in range(nc):
        pa = a == c
        pb = b == c
        inter[c] = (pa & pb).sum()
        union[c] = (pa | pb).sum()
    return inter, union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--confidence-threshold', type=float, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config_strat.yaml', encoding='utf-8'))
    threshold = resolve_confidence_threshold(cfg, a.confidence_threshold)

    IMG_DIR = Path(a.img_dir.rstrip('/\\'))
    OUT = Path(a.out_dir) if a.out_dir else (IMG_DIR / '_results')
    OUT.mkdir(parents=True, exist_ok=True)
    OV_V2 = OUT / 'overlays_v2fast'
    OV_V2B = OUT / 'overlays_v2b'
    OV_YO = OUT / 'overlays_yolo'
    for d in (OV_V2, OV_V2B, OV_YO):
        d.mkdir(exist_ok=True)

    imgs = sorted(glob.glob(str(IMG_DIR / '*.jpg')))
    assert imgs, f"无 jpg: {IMG_DIR}"
    n_img = len(imgs)
    print(f"[{IMG_DIR.name}] {n_img} 张图 -> {OUT}, confidence_threshold={threshold}", flush=True)

    print(f"加载 v2-fast: {V2_BEST}", flush=True)
    v2 = SegformerForSemanticSegmentation.from_pretrained(V2_BEST).to(device).eval()
    print(f"加载 v2b    : {V2B_BEST}", flush=True)
    v2b = SegformerForSemanticSegmentation.from_pretrained(V2B_BEST).to(device).eval()
    from ultralytics import YOLO
    print(f"加载 yolo   : {YOLO_BEST}", flush=True)
    yolo = YOLO(YOLO_BEST)

    # warmup
    w = cv2.imread(imgs[0])
    seg_predict(v2, w, threshold)
    seg_predict(v2b, w, threshold)
    yolo_predict(yolo, w)
    if device == 'cuda':
        torch.cuda.synchronize()

    def timeit(fn, *args):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        r = fn(*args)
        if device == 'cuda':
            torch.cuda.synchronize()
        return r, (time.time() - t0) * 1000.0

    rows_stat, rows_time = [], []
    # 两两一致性累计
    iu = {'v2fast_yolo': np.zeros(3, np.int64), 'v2b_yolo': np.zeros(3, np.int64), 'v2fast_v2b': np.zeros(3, np.int64)}
    uu = {'v2fast_yolo': np.zeros(3, np.int64), 'v2b_yolo': np.zeros(3, np.int64), 'v2fast_v2b': np.zeros(3, np.int64)}
    flag = {'v2fast': 0, 'v2b': 0, 'yolo': 0}
    tot = {'v2fast': 0, 'v2b': 0, 'yolo': 0}          # 全图缺陷像素合计
    removed_by_v2b = []   # v2fast 报缺陷、v2b 全清的图
    added_by_v2b = []     # v2fast 全清、v2b 报缺陷的图

    for i, ip in enumerate(imgs):
        name = Path(ip).name
        img = cv2.imread(ip)
        p2, t2 = timeit(seg_predict, v2, img, threshold)
        pb, tb = timeit(seg_predict, v2b, img, threshold)
        (py, n_inst), ty = timeit(yolo_predict, yolo, img)

        px = {
            'v2fast': int((p2 > 0).sum()),
            'v2b': int((pb > 0).sum()),
            'yolo': int((py > 0).sum()),
        }
        for k in flag:
            if px[k] > 0:
                flag[k] += 1
            tot[k] += px[k]

        s2, f2 = int((p2 == 1).sum()), int((p2 == 2).sum())
        sb, fb = int((pb == 1).sum()), int((pb == 2).sum())
        sy, fy = int((py == 1).sum()), int((py == 2).sum())

        if px['v2fast'] > 0 and px['v2b'] == 0:
            removed_by_v2b.append(name)
        if px['v2fast'] == 0 and px['v2b'] > 0:
            added_by_v2b.append(name)

        for key, (ma, mb) in {'v2fast_yolo': (p2, py), 'v2b_yolo': (pb, py), 'v2fast_v2b': (p2, pb)}.items():
            inter, union = iou_between(ma, mb)
            iu[key] += inter
            uu[key] += union

        stem = Path(ip).stem
        cv2.imwrite(str(OV_V2 / (stem + '.png')), overlay(img, p2))
        cv2.imwrite(str(OV_V2B / (stem + '.png')), overlay(img, pb))
        cv2.imwrite(str(OV_YO / (stem + '.png')), overlay(img, py))
        rows_time.append([name, f'{t2:.1f}', f'{tb:.1f}', f'{ty:.1f}'])
        rows_stat.append([name, s2, f2, sb, fb, sy, fy, n_inst,
                          px['v2fast'], px['v2b'], px['yolo'], px['v2b'] - px['v2fast']])
        if i < 3 or i % 10 == 0:
            print(f"  [{i+1}/{n_img}] {name}  v2={t2:.0f} v2b={tb:.0f} yolo={ty:.0f} ms  "
                  f"defpx v2={px['v2fast']} v2b={px['v2b']} yolo={px['yolo']}", flush=True)

    # 清理三套叠加图目录里对应当前图片集之外的陈旧 PNG（如已从目录移出的旧图）
    cur = {Path(p).stem for p in imgs}
    pruned = {}
    for tag, d in (('v2fast', OV_V2), ('v2b', OV_V2B), ('yolo', OV_YO)):
        stale = [f for f in d.glob('*.png') if f.stem not in cur]
        for f in stale:
            f.unlink()
        if stale:
            pruned[tag] = [f.stem for f in stale]
    if pruned:
        print('清理陈旧叠加图:', pruned, flush=True)

    def ious(key):
        return iu[key] / (uu[key] + 1e-9)

    t2s = [float(r[1]) for r in rows_time]
    tbs = [float(r[2]) for r in rows_time]
    tys = [float(r[3]) for r in rows_time]

    with open(OUT / 'timing_3models.csv', 'w', newline='', encoding='utf-8') as f:
        cw = csv.writer(f)
        cw.writerow(['image', 'v2fast_ms', 'v2b_ms', 'yolo_ms'])
        cw.writerows(rows_time)
    with open(OUT / 'stats_3models.csv', 'w', newline='', encoding='utf-8') as f:
        cw = csv.writer(f)
        cw.writerow(['image', 'v2fast_silie_px', 'v2fast_fanzhe_px', 'v2b_silie_px', 'v2b_fanzhe_px',
                     'yolo_silie_px', 'yolo_fanzhe_px', 'yolo_instances',
                     'v2fast_defpx', 'v2b_defpx', 'yolo_defpx', 'v2b_minus_v2fast_defpx'])
        cw.writerows(rows_stat)

    i_vy, i_by, i_vb = ious('v2fast_yolo'), ious('v2b_yolo'), ious('v2fast_v2b')
    folder = IMG_DIR.name
    L = []
    L.append(f'# 三模型对比报告：v2-fast / v2b(B方案) / YOLOv8n-seg（{folder} | {n_img} 张全图，无 GT）\n')
    L.append('## 说明\n')
    L.append(f'- 本测试集 {n_img} 张图**无标注**，故 IoU/mAP 等真实精度指标 **N/A**。')
    L.append(f'- 口径与 `compare_report_strat.md` 一致：v2-fast / v2b 同为 (bs16 fp16 stride448 tile640 noTTA, Softmax 置信度阈值 {threshold})，仅权重不同；yolo (imgsz1200 conf0.25 iou0.5)。')
    L.append('- v2-fast 权重 = `runs_strat/v2_strat/best`；v2b 权重 = `runs_strat/v2b_strat/best`。')
    L.append('- 两两一致性 IoU：值越高=两模型在缺陷区域越吻合；低=至少一个在误报。')
    L.append('- 叠加图(三套并对齐当前图片集)：`overlays_v2fast/`、`overlays_v2b/`、`overlays_yolo/`（silie红 / fanzhe蓝 / bg灰）。\n')

    L.append('## 1. 推理速度（RTX 3090，全图）\n')
    L.append('| 模型 | 平均 ms/图 | 最小 | 最大 |')
    L.append('|---|---|---|---|')
    L.append('| v2-fast | {:.1f} | {:.1f} | {:.1f} |'.format(np.mean(t2s), min(t2s), max(t2s)))
    L.append('| v2b | {:.1f} | {:.1f} | {:.1f} |'.format(np.mean(tbs), min(tbs), max(tbs)))
    L.append('| YOLOv8n-seg | {:.1f} | {:.1f} | {:.1f} |'.format(np.mean(tys), min(tys), max(tys)))
    L.append('')

    L.append('## 2. 检出与缺陷像素量\n')
    L.append(f'- {n_img} 张中判定"有缺陷"的图数：v2-fast **{flag["v2fast"]}** / v2b **{flag["v2b"]}** / YOLO **{flag["yolo"]}**')
    L.append(f'- 全图缺陷像素合计：v2-fast **{tot["v2fast"]:,}** / v2b **{tot["v2b"]:,}** / YOLO **{tot["yolo"]:,}**')
    L.append(f'- v2b 相对 v2-fast 缺陷像素变化：**{tot["v2b"] - tot["v2fast"]:+,}**'
             + (f'（{100*(tot["v2b"]-tot["v2fast"])/tot["v2fast"]:+.1f}%）' if tot['v2fast'] else ''))
    L.append('')
    L.append('| 模型 | 平均 silie 像素/图 | 平均 fanzhe 像素/图 |')
    L.append('|---|---|---|')
    for tag, si, fi in (('v2-fast', 1, 2), ('v2b', 3, 4), ('YOLO', 5, 6)):
        L.append('| {} | {:.0f} | {:.0f} |'.format(tag,
                 np.mean([float(r[si]) for r in rows_stat]), np.mean([float(r[fi]) for r in rows_stat])))
    L.append('')

    L.append('## 3. 两两一致性 IoU（累计，{0} 图）\n'.format(n_img))
    L.append('| 类别 | v2-fast↔YOLO | v2b↔YOLO | v2-fast↔v2b |')
    L.append('|---|---|---|---|')
    for c, nm in enumerate(['background', '**silie (撕裂)**', 'fanzhe (翻折)']):
        L.append('| {} | {:.4f} | {:.4f} | {:.4f} |'.format(nm, i_vy[c], i_by[c], i_vb[c]))
    L.append('| mIoU | {:.4f} | {:.4f} | {:.4f} |'.format(i_vy.mean(), i_by.mean(), i_vb.mean()))
    L.append('')

    L.append('## 4. v2b 相对 v2-fast 的逐图变化\n')
    L.append(f'- v2b 把 v2-fast 报的缺陷**完全清掉**的图：{len(removed_by_v2b)} 张'
             + ('  \n  ' + '、'.join(removed_by_v2b) if removed_by_v2b else ''))
    L.append(f'- v2b **新增**缺陷(v2-fast 原本干净)的图：{len(added_by_v2b)} 张'
             + ('  \n  ' + '、'.join(added_by_v2b) if added_by_v2b else ''))
    L.append('')
    L.append('明细见 `stats_3models.csv`（含每图三模型缺陷像素与 v2b-v2fast 差值）。')
    L.append('')

    L.append('## 5. 结论（如实）\n')
    L.append('- 本测试集**无 GT**，无法判定"更准"，只能看：缺陷输出量、模型间一致性、速度。')
    if tot['v2fast'] and tot['v2b'] < tot['v2fast']:
        L.append('- v2b 相对 v2-fast 缺陷输出像素 {:,}→{:,}（{:+.0f}%），'.format(
            tot['v2fast'], tot['v2b'], 100 * (tot['v2b'] - tot['v2fast']) / tot['v2fast'])
            + '即 B 方案更保守 —— 若该批多为背景/过杀，这是**收益**；若含真实缺陷，需人工看叠加图确认是否被误压。')
    else:
        L.append('- v2b 与 v2-fast 缺陷输出量接近或更大，需人工看叠加图判断。')
    L.append('- 速度：YOLO {:.1f}ms 远快于 v2-fast {:.1f}ms / v2b {:.1f}ms。'.format(
        np.mean(tys), np.mean(t2s), np.mean(tbs)))
    L.append('- 建议：人工复核 `overlays_v2b/` 与 `overlays_v2fast/` 同名图，逐图确认 v2b 是否压掉了真缺陷。')
    txt = '\n'.join(L)
    (OUT / 'report_3models.md').write_text(txt, encoding='utf-8')
    print('\n' + txt)
    print('\n-> ', OUT / 'report_3models.md')


if __name__ == '__main__':
    main()
