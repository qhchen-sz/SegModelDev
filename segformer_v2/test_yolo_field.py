# -*- coding: utf-8 -*-
"""通用 YOLO-seg 现场批测试: 指定权重, 输出到 <batch>/_results/ (带 tag 后缀, 不覆盖已有结果)。
用法: python test_yolo_field.py --batch <图片文件夹> --weights <best.pt> [--tag yolo_train2]
口径: yolo imgsz1200 conf0.25 iou0.5 (与 test_s4_field.py / final_report.md 一致)
"""
import time, csv, glob, argparse
import numpy as np, cv2
from pathlib import Path
from datetime import datetime

COLS = [(60, 60, 60), (0, 0, 255), (255, 0, 0)]


def yolo_predict(ym, img):
    H, W = img.shape[:2]
    res = ym(img, imgsz=1200, conf=0.25, iou=0.5, verbose=False)[0]
    sem = np.zeros((H, W), np.uint8); n = 0
    if res.masks is not None:
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        polys = res.masks.xy
        for k in np.argsort(cf):
            pl = polys[k]
            if pl is None or len(pl) < 3:
                continue
            cv2.fillPoly(sem, [np.round(pl).astype(np.int32)], int(cls[k]))
        n = len(cls)
    return sem, n


def overlay(img, pred):
    ov = img.copy()
    for c, col in enumerate(COLS):
        m = pred == c
        ov[m] = (img[m] * 0.5 + np.array(col, np.float32) * 0.5).astype(np.uint8)
    return ov


def load_old_stats(path):
    """读 test_s4_field.py 的 stats_s4.csv 做对照: 返回 {name: (s4_px, yolo_px, yolo_n)}"""
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        out[r['image']] = (int(r['s4_silie_px']) + int(r['s4_fanzhe_px']),
                           int(r['yolo_silie_px']) + int(r['yolo_fanzhe_px']),
                           int(r['yolo_instances']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--tag', default='yolo')
    a = ap.parse_args()

    IMG = Path(a.batch); OUT = IMG / '_results'; OUT.mkdir(exist_ok=True)
    OV = OUT / f'overlays_{a.tag}'; OV.mkdir(exist_ok=True)
    imgs = sorted(glob.glob(str(IMG / '*.jpg')))
    H0, W0 = cv2.imread(imgs[0]).shape[:2]
    wp = Path(a.weights)
    print(f'[{IMG.name}] {len(imgs)} 张, {W0}x{H0}')
    print(f'模型: {wp} ({wp.stat().st_size/1e6:.1f} MB, {datetime.fromtimestamp(wp.stat().st_mtime):%Y-%m-%d})')

    from ultralytics import YOLO
    ym = YOLO(str(wp))
    print(f'task={ym.task} names={ym.names}')

    _ = yolo_predict(ym, cv2.imread(imgs[0]))  # warmup
    rt = []; rs = []; flagged = 0
    for i, ip in enumerate(imgs):
        img = cv2.imread(ip)
        t0 = time.time(); pred, ni = yolo_predict(ym, img); ty = (time.time() - t0) * 1000
        s_px = int((pred == 1).sum()); f_px = int((pred == 2).sum())
        if s_px + f_px > 0: flagged += 1
        cv2.imwrite(str(OV / (Path(ip).stem + '.png')), overlay(img, pred))
        rt.append([Path(ip).name, f'{ty:.1f}'])
        rs.append([Path(ip).name, s_px, f_px, ni])
        if i < 3 or i % 10 == 0:
            print(f'  [{i+1}/{len(imgs)}] {Path(ip).name[:24]} {ty:.0f}ms [s{s_px}/f{f_px}] n={ni}', flush=True)

    tys = [float(r[1]) for r in rt]
    with open(OUT / f'timing_{a.tag}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['image', f'{a.tag}_ms']); w.writerows(rt)
    with open(OUT / f'stats_{a.tag}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['image', f'{a.tag}_silie_px', f'{a.tag}_fanzhe_px', f'{a.tag}_instances']); w.writerows(rs)

    n = len(imgs)
    silie_avg = np.mean([r[1] for r in rs]); fanzhe_avg = np.mean([r[2] for r in rs]); inst_avg = np.mean([r[3] for r in rs])
    old = load_old_stats(OUT / 'stats_s4.csv')

    L = [f'# {a.tag} 模型测试报告（{IMG.name} | {n} 张全图，无 GT）\n',
         '## 说明\n',
         f'- 权重: `{wp}`（{wp.stat().st_size/1e6:.1f} MB，{datetime.fromtimestamp(wp.stat().st_mtime):%Y-%m-%d}）',
         f'- 类别: {ym.names}；口径: imgsz1200 conf0.25 iou0.5（与 S4/旧 YOLO 对比一致）',
         f'- 本批无标注, 精度指标 N/A; 叠加图 `overlays_{a.tag}/`（silie红/fanzhe蓝）。\n',
         f'## 推理速度（全图 {W0}×{H0}）\n',
         '| 模型 | 平均 ms/图 | 最小 | 最大 |', '|---|---|---|---|',
         f'| {a.tag} | {np.mean(tys):.1f} | {min(tys):.1f} | {max(tys):.1f} |\n',
         '## 检出与统计\n',
         f'- {n} 张中判定有缺陷: **{flagged}** 张',
         f'- 平均每图 silie {silie_avg:.0f} px / fanzhe {fanzhe_avg:.0f} px / 实例 {inst_avg:.2f} 个',
         '| 模型 | 有检出张数 | 平均缺陷 px/图 | 平均实例/图 | 平均 ms/图 |', '|---|---|---|---|---|']

    if old:
        s4_flag = sum(1 for v in old.values() if v[0] > 0)
        oy_flag = sum(1 for v in old.values() if v[1] > 0)
        oy_px = np.mean([v[1] for v in old.values()])
        s4_px = np.mean([v[0] for v in old.values()])
        with open(OUT / 'timing_s4.csv', encoding='utf-8') as f:
            s4_ms = np.mean([float(r['s4_ms']) for r in csv.DictReader(f)])
            f.seek(0)
            oy_ms = np.mean([float(r['yolo_ms']) for r in csv.DictReader(f)])
        L += [f'| S4 (SegFormer) | {s4_flag} | {s4_px:.0f} | - | {s4_ms:.1f} |',
              f'| YOLO (train_strat) | {oy_flag} | {oy_px:.0f} | - | {oy_ms:.1f} |',
              f'| **{a.tag}** | **{flagged}** | **{silie_avg+fanzhe_avg:.0f}** | **{inst_avg:.2f}** | **{np.mean(tys):.1f}** |']
        new_flag = {r[0]: (r[1] + r[2] > 0) for r in rs}
        oy_flag = {k: v[1] > 0 for k, v in old.items()}
        both = sum(1 for k, v in new_flag.items() if v and oy_flag.get(k, False))
        new_only = sum(1 for k, v in new_flag.items() if v and not oy_flag.get(k, False))
        oy_only = sum(1 for k, v in new_flag.items() if not v and oy_flag.get(k, False))
        L += ['\n## 与旧 YOLO (train_strat) 的逐图检出对照\n',
              f'- 两者都报: {both} 张；仅 {a.tag} 报: {new_only} 张；仅旧 YOLO 报: {oy_only} 张']

    L += ['\n## 结论\n',
          f'- 速度: {a.tag} 平均 {np.mean(tys):.1f}ms/图。',
          f'- 检出: {flagged}/{n} 张。',
          '- 无 GT, 精度不可判; 逐图对比见 stats 与叠加图, 建议抽样标注复核。']
    (OUT / f'report_{a.tag}.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L)); print('\n->', OUT / f'report_{a.tag}.md')


if __name__ == '__main__':
    main()
