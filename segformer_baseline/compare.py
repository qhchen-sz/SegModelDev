# -*- coding: utf-8 -*-
"""汇总 SegFormer 与 YOLOv8n-seg 评估结果，生成对比报告 runs/compare_report.md"""
import csv, re, yaml
from pathlib import Path

def parse_kv(p):
    d = {}
    if not p.exists(): return d
    for line in open(p, encoding='utf-8'):
        if '=' in line:
            k, v = line.strip().split('=', 1)
            try: d[k] = float(v)
            except: d[k] = v
    return d

def yolo_native(csv_path):
    if not csv_path.exists(): return {}
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
    keys = list(rows[0].keys())
    k = lambda s: [x for x in keys if s in x][0]
    best = max(rows, key=lambda r: float(r[k('mAP50(M)')].strip()))
    return {
        'mask_mAP50': float(best[k('mAP50(M)')].strip()),
        'mask_mAP50-95': float(best[k('mAP50-95(M)')].strip()),
        'mask_precision': float(best[k('precision(M)')].strip()),
        'mask_recall': float(best[k('recall(M)')].strip()),
        'best_epoch': int(best[keys[0]].strip()),
        'total_epochs': int(rows[-1][keys[0]].strip()),
    }

def main():
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config.yaml', encoding='utf-8'))
    root = Path(cfg['project_root'])
    runs = Path(cfg['out_dir'])
    seg = parse_kv(runs / 'full' / 'eval_result.txt')
    yolo = parse_kv(runs / 'yolo_eval' / 'eval_result.txt')
    yolo_nat = yolo_native(root / 'runs' / 'segment' / 'train_split' / 'results.csv')

    def f(v): return f'{v:.4f}' if isinstance(v, (int, float)) else str(v)
    lines = []
    lines.append('# SegFormer vs YOLOv8n-seg 对比报告\n')
    lines.append('## 任务：极耳翻折(fanzhe)/撕裂(silie) 分割 | 验证集：val.txt 的 88 张全图\n')
    lines.append('## 像素级指标（同口径，全图对比）\n')
    lines.append('| 指标 | SegFormer (MiT-B1) | YOLOv8n-seg |')
    lines.append('|---|---|---|')
    lines.append(f"| mIoU | {f(seg.get('mIoU'))} | {f(yolo.get('mIoU'))} |")
    lines.append(f"| background IoU | {f(seg.get('background_IoU'))} | {f(yolo.get('background_IoU'))} |")
    lines.append(f"| silie IoU (撕裂) | {f(seg.get('silie_IoU'))} | {f(yolo.get('silie_IoU'))} |")
    lines.append(f"| fanzhe IoU (翻折) | {f(seg.get('fanzhe_IoU'))} | {f(yolo.get('fanzhe_IoU'))} |")
    lines.append('')
    lines.append('## 推理速度（RTX 3090，全图 ~3000×2600）\n')
    lines.append('| 模型 | 平均推理 ms/图 |')
    lines.append('|---|---|')
    lines.append(f"| SegFormer | {f(seg.get('avg_inference_ms'))} |")
    lines.append(f"| YOLOv8n-seg | {f(yolo.get('avg_inference_ms'))} |")
    lines.append('')
    lines.append('## YOLOv8n-seg 原生实例分割指标（来自 results.csv 最佳 epoch）\n')
    lines.append(f"- mask mAP@0.5 = **{yolo_nat.get('mask_mAP50','N/A')}**")
    lines.append(f"- mask mAP@0.5:0.95 = {yolo_nat.get('mask_mAP50-95','N/A')}")
    lines.append(f"- mask precision = {yolo_nat.get('mask_precision','N/A')}")
    lines.append(f"- mask recall = {yolo_nat.get('mask_recall','N/A')}")
    lines.append(f"- 最佳 epoch = {yolo_nat.get('best_epoch','N/A')} / 共训 {yolo_nat.get('total_epochs','N/A')} epoch（早停）")
    lines.append('')
    lines.append('## 说明\n')
    lines.append('- SegFormer 为**语义分割**，YOLOv8n-seg 为**实例分割**；原生指标体系不同（mIoU vs mask mAP），不可直接比。')
    lines.append('- 上表「像素级指标」把两者预测都转成语义图后算同一套 per-class IoU/mIoU，是公平对比口径。')
    lines.append('- SegFormer 用滑窗 512×512 stride 448 推理；YOLO 用 imgsz=1200 单次推理。两者推理设置不同，速度对比仅供参考（SegFormer 滑窗覆盖整图，YOLO 缩放到 1200）。')

    out = runs / 'compare_report.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    print('\n报告写入', out)

if __name__ == '__main__':
    main()
