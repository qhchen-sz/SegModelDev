# -*- coding: utf-8 -*-
"""三模型对比报告：SegFormer / YOLOv8n-seg / Mask2Former（像素 mIoU + 推理时间）"""
import yaml
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

def main():
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config.yaml', encoding='utf-8'))
    root = Path(cfg['project_root'])
    seg = parse_kv(root / 'segformer_baseline' / 'runs' / 'full' / 'eval_result.txt')
    yolo = parse_kv(root / 'segformer_baseline' / 'runs' / 'yolo_eval' / 'eval_result.txt')
    m2f = parse_kv(root / 'mask2former_baseline' / 'runs' / 'full' / 'eval_result.txt')
    out_dir = root / 'mask2former_baseline' / 'runs'; out_dir.mkdir(parents=True, exist_ok=True)
    f = lambda d, k: f"{d.get(k, 'N/A'):.4f}" if isinstance(d.get(k), float) else str(d.get(k, 'N/A'))
    lines = [
        '# 三模型对比报告（极耳翻折/撕裂 分割 | 验证集 val.txt 88 张全图）\n',
        '## 像素级指标（同口径，全图对比）\n',
        '| 指标 | SegFormer (MiT-B1) | YOLOv8n-seg | Mask2Former (Swin-Tiny) |',
        '|---|---|---|---|',
        f"| mIoU | {f(seg,'mIoU')} | {f(yolo,'mIoU')} | {f(m2f,'mIoU')} |",
        f"| background IoU | {f(seg,'background_IoU')} | {f(yolo,'background_IoU')} | {f(m2f,'background_IoU')} |",
        f"| silie IoU (撕裂) | {f(seg,'silie_IoU')} | {f(yolo,'silie_IoU')} | {f(m2f,'silie_IoU')} |",
        f"| fanzhe IoU (翻折) | {f(seg,'fanzhe_IoU')} | {f(yolo,'fanzhe_IoU')} | {f(m2f,'fanzhe_IoU')} |",
        '',
        '## 推理速度（RTX 3090，全图 ~3000×2600）\n',
        '| 模型 | 平均推理 ms/图 |',
        '|---|---|',
        f"| SegFormer | {f(seg,'avg_inference_ms')} |",
        f"| YOLOv8n-seg | {f(yolo,'avg_inference_ms')} |",
        f"| Mask2Former | {f(m2f,'avg_inference_ms')} |",
        '',
        '## 说明\n',
        '- 三个模型预测都转成语义图后算同一套 per-class IoU/mIoU，公平口径。',
        '- SegFormer/Mask2Former 滑窗 512×512 推理；YOLO imgsz=1200 单次推理。推理设置不同，速度仅供参考。',
        '- Mask2Former 为实例分割架构（与 YOLO 同类），本表只比像素 mIoU；若需 mask mAP 可后续补 pycocotools。',
    ]
    (out_dir / 'compare_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    print('\n报告 ->', out_dir / 'compare_report.md')

if __name__ == '__main__':
    main()
