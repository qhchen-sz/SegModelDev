# -*- coding: utf-8 -*-
"""五模型综合对比（分层重切版）: 读各 runs_strat 下的 eval_result.txt，
汇总成 runs_strat/compare_report_strat.md，附分层分布与口径差异说明。绝不改动 compare.py。"""
import yaml
from pathlib import Path

# 分层重切分布（split_stratified.py 输出，2026-09-23）
SPLIT_STATS = {
    'train': {'imgs': 357, 'silie_inst': 83, 'silie_img': 49, 'fanzhe_inst': 241, 'fanzhe_img': 142, 'neg': 168, 'neg_pct': 47.1},
    'val':   {'imgs': 89,  'silie_inst': 27, 'silie_img': 12, 'fanzhe_inst': 55,  'fanzhe_img': 35,  'neg': 42,  'neg_pct': 47.2},
}

def parse_kv(p):
    d = {}
    if not p.exists(): return d
    for line in open(p, encoding='utf-8'):
        if '=' in line:
            k, v = line.strip().split('=', 1)
            try: d[k] = float(v)
            except: d[k] = v
    return d

def num(d, k, fmt='{:.4f}'):
    v = d.get(k)
    return fmt.format(v) if isinstance(v, float) else 'N/A'

def main():
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config_strat.yaml', encoding='utf-8'))
    root = Path(cfg['project_root'])
    v1   = parse_kv(root / 'segformer_baseline'  / 'runs_strat' / 'full_strat'   / 'eval_result.txt')
    v2   = parse_kv(root / 'segformer_v2'        / 'runs_strat' / 'v2_strat'     / 'eval_result.txt')
    v2f  = parse_kv(root / 'segformer_v2'        / 'runs_strat' / 'v2_fast_strat'/ 'eval_result.txt')
    yolo = parse_kv(root / 'segformer_baseline'  / 'runs_strat' / 'yolo_eval'    / 'eval_result.txt')
    m2f  = parse_kv(root / 'mask2former_baseline'/ 'runs_strat' / 'full_strat'   / 'eval_result.txt')
    out_dir = root / 'segformer_v2' / 'runs_strat'
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = float(v2.get('avg_inference_ms', 1)) / max(v2f.get('avg_inference_ms', 1), 1)
    n_img = int(v2.get('num_images', 89))

    L = []
    L.append(f'# 分层重切综合对比报告（极耳翻折/撕裂 | 验证集 val_strat {n_img} 张全图）\n')
    L.append('## 与旧报告的核心差异\n')
    L.append('- 旧 `compare_report.md` 用 train.txt(192)/val.txt(88)：train 负样本 1.6% vs val 46.6% 严重失配，且排除 ~163 张负样本。')
    L.append('- 本报告用 `split_stratified.py` 对全部 446 张按类别组合分层 8:2：**train/val 负样本比例 47.1% / 47.2% 已对齐**，全部负样本纳入。')
    L.append('- 五模型均在同一新划分上**从零重训**（patience=100 早停，每 epoch 评估存 best），同一 val_full GT、同一语义 IoU 口径。\n')

    L.append('## 分层划分分布\n')
    L.append('| 集合 | 图数 | silie实例 | silie图 | fanzhe实例 | fanzhe图 | 负样本图 | 负样本占比 |')
    L.append('|---|---|---|---|---|---|---|---|')
    for k, s in SPLIT_STATS.items():
        L.append(f'| {k} | {s["imgs"]} | {s["silie_inst"]} | {s["silie_img"]} | {s["fanzhe_inst"]} | {s["fanzhe_img"]} | {s["neg"]} | {s["neg_pct"]}% |')
    L.append('')

    L.append('## 像素级指标（同口径，全图对比）\n')
    L.append('| 指标 | v1 (b1, CE) | **v2 (改进, +TTA)** | **v2-fast (无TTA)** | YOLOv8n-seg | Mask2Former |')
    L.append('|---|---|---|---|---|---|')
    L.append('| mIoU | {} | {} | {} | {} | {} |'.format(
        num(v1,'mIoU'), num(v2,'mIoU'), num(v2f,'mIoU'), num(yolo,'mIoU'), num(m2f,'mIoU')))
    L.append('| background IoU | {} | {} | {} | {} | {} |'.format(
        num(v1,'background_IoU'), num(v2,'background_IoU'), num(v2f,'background_IoU'),
        num(yolo,'background_IoU'), num(m2f,'background_IoU')))
    L.append('| **silie IoU (撕裂)** | {} | {} | {} | {} | {} |'.format(
        num(v1,'silie_IoU'), num(v2,'silie_IoU'), num(v2f,'silie_IoU'),
        num(yolo,'silie_IoU'), num(m2f,'silie_IoU')))
    L.append('| fanzhe IoU (翻折) | {} | {} | {} | {} | {} |'.format(
        num(v1,'fanzhe_IoU'), num(v2,'fanzhe_IoU'), num(v2f,'fanzhe_IoU'),
        num(yolo,'fanzhe_IoU'), num(m2f,'fanzhe_IoU')))
    L.append('')

    L.append('## 推理速度（RTX 3090，全图 ~3000x2600）\n')
    L.append('| 模型 | 平均推理 ms/图 | TTA | 加速手段 |')
    L.append('|---|---|---|---|')
    L.append('| SegFormer v1 | {} | 否 | bs1 fp32 stride448 |'.format(num(v1,'avg_inference_ms','{:.0f}')))
    L.append('| SegFormer v2 | {} | 是 | bs1 fp32 stride320 |'.format(num(v2,'avg_inference_ms','{:.0f}')))
    L.append('| **SegFormer v2-fast** | {} | 否 | **bs16 fp16 stride448** |'.format(num(v2f,'avg_inference_ms','{:.0f}')))
    L.append('| YOLOv8n-seg | {} | 否 | imgsz=1200 单次 |'.format(num(yolo,'avg_inference_ms','{:.0f}')))
    L.append('| Mask2Former | {} | 否 | 滑窗实例 |'.format(num(m2f,'avg_inference_ms','{:.0f}')))
    L.append('')

    L.append('## v2-fast 加速说明\n')
    L.append('- v2(+TTA) -> v2-fast: 推理 {}ms -> {}ms, 约 {:.1f}x 提速'.format(
        num(v2,'avg_inference_ms','{:.0f}'), num(v2f,'avg_inference_ms','{:.0f}'), sp))
    L.append('- 代价: mIoU {} -> {}, silie IoU {} -> {} (TTA 对 silie 贡献大, 去掉后 silie 回落)'.format(
        num(v2,'mIoU'), num(v2f,'mIoU'), num(v2,'silie_IoU'), num(v2f,'silie_IoU')))
    L.append('')

    L.append('## v2 vs v1 提升分析（v2 带 TTA）\n')
    L.append('- mIoU: {} -> {}'.format(num(v1,'mIoU'), num(v2,'mIoU')))
    L.append('- silie IoU: {} -> {}'.format(num(v1,'silie_IoU'), num(v2,'silie_IoU')))
    L.append('- fanzhe IoU: {} -> {}'.format(num(v1,'fanzhe_IoU'), num(v2,'fanzhe_IoU')))
    L.append('')

    L.append('## 说明\n')
    L.append('- v1/v2 同骨干(MiT-B1)。v2 差异: 损失(CE->Focal+Dice+Lovasz)、分辨率(512->640)、Copy-paste silie、断点续训。')
    L.append('- v2-fast 与 v2 同模型同权重, 仅推理不同: 批量(bs16)+FP16+stride448+无TTA。')
    L.append('- 全部模型预测转语义图后算同一套 per-class IoU/mIoU, 公平口径。')
    L.append('- 本轮为分层重切后从零重训; 与旧 `compare_report.md`(失配划分) 不可直接比绝对值, 只可比"分层是否拉平失配后各模型相对位次"。')
    L.append('- val_strat 含 42 张负样本(47.2%); background IoU 仍会偏高, 缺陷类(silie/fanzhe)IoU 为选型核心。')

    (out_dir / 'compare_report_strat.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))
    print('\n报告 ->', out_dir / 'compare_report_strat.md')

if __name__ == '__main__':
    main()
