# -*- coding: utf-8 -*-
"""汇总所有 sweep_eval.json (端点 v2/v2b + S1-S4) -> Pareto 表 (IoU vs 假阳) + 现场批对比。
输出 runs_strat/sweep_report.md。
"""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / 'runs_strat'
POINTS = [  # (tag, 显示名)
    ('v2_strat',  'v2基线 (0.12/0.20)'),
    ('sw_s3',     'S3 (0.12/0.40)'),
    ('sw_s4',     'S4 (0.12/0.80)'),
    ('sw_s1',     'S1 (0.25/0.20)'),
    ('sw_s2',     'S2 (0.40/0.20)'),
    ('v2b_strat', 'v2b/B (0.40/0.80)'),
]

def load(tag):
    p = RUNS / tag / 'sweep_eval.json'
    if not p.exists(): return None
    return json.loads(p.read_text(encoding='utf-8'))

def main():
    data = {tag: load(tag) for tag, _ in POINTS}
    have = [(t, n) for t, n in POINTS if data[t]]
    L = []
    L.append('# 超参扫描 Pareto 报告 (bg_keep × focal_alpha(bg))\n')
    L.append('背景假阳优化超参搜索: val_strat(有GT) 缺陷IoU + 42负样本假阳 + 3现场批缺陷输出量。\n')

    L.append('## Pareto 表 (按负样本假阳升序)\n')
    L.append('| 配置(bg_keep/alpha_bg) | mIoU | silie IoU | fanzhe IoU | 负样本假阳均值 | 最大 | >100px图 | 现场批总缺陷px(222/C8/fl) |')
    L.append('|---|---|---|---|---|---|---|---|')
    rows = sorted(have, key=lambda tn: data[tn[0]]['val']['neg_avg_fp'])
    for tag, name in rows:
        d = data[tag]; v = d['val']; f = d['field']
        fpx = '/'.join(str(f[b]['total_px']) for b in ['222', 'C8', 'fl'])
        L.append(f'| {name} | {v["mIoU"]} | {v["silie"]} | {v["fanzhe"]} | {v["neg_avg_fp"]} | {v["neg_max_fp"]} | {v["neg_gt100"]}/{v["n_neg"]} | {fpx} |')
    L.append('')

    L.append('## 现场批检出覆盖 (flagged/total, 防丢召回)\n')
    L.append('| 配置 | 222 | C8 | fl |')
    L.append('|---|---|---|---|')
    for tag, name in rows:
        f = data[tag]['field']
        L.append('| {} | {}/{} | {}/{} | {}/{} |'.format(name,
            f['222']['flagged'], f['222']['n'], f['C8']['flagged'], f['C8']['n'],
            f['fl']['flagged'], f['fl']['n']))
    L.append('')

    L.append('## 选点建议\n')
    base = data.get('v2_strat')
    if base:
        thr = base['val']['neg_avg_fp'] / 2
        L.append(f'- 约束: 负样本假阳均值 ≤ 基线一半 (≤ {thr:.0f} px/图)')
        cand = [(t, n) for t, n in rows if data[t]['val']['neg_avg_fp'] <= thr]
        if cand:
            best = max(cand, key=lambda tn: data[tn[0]]['val']['mIoU'])
            bd = data[best[0]]
            L.append(f'- 满足约束下 mIoU 最高: **{best[1]}** (mIoU={bd["val"]["mIoU"]}, '
                     f'假阳均值={bd["val"]["neg_avg_fp"]}, silie={bd["val"]["silie"]})')
        else:
            L.append('- 无配置满足"假阳≤基线一半"约束, 需放宽或换更小 bg_keep/alpha')
    L.append('- 若某 S 点相比基线的 silie/fanzhe IoU 几乎不掉、假阳显著降 -> 该点为平衡拐点')

    (RUNS / 'sweep_report.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))
    print('\n报告 ->', RUNS / 'sweep_report.md')

if __name__ == '__main__':
    main()
