# -*- coding: utf-8 -*-
"""汇总 3 个现场批 × 3 模型(v2-fast / S4 / YOLO) 的完整统计(含检出图片数量)。
读各 <batch>/_results/{stats.csv, stats_s4.csv}。
检出图片数量: 分 整体(任一缺陷) / silie / fanzhe 三类。
输出: 每个 <batch>/_results/summary_all_models.md + 控制台完整表。
"""
import csv
from pathlib import Path

BATCHES = [
    ('222',  'C:/Users/hymson/Desktop/222'),
    ('C8',   'C:/Users/hymson/Desktop/C8fanzhesilieguosha0925'),
    ('fl',   'C:/Users/hymson/Desktop/fanzhesilielousha0925'),
]

def read(p):
    if not p.exists(): return None
    rows = list(csv.reader(open(p, encoding='utf-8')))
    return rows[0], rows[1:]

def count(rows, si_i, fa_i):
    n = len(rows)
    n_any = n_si = n_fa = 0
    sum_si = sum_fa = 0
    for r in rows:
        s = int(r[si_i]); f = int(r[fa_i])
        if s + f > 0: n_any += 1
        if s > 0: n_si += 1
        if f > 0: n_fa += 1
        sum_si += s; sum_fa += f
    return dict(n=n, n_any=n_any, n_si=n_si, n_fa=n_fa,
                avg_si=sum_si/max(n,1), avg_fa=sum_fa/max(n,1))

def main():
    L = ['# 现场三批 × 三模型 完整统计（含检出图片数量）\n',
         '检出图片数量 = 该模型在该批中判定"存在缺陷"的图片数（分整体/silie/fanzhe）。',
         '所有批无 GT，精度指标 N/A；此表为输出量统计。\n',
         '| 批 | 图数 | 模型 | 检出图数(任一) | 检出silie图 | 检出fanzhe图 | 平均silie px | 平均fanzhe px |',
         '|---|---|---|---|---|---|---|---|']
    for name, d in BATCHES:
        res = Path(d) / '_results'
        v2 = read(res / 'stats.csv')
        s4 = read(res / 'stats_s4.csv')
        if not (v2 or s4):
            L.append(f'| {name} | - | (无结果) | | | | | |'); continue
        # 取共同图片集（避免某批图片被增删导致不对等，如 222 的 1384 被删）
        vrows = {r[0]: r for r in v2[1]} if v2 else {}
        srows = {r[0]: r for r in s4[1]} if s4 else {}
        common = sorted(set(vrows) & set(srows)) if (vrows and srows) else sorted(vrows or srows)
        # v2-fast: stats.csv cols 1,2 ; S4: stats_s4.csv cols 1,2 ; YOLO: cols 3,4
        v = count([vrows[k] for k in common], 1, 2) if vrows else None
        s = count([srows[k] for k in common], 1, 2) if srows else None
        y = count([(srows or vrows)[k] for k in common], 3, 4)
        for tag, cc in [('v2-fast', v), ('**S4**', s), ('YOLO', y)]:
            if cc is None: continue
            L.append(f'| {name} | {cc["n"]} | {tag} | **{cc["n_any"]}** | {cc["n_si"]} | {cc["n_fa"]} | {cc["avg_si"]:.0f} | {cc["avg_fa"]:.0f} |')
        L.append('| | | | | | | | |')  # 分隔三批
    txt = '\n'.join(L)
    print(txt)
    for name, d in BATCHES:
        res = Path(d) / '_results'
        if res.exists():
            (res / 'summary_all_models.md').write_text(txt, encoding='utf-8')
            print('->', res / 'summary_all_models.md')

if __name__ == '__main__':
    main()
