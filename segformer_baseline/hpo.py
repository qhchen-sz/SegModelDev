# -*- coding: utf-8 -*-
"""
SegFormer 超参 pilot 搜索：4 组配置各 40 epoch，按 val mIoU 选冠军。
通过 subprocess 调 train.py（每组独立进程，状态隔离）。
"""
import os, sys, json, subprocess, time
from pathlib import Path

# 4 组配置（充分档）
CONFIGS = [
    {'tag': 'C1', 'lr': 6e-5, 'cw1': 2.0, 'cw2': 1.0},  # 基线
    {'tag': 'C2', 'lr': 6e-5, 'cw1': 3.0, 'cw2': 1.2},  # silie 加重
    {'tag': 'C3', 'lr': 3e-5, 'cw1': 2.5, 'cw2': 1.0},  # 低 lr
    {'tag': 'C4', 'lr': 1e-4, 'cw1': 2.5, 'cw2': 1.0},  # 高 lr
]
PILOT_EPOCHS = 40
HERE = Path(__file__).parent

def run_one(c):
    # 读 best mIoU（本进程产出的可信文件）
    def read_best(tag):
        f = HERE / 'runs' / tag / 'final_miou.txt'
        if f.exists():
            try: return float(f.read_text(encoding='utf-8').strip())
            except: pass
        meta = HERE / 'runs' / tag / 'best_meta.pt'
        if meta.exists():
            import torch
            d = torch.load(meta, map_location='cpu', weights_only=False)  # 含 numpy 标量，需 weights_only=False
            return float(d.get('miou', -1))
        return -1

    # 若已完成则复用结果，不重跑（崩溃恢复）
    done_file = HERE / 'runs' / c['tag'] / 'final_miou.txt'
    if done_file.exists():
        best = read_best(c['tag'])
        print(f'PILOT {c["tag"]} 已完成，复用 best mIoU={best:.4f}（跳过重训）', flush=True)
        return best

    cmd = [sys.executable, str(HERE / 'train.py'),
           '--tag', c['tag'], '--lr', str(c['lr']),
           '--cw0', '0.2', '--cw1', str(c['cw1']), '--cw2', str(c['cw2']),
           '--epochs', str(PILOT_EPOCHS),
           '--batch_size', '24', '--eval_every', '5']
    print('\n' + '='*70)
    print(f'PILOT {c["tag"]}: lr={c["lr"]} cw1={c["cw1"]} cw2={c["cw2"]} epochs={PILOT_EPOCHS}')
    print('='*70, flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         bufsize=1, text=True, encoding='utf-8', errors='replace')
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush()
    rc = p.wait()
    print(f'PILOT {c["tag"]} exit={rc} time={time.time()-t0:.0f}s', flush=True)
    return read_best(c['tag'])

def main():
    results = []
    for c in CONFIGS:
        miou = run_one(c)
        results.append({**c, 'miou': miou})
        print(f'>> {c["tag"]} best mIoU = {miou:.4f}', flush=True)
    # 选冠军
    results.sort(key=lambda r: r['miou'], reverse=True)
    winner = results[0]
    print('\n' + '#'*70)
    print('PILOT 排名（按 best val mIoU）:')
    for r in results:
        print(f'  {r["tag"]}: mIoU={r["miou"]:.4f}  lr={r["lr"]} cw1={r["cw1"]} cw2={r["cw2"]}')
    print(f'\n冠军: {winner["tag"]}  mIoU={winner["miou"]:.4f}')
    print(f'冠军配置: --lr {winner["lr"]} --cw1 {winner["cw1"]} --cw2 {winner["cw2"]}')
    out = HERE / 'runs' / 'hpo_result.json'
    out.write_text(json.dumps({'winner': winner, 'all': results}, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    print('写入', out)

if __name__ == '__main__':
    main()
