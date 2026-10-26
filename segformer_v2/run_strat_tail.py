# -*- coding: utf-8 -*-
"""分层重切尾部编排（yolo 卡死修复后用）: 只跑 yolo 全新训练 -> 五模型全图评估 -> 汇总。
v1/v2/m2f 已训练完成（best 权重已在 runs_strat/*/best），不重训。
带日志 + 失败即停。绝不改动现有文件。
"""
import os, sys, time, subprocess
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2  = ROOT / 'segformer_v2'
V1  = ROOT / 'segformer_baseline'
M2F = ROOT / 'mask2former_baseline'
RUNS = V2 / 'runs_strat'
LOG = RUNS / 'tail.log'
PY = sys.executable


def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(cmd, tag, cwd):
    log(f'>>> {tag}: {" ".join(cmd)}')
    p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        bufsize=1, text=True, encoding='utf-8', errors='replace')
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush()
    rc = p.wait()
    log(f'<<< {tag} exit={rc}')
    if rc != 0:
        log(f'!!! {tag} FAILED -> 终止')
        sys.exit(1)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    open(LOG, 'a', encoding='utf-8').write(f'\n==== TAIL START {time.strftime("%Y-%m-%d %H:%M:%S")} ====\n')
    (RUNS / 'tail.pid').write_text(str(os.getpid()))

    # 1. yolo 全新训练（freeze_support 已修，exist_ok=True 复用 train_strat 目录）
    run([PY, str(ROOT / 'yolo_train_strat.py')], 'train_yolo', ROOT)

    # 2. 五模型全图评估（同一 val_full_strat GT，同口径）
    run([PY, str(V1 / 'infer_strat.py'),       '--run', 'full_strat'], 'eval_v1', V1)
    run([PY, str(V2 / 'infer_strat.py'),       '--run', 'v2_strat'],   'eval_v2', V2)
    run([PY, str(V2 / 'infer_fast_strat.py')],                          'eval_v2fast', V2)
    run([PY, str(M2F / 'eval_strat.py'),       '--run', 'full_strat'], 'eval_m2f', M2F)
    run([PY, str(V1 / 'eval_yolo_strat.py')],                           'eval_yolo', V1)

    # 3. 汇总
    run([PY, str(V2 / 'compare_strat.py')], 'compare', V2)
    log('=== TAIL DONE === runs_strat/compare_report_strat.md')


if __name__ == '__main__':
    main()
