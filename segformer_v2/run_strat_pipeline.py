# -*- coding: utf-8 -*-
"""分层重切全流程编排: 切分 -> 三套切块 -> m2f缓存 -> 训练(v1/v2/m2f/yolo) -> 评估 -> 汇总。
带日志 + 失败即停 + 断点续跑（各训练脚本自带 ckpt/last.pt resume）。nohup 跨会话。
绝不改动现有 run_pipeline.py / runs/。所有产物落 *_strat 目录。
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

ROOT = Path(__file__).resolve().parent.parent          # 工程根目录
V2  = ROOT / 'segformer_v2'
V1  = ROOT / 'segformer_baseline'
M2F = ROOT / 'mask2former_baseline'
RUNS = V2 / 'runs_strat'
LOG = RUNS / 'pipeline.log'
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
        log(f'!!! {tag} FAILED -> 终止流水线')
        sys.exit(1)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    open(LOG, 'a', encoding='utf-8').write(f'\n==== PIPELINE START {time.strftime("%Y-%m-%d %H:%M:%S")} ====\n')
    (RUNS / 'pipeline.pid').write_text(str(os.getpid()))

    # 1. 切分（幂等）
    run([PY, str(ROOT / 'split_stratified.py')], 'split', ROOT)

    # 2. 切块（v2 先跑，生成 canonical val_full）
    run([PY, str(V2 / 'convert_strat.py')], 'convert_v2', V2)
    run([PY, str(V1 / 'convert_strat.py')], 'convert_v1', V1)
    run([PY, str(M2F / 'convert_strat.py')], 'convert_m2f', M2F)
    run([PY, str(M2F / 'preprocess_cache_strat.py')], 'm2f_cache', M2F)

    # 3. 训练（严格口径：epochs=5000 + patience=100 纯早停）。各脚本自带 ckpt resume：
    #    ckpt 不存在->从零；存在->续跑（lr 不跳变，因 epochs 未变）。中断后重跑本脚本即可续。
    run([PY, str(V1 / 'train_strat.py'),  '--tag', 'full_strat'], 'train_v1', V1)
    run([PY, str(V2 / 'train_strat.py'),  '--tag', 'v2_strat'],  'train_v2', V2)
    run([PY, str(M2F / 'train_strat.py'), '--tag', 'full_strat'], 'train_m2f', M2F)
    run([PY, str(ROOT / 'yolo_train_strat.py')], 'train_yolo', ROOT)

    # 4. 评估（同一 val_full GT，同口径）
    run([PY, str(V1 / 'infer_strat.py'),       '--run', 'full_strat'], 'eval_v1', V1)
    run([PY, str(V2 / 'infer_strat.py'),       '--run', 'v2_strat'],   'eval_v2', V2)
    run([PY, str(V2 / 'infer_fast_strat.py')],                          'eval_v2fast', V2)
    run([PY, str(M2F / 'eval_strat.py'),       '--run', 'full_strat'], 'eval_m2f', M2F)
    run([PY, str(V1 / 'eval_yolo_strat.py')],                           'eval_yolo', V1)

    # 5. 汇总
    run([PY, str(V2 / 'compare_strat.py')], 'compare', V2)
    log('=== PIPELINE DONE === runs_strat/compare_report_strat.md')


if __name__ == '__main__':
    main()
