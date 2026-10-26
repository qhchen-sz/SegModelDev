# -*- coding: utf-8 -*-
"""Mask2Former 流水线：150ep 训练 -> 全图 eval -> compare。nohup 跨会话存活。日志 runs/pipeline.log"""
import os, sys, json, time, subprocess
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / 'runs'
LOG = RUNS / 'pipeline.log'

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(line + '\n')

def run(cmd, tag):
    log(f'>>> {tag}: {" ".join(cmd)}')
    p = subprocess.Popen(cmd, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        bufsize=1, text=True, encoding='utf-8', errors='replace')
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush()
    rc = p.wait()
    log(f'<<< {tag} exit={rc}')
    if rc != 0:
        log(f'!!! {tag} FAILED'); sys.exit(1)

def main():
    RUNS.mkdir(exist_ok=True)
    open(LOG, 'w', encoding='utf-8').close()
    open(RUNS / 'pipeline.pid', 'w').write(str(os.getpid()))
    py = sys.executable
    run([py, str(HERE / 'train.py'), '--tag', 'full', '--epochs', '150',
         '--lr', '5e-05', '--batch_size', '16', '--eval_every', '1'], 'train')
    run([py, str(HERE / 'eval.py'), '--run', 'full'], 'eval')
    run([py, str(HERE / 'compare.py')], 'compare')
    log('=== PIPELINE DONE === runs/compare_report.md')

if __name__ == '__main__':
    main()
