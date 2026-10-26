# -*- coding: utf-8 -*-
"""
完整流水线（单进程，nohup 启动可跨会话存活）：
HPO(4配置pilot) -> 读冠军 -> 200ep 全量训练 -> SegFormer infer -> YOLO eval -> compare
日志: runs/pipeline.log  PID: runs/pipeline.pid
"""
import os, sys, json, time, subprocess, signal
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
# Windows 控制台默认 GBK，转发子进程含中文/替换符的输出会 UnicodeEncodeError，强制 UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / 'runs'
HPO_RESULT = RUNS / 'hpo_result.json'
LOG = RUNS / 'pipeline.log'

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def run(cmd, tag):
    log(f'>>> {tag}: {" ".join(cmd)}')
    p = subprocess.Popen(cmd, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        bufsize=1, text=True, encoding='utf-8', errors='replace')
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush()
    rc = p.wait()
    log(f'<<< {tag} exit={rc}')
    if rc != 0:
        log(f'!!! {tag} FAILED, abort pipeline')
        sys.exit(1)

def main():
    RUNS.mkdir(exist_ok=True)
    open(LOG, 'w', encoding='utf-8').close()
    open(RUNS / 'pipeline.pid', 'w').write(str(os.getpid()))
    py = sys.executable

    # 1. HPO pilot
    run([py, str(HERE / 'hpo.py')], 'hpo')
    res = json.load(open(HPO_RESULT, encoding='utf-8'))
    w = res['winner']
    log(f'HPO 冠军: {w["tag"]} lr={w["lr"]} cw1={w["cw1"]} cw2={w["cw2"]} miou={w["miou"]:.4f}')

    # 2. 全量训练
    run([py, str(HERE / 'train.py'), '--tag', 'full', '--epochs', '200',
         '--lr', str(w['lr']), '--cw0', '0.2', '--cw1', str(w['cw1']), '--cw2', str(w['cw2']),
         '--batch_size', '24', '--eval_every', '2'], 'train_full')
    # 3. 评估
    run([py, str(HERE / 'infer.py'), '--run', 'full'], 'seg_infer')
    run([py, str(HERE / 'eval_yolo.py')], 'yolo_eval')
    run([py, str(HERE / 'compare.py')], 'compare')
    log('=== PIPELINE DONE === runs/compare_report.md')

if __name__ == '__main__':
    main()
