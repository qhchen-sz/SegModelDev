# -*- coding: utf-8 -*-
"""超参扫描驱动: 对 S1-S4 依次 切块(如需) -> 全patience训练 -> 评估(val+3现场批)。
后台运行, 日志+可续跑(训练有ckpt, 切块幂等)。绝不改动现有文件/run。
"""
import os, sys, time, subprocess, yaml
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
os.environ.setdefault('PYTHONIOENCODING','utf-8')
try:
    sys.stdout.reconfigure(encoding='utf-8',errors='replace'); sys.stderr.reconfigure(encoding='utf-8',errors='replace')
except Exception: pass
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
LOG=HERE/'runs_strat'/'sweep.log'
PY=sys.executable
SWEEPS=[('s1','sw_s1'),('s2','sw_s2'),('s3','sw_s3'),('s4','sw_s4')]

def log(m):
    line=f'[{time.strftime("%H:%M:%S")}] {m}'
    print(line,flush=True)
    with open(LOG,'a',encoding='utf-8') as f: f.write(line+'\n')

def run(cmd,tag):
    log(f'>>> {tag}: {" ".join(cmd)}')
    p=subprocess.Popen(cmd,cwd=str(HERE),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                       bufsize=1,text=True,encoding='utf-8',errors='replace')
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush()
    rc=p.wait(); log(f'<<< {tag} exit={rc}')
    if rc!=0:
        log(f'!!! {tag} FAILED -> 终止'); sys.exit(1)

def main():
    (HERE/'runs_strat').mkdir(parents=True,exist_ok=True)
    open(LOG,'a',encoding='utf-8').write(f'\n==== SWEEP START {time.strftime("%Y-%m-%d %H:%M:%S")} ====\n')
    (HERE/'runs_strat'/'sweep.pid').write_text(str(os.getpid()))
    # 先评估两个已训端点(基线 v2 / v2b)，统一 sweep_eval 口径
    run([PY,str(HERE/'sweep_eval.py'),'--config','config_strat.yaml','--run','v2_strat'], 'eval_endpoint_v2')
    run([PY,str(HERE/'sweep_eval.py'),'--config','config_strat_b.yaml','--run','v2b_strat'], 'eval_endpoint_v2b')
    for sid,tag in SWEEPS:
        cfgn=f'config_sw_{sid}.yaml'
        cfg=yaml.safe_load(open(HERE/cfgn,encoding='utf-8'))
        ds=Path(cfg['dataset_dir'])
        log(f'--- {sid} (bg_keep={cfg["bg_keep_ratio"]} alpha_bg={cfg["focal_alpha"][0]}) dataset={ds.name} exists={ds.exists()} ---')
        if not ds.exists():
            run([PY,str(HERE/'convert_strat_b.py'),'--config',cfgn], f'convert_{sid}')
        run([PY,str(HERE/'train_strat_b.py'),'--config',cfgn,'--tag',tag], f'train_{sid}')
        run([PY,str(HERE/'sweep_eval.py'),'--config',cfgn,'--run',tag], f'eval_{sid}')
        # 释放显存，避免下个训练 OOM
        time.sleep(5)
    log('=== SWEEP DONE ===')

if __name__=='__main__':
    main()
