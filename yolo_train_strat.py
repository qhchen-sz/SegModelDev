# -*- coding: utf-8 -*-
"""YOLOv8n-seg 训练（分层重切版）: 用 newsiliec9_strat.yaml，patience=100，imgsz=1200，workers=4。
支持断点续训（有 last.pt 则 resume）。输出 runs_strat/train_strat/。绝不改动旧 yolo 训练。
注意：Windows spawn 模式下必须 freeze_support()，否则 DataLoader worker 子进程重跑 __main__ 死锁。"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = str(ROOT / 'newsiliec9_strat.yaml')
PROJ = str(ROOT / 'runs_strat')
NAME = 'train_strat'
LAST = ROOT / 'runs_strat' / NAME / 'weights' / 'last.pt'


def main():
    if LAST.exists():
        print(f'[yolo_strat] resume from {LAST}')
        model = YOLO(str(LAST))
        model.train(resume=True)
    else:
        print(f'[yolo_strat] fresh train from yolov8n-seg.pt, data={DATA}')
        model = YOLO(str(ROOT / 'yolov8n-seg.pt'))
        model.train(data=DATA, project=PROJ, name=NAME, exist_ok=True,
                    epochs=5000, patience=100, imgsz=1200, workers=4,
                    seed=42, pretrained=True)


if __name__ == '__main__':
    freeze_support()
    main()

