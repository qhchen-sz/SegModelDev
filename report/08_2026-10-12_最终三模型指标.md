# 三模型在 val_strat 验证集上的指标对比

- 验证集: `val_strat.txt`（89 张全图，含 GT）
- 口径: S4 & v2-fast = bs16 fp16 stride448 noTTA 全图滑窗; YOLO = imgsz1200 conf0.25 iou0.5
- 指标为像素级 per-class：IoU = TP/(TP+FP+FN)，Recall = TP/(TP+FN)，Precision = TP/(TP+FP)

## per-class IoU

| 类别 | S4 | v2-fast | YOLO |
|---|---|---|---|
| background | 0.9908 | 0.9901 | 0.9789 |
| silie | 0.5833 | 0.5611 | 0.1291 |
| fanzhe | 0.7039 | 0.7081 | 0.4535 |
| **mIoU** | **0.7593** | **0.7531** | **0.5205** |

## per-class Recall

| 类别 | S4 | v2-fast | YOLO |
|---|---|---|---|
| background | 0.9957 | 0.9931 | 0.9880 |
| silie | 0.7233 | 0.7434 | 0.1391 |
| fanzhe | 0.8156 | 0.8903 | 0.6562 |
| **mRecall** | **0.8449** | **0.8756** | **0.5944** |

## per-class Precision

| 类别 | S4 | v2-fast | YOLO |
|---|---|---|---|
| background | 0.9950 | 0.9970 | 0.9907 |
| silie | 0.7508 | 0.6958 | 0.6429 |
| fanzhe | 0.8371 | 0.7758 | 0.5948 |
| **mPrecision** | **0.8610** | **0.8229** | **0.7428** |

## 推理速度（RTX 3090，全图 ~2601×3001）

| 模型 | 平均 ms/图 |
|---|---|
| S4 | 871.9 |
| v2-fast | 871.1 |
| YOLO | 19.2 |