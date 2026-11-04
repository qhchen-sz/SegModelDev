# SegFormer vs YOLOv8n-seg 对比报告

## 任务：极耳翻折(fanzhe)/撕裂(silie) 分割 | 验证集：val.txt 的 88 张全图

## 像素级指标（同口径，全图对比）

| 指标 | SegFormer (MiT-B1) | YOLOv8n-seg |
|---|---|---|
| mIoU | 0.6137 | 0.6055 |
| background IoU | 0.9827 | 0.9887 |
| silie IoU (撕裂) | 0.3357 | 0.2726 |
| fanzhe IoU (翻折) | 0.5228 | 0.5552 |

## 推理速度（RTX 3090，全图 ~3000×2600）

| 模型 | 平均推理 ms/图 |
|---|---|
| SegFormer | 653.6000 |
| YOLOv8n-seg | 73.8000 |

## YOLOv8n-seg 原生实例分割指标（来自 results.csv 最佳 epoch）

- mask mAP@0.5 = **0.41317**
- mask mAP@0.5:0.95 = 0.16706
- mask precision = 0.55179
- mask recall = 0.44718
- 最佳 epoch = 235 / 共训 478 epoch（早停）

## 说明

- SegFormer 为**语义分割**，YOLOv8n-seg 为**实例分割**；原生指标体系不同（mIoU vs mask mAP），不可直接比。
- 上表「像素级指标」把两者预测都转成语义图后算同一套 per-class IoU/mIoU，是公平对比口径。
- SegFormer 用滑窗 512×512 stride 448 推理；YOLO 用 imgsz=1200 单次推理。两者推理设置不同，速度对比仅供参考（SegFormer 滑窗覆盖整图，YOLO 缩放到 1200）。