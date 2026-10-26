# SegFormer Baseline —— 极耳翻折/撕裂 语义分割

## 任务
极耳（电池引出金属片）缺陷像素级分割，3 类：
- `0` background 背景
- `1` silie 撕裂
- `2` fanzhe 翻折

## 模型结构（与 NVlabs/SegFormer 源码一致）
- 骨干：**MiT-B1**（Mix Vision Transformer，对应源码 `mit_b1`）
- 解码头：**SegFormerHead**（线性 MLP，对应源码 `decode_head`）
- 预训练：`nvidia/segformer-b1-finetuned-ade-512-512`（NVIDIA 从 NVlabs 原始权重转换）
  - 因 transformers 4.57 + torch 2.5.1 拒绝加载 `.bin`（CVE-2025-32434），且 b1 仓库无 safetensors，
    故用 `convert_weights.py` 把 `.bin` 一次性转成本地 `model.safetensors`，不升级 torch、不动 yolov10 环境。
- 末层分类头 150→3 类，重新初始化后微调。

## 文件
| 文件 | 作用 |
|---|---|
| `config.yaml` | 全部配置 |
| `convert.py` | ISAT JSON→class mask→滑窗 512 切块，按 train.txt/val.txt 划分 |
| `convert_weights.py` | 一次性：b1 `.bin`→本地 safetensors（已完成，产物在 `models/segformer-b1-ade/`） |
| `dataset.py` | torch Dataset + Albumentations 增强 + 加权采样 |
| `train.py` | 训练（加权 CE + poly lr + AMP + mIoU 存 best） |
| `infer.py` | 全图滑窗推理 + per-class IoU/mIoU + overlay 可视化 |
| `smoke.py` | 2 epoch + 1 图冒烟测试 |

## 运行（在 yolov10 conda env）
```bash
conda activate yolov10
cd D:/qhchen/NewModel/ADevFolder/segformer_baseline

# 0.（仅首次）转权重 —— 已执行过，无需重复，除非删了 models/
python convert_weights.py

# 1. 切块（已执行过，产物在 dataset/）—— 想重切再跑
python convert.py

# 2. 正式训练（150 epoch，约几小时）
python train.py

# 3. 全图推理评估 + overlay
python infer.py
```

## 输出
- `runs/best/` —— 最佳权重（按 val mIoU）
- `runs/train_log.csv` —— 每 epoch loss/mIoU/per-class IoU
- `runs/overlays/*.png` —— val 全图预测叠加图（背景灰/silie 红/fanzhe 蓝）
- `runs/eval_result.txt` —— 最终全图 mIoU

## 关键设计点
- **滑窗切块**：原图 ~3000×2600，缺陷占比小；512×512、stride 384（overlap 128）训练；纯背景块按 12% 比例保留防背景过载。
- **类别不平衡**：silie 仅 110 实例 vs fanzhe 296；用加权 CE（`class_weights=[0.2, 2.0, 1.0]`）+ WeightedRandomSampler（含 silie 块权重 3×）。
- **全图评估**：val 全图滑窗推理 logits overlap 平均后 argmax，与 GT 全图 mask 比 IoU，避免切块边界造成的虚高/虚低。

## 与现有 YOLOv8n-seg 的关系
SegFormer 是**语义分割**，YOLOv8n-seg 是**实例分割**，指标体系不同（mIoU vs mask mAP），
本 baseline 作**架构对比基线**。若要直接对标 mask mAP，再上 `facebook/mask2former-swin-tiny-coco-instance`。

## 已知风险
- 446 张里仅 239 张有缺陷，小数据易过拟合 → 重增强 + early best（按 val mIoU）。
- silie 实例少且小 → 加权 + 上采样；若 val silie IoU 仍低，可再加大 silie 权重或裁更多含 silie 块。
