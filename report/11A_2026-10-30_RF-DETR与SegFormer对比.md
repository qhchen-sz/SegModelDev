# RF-DETR 与 SegFormer 在 hefeic1 正/侧面数据上的对比实验

> 实验日期：2026-10-29 ~ 2026-10-30  
> 数据：`D:\qhchen\NewModel\hefeic1\zhengmian`、`D:\qhchen\NewModel\hefeic1\cemian`  
> RF-DETR 结果：`D:\qhchen\NewModel\ADevFolder\rfdetr\runs\{zhengmian,cemian}_base1120`  
> SegFormer 结果：`hefeic1\{view}\validation_results_more`、`hefeic1\{view}\SOP_{view}_more.md`  
> 报告生成：2026-11-04

---

## 0. 结论速览

1. **RF-DETR 与 SegFormer 不是同一任务，指标不能直接比较。** RF-DETR 是目标检测，输出矩形框，评估用 COCO mAP/F1；SegFormer 是语义分割，输出像素 mask，评估用 per-class IoU/mIoU/P/R。本报告**不做跨口径排名**。
2. **RF-DETR 的框级检测结果**：zhengmian 最佳 EMA mAP@0.5 = 0.6682、mAP@0.5:0.95 = 0.3489；cemian 最佳 EMA mAP@0.5 = 0.8158、mAP@0.5:0.95 = 0.5081。cemian 的框级定位明显好于 zhengmian。
3. **SegFormer `sw_s4_more` 的像素级结果**：zhengmian 合并验证集 60 张 mIoU 0.7471、silie IoU 0.5691、fanzhe IoU 0.6907，负样本假阳 0；cemian 47 张 mIoU 0.7893、silie IoU 0.6111、fanzhe IoU 0.7670，负样本假阳 394.8 px/图。
4. **RF-DETR 本次没有 per-class AP、没有推理耗时、没有负样本假阳统计**，因此无法判断它在 silie 小目标、过杀控制、速度上的实际表现。现有数据**不足以得出"RF-DETR 优于/劣于 SegFormer"的结论**。
5. **从业务输出形态看**：若产线需要缺陷像素区域、面积或接斑点分析，SegFormer 的 mask 仍是更直接的方案；RF-DETR 的框需要额外分割才能得到同样输出。

---

## 1. 实验背景与任务口径

### 1.1 两条技术路线

| 项 | RF-DETR | SegFormer `sw_s4_more` |
|---|---|---|
| 任务 | 目标检测（bbox） | 语义分割（pixel mask） |
| 输出 | 每图若干 `{class, score, x1,y1,x2,y2}` | 与原图同尺寸的 3 类语义图（0 背景 / 1 silie / 2 fanzhe） |
| 标注形式 | YOLO 归一化框 → COCO bbox | ISAT 多边形 → 类别 mask |
| 评估指标 | COCO mAP@0.5、mAP@0.5:0.95、mAR、P/R/F1 | per-class IoU、mIoU、P/R、负样本假阳像素 |
| 能否直接算缺陷面积 | 否，只能给框面积 | 是，像素级面积 |
| 能否直接接斑点分析 | 否，需额外分割/后处理 | 是，输出 8bpp 单通道 mask |
| 本次训练数据 | `train_strat_more.txt` | `train_strat_more.txt` |
| 本次验证数据 | `val_strat_more.txt` | `val_strat_more.txt` |

> 关键：mAP 衡量"框是否框住目标且类别正确"，mIoU 衡量"每个像素类别是否正确"。两者没有可换算关系，**不能把 0.6682 的 mAP 与 0.7471 的 mIoU 比大小**。

### 1.2 为什么做这次对比

此前极耳缺陷主方案是 SegFormer 语义分割（报告 01~11）。RF-DETR 是近期较新的检测架构（DINOv2 编码器 + DETR 解码器），本次尝试用它直接检测 silie/fanzhe 框，验证两条路线在同数据上的可行性、精度形态与工程代价。

---

## 2. 数据与验证集

### 2.1 数据规模

两个视角均使用"旧数据 + `images_more` 新增数据"的合并划分（与 SegFormer `sw_s4_more` 完全一致）：

| 视角 | 原始 JPG | 训练图 | 验证图 | 训练框 | 验证框 |
|---|---:|---:|---:|---:|---:|
| zhengmian | 301 | 241 | 60 | 389 | 86 |
| cemian | 232 | 185 | 47 | 205 | 52 |

### 2.2 验证集目标分布

| 视角 | 验证图 | silie 框 | silie 图 | fanzhe 框 | fanzhe 图 | 负样本图 |
|---|---:|---:|---:|---:|---:|---:|
| zhengmian | 60 | 6 | 5 | 80 | 47 | 8 |
| cemian | 47 | 12 | 8 | 40 | 25 | 14 |

数据来源：`rfdetr/datasets/{view}/valid/_annotations.coco.json` 与 `hefeic1/{view}/SOP_{view}_more.md` 的划分表，两边统计一致。

### 2.3 子集口径

`val_strat_more` 由两段组成：

- `oldval`：旧划分的验证集（zhengmian 43 张 / cemian 37 张），是**主 holdout**；
- `newval`：新增图片的验证集（17 张 / 10 张），全部为正样本，且与新增训练图来自同批次相邻帧，指标偏乐观，**只作趋势参考**；
- `combined`：两者合并（60 张 / 47 张），即 RF-DETR 的完整 valid 集。

> 本报告的 SegFormer 结论以 `oldval` 为主、`combined` 为辅；RF-DETR 只有 `combined` 级指标（训练日志不区分子集）。

---

## 3. RF-DETR 实验设置

### 3.1 模型与输入

| 项 | 值 |
|---|---|
| 模型 | `RFDETRBase`（`RFDETRBaseConfig`） |
| 编码器 | `dinov2_windowed_small`，`patch_size=14`，`num_windows=4` |
| 解码器 | 3 层，300 queries，`num_select=300` |
| 输入分辨率 | **1120×1120**（全图缩放，不切块；1120 = 56×20） |
| 类别 | 2 类：`silie`、`fanzhe`（背景无框） |
| 预训练 | `rf-detr-base.pth`（Roboflow 官方权重） |
| 分割头 | 关闭（`segmentation_head=false`），纯检测 |

### 3.2 数据转换

`rfdetr/make_coco.py` 把 YOLO 归一化框 `(cls, cx, cy, w, h)` 转成 COCO `[x1,y1,w,h]`：

- 越界框裁剪到图像内，非法框跳过；
- 每视角生成 `datasets/{view}/{train,valid}/_annotations.coco.json`；
- 图片用硬链接，不复制原文件。

转换结果与 SegFormer 划分完全对齐：

| 视角/子集 | 图数 | 框数 | silie | fanzhe | 空图（负样本） |
|---|---:|---:|---:|---:|---:|
| zhengmian/train | 241 | 389 | 38 | 351 | 34 |
| zhengmian/valid | 60 | 86 | 6 | 80 | 8 |
| cemian/train | 185 | 205 | 56 | 149 | 54 |
| cemian/valid | 47 | 52 | 12 | 40 | 14 |

### 3.3 训练配置

| 参数 | 值 |
|---|---|
| epochs 上限 | 5000（实际由早停决定） |
| early stopping | 开启，patience = 100，min_delta = 0.001，监控 EMA 指标 |
| best model metric | `map`（EMA 最佳） |
| batch size | 15（自动探测），grad_accum_steps = 2 |
| 学习率 | 1e-4（encoder 1.5e-4） |
| 优化器 | AdamW，weight decay 1e-4，grad clip 0.1 |
| 混合精度 | AMP 自动 |
| 多尺度 | 关闭（固定 1120） |
| 数据增强 | 默认 scale jitter，CPU |
| EMA | 开启，decay 0.993 |

> 训练日志没有记录每 epoch 耗时，因此本报告**不给出训练总时长**。

### 3.4 训练状态

| 视角 | 日志最后 val epoch | 最佳 mAP@0.5:0.95 epoch | 最佳 + patience | 状态判断 |
|---|---:|---:|---:|---|
| zhengmian | 160 | 60 | 160 | 恰为 patience 边界，按配置推断已早停 |
| cemian | 173 | 73 | 173 | 恰为 patience 边界，按配置推断已早停 |

日志文件没有显式 "early stop" 标记，上表是按 `best epoch + patience 100` 推断；最佳权重文件 `checkpoint_best_ema.pth` 已生成。

---

## 4. RF-DETR 结果

### 4.1 最佳 EMA 指标

| 视角 | 最佳 epoch（mAP50-95） | mAP@0.5 | mAP@0.5:0.95 | mAP@0.75 | mAR | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zhengmian | 60 | 0.6682 | **0.3489** | 0.3277 | 0.5913 | 0.7571 | 0.6458 | 0.6970 |
| cemian | 73 | 0.8158 | **0.5081** | 0.5618 | 0.7071 | 0.9655 | 0.6292 | 0.7597 |

> 指标来自 `runs/{view}_base1120/metrics.csv` 的 `val/ema_*` 列，均为 EMA 模型在完整 valid 集（60/47 张）上的 COCO 检测指标，**不是 mask mAP**。

### 4.2 不同指标的最佳 epoch 并不一致

| 视角 | 指标 | 最佳值 | 最佳 epoch |
|---|---|---:|---:|
| zhengmian | mAP@0.5:0.95 | 0.3489 | 60 |
| zhengmian | mAP@0.5 | 0.6976 | 53 |
| zhengmian | F1 | 0.7223 | 68 |
| cemian | mAP@0.5:0.95 | 0.5081 | 73 |
| cemian | mAP@0.5 | 0.8172 | 64 |
| cemian | F1 | 0.7973 | 59 |

说明验证指标存在 ±0.02~0.03 的波动；报告以配置指定的 `best_model_metric=map`（mAP50-95）选出的权重为准。

### 4.3 训练末期指标回落

| 视角 | 最佳 mAP50-95 | 末期 mAP50-95 | 末期 epoch |
|---|---:|---:|---:|
| zhengmian | 0.3489 | 0.3136 | 160 |
| cemian | 0.5081 | 0.4795 | 173 |

早停前指标已低于峰值，说明 best checkpoint 的选择是必要的；末期权重不应直接用于部署。

### 4.4 本次未记录的关键信息

| 项 | 状态 | 影响 |
|---|---|---|
| per-class AP（silie / fanzhe 分开） | **未记录**（`log_per_class_metrics=false`） | 无法判断 silie 小目标是否被 fanzhe 主导的总体 mAP 掩盖 |
| 负样本假阳（8/14 张空图上的误检） | **未统计** | 无法判断过杀水平 |
| 推理耗时（ms/图） | **未测** | 无法与 SegFormer 的 TRT 速度比较 |
| 训练总时长 / 每 epoch 耗时 | **未记录** | 无法比较训练成本 |
| 检测框可视化案例 | 已有 `runs/{view}_vis/` 叠加图，但未做系统漏检/误检分析 | 只能目检，不能量化 |

---

## 5. SegFormer `sw_s4_more` 实验设置与结果

### 5.1 训练设置

| 参数 | 值 |
|---|---|
| 模型 | SegFormer-B1，MiT-B1 编码器 + 全 MLP 解码头 |
| 输入 | 640×640 tile，stride 512，全图滑窗 |
| 类别 | 3 类：0 background / 1 silie / 2 fanzhe |
| 初始化 | ADE20K 预训练，3 类头重新初始化 |
| 损失 | Focal 1.0 + Dice 1.0 + Lovasz 1.0（gamma 2.0，alpha [0.8, 2.0, 1.0]） |
| 优化器 | AdamW，lr 6e-5，wd 0.01，poly `(1-step/total)^0.9` |
| batch / AMP | 16 / FP16 autocast + GradScaler |
| 背景采样 | 负样本图背景 tile 全保留，正样本图背景 0.30，val 背景 0.12 |
| Copy-Paste | silie 素材概率 0.5 |
| 早停 | epochs 5000，patience 100，按 val tile mIoU |
| 推理 | 640 tile / stride 448 / bs16 / FP16 / 无 TTA |

### 5.2 训练结果

| 视角 | 实际 epoch | best epoch | 早停 epoch | 训练期 best tile mIoU | 单 epoch 耗时 | 累计训练 |
|---|---:|---:|---:|---:|---:|---:|
| zhengmian | 483 | 382 | 482 | 0.7628 | 90~95 s | 约 12.5 h |
| cemian | 283 | 182 | 282 | 0.7860 | 51 s | 约 4.0 h |

> 训练期 tile mIoU 只用于早停，不是全图验收值。

### 5.3 全图验证结果（新模型 vs 旧模型）

**zhengmian**

| 子集 | 模型 | mIoU | silie IoU | fanzhe IoU | silie P/R | fanzhe P/R | 负样本假阳均值/最大 | pred/GT 面积 | 平均耗时 |
|---|---|---:|---:|---:|---|---|---|---:|---:|
| oldval 43 | 旧 `sw_s4` | 0.7461 | 0.5264 | 0.7289 | 0.711/0.670 | 0.833/0.854 | 9557.6 / 64433 | 1.021 | 1039.7 ms |
| oldval 43 | **新 `sw_s4_more`** | **0.7568** | **0.5691** | 0.7185 | **0.736/0.715** | **0.857**/0.817 | **0 / 0** | **0.954** | 1088.8 ms |
| combined 60 | 旧 `sw_s4` | 0.7285 | 0.5264 | 0.6793 | — | 0.796/0.822 | 9557.6 / 64433 | 1.029 | 997.2 ms |
| combined 60 | **新 `sw_s4_more`** | **0.7471** | **0.5691** | **0.6907** | — | **0.844/0.792** | **0 / 0** | **0.938** | 1089.6 ms |
| newval 17 | 旧 `sw_s4` | 0.7650 | – | 0.5583 | – | 0.699/0.735 | – | 1.052 | 1103.2 ms |
| newval 17 | **新 `sw_s4_more`** | **0.7975** | – | **0.6169** | – | **0.807/0.723** | – | **0.896** | 1083.0 ms |

**cemian**

| 子集 | 模型 | mIoU | silie IoU | fanzhe IoU | silie P/R | fanzhe P/R | 负样本假阳均值/最大 | pred/GT 面积 | 平均耗时 |
|---|---|---:|---:|---:|---|---|---|---:|---:|
| oldval 37 | 旧 `sw_s4` | 0.7785 | 0.6048 | 0.7398 | 0.724/0.787 | 0.832/0.869 | 2652.3 / 21191 | 1.038 | 493.6 ms |
| oldval 37 | **新 `sw_s4_more`** | **0.7850** | **0.6111** | **0.7526** | **0.787**/0.732 | 0.833/**0.886** | **394.8 / 5520** | 1.061 | 489.0 ms |
| combined 47 | 旧 `sw_s4` | 0.7029 | 0.5931 | 0.5352 | — | 0.823/0.605 | 2652.3 / 21191 | 0.733 | 494.6 ms |
| combined 47 | **新 `sw_s4_more`** | **0.7893** | **0.6111** | **0.7670** | — | **0.853/0.884** | **394.8** / 5520 | **1.034** | 491.4 ms |
| newval 10 | 旧 `sw_s4` | 0.3732 | – | 0.1783 | – | 0.762/0.189 | – | 0.248 | 493.7 ms |
| newval 10 | **新 `sw_s4_more`** | **0.8872** | – | **0.7909** | – | **0.887/0.879** | – | **0.991** | 494.0 ms |

要点：

- zhengmian 新模型在 oldval 上**负样本假阳清零**（8 张全干净），silie IoU +0.043，fanzhe IoU 略降 0.010（预测面积收缩、precision 提升）；
- cemian 新模型在 oldval 上负样本假阳 -85%（2652→395），fanzhe IoU +0.013，silie precision 提升、recall 略降；
- cemian 旧模型在 newval 上严重漏检（fanzhe recall 0.189、面积比 0.248），新模型恢复到 0.879 / 0.991，这是 combined 从 0.703 升到 0.789 的主因。

### 5.4 C++ / TensorRT FP16 复核

| 视角 | 子集 | PyTorch mIoU | C++ TRT FP16 mIoU | 差值 | TRT 耗时 |
|---|---|---:|---:|---:|---:|
| zhengmian | combined | 0.747124 | 0.746960 | 0.00016 | 263.0 ms（GPU 忙时；空闲约 100 ms） |
| zhengmian | oldval | 0.756846 | 0.756715 | 0.00013 | — |
| cemian | combined | 0.789255 | 0.789205 | 0.00005 | 100.7 ms |
| cemian | oldval | 0.784996 | 0.784953 | 0.00004 | — |

部署路径精度损失 ≤0.0002，已满足验收线；模型可通过 C# `SegModel` 直接输出 8bpp mask 接斑点分析。

---

## 6. 分任务对照

> 下表只做**同视角并列展示**，不同列不是同一指标，禁止跨列比较或排名。

| 视角 | RF-DETR 框级 | SegFormer 像素级（combined） | SegFormer 负样本假阳 | RF-DETR 速度 |
|---|---|---|---|---|
| zhengmian | mAP@0.5 0.6682 / mAP@0.5:0.95 0.3489 / F1 0.6970 | mIoU 0.7471 / silie 0.5691 / fanzhe 0.6907 | 0 px/图（8 张全干净） | 未测 |
| cemian | mAP@0.5 0.8158 / mAP@0.5:0.95 0.5081 / F1 0.7597 | mIoU 0.7893 / silie 0.6111 / fanzhe 0.7670 | 394.8 px/图（14 张中 1 张 >100px） | 未测 |

### 6.1 输出形态

- RF-DETR 只输出框。要得到缺陷面积、形状、斑点分析结果，还需要额外的分割步骤（例如框内分割或传统阈值分割），这会引入第二次误差；
- SegFormer 直接输出逐像素类别图，产线已经用它生成 `GetMask()` 单通道掩膜，接斑点分析无需额外模型。

### 6.2 silie 小目标

- RF-DETR 的总体 mAP 是两类平均，但验证集 silie 只有 6 个（zhengmian）/ 12 个（cemian）目标，且日志没有 per-class AP，**无法确认 silie 的实际检出水平**；
- SegFormer 有 per-class 数字：zhengmian silie IoU 0.5691、P/R 0.736/0.715；cemian silie IoU 0.6111、P/R 0.787/0.732。撕裂类仍是弱项，但至少可量化。

### 6.3 负样本与过杀

- RF-DETR 未统计 8/14 张空图上的误检数量，无法与 SegFormer 的 0 / 394.8 px 对比；
- SegFormer `sw_s4_more` 的过杀控制来自"负样本背景 tile 全保留 + 正样本背景 0.30"，这是分割路线可以直接优化的手段；检测路线要控制空图误检，需要额外调阈值或负样本策略。

### 6.4 速度与部署

| 项 | RF-DETR | SegFormer |
|---|---|---|
| 推理耗时 | **未测** | PyTorch 489~1089 ms/图；C++ TRT FP16 约 100~263 ms/图 |
| 部署形态 | PyTorch/或需自行导出；本次未做 C++/TRT 部署 | 已有 C++ DLL + C# 接口 + TensorRT engine，产线可用 |
| 输入方式 | 1120×1120 全图缩放 | 640 tile + 448 stride 滑窗 |
| 数据准备 | 直接全图 + COCO 框，无需切块 | 需生成 4039 / 2272 个训练 tile 与 mask |

> 因为 RF-DETR 未测速度，**不能断言检测路线更快**；全图 1120 单次前向在理论上可能比全图滑窗少很多次调用，但需要实测确认。

### 6.5 训练成本

| 项 | RF-DETR | SegFormer |
|---|---|---|
| 实际训练轮次 | 160 / 173（早停） | 483 / 283（早停） |
| 单 epoch 耗时 | 未记录 | 90~95 s / 51 s |
| 总训练时长 | 未记录 | 约 12.5 h / 4.0 h |

> 两者每 epoch 的数据量和模型规模不同，轮次与时长都不能直接比。

---

## 7. 结论与建议

### 7.1 本次能确认的

1. RF-DETR 可以在两个视角上完成框级训练与早停收敛，cemian 的框级 mAP@0.5 达到 0.8158，说明检测路线在侧面数据上有可行性；
2. SegFormer `sw_s4_more` 在相同数据划分上提供了可量化的像素级结果，并且在 zhengmian 上把负样本假阳压到 0、在 cemian 上降低 85%；
3. 两条路线的验证集完全对齐（60 / 47 张，目标数一致），数据层面具备后续做统一口径复评的基础。

### 7.2 本次不能确认的

1. RF-DETR 的 **silie 检出能力**——没有 per-class AP，silie 验证目标只有 6/12 个；
2. RF-DETR 的 **过杀水平**——没有统计负样本图误检；
3. RF-DETR 的 **推理速度与部署成本**——未测、未导出、未接产线；
4. **哪条路线更优**——任务口径不同，现有证据不足以排名。

### 7.3 建议的下一步（按优先级）

1. **补 RF-DETR per-class COCO AP**：用 `checkpoint_best_ema.pth` 对 `valid` 重跑 COCO 评估，分别输出 silie / fanzhe 的 AP50、AP50-95、P/R；这是判断检测路线是否可行的最短路径。
2. **补 RF-DETR 负样本误检统计**：对 8/14 张空图统计检出框数与置信度分布，与 SegFormer 的假阳像素对比。
3. **补速度实测**：在同一台机器上测 RF-DETR（PyTorch 或导出后）与 SegFormer（PyTorch / TRT）的 ms/图，至少给出 batch=1 口径。
4. **若要严格比较**：把 SegFormer mask 转成连通域框，在同一 `valid` 集上重算 COCO mAP；或把 RF-DETR 框填成 mask 算像素 IoU。两种方式都各有偏向，需在报告中明确说明。
5. **业务选型**：如果最终需要缺陷面积/斑点分析，优先保留 SegFormer 路线；如果只需要"有无 + 位置"的快速筛查，且后续补测证明 RF-DETR 的 silie 与过杀可接受，再考虑检测路线。

---

## 8. 复现信息

### 8.1 RF-DETR

```bash
cd D:\qhchen\NewModel\ADevFolder\rfdetr
python make_coco.py --view zhengmian
python train_rfdetr.py --view zhengmian
python train_rfdetr.py --view cemian
# 可视化（可选）
python infer_rfdetr.py --view zhengmian --ckpt runs/zhengmian_base1120/checkpoint_best_ema.pth --num 20
```

产物：

- 权重：`runs/{view}_base1120/checkpoint_best_ema.pth`
- 指标：`runs/{view}_base1120/metrics.csv`
- 训练配置：`runs/{view}_base1120/training_config.json`
- 数据：`datasets/{view}/{train,valid}/_annotations.coco.json`

### 8.2 SegFormer `sw_s4_more`

```bash
python D:\qhchen\NewModel\hefeic1\{view}\scripts\eval_compare_more.py \
  --config D:\qhchen\NewModel\hefeic1\{view}\config_sw_s4_more.yaml \
  --out-dir D:\qhchen\NewModel\hefeic1\{view}\validation_results_more
```

产物：

- 权重：`hefeic1/{view}/runs_strat/sw_s4_more/best`
- 指标：`hefeic1/{view}/validation_results_more/summary.md`、`summary.csv`、6 份 JSON
- 训练日志：`hefeic1/{view}/runs_strat/sw_s4_more/train_log.csv`
- 部署复核：`hefeic1/{view}/validation_results_more/cpp_combined_metrics.json`
- 完整 SOP：`hefeic1/{view}/SOP_{view}_more.md`

---

## 9. 局限声明

1. 本报告**没有做统一口径复评**（未把 mask 转框、未把框转 mask），跨任务比较仅限定性；
2. RF-DETR 的 per-class AP、负样本假阳、推理速度、训练时长均未记录或未测；
3. RF-DETR 的 `newval` 与 `train` 的相邻帧关系无法从日志中区分，报告只用 combined 级指标；
4. SegFormer 的 `newval` 指标偏乐观，结论以 `oldval` 为主；
5. 两个视角的数据量、目标数不同（cemian silie 更多、zhengmian fanzhe 更多），同一模型在视角间的差异不能全部归因于模型。
