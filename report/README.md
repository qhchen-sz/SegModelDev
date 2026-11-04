# 极耳缺陷分割实验报告总览

> 本文档汇总 `D:\qhchen\NewModel\newsiliec9` 与 `D:\qhchen\NewModel\ADevFolder` 两个工程中的实验与部署报告，按**报告内容所对应的实验日期**排序归档，共 11 份。每份报告说明"做了什么、关键配置是什么、取得了什么结果、结论是什么"，并给出与源码、权重、数据集的对应关系。

---

## 一、归档说明

| 项 | 说明 |
|---|---|
| 报告来源 | `newsiliec9`（实验主目录，2026-09 ~ 2026-10）与 `ADevFolder`（当前工程，含最新 RTX50 适配版） |
| 排序依据 | 报告内容中的实验/生成日期；无显式日期的取文件生成时间，并在下表中标注 |
| 去重规则 | 两工程同名的 7 份报告只保留一份；内容完全一致或仅换行符/路径差异的直接去重；RTX50 报告保留当前工程 2026-11-02 最新版 |
| 文件命名 | `序号_日期_主题.md`，序号即时间顺序，按文件名排序即可还原时间线 |
| 原文处理 | 原样复制，未修改报告正文；原工程文件均保留未动 |

---

## 二、任务与数据基础

### 2.1 任务定义

电池极耳的像素级缺陷分割，共 3 类：

| 类别 ID | 名称 | 含义 |
|---:|---|---|
| 0 | background | 背景 |
| 1 | silie | 撕裂 |
| 2 | fanzhe | 翻折 |

原始数据共 **446 张**，其中缺陷图约 239 张，其余为负样本（全背景，约 207~210 张，不同划分的统计口径略有差异）。silie 实例明显少于 fanzhe（旧划分下 silie 110 个实例、fanzhe 296 个实例）。

### 2.2 两套数据划分（重要）

实验过程中先后使用过两套划分，**绝对指标不可跨套比较**：

| 划分 | 训练/验证 | 负样本比例 | 问题 |
|---|---|---:|---|
| 旧划分 `train.txt` / `val.txt` | 192 / 88 | train 1.6%，val 46.6% | 训练集与验证集负样本比例严重失配，且约 163 张负样本被排除 |
| 分层划分 `train_strat.txt` / `val_strat.txt` | 357 / 89 | 47.1% / 47.2% | 按类别组合分层 8:2，全部负样本纳入，train/val 分布对齐 |

分层划分的验证集分布：

| 集合 | 图数 | silie 实例/图数 | fanzhe 实例/图数 | 负样本图数 | 负样本占比 |
|---|---:|---:|---:|---:|---:|
| train_strat | 357 | 83 / 49 | 241 / 142 | 168 | 47.1% |
| val_strat | 89 | 27 / 12 | 55 / 35 | 42 | 47.2% |

> 结论：**09-25 之后的所有对比（报告 04~11）均基于分层划分从零重训**。09-21~09-23 的旧划分结果只能用于说明当时的选择过程，不能与后期数字直接横向比较。

---

## 三、时间线与演进脉络

整个研究分为六个阶段，每一阶段的结论直接决定下一步做什么：

### 阶段一：基线选型与 SegFormer v2 改进（2026-09-21 ~ 09-23）

- **问题**：现有产线使用 YOLOv8n-seg（实例分割），需要验证语义分割架构是否更适合"极耳撕裂/翻折"这种像素级、形状不规则的缺陷。
- **做法**：先建立 SegFormer-B1 语义分割基线，与 YOLOv8n-seg 在统一像素口径下比较；再加入 Mask2Former 作为第二个实例分割参照；随后对 SegFormer 做 v2 改进（Focal+Dice+Lovasz 损失、640 分辨率、silie Copy-Paste、断点续训）并引入 TTA。
- **结果**：旧划分上 SegFormer v1 mIoU 0.6137，略高于 YOLO 0.6055 和 Mask2Former 0.6176；v2+TTA 提升到 0.6839，silie IoU 0.3357→0.4951。但 SegFormer 推理 654~3836 ms，远慢于 YOLO 的 74 ms。
- **下一步动因**：发现旧划分 train/val 负样本比例严重失配（1.6% vs 46.6%），对比结论可能失真，决定重新分层划分。

### 阶段二：分层重切、五模型从零重训（2026-09-25）

- **问题**：旧划分排除了约 163 张负样本，且训练集几乎没有负样本，模型在真实背景图上可能大量误报。
- **做法**：`split_stratified.py` 按类别组合分层 8:2 重切全部 446 张，五模型（SegFormer v1、v2、v2-fast、YOLOv8n-seg、Mask2Former）在同一新划分上从零重训，统一 val_full GT、统一语义 IoU 口径。
- **结果**：分层后所有模型指标上升，但相对位次保持：v2+TTA 0.7684 > v2-fast 0.7531 > v1 0.7313 > Mask2Former 0.6484 > YOLO 0.5205。YOLO 在分层验证集上 silie IoU 仅 0.1291，暴露出实例分割对细长撕裂类缺陷的明显短板。
- **下一步动因**：精度问题基本解决，但现场反馈 SegFormer 在无缺陷背景图上仍有假阳，需要专项优化。

### 阶段三：背景假阳优化与现场批验证（2026-10-09 ~ 10-10）

- **问题**：背景图误报（假阳）会造成产线误判，需要压制。
- **做法**：提出 B 方案 `v2b`——提高背景块保留比例 `bg_keep` 0.12→0.40、提高背景类 focal 权重 `alpha_bg` 0.20→0.80，重新训练；随后用三批无标注现场图（222 背景批、C8 过杀批、漏杀批）对比 v2-fast、v2b、YOLO 的实际缺陷输出量。
- **结果**：v2b 把 val_strat 42 张负样本的平均假阳从 11393 px 降到 1932 px（-83%），最大假阳 211461→34401 px（-84%），222 批总缺陷像素 -89%；但缺陷 IoU 全线下滑 3~6 个点（mIoU 0.7531→0.7210）。三批现场图上 v2b 在背景/过杀批大幅压假阳（-89% / -56%），漏杀批缺陷像素反而 +5.2%，说明该方案不是免费午餐，而是"用像素贴合度换假阳抑制"的 tradeoff。
- **下一步动因**：需要拆解 B 方案中到底是哪个超参伤害精度，寻找不牺牲 IoU 的替代点。

### 阶段四：超参扫描与 S4 选点（2026-10-11）

- **问题**：B 方案同时改了两个超参，无法判断精度损失来自哪一个。
- **做法**：围绕 `bg_keep × alpha_bg` 设计 5 个扫描点（S1~S4 + v2b/B），在同一 val_strat 上评估缺陷 IoU、42 张负样本假阳、三批现场图输出量，构建 Pareto 表。
- **结果**：解耦归因发现——`bg_keep` 是"用精度换假阳"的杠杆，单独提高它会单调拉低 mIoU；`alpha_bg` 是"近乎免费降假阳"的杠杆。**S4（bg_keep 0.12、alpha_bg 0.80）** 在不改背景块比例的前提下，把假阳均值再降 82%（11394→2003）、最大假阳降 72%（211461→59578），同时 mIoU 0.7531→0.7593、silie 0.5611→0.5833，全面优于基线。
- **下一步动因**：S4 成为最终候选配置，需要固化最终指标并进入部署。

### 阶段五：最终指标固化（2026-10-12）

- **做法**：在 val_strat 89 张上统一评估 S4、v2-fast、YOLO 的 per-class IoU / Recall / Precision 和推理速度。
- **结果**：S4 mIoU 0.7593、mRecall 0.8449、mPrecision 0.8610；v2-fast mIoU 0.7531；YOLO mIoU 0.5205。S4 以召回略降为代价换取精确率和 IoU 提升，综合最优。
- **下一步动因**：模型指标已定，进入 C++/TensorRT 部署。

### 阶段六：C++ / TensorRT 部署与 RTX50 适配（2026-10-22 ~ 11-02）

- **问题**：产线需要独立 C++ 程序/DLL，推理速度要从 PyTorch 的 ~864 ms 降下来；同时产线机器存在 RTX 30 系与 RTX 50 系两类显卡。
- **做法**：导出 ONNX，用 TensorRT 8.6.1.6 构建 FP16/FP32 引擎，封装 C++ DLL 与 C# 调用层；编写跨机器 engine 导出 SOP；再适配 RTX 50（TensorRT 10.10 + CUDA 12.9 + cuDNN 9）。
- **结果**：sw_s4 TRT FP16 在 RTX 3090 上 mIoU 0.7594、155.2 ms/图（PyTorch 864.2 ms，约 5.6x 加速），负样本假阳指标完整保留；C#→C++ DLL 路径 mIoU 0.7593、205 ms（含封送）。RTX50 适配中发现 **batch>1 数值错误**：batch=1 时缺陷类输出 31.9% 前景，batch=2/4/8/16 全部被判成背景；最终改为固定 batch=1 引擎 + 4 路独立 context/stream 并发，保证逐像素结果与 Python/RTX3090 一致。

---

## 四、关键指标总表

### 4.1 旧划分（val.txt，88 张全图）

| 模型 | mIoU | silie IoU | fanzhe IoU | 推理 ms/图 | 备注 |
|---|---:|---:|---:|---:|---|
| SegFormer v1（MiT-B1, CE） | 0.6137 | 0.3357 | 0.5228 | 653.6 | 滑窗 512，stride 448 |
| YOLOv8n-seg | 0.6055 | 0.2726 | 0.5552 | 73.8 | imgsz=1200 单次 |
| Mask2Former（Swin-Tiny） | 0.6176 | 0.3994 | 0.4710 | 3287.6 | 滑窗实例分割 |
| SegFormer v2 + TTA | 0.6839 | 0.4951 | 0.5710 | 3836 | 640 分辨率，stride 320 |
| SegFormer v2-fast（无 TTA） | 0.6184 | 0.3128 | 0.5575 | 804 | bs16 + FP16 + stride448 |

> YOLO 原生 mask mAP@0.5 = 0.41317，mask mAP@0.5:0.95 = 0.16706，最佳 epoch 235/478。该表仅用于阶段一内部对比。

### 4.2 分层划分（val_strat.txt，89 张全图）

| 模型 | mIoU | silie IoU | fanzhe IoU | mRecall | mPrecision | 推理 ms/图 |
|---|---:|---:|---:|---:|---:|---:|
| SegFormer v1 | 0.7313 | 0.5300 | 0.6750 | — | — | 690 |
| SegFormer v2 + TTA | 0.7684 | 0.5983 | 0.7162 | — | — | 4005 |
| SegFormer v2-fast（v2 基线） | 0.7531 | 0.5611 | 0.7081 | 0.8756 | 0.8229 | 842~871 |
| YOLOv8n-seg | 0.5205 | 0.1291 | 0.4535 | 0.5944 | 0.7428 | 16~19 |
| Mask2Former | 0.6484 | 0.3846 | 0.5738 | — | — | 1906 |
| SegFormer v2b（B 方案） | 0.7210 | 0.5013 | 0.6724 | — | — | ~773（现场批实测） |
| **SegFormer S4（最终选点）** | **0.7593** | **0.5833** | **0.7039** | **0.8449** | **0.8610** | **872** |

### 4.3 背景假阳（val_strat 中 42 张 GT 负样本，mask 全 0）

| 配置 | 假阳均值 px/图 | 最大假阳 px | >100px 图 | 全干净图 |
|---|---:|---:|---:|---:|
| v2 基线（bg_keep 0.12 / alpha_bg 0.20） | 11393.5 | 211461 | 8/42 | 34/42 |
| v2b / B 方案（0.40 / 0.80） | 1932.5 | 34401 | 11/42 | 31/42 |
| S4（0.12 / 0.80） | **2003.3** | **59578** | **7/42** | **33/42** |
| S4 + TensorRT FP16 部署 | 2016.4 | 59671 | 7/42 | 33/42 |

> S4 相对基线：假阳均值 **-82%**、最大假阳 **-72%**，而 mIoU 不降反升（+0.006），是"近乎免费降假阳"的配置；v2b 虽然假阳最低，但牺牲了 3~6 个点的缺陷 IoU。

### 4.4 三批现场图（无 GT，只统计输出量）

| 批次 | 图数 | 含义 | v2-fast 报缺陷图 | v2b 报缺陷图 | YOLO 报缺陷图 |
|---|---:|---|---:|---:|---:|
| 222 | 28 | 背景批（应不报） | 8/28 | **5/28** | 3/28 |
| C8 过杀批 | 30 | YOLO 误报案例（应不报） | 3/30 | 4/30 | 4/30 |
| 漏杀批 | 38 | YOLO 漏检案例（应报出） | 37/38 | 37/38 | 30/38 |

| 批次 | v2-fast 缺陷像素合计 | v2b 缺陷像素合计 | YOLO 缺陷像素合计 |
|---|---:|---:|---:|
| 222 | 124413 | **13709（-89.0%）** | 99103 |
| C8 | 42507 | **18631（-56.2%）** | 3194748 |
| 漏杀 | 22826510 | **24020535（+5.2%）** | 22312367 |

> 关键证据：C8 批 YOLO 在 2 张图上分别输出 180 万、69 万缺陷像素（整块矩形涂抹），v2 系基本干净；漏杀批 YOLO 完全漏掉 7 张真实翻折，v2/v2b 全部检出。速度上 YOLO 约 17 ms，SegFormer 约 800 ms，相差 43~48 倍。

### 4.5 部署实测（RTX 3090，89 张 val_strat）

| 路径 | mIoU | silie IoU | 假阳均值 px | 最大假阳 px | 耗时 |
|---|---:|---:|---:|---:|---:|
| sw_s4 PyTorch | 0.7593 | 0.5833 | 2003.3 | 59578 | 864.2 ms |
| **sw_s4 TensorRT FP16** | **0.7594** | **0.5824** | **2016.4** | **59671** | **155.2 ms** |
| sw_s4 TensorRT FP32 | 0.7593 | 0.5831 | — | — | 297.0 ms |
| sw_s4 C#→C++ DLL | 0.7593 | 0.5820 | 1988.0 | 59740 | 205 ms（含封送） |
| v2_strat PyTorch | 0.7531 | 0.5611 | 11394 | 211461 | 811.1 ms |
| v2_strat TensorRT FP16 | 0.7531 | 0.5611 | — | — | 156.3 ms |

---

## 五、逐份报告详解

### 01 · 2026-09-21 · [SegFormer 与 YOLOv8n-seg 对比](01_2026-09-21_SegFormer_vs_YOLOv8n-seg.md)

**做了什么**：在旧划分 88 张 val 全图上，把 SegFormer-B1（语义分割）与产线现用的 YOLOv8n-seg（实例分割）都转成语义图，用同一套 per-class IoU/mIoU 口径比较，并记录推理速度。

**关键配置**：SegFormer MiT-B1，ADE20K 预训练后改 3 类头；滑窗 512×512、stride 448；加权 CE（class_weights=[0.2, 2.0, 1.0]）。YOLO imgsz=1200 单次推理。

**结果**：SegFormer mIoU 0.6137，YOLO 0.6055；silie IoU SegFormer 0.3357 明显优于 YOLO 0.2726，fanzhe 则 YOLO 更好（0.5552 vs 0.5228）。速度 SegFormer 653.6 ms，YOLO 73.8 ms。

**结论**：语义分割架构在细长撕裂类缺陷上更有潜力，但速度差距约 9 倍；该结论建立在失配的旧划分上，后续被报告 04 修正。

---

### 02 · 2026-09-22 · [三模型对比（加入 Mask2Former）](02_2026-09-22_三模型对比.md)

**做了什么**：在报告 01 基础上加入 Mask2Former-Swin-Tiny（实例分割架构），三模型统一像素口径对比。

**结果**：Mask2Former mIoU 0.6176，略高于 SegFormer 0.6137 和 YOLO 0.6055；其 silie IoU 0.3994 为三者最高，但 fanzhe IoU 0.4710 最低，推理 3287.6 ms 也是最慢。

**结论**：Mask2Former 精度没有形成代差，速度却慢一个数量级，不作为主方案；SegFormer 继续作为优化对象。

---

### 03 · 2026-09-23 · [SegFormer v2 改进对比](03_2026-09-23_SegFormer_v2改进对比.md)

**做了什么**：对 SegFormer 做 v2 改进并与 v1、v2-fast、YOLO、Mask2Former 对比。v2 改进包括：损失由加权 CE 改为 **Focal + Dice + Lovasz**、训练分辨率 512→640、加入 **silie Copy-Paste** 增强、支持断点续训。

**结果**：v2 + TTA mIoU 0.6839（v1 0.6137），silie IoU 0.3357→0.4951，fanzhe 0.5228→0.5710，但推理 3836 ms。去掉 TTA 并改用 bs16+FP16+stride448 的 v2-fast 为 0.6184 / 804 ms，约 4.8x 提速，代价是 silie IoU 回落到 0.3128。

**结论**：v2 的精度提升主要来自改进配方，TTA 对 silie 贡献最大；v2-fast 用速度换精度，仍略优于 v1。此时仍未解决数据划分失配问题。

---

### 04 · 2026-09-25 · [分层重切五模型对比](04_2026-09-25_分层重切五模型对比.md)

**做了什么**：指出旧划分 train 负样本仅 1.6%、val 高达 46.6%，且排除约 163 张负样本；用 `split_stratified.py` 按类别组合分层 8:2 重切全部 446 张（357/89），五模型在同一新划分上**从零重训**（patience=100，每 epoch 评估存 best）。

**结果**：v2+TTA mIoU 0.7684、v2-fast 0.7531、v1 0.7313、Mask2Former 0.6484、YOLO 0.5205。YOLO 的 silie IoU 暴跌到 0.1291；SegFormer 全系在缺陷类上明显领先。

**结论**：分层划分拉平了 train/val 分布，结论更可信；SegFormer v2-fast 在精度与速度间取得可用平衡，成为后续主基线。报告明确提醒：与旧报告不可比绝对值，只能看相对位次。

---

### 05 · 2026-10-09 · [B 方案：背景假阳优化](05_2026-10-09_B方案背景假阳优化.md)

**做了什么**：针对背景图误报，提出 B 方案 `v2b`：`bg_keep` 0.12→0.40（背景块保留更多）、`focal_alpha(bg)` 0.20→0.80（背景类损失权重更高）；训练块数 2756→5981，早停 ep287(best@187)→ep202(best@102)。在 val_strat 89 张和三批现场图上对比 v2 基线。

**结果**：负样本平均假阳 11393→1932 px（-83%），最大假阳 211461→34401 px（-84%），222 批总缺陷像素 124413→13709（-89%）；但 mIoU 0.7531→0.7210（-0.032），silie 0.5611→0.5013（-0.060），fanzhe 0.7081→0.6724（-0.036），训练期指标同步下滑，不是评估假象。另有"少量假阳的图"从 8 张增至 11 张，说明大块假阳被压小、零散小点增多。

**结论**：B 方案确实压制了背景大块假阳，但属于**非免费午餐**的 tradeoff；是否采用取决于部署侧更在意误报还是召回。报告提出下一步应做更温和的 B 方案或后处理门控。

---

### 06 · 2026-10-10 · [三批现场图对比](06_2026-10-10_三批现场图对比.md)

**做了什么**：用三批**无标注**现场图验证实际表现：222 背景批 28 张（应不报）、C8 过杀批 30 张（YOLO 误报案例，应不报）、漏杀批 38 张（YOLO 漏检案例，应报出）。对比 v2-fast、v2b、YOLOv8n-seg 的缺陷输出量、两两一致性、速度和叠加图。

**结果**：v2b 相对 v2-fast 在 222 批缺陷像素 -89.0%、C8 批 -56.2%，漏杀批 +5.2%（没有掉召回）；YOLO 在 C8 批 4 张图上输出 319 万缺陷像素，漏杀批漏掉 7 张真实翻折（其中 5 张缺陷区域 >27 万像素）。速度 YOLO 约 17 ms，SegFormer 约 800 ms，相差 43~48 倍。

**结论**：现场三批上 v2b 是净收益——背景/过杀图大幅压假阳、漏杀图不丢检出；但结合报告 05 的有 GT 数据，v2b 仍以 3~6 点 IoU 为代价。报告还记录了 222 目录图片数从 29 变 28、叠加图重算对齐等数据一致性问题。

---

### 07 · 2026-10-11 · [超参扫描 Pareto 报告](07_2026-10-11_超参扫描Pareto.md)

**做了什么**：在 `bg_keep × focal_alpha(bg)` 两个维度上设计 5 个扫描点，连同 v2 基线共 6 个配置，在 val_strat 有 GT 数据上评估缺陷 IoU / 42 张负样本假阳，在三批现场图上评估缺陷输出量，构建 Pareto 表。

**结果**：解耦归因——只动 `bg_keep`（S1 0.25、S2 0.40，alpha 固定 0.20）时 mIoU 0.7531→0.7479→0.7320 单调下滑，假阳同步下降，说明 `bg_keep` 是"用精度换假阳"的杠杆；只动 `alpha_bg`（S3 0.40、S4 0.80，bg_keep 固定 0.12）时，**S4 mIoU 0.7593 反而高于基线，假阳均值仅 2003（-82%）**，说明 `alpha_bg` 是近乎免费的杠杆。S4 还消除了 C8 图 `33350` 的 4479 px 假阳。

**结论**：推荐 **S4 = (bg_keep 0.12, alpha_bg 0.80)**，全面优于基线；并指出 S3/S4 非单调，训练存在 ±0.01 量级随机方差，但 S4 的假阳 -82% 远超方差，结论稳健。

---

### 08 · 2026-10-12 · [最终三模型指标](08_2026-10-12_最终三模型指标.md)

**做了什么**：在 val_strat 89 张上，按统一推理口径（S4/v2-fast 均为 bs16 + FP16 + stride448 + 无 TTA；YOLO imgsz1200/conf0.25/iou0.5）输出 per-class IoU、Recall、Precision 和平均耗时。

**结果**：S4 mIoU 0.7593、mRecall 0.8449、mPrecision 0.8610；v2-fast mIoU 0.7531、mRecall 0.8756、mPrecision 0.8229；YOLO mIoU 0.5205、mRecall 0.5944、mPrecision 0.7428。S4 相比 v2-fast 召回略降（0.8756→0.8449），但精确率（0.8229→0.8610）和 mIoU 提升，silie IoU 0.5611→0.5833。速度 S4 871.9 ms、v2-fast 871.1 ms、YOLO 19.2 ms。

**结论**：S4 是精度与假阳抑制的综合最优配置，作为最终交付模型。

---

### 09 · 2026-10-22 · [C++ / TensorRT 部署报告](09_2026-10-22_C++_TensorRT部署报告.md)

**做了什么**：把 `runs_strat/<model>/best` 的 PyTorch 权重导出 ONNX，用 TensorRT 8.6.1.6 构建 FP16/FP32 引擎，封装独立 C++ 可执行程序 `seg_infer.exe` 和 `segformer_Dll_func.dll`，并提供 C# 调用层 `EVDll_Segformer`（接口与现有 `EVDll.AI_Gpu1` 同构）；整图滑窗输入，输出语义图、掩膜和叠加图。

**结果**：sw_s4 PyTorch mIoU 0.7593 / 864.2 ms → TRT FP16 mIoU 0.7594 / **155.2 ms**（约 5.6x 加速），FP32 297.0 ms；v2_strat TRT FP16 0.7531 / 156.3 ms。部署后负样本假阳均值 2016.4、最大 59671，与 PyTorch 的 2003.3 / 59578 差 +0.65%，`>100px` 图和全干净图数完全一致，**-82% 的假阳削减被完整保留**。C#→C++ DLL 路径 mIoU 0.7593、假阳均值 1988.0、205 ms（含封送）；纯推理 exe 路径 150.7 ms，其中 GPU 约 97 ms。报告还发现 sw_s4 对 FP16 比 v2_strat 敏感约一个数量级（单 tile logits max|Δ| 2.312 vs 0.289），但差异集中在类别边界附近，最终 mIoU 只差 0.0001，决策指标（假阳）无实质影响，默认仍用 FP16。

**结论**：部署路径精度零损失、速度提升约 5.6 倍；C# 包装逐像素零偏差（无损 PNG 输入下 mask 完全一致）。报告同时记录了 `bool` 封送、GDI+ 调色板 PNG、斑点分析需用 `GetMask()` 单通道掩膜等踩坑点。

---

### 10 · 2026-10-22 · [TensorRT Engine 导出 SOP](10_2026-10-22_TensorRT引擎导出SOP.md)

**做了什么**：为"换到另一台 30 系显卡机器"编写完整的 engine 导出标准作业流程，覆盖版本确认、环境准备、ONNX 导出、trtexec 构建、验证、部署和常见问题。

**关键结论**：`.engine` 绑定 TensorRT 版本、GPU 架构和 shape profile，**不能跨机器拷贝**；ONNX 是标准格式，可跨机器复用，新机器只需 TensorRT 即可构建 engine。本机基线为 RTX 3090 / TensorRT 8.6.1.6 / CUDA 12.1 / Python 3.9.23 / torch 2.5.1 / transformers 4.57.6 / onnx 1.14.0。目标软件的 `nvinfer.dll` 与 TensorRT 8.6.1.6 的 `nvinfer.dll` 均为 211,297,832 字节，可直接复用。验收标准：`trtexec --loadEngine` 无错误、单图推理正常、全量 mIoU ≈ 0.759、`SegStat` 布局 1840 字节、软件内 C# Init/Run 正常。特别提醒 `runs_strat\sw_s4\final_miou.txt` 中的 0.7651 是训练期 tile 口径，不能与部署后的 0.7594 直接比较。

---

### 11 · 2026-11-02 · [RTX 50 系列适配报告](11_2026-11-02_RTX50适配报告.md)

**做了什么**：把 TensorRT 8.6 版 C++ 工程移植到 RTX 50 平台（TensorRT 10.10.0.31 + CUDA 12.9 + cuDNN 9），导出接口、`SegStat` 布局、DLL 文件名保持不变，C# 层无需改动。初版（2026-10-23）完成 TRT10 编译适配后，11-02 版新增了 batch>1 数值错误的排查与多流并发方案。

**关键发现**：RTX50 + TRT10 上该 SegFormer 模型存在 **batch>1 数值错误**——同一输入、同一引擎，batch=1 时缺陷类输出 31.9% 前景，batch=2/4/8/16（动态 profile 或固定 batch）全部被判成背景，logits 系统性偏移（缺陷类均值下降约 5、背景类上升约 1.3）；FP32/FP16、opset16、固定/动态 batch 均复现，而 RTX3090 + TRT8.6 的 batch=16 完全正常。

**解决方案**：引擎改为**固定 batch=1** 构建；DLL 用 `numWorkers` 路（默认 4）独立 execution context + CUDA stream 并行处理滑窗 tile，主线程仍按 tile 原顺序串行累加，保证与 Python / RTX3090 结果逐像素一致；并发路数可用 `SEGFORMER_WORKERS`（1~16）覆盖，显存不足自动降级（4→2→1），workspace 默认 1024 MiB。报告给出了 RTX50 上的编译、建引擎、自测、部署和路径修改步骤，以及 `SEGFORMER_WORKERS=1/2/4/6` 的速度对比方法（报告本身未给出 RTX50 实测耗时数字）。

**结论**：RTX50 适配的难点不是接口，而是 TRT10 上的 batch 数值问题；固定 batch=1 + 多流并发在保持结果一致的前提下恢复了吞吐，是当前 RTX50 部署方案。

---

## 六、源文件对应关系

| 归档报告 | 原始路径 | 生成/相关脚本 | 依赖数据与权重 |
|---|---|---|---|
| 01 | `newsiliec9/segformer_baseline/runs/compare_report.md` | `segformer_baseline/compare.py`、`infer.py`、`eval_yolo.py` | `train.txt`/`val.txt`、`segformer_baseline/runs/full/best`、根目录 `runs/segment/train*/weights/best.pt` |
| 02 | `newsiliec9/mask2former_baseline/runs/compare_report.md` | `mask2former_baseline/compare.py`、`eval.py` | 同旧划分；`mask2former_baseline/runs/full/best`；预训练 `mask2former_baseline/models/m2f-swin-tiny-coco-instance` |
| 03 | `newsiliec9/segformer_v2/runs/compare_report.md` | `segformer_v2/compare.py`、`infer.py`、`infer_fast.py` | 旧划分；`segformer_v2/runs/v2/best`、`runs/v2_fast/`、`runs/v2_notTA/` |
| 04 | `newsiliec9/segformer_v2/runs_strat/compare_report_strat.md` | `segformer_v2/run_strat_pipeline.py` → `compare_strat.py`；数据侧 `split_stratified.py`、`convert_strat.py`、`train_strat.py`、`segformer_baseline/eval_yolo_strat.py`、`mask2former_baseline/eval_strat.py` | `train_strat.txt`/`val_strat.txt`、`dataset_strat/`、`segformer_v2/runs_strat/v2_strat`与`v2_fast_strat`、`segformer_baseline/runs_strat/full_strat`、`mask2former_baseline/runs_strat/full_strat`、根目录 `runs_strat/train_strat/weights/best.pt` |
| 05 | `newsiliec9/segformer_v2/runs_strat/b_compare_report.md` | `convert_strat_b.py`、`train_strat_b.py`、`eval_strat_b.py` | `config_strat_b.yaml`、`runs_strat/v2b_strat/best`、`dataset_strat_b/` |
| 06 | `newsiliec9/segformer_v2/runs_strat/batch_3models_summary.md` | `segformer_v2/batch_compare_3models.py` | `runs_strat/v2_strat/best`、`runs_strat/v2b_strat/best`、`runs_strat/train_strat/weights/best.pt`、222/C8/漏杀三批现场图 |
| 07 | `newsiliec9/segformer_v2/runs_strat/sweep_report.md` | `run_sweep.py`、`sweep_make_configs.py`、`sweep_eval.py`、`sweep_report.py` | `config_sw_s1~s4.yaml`、`dataset_sw_s1~s4/`、`runs_strat/sw_s1~s4/best` |
| 08 | `newsiliec9/final_report.md` | `segformer_v2/eval_final_report.py` | `val_strat.txt`、`runs_strat/sw_s4/best`、`runs_strat/v2_strat/best`、YOLO `runs_strat/train_strat/weights/best.pt` |
| 09 | `newsiliec9/deploy_cpp/README.md` | `segformer_v2/export_best_onnx.py`、`deploy_cpp/tools/build_engine.bat`、`build.bat`、`run.bat`、`EVDll_Segformer/build_csharp.bat` | `deploy_cpp/models/segformer_sw_s4_b1_640.onnx` / `*.engine`、`dataset_strat/val_full.txt` |
| 10 | `newsiliec9/deploy_cpp/SOP_ENGINE_EXPORT.md` | `deploy_cpp/tools/build_engine.bat`、`trtexec`、`run.bat`、`SegTest.bat` | 同报告 09；目标机 TensorRT 版本须与软件 `dll/nvinfer.dll` 一致 |
| 11 | `ADevFolder/deploy_cpp/segformer_dll50/README_RTX50.md`（最新版） | `deploy_cpp/segformer_dll50/segformer_dll.sln`、`build_engine50.bat`、`run_seg_test.bat`、`trt_engine.cpp`、`seg_pipeline.cpp` | `deploy_cpp/models/*.onnx`、`deploy_cpp/models50/*.engine`（在 RTX50 机器上构建） |

---

## 七、口径与使用注意

1. **两套划分不可横向比较**：报告 01~03 使用旧划分（train 负样本 1.6%、val 46.6%，排除约 163 张负样本）；报告 04~11 使用分层划分（train/val 负样本 47.1%/47.2%）。同一模型在两套划分上的 mIoU 差异主要来自数据分布，不是模型变化。
2. **mIoU 与 mask mAP 不可直接比**：SegFormer/Mask2Former 与 YOLO 的像素级指标是"把预测转成语义图后算 per-class IoU"，而 YOLO 原生 mask mAP 是实例级指标，两者不能混用。
3. **速度对比需看设置**：YOLO 为 imgsz=1200 单次推理；SegFormer 为全图滑窗（不同 stride/TTA/batch/精度差异很大）。报告中标注"仅供参考"的速度数字，不要直接作为最终选型依据。跨报告的速度数字（例如 YOLO 73.8 ms 与 16~19 ms）测量批次不同，也不完全可比。
4. **v2 / v2-fast / v2_strat / sw_s4 的关系**：v2-fast 与 v2 是**同一份权重**，仅推理策略不同（bs16 + FP16 + stride448 + 无 TTA）；`v2_strat` 即 v2 基线权重；`sw_s4` 是超参扫描 S4 点训练出的权重，架构同为 SegFormer-B1。
5. **训练期 tile mIoU ≠ 全图 mIoU**：`final_miou.txt` 中的 0.7651 是训练期切块评估口径，部署/全图评估的对应值是 0.7593~0.7594，不能直接对比。
6. **日期依据**：报告 01~05、07、08 原文未标注生成日期，归档日期取自文件生成时间；报告 06 标注 2026-10-10；报告 09、10 标注最后更新 2026-10-22；报告 11 初版 2026-10-23，归档采用当前工程 2026-11-02 更新版。
7. **RTX50 两版差异**：`newsiliec9` 中的 10-23 初版只完成 TRT10 编译适配；当前工程 11-02 版新增 batch>1 数值错误的排查结论、固定 batch=1 引擎和 4 路多流并发方案，归档取最新版，旧版未重复收录。

---

## 八、结论与后续建议

### 8.1 当前结论

- **模型**：SegFormer-B1 + S4 配置（`bg_keep=0.12`、`focal_alpha(bg)=0.80`、Focal+Dice+Lovasz、640 分辨率、silie Copy-Paste）为综合最优。val_strat 89 张上 mIoU 0.7593、silie IoU 0.5833、fanzhe IoU 0.7039，负样本假阳较 v2 基线下降 82%。
- **部署**：TensorRT 8.6.1.6 FP16 在 RTX 3090 上 mIoU 0.7594、155.2 ms/图，假阳削减完整保留；C#→C++ DLL 路径精度一致，205 ms（含封送）。
- **RTX50**：TensorRT 10.10 下必须用固定 batch=1 引擎 + 多流并发，避免 batch>1 的数值错误。
- **与 YOLO 对比**：YOLO 速度约快 43~48 倍，但在分层验证集上 silie IoU 仅 0.1291，现场漏杀批漏检 7 张真实翻折，过杀批出现大面积整块误报；SegFormer 在缺陷召回和假阳控制上明显更优。

### 8.2 已提出但未实施的后续方向

- **后处理门控**（报告 05 提出）：置信度门槛 + 最小连通域过滤，直接压假阳且不伤模型 IoU；报告的实测支持"重训单独解决不了假阳而不伤精度"的判断。
- **更温和的 B 方案**（报告 05 提出）：`bg_keep=0.25` + `alpha_bg=0.40` 的折中配置，寻找更好的 FP/IoU 平衡点，预计需再训练 5~8 小时。
- **v2 + 后处理组合**：保留 v2 的精度，用后处理压假阳。
- **DLL 零拷贝接口**（报告 09 提出）：当前 DLL 路径比 exe 路径多约 55 ms 的全图内存拷贝，若要进一步提速可改造接口。
- **RTX50 速度实测**：报告 11 给出了 `SEGFORMER_WORKERS=1/2/4/6` 的对比方法，但尚未记录具体耗时结果，建议在目标机补齐。

### 8.3 相关但未归档的内容

- 当前工程 `README.md`（2026-10-26）：工程总说明，包含 sw_s4 + TensorRT FP16 的最终结论表。
- `segformer_baseline/README.md`：SegFormer-B1 基线方案设计说明，非实验对比报告。
- `deploy_cpp/papers/paper_source.md`：SegFormer 论文出处核查笔记。
- `fix50_batch16_bug.txt`、`segformer_readme.txt`：非 md 的过程记录；git 提交 `abce3cd`（修复 50 系 batch16 推理不一致）、`711e89e`（EV 时间波动与速度优化）、`4fc52cf`（迁移 50 系优化到 30 系）记录了 2026-10-31 ~ 11-03 的后续工作，但尚无独立实验报告。
