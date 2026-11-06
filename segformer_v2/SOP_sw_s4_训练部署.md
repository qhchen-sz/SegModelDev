# sw_s4 模型训练部署 SOP（config_sw_s4.yaml）

> **适用模型**：`segformer_v2/runs_strat/sw_s4`（超参扫描 S4 点，当前部署主版本）
>
> **链路**：数据切分 → 切块 → 训练（早停）→ 评估 → 导出 ONNX → 构建 TensorRT 引擎 → C++/C# 部署
>
> **本机基线（已验证）**：RTX 3090 / TensorRT 8.6.1.6 / CUDA 12.1 /
> Python(`F:/HymsonSoftWare/MiniForge/envs/yolov10/python.exe`) / torch 2.5.1+cu121 / transformers 4.57.6
>
> **部署后指标**：TRT fp16 mIoU 0.7594（PyTorch 基线 0.7593），3090 约 155 ms/图
>
> **最后更新**：2026-10-22

---

## 0. 先看结论（TL;DR）

```bat
:: 1. 训练（中断后重跑同一命令即可续训）
python segformer_v2\train_strat_b.py --config config_sw_s4.yaml --tag sw_s4

:: 2. 导出 ONNX（输出原始 logits，Softmax 在部署端做）
python segformer_v2\export_best_onnx.py ^
  --src segformer_v2\runs_strat\sw_s4\best ^
  --out deploy_cpp\models\segformer_sw_s4_b1_640.onnx

:: 3. 构建引擎（30 系；50 系用 build_engine50.bat，必须在 50 机器上执行）
deploy_cpp\tools\build_engine.bat sw_s4

:: 4. 部署（C++ DLL + C# 层，置信度阈值默认 0.5）
```

**三条铁律**：

1. `.engine` 绑定 TensorRT 版本 + GPU，换机器必须在目标机重建；`.onnx` 可跨机器复用。
2. 训练 sw_s4 必须用 `train_strat_b.py --config config_sw_s4.yaml`，不能用 `train_strat.py`（它读死 `config_strat.yaml`）。
3. 引擎只输出原始 logits；**滑窗平均 logits → Softmax → 置信度过滤（默认 0.5，低于判背景）**在部署端完成，ONNX 不用为此重新导出。

---

## 1. 配置定位与关键参数

`config_sw_s4.yaml` 是 S1–S4 超参扫描（`bg_keep_ratio` × `focal_alpha[bg]` 解耦扫描）中的 **S4 点**：
背景保留率 0.12、背景 focal 权重 0.8，**复用基线切块数据集** `dataset_strat`（不重新切块）。

| 参数 | 值 | 说明 |
|---|---|---|
| `dataset_dir` | `segformer_v2/dataset_strat` | 训练切块（tile 640 / stride 512 切好） |
| `num_classes` / `id2label` | 3 / background, silie, fanzhe | 背景+撕裂+翻折 |
| `bg_keep_ratio` | 0.12 | 纯背景 tile 保留 12% |
| `loss_focal / loss_dice / loss_lovasz` | 1.0 / 1.0 / 1.0 | 三损失等权 |
| `focal_gamma` / `focal_alpha` | 2.0 / [0.8, 2.0, 1.0] | S4 的扫描点即 alpha_bg=0.8 |
| `copy_paste_prob` | 0.5 | silie 小目标 copy-paste 增广 |
| `model_ckpt` | `segformer_baseline/models/segformer-b1-ade` | SegFormer-B1，ADE 预训练初始化 |
| `epochs` / `patience` | 5000 / 100 | 纯早停口径 |
| `batch_size` / `lr` / `weight_decay` | 16 / 6e-5 / 0.01 | poly lr |
| `infer_stride` / `tta` | 320 / true | 整图推理滑窗步长 / 验证 TTA |
| `confidence_threshold` | 0.5 | Softmax 置信度过滤阈值（默认 0.5） |

---

## 2. 环境准备（全新机器）

### 2.1 Python 环境

```bat
:: 本机固定用该环境（有 torch）：
F:\HymsonSoftWare\MiniForge\envs\yolov10\python.exe

:: 全新机器复刻：
conda create -n segformer python=3.9 -y
pip install torch==2.5.1 transformers==4.57.6 safetensors pyyaml numpy ^
            onnx==1.14.0 onnxruntime==1.15.1 opencv-python albumentations
```

所有脚本依赖环境变量（脚本内已自带默认值，可不设）：

```bat
set HF_ENDPOINT=https://hf-mirror.com
set NO_ALBUMENTATIONS_UPDATE=1
set PYTHONIOENCODING=utf-8
```

### 2.2 数据前置（只做一次，幂等）

**先决判断**：沿用现有 `dataset_strat/`、`train_strat.txt`/`val_strat.txt` 继续训练时，
本节全部跳过，直接进 §3；只有全新机器或新增/更换训练图片时才需要准备。

#### 原料（放在项目根目录）

| 原料 | 位置 | 要求 |
|---|---|---|
| 训练原图 | `images/*.jpg` | 现场采集图 |
| ISAT 标注 | `images/*.json` | **与图片同名、同目录**；类别名只能是 `silie`/`fanzhe`（大小写不敏感），background 忽略；多边形存于 `objects[].segmentation` |
| YOLO 标签 | `labels/<stem>.txt` | YOLO 格式（类别 + 归一化框），供 `split_stratified.py` 分层和 YOLO 训练用。**仓库内没有 ISAT→YOLO 的转换脚本**，需用 ISAT 自带导出或自行转换 |

#### 三级产物（数据链路）

```
images/*.jpg + images/*.json(ISAT) ──┬──> labels/*.txt(YOLO, 需另行导出)
                                     │
   split_stratified.py  ←────────────┘   按类别组合分层 8:2 (seed=42)
        └─> train_strat.txt / val_strat.txt
   convert_strat.py       读 images/*.json → 语义 mask → 640 滑窗切块 + bg_keep 采样
        └─> dataset_strat/{train, val, val_full}   (val_full = 全图 GT，评估口径)
   preprocess_silie_crops.py  读 images/*.json → 抠 silie 小目标
        └─> silie_crops/   (copy-paste 增广用)
```

对应命令（在项目根目录）：

```bat
:: 1) 分层切分（非破坏性，只写 train_strat.txt / val_strat.txt，不动旧 train.txt/val.txt）
python split_stratified.py

:: 2) 切块。读 segformer_v2\config_strat.yaml（bg_keep_ratio 等以它为准）；
::    输出 segformer_v2\dataset_strat\，不覆盖旧 dataset\
python segformer_v2\convert_strat.py

:: 3) silie 小目标 crops（读 segformer_v2\config.yaml 的 silie_crops_dir）
python segformer_v2\preprocess_silie_crops.py
```

### 2.3 目录布局（硬编码相对关系）

```
<repo-root>\
├── train_strat.txt / val_strat.txt
├── split_stratified.py
├── segformer_v2\
│   ├── config_sw_s4.yaml
│   ├── train_strat_b.py / export_best_onnx.py / sweep_eval.py ...
│   ├── dataset_strat\            # 切块数据 + val_full.txt（canonical 全图 GT）
│   └── runs_strat\sw_s4\         # 训练产物
└── deploy_cpp\
    ├── tools\build_engine.bat    # 30 系建引擎
    ├── models\                   # ONNX / engine 输出
    └── segformer_dll50\          # 50 系工程（含 build_engine50.bat）
```

`deploy_cpp` 与 `segformer_v2` 必须同级；`runs_strat\` 在 `.gitignore` 里，
**换机器要手动拷贝** `runs_strat\sw_s4\best\`。

### 2.4 从零复现检查清单（2026-11-06 本机实测）

按本清单逐项确认后再进 §3，每一步都给出可核对的数量基准：

- [ ] **`images\` / `labels\` 必须在仓库根目录**。`split_stratified.py` 硬编码
      `ROOT\images`、`ROOT\labels`；`train_strat.txt` 里存的是
      `<仓库根>\images\xxx.jpg` 绝对路径，切块、crops 全靠它找图。
      数据放在别处（如 `D:\qhchen\NewModel\newsiliec9\`）时不要改 txt，
      用目录联接挂载（无需管理员）：
      ```bat
      mklink /J images D:\qhchen\NewModel\newsiliec9\images
      mklink /J labels D:\qhchen\NewModel\newsiliec9\labels
      ```
      （Git Bash 里直接调 mklink 会被引号/斜杠坑，写成 .bat 执行最稳）
- [ ] **`model_ckpt` 目录必须有权重文件**。只有 `config.json` 时训练直接报
      `Error no file named ... found in directory`。权重可从 hf-mirror 下载
      `nvidia/segformer-b1-finetuned-ade-512-512` 的 `pytorch_model.bin`；
      注意 **torch < 2.6 环境（transformers ≥4.51）会因 CVE-2025-32434 拒绝加载
      .bin**，需转成 `model.safetensors`（`safetensors` 优先于 .bin，共存无妨）：
      ```python
      import torch; from safetensors.torch import save_file
      sd = torch.load('pytorch_model.bin', map_location='cpu', weights_only=True)
      save_file(sd, 'model.safetensors')
      ```
- [ ] **split 脚本有"空写"隐患**：`images\` 缺失时它照样写空 `train_strat.txt` /
      `val_strat.txt` 然后才崩（ZeroDivisionError）。跑完必须核对输出非空：
      预期 `共发现 446 张图片`、train **357** 行 / val **89** 行。
      若已被写空，用 `git checkout -- train_strat.txt val_strat.txt` 恢复。
- [ ] **切块数量基准**：train 约 **2756** 块 / val 约 **694** 块（含缺陷/保留背景各半）、
      `val_full` 89 张全图 GT mask。`preprocess_silie_crops.py` 预期输出
      88 个 crops（176 个文件）。训练日志载入数（约 2698/685）与切块数略有
      差异是 dataset 过滤所致，属正常。
- [ ] 以上数据链路（切分→切块→crops→1 epoch→导出 ONNX→trtexec→seg_infer 单图）
      已于 2026-11-06 在本机用临时目录全链路冒烟通过；正式训练用
      `train_strat_b.py --config config_sw_s4.yaml --tag sw_s4` 全量跑即可。

---

## 3. 训练

### 3.1 单点训练（sw_s4 本体）

```bat
cd /d D:\qhchen\NewModel\ADevFolder\segformer_v2
python train_strat_b.py --config config_sw_s4.yaml --tag sw_s4
```

- **必须用 `train_strat_b.py --config`**：`train_strat.py` 写死读 `config_strat.yaml`，
  用它训出来的是 v2_strat 而不是 sw_s4。
- `resume` 默认开启：目录里已有 ckpt 就续跑（lr 不跳变）；加 `--no_resume` 从零开始。
- 中断/断电：直接重跑同一命令即可续训。
- 脚本自带 GPU 温度保护（≥85°C 暂停冷却）与 seed=42 固定。
- 产物（`runs_strat\sw_s4\`）：

```
best\                 # 最优权重（部署唯一需要的）
  config.json + model.safetensors
last.pt / ckpt 日志   # 续跑用
final_miou.txt        # 训练期 tile 评估口径（≈0.7651，勿与部署全图 mIoU 混比）
```

### 3.2 一键扫描（可选，S1–S4 全流程）

```bat
python run_sweep.py
```

依次：切块（缺数据集时）→ `train_strat_b.py --config config_sw_sN.yaml --tag sw_sN` →
`sweep_eval.py` 统一口径评估，日志 `runs_strat\sweep.log`，失败即停、可续跑。
日常只重训 sw_s4 时用 §3.1，不必跑整个 sweep。

---

## 4. 评估

```bat
:: 统一口径评估（val 全图 + 现场批），config 必须传 config_sw_s4.yaml
python sweep_eval.py --config config_sw_s4.yaml --run sw_s4

:: 全图滑窗推理评估（TTA 默认开；--no_tta 关闭，与部署口径一致用 --no_tta）
python infer_strat.py --run sw_s4 [--no_tta]

:: 现场批抽查（三批现场图对比用）
python test_s4_field.py --batch <现场图片目录> [--confidence-threshold 0.5]
```

- 置信度阈值统一走配置 `confidence_threshold: 0.5`，命令行 `--confidence-threshold` 可临时覆盖。
- 训练期 `final_miou.txt`（tile 口径）≠ 部署期全图滑窗 mIoU（≈0.759），验收以 §7 为准。

---

## 5. 导出 ONNX

```bat
python export_best_onnx.py ^
  --src segformer_v2\runs_strat\sw_s4\best ^
  --out deploy_cpp\models\segformer_sw_s4_b1_640.onnx
```

- 输入 `pixel_values` (B,3,H,W) float32 / ImageNet 归一化，输出 `logits` (B,3,H/4,W/4)；
  batch/H/W 全部动态轴，opset 默认 17。
- 导出后自动用 onnxruntime 与 torch 做 `max|Δ|` 校验；trtexec 报算子不支持时加 `--opset 16` 重导。
- **不需要 GPU**，CPU 版 torch 即可导出。
- 产物约 55 MB，同一份 ONNX 服务 fp16 / fp32 两种引擎。

---

## 6. 构建 TensorRT 引擎

### 6.1 30 系列（TRT 8.6.1.6，本机已验证）

```bat
cd /d D:\qhchen\NewModel\ADevFolder\deploy_cpp
tools\build_engine.bat sw_s4            :: fp16，已有产物则跳过
tools\build_engine.bat sw_s4 --force    :: 强制重建
tools\build_engine.bat sw_s4 --fp32 --force
```

实际等价于 `trtexec --fp16 --minShapes=1x3x640x640 --optShapes=16x3x640x640
--maxShapes=16x3x640x640 --memPoolSize=workspace:4096`（batch 1~16 动态，
与 `SegConfig::batch=16` 匹配）。构建约 3–5 分钟，产物约 35 MB（fp16）。

### 6.2 50 系列（TRT 10.10.0.31，必须在 RTX50 机器上执行）

```bat
segformer_dll50\build_engine50.bat sw_s4
```

- 固定 **batch=1** 引擎 + DLL 内 4 路多流并发（RTX50/TRT10 上 batch>1 有数值错误，禁用）；
- 50 机器前置与路径修改见 `deploy_cpp\segformer_dll50\README_RTX50.md` §2/§8；
- **3090 上建的引擎不能拿到 50 系用**，反之亦然。

### 6.3 换机器重建

engine 绑定 TRT 版本 + GPU 架构，一律在目标机重建；ONNX 直接拷过去用。
30 系目标机只装 TensorRT 时，可跳过 Python 直接 `trtexec`（命令见
`report\10_2026-10-22_TensorRT引擎导出SOP.md` §4.2 方式 B）。

---

## 7. C++ / C# 部署

### 7.1 推理后处理（已实现，默认生效）

C++ 三套管线（`deploy_cpp\src`、`segformer_dll`、`segformer_dll50`）口径一致：

```
滑窗 tile 推理 → bilinear 上采样到 640 → 重叠区 logits 累加(acc_) + 覆盖计数(cov_)
→ 逐像素 平均 logits (acc_/cov_) → 减最大值稳定 Softmax
→ 最大概率 < 阈值(默认0.5) 判背景0，否则取 argmax 类别
```

### 7.2 置信度阈值设置

| 层 | 接口 | 说明 |
|---|---|---|
| exe | `seg_infer.exe --confidence-threshold 0.5` | 命令行 |
| 原生 DLL | `SegSetConfidenceThreshold(handle, 0.5f)` | C ABI，Load 前后均可调 |
| C# 实例式 | `SegModel.SetConfidenceThreshold(0.5f, ref erro)` | |
| C# 静态槽 | `AI_Seg1.SetConfidenceThreshold(0.5f, ref erro)` | `AI_Seg1`–`AI_Seg48` 全 48 个槽都有 |

现场脚本标准用法（`AI_Seg1` 风格，`SoftwareScriptSample.cs` 模板）：

```csharp
private const float ConfidenceThreshold = 0.5f;

// AI_Seg1.Init 成功之后、首次 Run 之前调用一次：
if (!AI_Seg1.SetConfidenceThreshold(ConfidenceThreshold, ref erro))
{
    LogError(GetLogPre() + erro);
    return false;
}
```

- 阈值合法范围 `[0,1]`；`0` 等价于旧版纯 argmax（已验证逐像素一致）。
- 默认值就是 0.5，不调用也行——显式调用是为了让现场可调、意图可见。
- 若脚本显式 `Close()` 后重新 `Init()`，需重新调用一次（用 `_thresholdSet` 标志控制）。

### 7.3 交付文件清单

| 文件 | 来源 |
|---|---|
| `segformer_Dll_func.dll` | `deploy_cpp\segformer_dll`（30 系）或 `segformer_dll50`（50 系，本目录产物） |
| `EVDll_Segformer.dll` | `EVDll_Segformer(50)\build_csharp.bat` 产物，含阈值接口，必须与原生 DLL 同版本配套 |
| `segformer_sw_s4_b1_fp16_640.engine` | §6 产物 |
| TRT 运行时 / CUDA / cuDNN / OpenCV dll | 30 系 TRT8.6 一套；50 系 TRT10.10 + CUDA12.9 + cuDNN9（清单见 `segformer_dll50\README_RTX50.md` §7） |

**注意**：C# 层新增了接口，`EVDll_Segformer.dll` 必须随本次更新一起替换；
原生 C ABI 仅新增导出，旧调用代码不受影响。引擎路径**不能含中文**。

### 7.4 验收标准

- [ ] `trtexec --loadEngine` 无 `[E]` 报错
- [ ] 单图推理有叠加图 + mask，无报错
- [ ] 全量 mIoU ≈ 0.759（89 张 val_strat 全图，无 TTA；允许 ±0.002 级差异）
- [ ] C# `SegTest` 的 `SegStat` 布局自检 1840 字节 OK，引擎加载成功
- [ ] 阈值 0 时输出与旧版 argmax 逐像素一致；阈值 0.9 时缺陷像素数应下降
- [ ] 现场脚本 Init 后 `SetConfidenceThreshold` 返回 true，斑点分析链接 `GetMask()` 单通道掩膜

---

## 8. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 训出来的是 v2_strat 不是 sw_s4 | 误用 `train_strat.py` | 用 `train_strat_b.py --config config_sw_s4.yaml --tag sw_s4` |
| 续训 lr 跳变/从头开始 | 用了 `--no_resume` 或 ckpt 丢了 | 去掉 `--no_resume`；ckpt 在 `runs_strat\sw_s4\` |
| trtexec 报 ONNX 算子不支持 | opset 17 解析不了 | `export_best_onnx.py --opset 16` 重导 |
| `Serialization Error` / 加载失败 | engine 与目标 `nvinfer.dll` 版本或 GPU 不匹配 | 目标机重建；不要跨 TRT 版本/跨卡混用 |
| mIoU 明显偏低 | 预处理/滑窗口径不一致 | 核对 BGR→RGB、/255、ImageNet mean/std、tile 640 / stride / batch 与引擎 shape 一致 |
| 换机器 `python not found` | base 环境无 torch | `PY` 指到带 torch 的环境（本机 `F:\HymsonSoftWare\MiniForge\envs\yolov10\python.exe`） |
| 50 系全部推成背景 | 用了 batch>1 的引擎 | 必须 `build_engine50.bat` 固定 b1 重建（README_RTX50 §1.1） |

---

## 9. 参考文档

- `report\10_2026-10-22_TensorRT引擎导出SOP.md` —— 30 系换机重建引擎详细版
- `deploy_cpp\segformer_dll50\README_RTX50.md` —— 50 系工程编译/部署
- `deploy_cpp\README.md` —— C++ 管线与 C# API 总说明
- `report\07_2026-10-11_超参扫描Pareto.md` / `report\08_2026-10-12_最终三模型指标.md` —— S4 选点依据与最终指标
