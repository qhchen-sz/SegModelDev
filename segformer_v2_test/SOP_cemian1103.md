# `cemian1103` SegFormer-B1（sw_s4 配方）SOP

> 数据集：`D:\qhchen\NewModel\hefeic1\cemian1103`
> 模型：SegFormer-B1，3 类 `0 background / 1 silie / 2 fanzhe`
> 训练配方：与 `newsiliec9/segformer_v2/runs_strat/sw_s4` 一致
> 生成时间：2026-10-23 08:44
>
> 本次范围：数据划分、训练、测试、导出与部署验证。
> **未修改生产软件目录 `E:\Release - 14.0.5.1-exe\dll`。**

---

## 1. 数据清单

| 项目 | 数值 |
|---|---:|
| JPG | 183 |
| JSON | 183 |
| 缺失 JSON | 0 |
| label 文件 | 183 |
| 空 label（纯背景） | 68 |
| label 行数（目标数） | 177 |
| silie 对象 / 图 | 68 / 42 |
| fanzhe 对象 / 图 | 109 / 74 |

类别来自 `images/isat.yaml`；JSON 里的 `fanzhe ` 带尾随空格，标签生成和切块脚本都用
`.strip().lower()` 归一化。原始 `images/` 下的 JPG / JSON / isat.yaml 未修改。

---

## 2. 数据划分（与现有 `split_stratified.py` 一致）

- 脚本：`D:\qhchen\NewModel\hefeic1\cemian1103\split_stratified.py`（从 `newsiliec9/split_stratified.py` 逐字复制）
- 规则：按每张图的缺陷类别组合分组（neg / silie-only / fanzhe-only / both），
  组内 8:2，`VAL_RATIO=0.2`，`SEED=42`；组内样本 <5 时该组全部进 train。
- 输出：`D:\qhchen\NewModel\hefeic1\cemian1103\train_strat.txt`、`D:\qhchen\NewModel\hefeic1\cemian1103\val_strat.txt`（绝对路径，一行一张原图）。
- 划分结果：

| 组 | 总数 | train | val |
|---|---:|---:|---:|
| neg（纯背景） | 68 | 54 | 14 |
| silie-only | 41 | 33 | 8 |
| fanzhe-only | 73 | 58 | 15 |
| both | 1 | 1 | 0 |
| **合计** | **183** | **146** | **37** |

| split | 图数 | silie 实例/图 | fanzhe 实例/图 | 负样本图 |
|---|---:|---:|---:|---:|
| train | 146 | 56 / 34 | 83 / 59 | 54 |
| val | 37 | 12 / 8 | 26 / 15 | 14 |

命令：

```bat
python D:\qhchen\NewModel\hefeic1\cemian1103\scripts\isat_to_yolo_labels.py --images D:\qhchen\NewModel\hefeic1\cemian1103\images --labels D:\qhchen\NewModel\hefeic1\cemian1103\labels --report D:\qhchen\NewModel\hefeic1\cemian1103\logs\labels_report.json
python D:\qhchen\NewModel\hefeic1\cemian1103\split_stratified.py
```

---

## 3. 标签、切块与掩膜

- 切块：`D:\qhchen\NewModel\hefeic1\cemian1103\scripts\pipeline\convert_strat_b.py`，
  `tile_size=640`、`stride=512`、`bg_keep_ratio=0.12`、`seed=42`。
- 掩膜：单通道 `uint8`，`0=背景 1=silie 2=fanzhe`；每个 val 原图另有全分辨率
  `dataset_strat/val_full/<stem>.png`。
- 实际落盘（`logs/validate_report.json`）：

| split | 图像 tile | mask tile | 含缺陷 tile | 背景 tile |
|---|---:|---:|---:|---:|
| train | 934 | 934 | 622 | 312 |
| val | 232 | 232 | 149 | 83 |

> 切块脚本在图像右/下边缘会把最后两个 stride 位置 clamp 到同一 `(y0,x0)`，
> 同名文件会覆盖；因此 `convert.log` 的“总块数”会比落盘 unique 文件数多。
> 上表是实际 unique 文件数，训练实际使用这些 tile。重复 tile 内容相同，
> 不损失覆盖范围。

- val 全图：`val_full.txt` 37 行，37 个 PNG；
  尺寸等于原图，mask 值 ⊆ {0,1,2}。
- silie copy-paste：从本数据集 train split 抠出 `56`
  对 `_img.jpg/_mask.png`，供 `copy_paste_prob=0.5` 使用。

命令：

```bat
python D:\qhchen\NewModel\hefeic1\cemian1103\scripts\preprocess_silie_crops.py --config D:\qhchen\NewModel\hefeic1\cemian1103\config_sw_s4.yaml
python D:\qhchen\NewModel\hefeic1\cemian1103\scripts\pipeline\convert_strat_b.py --config D:\qhchen\NewModel\hefeic1\cemian1103\config_sw_s4.yaml
python D:\qhchen\NewModel\hefeic1\cemian1103\scripts\validate_dataset.py --config D:\qhchen\NewModel\hefeic1\cemian1103\config_sw_s4.yaml --stage all
```

---

## 4. 训练（与 sw_s4 完全一致）

| 参数 | 值 |
|---|---|
| 初始化 | `D:/qhchen/NewModel/newsiliec9/segformer_baseline/models/segformer-b1-ade`，3 类头，`ignore_mismatched_sizes=True` |
| epochs / patience | 5000 / 100 |
| batch size / workers | 16 / 4 |
| optimizer / lr / wd | AdamW / 6e-05 / 0.01 |
| LR schedule | polynomial `(1-step/total)^0.9` |
| AMP / grad clip | fp16 autocast + GradScaler / 1.0 |
| loss | Focal 1.0 + Dice 1.0 + Lovasz 1.0 |
| focal gamma / alpha | 2.0 / [0.8, 2.0, 1.0] |
| tile / stride / bg_keep | 640 / 512 / 0.12 |
| copy-paste | `prob=0.5`，素材为本数据集 `silie_crops` |
| 增强 | HFlip .5 / VFlip .3 / Rotate90 .3 / RandomScale 1.0–1.3 / RandomCrop 640 / ColorJitter / GaussianBlur |
| resume | `True`，读 `runs_strat/sw_s4/ckpt.pt` |
| GPU 温度保护 | ≥85°C 暂停，≤72°C 恢复 |

- 实际训练：208 个 epoch，best epoch 107，
  early stop @ep207，累计约 1.3 小时
  （按日志每 epoch 时间累加，不含首次模型加载）。

冒烟（1 epoch，独立 tag，不污染正式 run）：

```bat
D:\qhchen\NewModel\hefeic1\cemian1103\scripts\smoke_sw_s4.bat
```

正式训练（后台/前台均可，日志 `D:\qhchen\NewModel\hefeic1\cemian1103\logs\train_sw_s4.log`）：

```bat
D:\qhchen\NewModel\hefeic1\cemian1103\scripts\train_sw_s4.bat
```

后台执行命令
```bat
Start-Process -FilePath "D:\qhchen\NewModel\hefeic1\cemian1103\scripts\train_sw_s4.bat" -WindowStyle Hidden
```

查看后台命令的执行情况
```
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*train_strat_b*" } | Select-Object ProcessId, CommandLine
```


直接命令：

```bat
python D:\qhchen\NewModel\hefeic1\cemian1103\scripts\pipeline\train_strat_b.py --config D:\qhchen\NewModel\hefeic1\cemian1103\config_sw_s4.yaml --tag sw_s4
```

产出：`D:\qhchen\NewModel\hefeic1\cemian1103\runs_strat\sw_s4\best`、`best_meta.pt`、`ckpt.pt`、`train_log.csv`、`final_miou.txt`。
中断后用同一命令续训；要彻底重来先删除 `D:\qhchen\NewModel\hefeic1\cemian1103\runs_strat\sw_s4`。

---

## 5. 测试结果

### 5.1 PyTorch 全图评估

```bat
python D:\qhchen\NewModel\hefeic1\cemian1103\scripts\eval_full.py --config D:\qhchen\NewModel\hefeic1\cemian1103\config_sw_s4.yaml --run sw_s4
```

已写入 `D:\qhchen\NewModel\hefeic1\cemian1103\runs_strat\sw_s4\eval_metrics.json`。

| 指标 | 数值 |
|---|---|
| 训练期 best tile mIoU（final_miou.txt） | 0.7943 |
| best epoch | 107 |
| 全图 mIoU（PyTorch, stride 448, no TTA） | 0.7785 |
| background IoU | 0.9909 |
| silie IoU | 0.6048 |
| fanzhe IoU | 0.7398 |
| pos_silie（正样本图上的 IoU） | 0.6048 |
| pos_fanzhe | 0.7445 |
| 负样本图数 | 14 |
| 负样本假阳均值 / 最大 | 2652.3 / 21191 |
| 负样本 >100px 图数 / 全干净图数 | 3 / 11 |
| 平均耗时（含滑窗） | 505.3 ms |


> `final_miou.txt` 是训练期 **tile-level** 指标，不是全图验收值。
> 参考：sw_s4 在 89 张 val_strat 全图上 mIoU 0.7593 / bg 0.9908 / silie 0.5833 / fanzhe 0.7039。
> 本数据集样本更小且类别分布不同，不要求达到同一 mIoU；重点看漏检、假阳和两类 IoU。

### 5.2 C++ / C# 对比

```bat
cd /d D:\qhchen\NewModel\newsiliec9\deploy_cpp
run.bat --model cemian1103 --eval D:\qhchen\NewModel\hefeic1\cemian1103\dataset_strat\val_full.txt --out D:\qhchen\NewModel\hefeic1\cemian1103\runs_strat\sw_s4\cpp_eval --mask

EVDll_Segformer\SegTest\SegTest.bat D:\qhchen\NewModel\hefeic1\cemian1103\export\segformer_cemian_b1_fp16_640.engine --eval D:\qhchen\NewModel\hefeic1\cemian1103\dataset_strat\val_full.txt D:\qhchen\NewModel\hefeic1\cemian1103\runs_strat\sw_s4\cs_eval

python D:\qhchen\NewModel\hefeic1\cemian1103\scripts\score_masks.py --val-full D:\qhchen\NewModel\hefeic1\cemian1103\dataset_strat\val_full.txt --pred-dir D:\qhchen\NewModel\hefeic1\cemian1103\runs_strat\sw_s4\cs_eval --ref-dir D:\qhchen\NewModel\hefeic1\cemian1103\runs_strat\sw_s4\cpp_eval --out D:\qhchen\NewModel\hefeic1\cemian1103\runs_strat\sw_s4\cs_eval_metrics.json
```

| 路径 | mIoU | bg | silie | fanzhe |
|---|---|---|---|---|
| PyTorch 全图 | 0.7785 | 0.9909 | 0.6048 | 0.7398 |
| C++ `seg_infer.exe` | 0.778492 | 0.990932 | 0.604702 | 0.739842 |
| C# `SegTest.exe` | 0.778268 | 0.990947 | 0.603626 | 0.740232 |


C# vs C++ mask 差异：11273 像素 / 26 张图（ref_pairs=37）。

> 这里输入的是原始 JPG：GDI+（C# `new Bitmap`）与 OpenCV（C++ `cv::imread`）的 JPEG
> 解码结果不完全相同，因此会有少量像素/mask 差异。换成同一张无损 PNG 输入时，
> `SegTest.exe` 与 `seg_infer.exe` 的 mask 应逐像素一致；若仍有差异，先检查
> `src/` 与 `segformer_dll/` 两份推理代码是否同步。

---

## 6. 导出与部署

### 6.1 导出 ONNX + fp16 engine

```bat
D:\qhchen\NewModel\hefeic1\cemian1103\scripts\export_engine.bat
```

等价命令：

```bat
python D:\qhchen\NewModel\hefeic1\cemian1103\scripts\pipeline\export_best_onnx.py --src D:\qhchen\NewModel\hefeic1\cemian1103\runs_strat\sw_s4\best --out D:\qhchen\NewModel\hefeic1\cemian1103\export\segformer_cemian_b1_640.onnx
trtexec --onnx=D:\qhchen\NewModel\hefeic1\cemian1103\export\segformer_cemian_b1_640.onnx --saveEngine=D:\qhchen\NewModel\hefeic1\cemian1103\export\segformer_cemian_b1_fp16_640.engine --fp16 ^
  --minShapes=pixel_values:1x3x640x640 ^
  --optShapes=pixel_values:16x3x640x640 --maxShapes=pixel_values:16x3x640x640 ^
  --memPoolSize=workspace:4096
```

| 项目 | 值 |
|---|---|
| ONNX | `D:\qhchen\NewModel\hefeic1\cemian1103\export\segformer_cemian_b1_640.onnx` |
| fp16 engine | `D:\qhchen\NewModel\hefeic1\cemian1103\export\segformer_cemian_b1_fp16_640.engine` |
| 测试用 engine 副本 | `D:\qhchen\NewModel\newsiliec9\deploy_cpp\models\segformer_cemian_b1_fp16_640.engine` |
| ONNX 大小 / SHA256 | 54961131 B / `c723b35ac02baa6994601ef78fe9eecfc254a18e507b702c28e509830bf37d4d` |
| engine 大小 / SHA256 | 34864308 B / `1463f381f6ceec988171012e82bc97526e8ccad54dca1ae7cc0486ccf3ce0777` |


### 6.2 引擎加载

```bat
trtexec --loadEngine=D:\qhchen\NewModel\hefeic1\cemian1103\export\segformer_cemian_b1_fp16_640.engine --iterations=10 --warmUp=500 --avgRuns=10
```

最后应为 `&&&& PASSED`。engine 与 TensorRT 版本、GPU 绑定；换机器需用
`deploy_cpp/SOP_ENGINE_EXPORT.md` 的流程重建。

### 6.3 软件 C# 调用

单模型（与现有 `AI_Seg1` 一致）：

```csharp
using EVDll_Segformer;

AI_Seg1.Init(@"D:\qhchen\NewModel\newsiliec9\deploy_cpp\models\segformer_cemian_b1_fp16_640.engine", ref erro);
var overlay = AI_Seg1.Run(GetBitMap("采集图像.图像"), ref erro);
SetBitMap(AI_Seg1.GetMask(), "采集图像.图像");   // 斑点分析用单通道掩膜
```

两个模型在同一进程（`AI_Seg1` 只能装一个，推荐 `SegModel`）：

```csharp
using EVDll_Segformer;

SegModel front = new SegModel(@"D:\qhchen\NewModel\newsiliec9\deploy_cpp\models\segformer_zhengmian_b1_fp16_640.engine");
SegModel side  = new SegModel(@"D:\qhchen\NewModel\newsiliec9\deploy_cpp\models\segformer_cemian_b1_fp16_640.engine");

var ovFront = front.Run(GetBitMap("正面上相机.图像"), ref erro);
var ovSide  = side.Run(GetBitMap("侧面上相机.图像"), ref erro);
SetBitMap(front.GetMask(), "正面结果.掩膜");
SetBitMap(side.GetMask(),  "侧面结果.掩膜");
```

斑点分析必须链接 `GetMask()` 的 8bpp 单通道掩膜，阈值：

- 同时检测两类：1~2
- 只撕裂：1~1
- 只翻折：2~2

不能链接 `Run()` 的 3 通道叠加图，否则会报
`输入图像是多通道图像,无法进行blob分析`。

---

## 7. 验收清单

- [ ] `labels/183` 个，空 label 68 个
- [ ] train/val = 146/37，交集 0
- [ ] train tiles 934 对，val tiles 232 对，mask 值 ⊆ {0,1,2}
- [ ] val_full 37 行，尺寸等于原图
- [ ] 冒烟 1 epoch 成功，loss 非 NaN
- [ ] 正式训练生成 `best/`、`ckpt.pt`、`train_log.csv`、`final_miou.txt`
- [ ] `eval_metrics.json` 已生成
- [ ] `trtexec --loadEngine` 输出 PASSED
- [ ] C++ mIoU 与 PyTorch 差异 ≤ 0.01
- [ ] `SegTest` 布局 OK、engine 加载成功、mask 与 C++ 一致
- [ ] 生产目录 `E:\Release - 14.0.5.1-exe\dll` 未被修改

---

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `FileNotFoundError ... pipeline\config_sw_s4.yaml` | 必须给 `train_strat_b.py` 传 **绝对** config 路径；本目录的 .bat 已传绝对路径 |
| 续训到了错误的 run | `resume: true` 会读 `runs_strat/sw_s4/ckpt.pt`；彻底重来先删除该目录；冒烟用 `sw_s4_smoke` |
| `trtexec` 找不到 `cudart64_12.dll` | 把 CUDA 12.1 `bin` 加到 PATH；export 脚本已自动加 |
| 引擎加载失败 / 版本不匹配 | engine 与目标 `nvinfer.dll` 必须是同一 TensorRT 版本；按 `deploy_cpp/SOP_ENGINE_EXPORT.md` 重建 |
| 斑点分析报多通道 | 链接 `GetMask()` 掩膜，不要链接 `Run()` 叠加图 |
| 训练显存 OOM | 关闭其他 GPU 程序；不要擅自改 batch/分辨率，先确认是否与其他任务并发 |
| silie IoU 低 / 波动大 | val 中 silie 实例很少，属正常高方差；先看漏检和假阳，调参需另行批准 |

---

## 9. 版本记录

| 日期 | 内容 |
|---|---|
| 2026-10-23 | 基于 sw_s4 配方完成数据准备、训练、评估与导出 SOP |
