# SegFormer sw_s4 → TensorRT Engine 导出 SOP

> **适用模型**：`deploy_cpp\models\segformer_sw_s4_b1_fp16_640.engine`
>
> **目标**：换到另一台 **30 系显卡** 的机器时，重新导出 ONNX 并构建可用的 `.engine`
>
> **本机基线（已验证）**：RTX 3090 / TensorRT 8.6.1.6 / CUDA 12.1 /
> Python 3.9.23 / torch 2.5.1+cu121 / transformers 4.57.6 / onnx 1.14.0
>
> **最后更新**：2026-10-22

---

## 0. 先看结论（TL;DR）

1. **engine 不能跨机器直接拷贝**。它绑定 TensorRT 版本和 GPU，换机器要在目标机上重建。
2. **ONNX 可以跨机器复用**。新机器只要有 TensorRT 和同一份 ONNX，就能直接构建 engine，**不需要 Python / PyTorch**。
3. 最省事的流程：

   ```
   拷贝 deploy_cpp\models\segformer_sw_s4_b1_640.onnx
   安装 TensorRT 8.6.1.6（必须与目标软件 dll\nvinfer.dll 同版本）
   在目标机执行 trtexec（见 §4.2）
   用 run.bat / SegTest.bat 验证（见 §5）
   把新 engine 放进软件目录，改 C# 脚本里的 EnginePath
   ```

4. 如果要**从权重完整复现**（重新导 ONNX），再按 §4.1 准备 Python 环境。
5. **TensorRT 版本不要随意升级**：engine、`segformer_Dll_func.dll`、软件目录里的
   `nvinfer.dll` 必须是同一套 TensorRT。本机三者都是 8.6.1.6；目标软件
   `dll\nvinfer.dll` 与 `E:\Tensorrt\TensorRT-8.6.1.6\lib\nvinfer.dll` 的大小均为
   `211,297,832` 字节，可直接复用。

---

## 1. 这个 engine 是怎么来的

### 1.1 权重来源

`sw_s4` 是超参扫描里选中的 S4 点，权重在：

```
segformer_v2\runs_strat\sw_s4\best\
├── config.json          # SegFormer-B1, 3 类 (background / silie / fanzhe)
└── model.safetensors    # 微调后的权重，约 54.7 MB
```

架构是 **SegFormer-B1**（`hidden_sizes [64,128,320,512]`，`depths [2,2,2,2]`），
输出 3 类 logits。`best` 是从 `segformer_baseline/models/segformer-b1-ade`
初始化后，在 `dataset_strat` 上训练得到的，部署只用 `best`，不需要训练脚本或数据集。

> 其他模型（`v2_strat` / `sw_s1` / `sw_s2` / ...）流程完全相同，
> 把下文的 `sw_s4` 换成对应的 `runs_strat\<model>` 目录名即可。

### 1.2 第一步：导出 ONNX

脚本：`segformer_v2\export_best_onnx.py`

它做四件事：

1. `SegformerForSemanticSegmentation.from_pretrained(<best>)` 加载 PyTorch 权重；
2. 用 `1x3x640x640` 的随机输入导出 ONNX，**opset 17**；
3. 输入名 `pixel_values`，输出名 `logits`，**batch / H / W 都是动态轴**；
4. 如果装了 `onnxruntime`，会用 CPU 跑一遍，和 PyTorch 输出做 `max|Δ|` 校验。

等价命令：

```bat
<PYTHON> segformer_v2\export_best_onnx.py ^
  --src segformer_v2\runs_strat\sw_s4\best ^
  --out deploy_cpp\models\segformer_sw_s4_b1_640.onnx
```

- 默认 `--opset 17`；如果 `trtexec` 报 LayerNormalization / 算子不支持，再试
  `--opset 16` 重新导出。
- **这一步不需要 GPU**，CPU 版 PyTorch 即可；新机器完全可以只做 ONNX 导出，
  再到目标机用 `trtexec` 构建 engine。
- 产物：`models\segformer_sw_s4_b1_640.onnx`，约 54.96 MB，一份 ONNX 同时服务
  fp16 / fp32 两种 engine（精度是 `trtexec` 的构建参数，不写进 ONNX）。

### 1.3 第二步：用 trtexec 构建 engine

脚本：`deploy_cpp\tools\build_engine.bat`

实际执行的命令：

```bat
trtexec.exe ^
  --onnx="deploy_cpp\models\segformer_sw_s4_b1_640.onnx" ^
  --saveEngine="deploy_cpp\models\segformer_sw_s4_b1_fp16_640.engine" ^
  --fp16 ^
  --minShapes=pixel_values:1x3x640x640 ^
  --optShapes=pixel_values:16x3x640x640 ^
  --maxShapes=pixel_values:16x3x640x640 ^
  --memPoolSize=workspace:4096
```

参数解释：

| 参数 | 原因 |
|---|---|
| `--fp16` | 默认精度。3090 上约 155 ms/图；fp32 约 297 ms/图 |
| `640x640` 固定空间维 | Python/C++ 都把每个滑窗块补到 640×640；固定后 TRT 可常量折叠 ONNX 里的 Shape/Reshape 节点 |
| batch `1~16` | 推理管线 `SegConfig::batch = 16`，最后一批可能不满；batch 保持动态 |
| `--memPoolSize=workspace:4096` | 构建期临时显存 4 GB；8 GB 显卡构建时如果 OOM，改成 2048 |
| 不写 `--fp16` | 构建 fp32 engine（文件名用 `_fp32_`） |

> 如果要改 `tile_size` / `stride` / `batch`，必须**同时**改这里的三组 shapes 和
> C++ 里的 `seg_pipeline.hpp::SegConfig`，再重新构建 engine；只换 engine 不改
> 管线，推理结果会错位或直接失败。

产物命名约定：

```
models\segformer_<model>_b1_<prec>_640.engine
models\segformer_sw_s4_b1_fp16_640.engine
models\segformer_sw_s4_b1_fp32_640.engine
```

> `sw_s4` 的 fp16 对权重比 `v2_strat` 敏感（单 tile `max|Δ|` 约 2.31，fp32 约 0.12），
> 但最终 mIoU / 负样本假阳没有实质差异，所以默认仍用 fp16。需要更贴近 PyTorch
> 数值时再构建 fp32。

### 1.4 构建后的验证数字（本机 3090）

| 指标 | PyTorch 基线 | TRT fp16 |
|---|---|---|
| mIoU（89 张 val_strat 全图） | 0.7593 | **0.7594** |
| silie IoU | 0.5833 | 0.5824 |
| fanzhe IoU | 0.7039 | 0.7050 |
| 平均耗时 | 864 ms | **155 ms** |

> `runs_strat\sw_s4\final_miou.txt` 里的 `0.7651` 是训练期 tile 评估口径，
> 不要拿它和部署后的 0.7594 直接对比。

---

## 2. 为什么换机器必须重建 engine

TensorRT 的序列化 engine **不是通用模型文件**：

- 绑定 **TensorRT 版本**（8.6.1.6 构建的 engine 不能给 8.5/10.x 加载）；
- 绑定 **GPU 架构/型号**（TRT 会在构建期选 kernel、tactic）；
- 绑定构建时的 shape profile（本项目的 640×640、batch 1~16）。

30 系显卡都是 Ampere（compute capability 8.6），理论上同代卡之间可能能加载，
但 TensorRT 官方不保证跨型号可移植，**生产环境一律在目标机重建**。
ONNX 是标准格式，可以随便跨机器；只有 `.engine` 需要重建。

### 2.1 先确认目标软件的 TensorRT 版本

目标软件加载的是它自己目录下的 `nvinfer.dll`。先比对：

```bat
dir "E:\Release - 14.0.5.1-exe\dll\nvinfer.dll"
dir "E:\Tensorrt\TensorRT-8.6.1.6\lib\nvinfer.dll"
```

已验证环境下两者都是 **211,297,832 字节**，即同一版 TensorRT 8.6.1.6。
如果大小/版本不一致，**不要混用**：要么把目标软件升级到同一 TRT 版本，
要么用目标软件自带的 TensorRT 版本重新构建 engine 和原生 DLL。

> 查看 TensorRT 版本：运行 `trtexec.exe`，启动横幅里会打印
> `TensorRT v8601`（8.6.1）。

---

## 3. 新机器环境准备

### 3.1 版本矩阵

| 组件 | 本机已验证 | 新机器要求 |
|---|---|---|
| Windows | 10 x64 | x64 |
| GPU | RTX 3090 (SM 8.6) | 任意 30 系（SM 8.6） |
| NVIDIA 驱动 | 610.88 | 支持 CUDA 12.1 的版本 |
| TensorRT | 8.6.1.6 | **必须与目标 `nvinfer.dll` 同版本** |
| CUDA | 12.1 | TRT 8.6 支持的 12.x |
| Python（仅重导 ONNX 时需要） | 3.9.23 | 3.9 / 3.10 |
| torch | 2.5.1+cu121 | 2.5.1（CPU 版也可导出） |
| transformers | 4.57.6 | 4.57.x |
| onnx | 1.14.0 | 1.14.x |
| onnxruntime（可选校验） | 1.15.1 | 1.15.x |
| VS2022 + CMake + OpenCV 4.8 | 见 README §5 | 仅在需要重编 C++ DLL 时 |

### 3.2 只构建 engine（推荐，最快）

新机器只需要：

1. TensorRT 8.6.1.6（含 `trtexec.exe`、`nvinfer.dll`、`nvinfer_plugin.dll` 等）；
2. CUDA 12.1 运行库（确保 `cudart64_12.dll` 在 PATH 中）；
3. 一份已有的 `segformer_sw_s4_b1_640.onnx`；
4. 目标 GPU。

**不需要 Python、PyTorch、transformers**，直接执行 §4.2 的 `trtexec` 命令即可。

### 3.3 从权重重新导出（完整复现）

如果新机器没有现成 ONNX，或者要改 opset，安装 Python 环境：

```bat
conda create -n segformer_trt python=3.9 -y
conda activate segformer_trt

:: CPU-only torch is enough for ONNX export:
pip install torch==2.5.1
:: For the exact CUDA 12.1 build:
:: pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.57.6 safetensors pyyaml numpy onnx==1.14.0 onnxruntime==1.15.1
```

验证：

```bat
python -c "import torch, transformers, onnx; print(torch.__version__, transformers.__version__, onnx.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

> 默认 `python` 很可能是没有 torch 的 base 环境。`build_engine.bat` 里写的是
> **绝对路径**，务必指向上面创建的环境。

### 3.4 目录布局

`build_engine.bat` 按相对路径找权重和输出，目录必须是：

```
<repo-root>\
├── segformer_v2\
│   ├── export_best_onnx.py
│   ├── config_strat.yaml
│   └── runs_strat\sw_s4\best\
│       ├── config.json
│       └── model.safetensors
└── deploy_cpp\
    ├── tools\build_engine.bat
    └── models\                 # 输出 ONNX / engine
```

`deploy_cpp` 和 `segformer_v2` 必须是**同级目录**；
`build_engine.bat` 自己会从 `tools\..\..` 推算 repo 根目录。

### 3.5 要改的路径

不同机器的盘符/路径不同，至少要改 `build_engine.bat` 里这两行：

```bat
set "TRT=E:\Tensorrt\TensorRT-8.6.1.6"
set "PY=F:\HymsonSoftWare\MiniForge\envs\yolov10\python.exe"
```

如果还要跑验证/部署，以下脚本也写死了本机路径，按新机器修改：

| 文件 | 变量 | 用途 |
|---|---|---|
| `deploy_cpp\tools\build_engine.bat` | `TRT`, `PY` | 构建 engine |
| `deploy_cpp\run.bat` | `TRT`, `CUDA`, `OCV` | 命令行推理/评测 |
| `deploy_cpp\build.bat` | `VCVARS`, `CMAKE` | 重编 C++ DLL |
| `deploy_cpp\CMakeLists.txt` | `TRT_ROOT`, `CUDA_ROOT`, `OPENCV_BUILD_DIR` | CMake 依赖 |
| `deploy_cpp\segformer_dll\run_seg_test.bat` | `TRT`, `CUDA`, `OCV` | C++ DLL 自测 |
| `deploy_cpp\EVDll_Segformer\SegTest\SegTest.bat` | `TRT`, `CUDA`, `OCV` | C# → DLL 自测 |
| `deploy_cpp\EVDll_Segformer\SegTest\SegTest.csproj` | `TrtRoot`, `OpenCvBuild`, `CudaRoot` | Debug 运行时 DLL 拷贝 |

> `.bat` 文件**只用 ASCII 注释/路径**。cmd.exe 按 OEM 代码页(936/GBK)解析，
> 写 UTF-8 中文注释可能把命令解析坏。

---

## 4. 导出步骤

### 4.1 从权重导出 ONNX（可选）

```bat
cd /d <repo-root>
<PYTHON> segformer_v2\export_best_onnx.py ^
  --src segformer_v2\runs_strat\sw_s4\best ^
  --out deploy_cpp\models\segformer_sw_s4_b1_640.onnx
```

正常输出：

```
加载: ...\segformer_v2\runs_strat\sw_s4\best
导出 ONNX (opset 17, 示例输入 torch.Size([1, 3, 640, 640]), 动态 batch+H+W) ...
已保存: ...\segformer_sw_s4_b1_640.onnx  (52.4 MB)
校验 onnx vs torch: max|Δ|=...  形状 (1, 3, 160, 160)  (一致=OK)
动态尺寸测试 512x768 -> 输出 (1, 3, 128, 192)  OK
```

如果 `trtexec` 报 ONNX 算子不支持：

```bat
<PYTHON> segformer_v2\export_best_onnx.py ^
  --src segformer_v2\runs_strat\sw_s4\best ^
  --out deploy_cpp\models\segformer_sw_s4_b1_640.onnx ^
  --opset 16
```

### 4.2 构建 engine

#### 方式 A：用 `build_engine.bat`（有 Python 环境）

```bat
cd /d <repo-root>\deploy_cpp
tools\build_engine.bat sw_s4
```

- 已有 engine 时会跳过；要覆盖加 `--force`：
  ```bat
  tools\build_engine.bat sw_s4 --force
  ```
- 构建 fp32：
  ```bat
  tools\build_engine.bat sw_s4 --fp32 --force
  ```
- 已有 ONNX 时会直接复用，不再调用 Python：
  ```
  [skip] reusing existing ONNX
  ```
- 注意：脚本在复用 ONNX 之前仍会检查 `segformer_v2\runs_strat\sw_s4\best\config.json`
  和 `PY` 是否存在。如果没有 Python/权重，请用方式 B。

#### 方式 B：直接 trtexec（新机器只装 TensorRT 时）

```bat
set "TRT=E:\Tensorrt\TensorRT-8.6.1.6"
set "PATH=%TRT%\lib;%TRT%\bin;%PATH%"
:: If cudart64_12.dll is missing, add CUDA to PATH:
:: set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;%PATH%"

"%TRT%\bin\trtexec.exe" ^
  --onnx="<repo-root>\deploy_cpp\models\segformer_sw_s4_b1_640.onnx" ^
  --saveEngine="<repo-root>\deploy_cpp\models\segformer_sw_s4_b1_fp16_640.engine" ^
  --fp16 ^
  --minShapes=pixel_values:1x3x640x640 ^
  --optShapes=pixel_values:16x3x640x640 ^
  --maxShapes=pixel_values:16x3x640x640 ^
  --memPoolSize=workspace:4096
```

- 构建时间：3090 约 3–5 分钟；低端 30 系更久。
- 显存不足（`[E] Error 2: ... out of memory`）：
  - 关闭其他占 GPU 的程序；
  - 把 `workspace:4096` 改成 `workspace:2048` 再试。
- fp32 就把 `--fp16` 去掉，输出名改为 `_fp32_`。

### 4.3 检查产物

```bat
dir deploy_cpp\models\segformer_sw_s4_b1_fp16_640.engine
```

本机参考大小：

| 文件 | 约大小 |
|---|---|
| `segformer_sw_s4_b1_640.onnx` | 54.96 MB |
| `segformer_sw_s4_b1_fp16_640.engine` | 35.08 MB |
| `segformer_sw_s4_b1_fp32_640.engine` | 63.21 MB |

> 不同 TensorRT/GPU 构建出的 engine 大小会有差异，大小只做 sanity check，
> 不作为成功标准；以 §5 的加载和推理为准。

---

## 5. 验证

### 5.1 engine 能否加载

```bat
set "TRT=E:\Tensorrt\TensorRT-8.6.1.6"
set "PATH=%TRT%\lib;%TRT%\bin;%PATH%"
"%TRT%\bin\trtexec.exe" ^
  --loadEngine="deploy_cpp\models\segformer_sw_s4_b1_fp16_640.engine" ^
  --iterations=10 --warmUp=500 --avgRuns=10
```

看到 `Engine deserialized`、推理完成、最后一行是 `&&&& PASSED` 且没有 `[E]` 即加载成功。
如果报 `Serialization Error` / `Unsupported engine version`，说明 TRT 版本或 GPU
不匹配，回到 §2.1 检查，按目标机版本重建。

### 5.2 单图推理

```bat
cd /d <repo-root>\deploy_cpp
run.bat --model sw_s4 --input <一张图> --out out_check --mask
```

> 前提：`deploy_cpp\build\Release\seg_infer.exe` 已存在。`build\` 在 `.gitignore`
> 里，新机器如果只拷了源码，需要先按 §8.3 重编 C++，或直接用 §5.4 的 C# 路径验证。
> `run.bat` 里的 `TRT/CUDA/OCV` 路径也要先改。

检查：

- `out_check\<stem>.png` 有叠加图；
- `out_check\mask\<stem>.png` 有语义图；
- 控制台打印 `ng=... px[bg=... silie=... fanzhe=...]`，没有报错。

### 5.3 全量 mIoU（可选，需要数据集）

`val_full.txt` 里是**本机绝对路径**（`C:/Users/hymson/Desktop/newsiliec9/...`）。
新机器要么把数据放到同样路径，要么改 `config_strat.yaml` 的
`project_root / dataset_dir` 后重新生成：

```bat
python segformer_v2\convert_strat.py
```

然后：

```bat
run.bat --model sw_s4 --eval segformer_v2\dataset_strat\val_full.txt
```

预期（89 张全图，无 TTA）：

```
mIoU ≈ 0.759
background IoU ≈ 0.9908
silie IoU ≈ 0.582
fanzhe IoU ≈ 0.705
```

不同 30 系显卡上 fp16 kernel 选择可能不同，mIoU 允许 ±0.002 左右的差异；
如果掉到 0.70 以下，优先检查滑窗/tile 尺寸、BGR/RGB、归一化是否一致。

> `runs_strat\sw_s4\final_miou.txt` 的 `0.7651` 是训练期口径，不是这里的验收值。

### 5.4 C# → C++ DLL 路径

```bat
cd /d <repo-root>\deploy_cpp
EVDll_Segformer\SegTest\SegTest.bat ^
  models\segformer_sw_s4_b1_fp16_640.engine ^
  <一张图> out_csharp --mask
```

> 前提：`EVDll_Segformer\SegTest\bin\SegTest.exe`、`EVDll_Segformer.dll`、
> `segformer_Dll_func.dll` 已生成。这些目录也在 `.gitignore` 里；没有的话先跑
> `build.bat` + `EVDll_Segformer\build_csharp.bat`（见 §8.3）。

应在开头看到：

```
SegStat 布局: C++ 1840 字节, C# 1840 字节  -> OK
引擎已加载: ...\segformer_sw_s4_b1_fp16_640.engine
```

同一张无损 PNG 下，`SegTest.exe` 与 `seg_infer.exe` 的 mask 应逐像素相同；
如果不同，说明 C++ DLL 与 exe 两份源码副本不同步，或输入解码路径不同。

### 5.5 验收标准

- [ ] `trtexec --loadEngine` 无错误
- [ ] 单图推理有叠加图 + mask，无报错
- [ ] 全量 mIoU ≈ 0.759（有数据集时）
- [ ] `SegTest.bat` 的 `SegStat` 布局 OK，引擎加载成功
- [ ] 软件里 C# 脚本能 Init / Run，斑点分析用掩膜不再是多通道

---

## 6. 部署到软件

1. 把新 engine 拷到目标软件目录（**路径不能含中文**），例如：
   ```
   E:\Release - 14.0.5.1-exe\dll\models\segformer_sw_s4_b1_fp16_640.engine
   ```
2. 修改 C# 脚本里的 `EnginePath` 指向新文件。
3. `EVDll_Segformer.dll` / `segformer_Dll_func.dll` 如果目标软件已有且 TRT 版本一致，
   可以直接复用；如果重编过，两个 DLL 一起替换。
4. 斑点分析模块必须链接 **`GetMask()` 输出的单通道掩膜**，不要链接 `Run()` 的
   3 通道叠加图，否则会报：
   ```
   输入图像是多通道图像,无法进行blob分析
   ```
   具体见 `README.md` 的“送给斑点分析的必须是掩膜”一节。

---

## 7. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `trtexec.exe not found` | `TRT` 路径不对 | 改 `build_engine.bat` 的 `TRT` |
| `python not found` / 没有 torch | 用了 base env | 改 `PY` 指向装了 torch 的环境；或直接用 §4.2 方式 B |
| ONNX 导出报 `LayerNormalization` / 算子不支持 | opset 太高/解析器不支持 | 加 `--opset 16` 重新导出 |
| `ImportError: onnxruntime` | 未装 onnxruntime | `pip install onnxruntime==1.15.1`；不装也能导出，只是跳过校验 |
| `[E] Error 1: ... out of memory` | 构建 workspace 太大 / 显卡被占 | 关程序；`workspace` 改 2048 |
| `Serialization Error` / `Unsupported engine version` | engine 与目标 `nvinfer.dll` 版本不同 | 用目标软件的 TRT 版本重建；不要混用 |
| 加载 engine 报 GPU 不匹配 | engine 在别的 GPU 上构建 | 在目标机重建 |
| 找不到 `cudart64_12.dll` | CUDA bin 不在 PATH | 把 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin` 加到 PATH |
| 推理 mIoU 明显下降 | 预处理/滑窗/颜色顺序不一致 | 对比 `infer_fast_strat.py` 与 `seg_pipeline.cpp`；确认 BGR→RGB、/255、ImageNet mean/std |
| 同一 engine 两次构建精度略有差异 | TRT tactic 选择不完全确定 | 以 mIoU/假阳指标验收，不要求逐位一致 |
| 斑点分析报多通道 | 链接了 `Run()` 的叠加图 | 改链 `GetMask()` 的掩膜，阈值 1~2 |

---

## 8. 附录

### 8.1 最小文件清单

**方式 B（直接 trtexec，只重建 engine）真正需要的只有：**

```
deploy_cpp\models\segformer_sw_s4_b1_640.onnx    # ONNX 本身可跨机器
TensorRT 8.6.1.6                                  # 与目标 nvinfer.dll 同版本
目标 30 系显卡
```

**如果要从权重完整重导 ONNX，再加：**

```
segformer_v2\export_best_onnx.py
segformer_v2\config_strat.yaml
segformer_v2\runs_strat\sw_s4\best\config.json
segformer_v2\runs_strat\sw_s4\best\model.safetensors
Python 环境（torch / transformers / onnx / onnxruntime）
```

> `segformer_v2\runs_strat\` 在 `.gitignore` 里，**权重可能不在 git 仓库中**，
> 换机器时要手动拷贝 `best` 目录。

### 8.2 路径修改检查表

- [ ] `deploy_cpp\tools\build_engine.bat`：`TRT`、`PY`
- [ ] `deploy_cpp\run.bat`：`TRT`、`CUDA`、`OCV`
- [ ] `deploy_cpp\build.bat`：`VCVARS`、`CMAKE`（重编 DLL 时）
- [ ] `deploy_cpp\CMakeLists.txt`：`TRT_ROOT`、`CUDA_ROOT`、`OPENCV_BUILD_DIR`
- [ ] `deploy_cpp\segformer_dll\run_seg_test.bat`：`TRT`、`CUDA`、`OCV`
- [ ] `deploy_cpp\EVDll_Segformer\SegTest\SegTest.bat`：`TRT`、`CUDA`、`OCV`
- [ ] `deploy_cpp\EVDll_Segformer\SegTest\SegTest.csproj`：`TrtRoot`、`OpenCvBuild`、`CudaRoot`

### 8.3 可选：重编 C++ DLL

如果目标机的 TensorRT/OpenCV 版本与交付 DLL 不一致，需要重编：

```bat
cd /d <repo-root>\deploy_cpp
build.bat
EVDll_Segformer\build_csharp.bat
```

前提：VS2022、CMake、OpenCV 4.8 已装好，`CMakeLists.txt` 的三个依赖路径已改。
`build.bat` 输出：

```
build\Release\seg_infer.exe
build\Release\segformer_Dll_func.dll
EVDll_Segformer\dll\EVDll_Segformer.dll
```

### 8.4 版本记录

| 日期 | 内容 |
|---|---|
| 2026-10-22 | 初版：基于 RTX 3090 / TensorRT 8.6.1.6 / sw_s4 验证 |
