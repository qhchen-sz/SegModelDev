# SegFormer 语义分割的 C++ / TensorRT 部署

把 `segformer_v2/runs_strat/<model>/best` 里的权重做成**独立 C++ 可执行程序 + TensorRT 引擎**，
整图滑窗输入，输出语义图与叠加图。当前已接入两个模型：`v2_strat`（原 v2-fast）与 `sw_s4`（超参扫描选点）。

**历史基线（旧版直接 argmax）：两模型均已验证。** sw_s4 部署后 mIoU 0.7594（PyTorch 基线 0.7593），
155 ms/图（PyTorch 864 ms）。S4 被选中的理由——负样本假阳降 82%——在部署后完整保留。
当前默认使用 Softmax 最大置信度阈值 0.5 过滤背景；下文历史精度与速度不代表该阈值的重新评估结果。

---

## 1. 快速开始

```bat
:: --- 离线：构建引擎 + 编译（各一次）---
tools\build_engine.bat sw_s4           :: 导出 ONNX + 构建 FP16 引擎（约 4 分钟）
build.bat                              :: 编译 seg_infer.exe 与 segformer_Dll_func.dll
EVDll_Segformer\build_csharp.bat       :: 编译 EVDll_Segformer.dll 与 SegTest.exe

:: --- 命令行验证 ---
run.bat --model sw_s4 --input <图或目录> --out <输出目录> [--mask]
run.bat --model sw_s4 --eval ..\segformer_v2\dataset_strat\val_full.txt
run.bat --eval ..\segformer_v2\dataset_strat\val_full.txt        :: 默认 v2_strat

:: --- 走软件用的那条 C# → C++ 路径自测（见 §10）---
EVDll_Segformer\SegTest\SegTest.bat <engine> <图> <输出目录> [--mask]
```

`<model>` = `segformer_v2/runs_strat/` 下的 run 目录名（`sw_s4` / `v2_strat` / `sw_s1` / `sw_s2` /
`sw_s3` / `v2b_strat`）。引擎名规则：`models/segformer_<model>_b1_<prec>_640.engine`。

> ⚠️ 本目录下 `build.bat` / `run.bat` / `tools\build_engine.bat` **刻意只用 ASCII 注释**：
> cmd.exe 按 OEM 代码页(936/GBK)解析 .bat，UTF-8 中文注释会打乱命令解析（踩过这个坑）。
> 中文注释只放在 `.cpp` / `.py` / `.md` 里。

---

## 2. 先澄清：`v2_fast_strat/` 里为什么没有模型文件

`segformer_v2/runs_strat/v2_fast_strat/` 是**评测产物目录**，不是模型目录。
`infer_fast_strat.py:91` 只从别处**读**权重：

```python
model = SegformerForSemanticSegmentation.from_pretrained(runs / 'v2_strat' / 'best')
```

它把结果写进该目录的 `eval_result.txt` 与 `overlays/`（89 张叠加图），权重一个字节都不落在这里。
**`v2-fast` 与 `v2` 是同一份权重**，只是推理策略不同（bs16 + FP16 + stride448 + 无 TTA）。

所有模型的权重都在各自的 run 目录下：

| run 目录 | 权重 | 说明 |
|---|---|---|
| `runs_strat/v2_strat/best/` | `model.safetensors` + `config.json` | v2 基线（= v2-fast 用的权重） |
| `runs_strat/sw_s4/best/` | 同上 | 超参扫描 S4：`bg_keep=0.12, focal_alpha(bg)=0.80` |

两份 `config.json` **逐字段相同**（同架构 SegFormer-B1，3 类），只有权重不同。

---

## 3. 实测结果

### 3.1 精度

89 张 `val_strat` 全图评测（同一 GT、同一语义 IoU 口径、均无 TTA）：

| 指标 | **sw_s4** PyTorch | **sw_s4** TRT fp16 | **sw_s4** TRT fp32 | v2_strat PyTorch | v2_strat TRT fp16 |
|---|---|---|---|---|---|
| background IoU | 0.9908 | 0.9908 | 0.9908 | 0.9901 | 0.9901 |
| silie IoU | 0.5833 | 0.5824 | 0.5831 | 0.5611 | 0.5611 |
| fanzhe IoU | 0.7039 | 0.7050 | 0.7039 | 0.7081 | 0.7081 |
| **mIoU** | **0.7593** | **0.7594** | **0.7593** | **0.7531** | **0.7531** |
| 平均耗时 | 864.2 ms | **155.2 ms** | 297.0 ms | 811.1 ms | **156.3 ms** |

### 3.2 sw_s4 的部署保留住了选点理由（这才是关键指标）

S4 被选中的理由是**负样本假阳降 82%**，不是 mIoU。部署后逐张验算 42 张负样本：

| | 假阳均值 | 最大 | >100px 图 | 全干净图 |
|---|---|---|---|---|
| PyTorch 基线（= `sweep_eval.json`） | 2003.3 | 59578 | 7 | 33 |
| **TRT fp16 部署** | **2016.4** | 59671 | **7** | **33** |
| 对照：v2_strat（`sweep_report.md`） | 11394 | 211461 | 8 | — |

差 **+0.65%**，`>100px` 与全干净图数完全一致 → **-82% 的假阳削减完整保留**。

---

## 4. 一个值得知道的发现：sw_s4 的 fp16 比 v2_strat 敏感得多

单 tile 原始 logits 与 PyTorch 对比（同一张含缺陷裁图）：

| 引擎 | max\|Δ\| | mean\|Δ\| | silie 通道 max\|Δ\|/std | argmax 一致率 |
|---|---|---|---|---|
| v2_strat fp16 | 0.289 | 0.048 | 0.075 | 99.992% |
| **sw_s4 fp16** | **2.312** | **0.483** | **0.935** | 99.977% |
| sw_s4 fp32 | 0.124 | 0.013 | 0.050 | 99.977% |

sw_s4 的 fp16 误差比 v2_strat 大一个数量级，且 silie 通道误差几乎等于该通道自身的标准差。
**排查过程与结论**：

- 不是 trtexec 选核抽签——重建两次（引擎 35077172 / 35140892 字节，确实选了不同核），
  保真度几乎完全一致（max\|Δ\| 2.31 vs 2.27）。是这批权重自身对 fp16 更敏感
  （大概率与 `focal_alpha(bg)=0.80` 训练出的权重尺度分布有关）。
- **不影响决策**：差异全部集中在接近平局的像素上——非平局像素的 argmax 不一致率为 **0**；
  89 张标签图逐像素差异 sw_s4 37710 px (0.0055%) vs v2_strat 8452 px (0.0012%)，
  其中 73% 落在距类别边界 1px 内、92% 在 4px 内。最终 mIoU 差 0.0001。
- 想要更贴近 PyTorch 数字可以用 fp32：`tools\build_engine.bat sw_s4 --fp32`
  （mIoU/fanzhe 完全一致，silie 0.5831 vs 0.5833）。代价是 1.9x 慢（297 vs 155 ms）。

**默认仍用 fp16**：它 1.9x 快，且决策指标（假阳）无实质差异。

---

## 5. 工具链（本机实测）

| 组件 | 路径 | 备注 |
|---|---|---|
| TensorRT 8.6.1.6 | `E:\Tensorrt\TensorRT-8.6.1.6` | `include/`、`lib\*.lib`、`bin\trtexec.exe` |
| CUDA 12.1 + cuDNN 8.9.6 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1` | `cudnn.h`、`cudnn.lib` 齐 |
| OpenCV 4.8.0 | `E:\opencv_exe\opencv\build` | `opencv_world480.lib`/`.dll`，vc16 |
| MSVC | `F:\HymsonSoftWare\VS2022` | 14.32.31326 |
| CMake | `F:\HymsonSoftWare\Cmake\bin\cmake.exe` | 3.31.0 |
| Python（导出 ONNX 用） | `F:\HymsonSoftWare\MiniForge\envs\yolov10\python.exe` | 默认 `python` 是 base env，无 torch |

> ⚠️ `D:\SoftWareDownloads\opencv-4.8.0\build\win-install` 只有 CMake 配置文件、**没有任何 lib/dll**，
> 是半成品，不能用。必须用 `E:\opencv_exe\opencv\build`。
>
> OpenCV 是 **vc16**（VS2019 ABI），MSVC v140–v143 ABI 兼容，VS2022(vc17) 可直接链接，无需重编。

---

## 6. 目录结构

```
deploy_cpp/
├── CMakeLists.txt           # 链接 nvinfer / opencv_world480 / cudart；构建 exe + dll 两个目标
├── build.bat                # vcvars64 + cmake configure + build (Release)
├── run.bat                  # 设好 TRT/CUDA/OpenCV 的 PATH，转发参数给 exe
├── tools/build_engine.bat   # <model> -> 导出 ONNX + trtexec 构建引擎
├── tools/parity_check.py    # 导出 PyTorch 参考 logits（单 tile），用于数值对齐
├── tools/crosscheck_eval.py # 用 sweep_eval.py 的参考实现跑 val、导出 label map
├── src/main.cpp             # CLI（--model/--engine）、评测模式、叠加图输出
├── src/trt_engine.hpp/.cpp  # 引擎加载 + 推理
├── src/seg_pipeline.hpp/.cpp# 滑窗切块 + 归一化 + 平均 logits + Softmax 过滤 + segOverlay
├── segformer_dll/           # ← C ABI 导出层，供 C# 经 P/Invoke 调用（见 §10）
│   ├── segformer_dll.hpp/.cpp
│   ├── segformer_dll.vcxproj#   原生 VS 工程（与 CMakeLists 描述同一份源码，见 §8）
│   └── segformer_dll.sln
├── EVDll_Segformer/         # ← C# 调用层（见 §10）
│   ├── AI_Seg.cs            #    Native(P/Invoke) + SegModel(实例式) + AI_Seg1(静态槽)
│   ├── EVDll_Segformer.csproj
│   ├── EVDll_Segformer.sln  #    C# 库 + SegTest（SegTest 为启动项目）
│   ├── build_csharp.bat
│   └── SegTest/             #    端到端自测台（也是用法示例）
└── models/
    ├── segformer_sw_s4_b1_640.onnx            # 一份 ONNX 服务两种精度
    ├── segformer_sw_s4_b1_fp16_640.engine     # 默认
    ├── segformer_sw_s4_b1_fp32_640.engine     # 可选（保真度更高，1.9x 慢）
    └── segformer_v2_strat_b1_fp16_640.engine
```

> `seg_infer.exe` 用 `src/trt_engine.cpp` + `src/seg_pipeline.cpp`；`segformer_Dll_func.dll`
> 用 `segformer_dll/` 下的同名副本（VS 工程与 CMake 的 DLL 目标编译的是这一份）。
> 两处必须同步修改——副本内容一致时，用 `seg_infer.exe --eval` 验证的结果对 DLL 同样成立。

---

## 7. 构建引擎

```
tools\build_engine.bat <model> [--force] [--fp32]
```

实际执行：

```
python segformer_v2\export_best_onnx.py --src <best> --out models\segformer_<model>_b1_640.onnx
trtexec --onnx=... --saveEngine=models\segformer_<model>_b1_fp16_640.engine
        --fp16
        --minShapes=pixel_values:1x3x640x640
        --optShapes=pixel_values:16x3x640x640
        --maxShapes=pixel_values:16x3x640x640
```

- 引擎/ONNX 已存在则跳过，`--force` 强制重建
- **为什么固定 640×640**：`infer_fast_strat.py:39-41` 对边缘块用 `BORDER_REFLECT` 补到 tile×tile，
  所以**每一块恒为 640×640**，空间维不需要动态。固定空间维能让 TRT 把 ONNX 里 84 个 `Shape` +
  76 个 `Reshape` 全部常量折叠掉。batch 留 1–16 动态，方便最后一批不满。
- 引擎**不可跨 TRT 版本/显卡移植**，换机器需重建
  → 换机器（同为 30 系）的完整步骤见 [SOP_ENGINE_EXPORT.md](SOP_ENGINE_EXPORT.md)
- ⚠️ `export_best_onnx.py:24` 的 `--out` 默认值指向 v2 的 ONNX，**脚本里永远显式传 `--out`**，
  否则会静默覆盖 v2 的 ONNX

## 8. 编译

### 命令行

```
build.bat                          :: CMake -> build\Release\{seg_infer.exe, segformer_Dll_func.dll}
EVDll_Segformer\build_csharp.bat   :: MSBuild -> EVDll_Segformer.dll 与 SegTest.exe
```

### 用 VS 直接打开

| 解决方案 | 含哪些工程 | 用途 |
|---|---|---|
| `segformer_dll\segformer_dll.sln` | `segformer_dll`（C++ DLL） | 编原生 DLL，可下断点调试 |
| `EVDll_Segformer\EVDll_Segformer.sln` | `EVDll_Segformer` + `SegTest`（**启动项目**） | 编 C# 调用层；F5 直接跑端到端自测 |

**VS 侧只以 Release|x64 为准**：

- C++ 工程**只有** `Release|x64` —— TensorRT 与 OpenCV 都是 Release 库，Debug 配置混 CRT
  很容易编过但运行时崩。这是交付用的 DLL，不给 Debug 反而少一个坑
- C# 两个工程补了 `Debug|x64`（VS 配置下拉里好看，也方便调托管代码），但**交付用 Release**

> **F5 调试 `SegTest`**：Debug 构建的 post-build 会把 `nvinfer.dll` / `opencv_world480.dll` /
> `cudart64_12.dll` 拷进 `SegTest\bin\Debug\`（`SegTest.csproj` 的 `CopyNativeRuntimeDlls`），
> 所以调试会话不需要预置 PATH；Release 仍由 `SegTest.bat` 设置 PATH。
> 否则 F5 会因加载不到依赖报 `DllNotFoundException`（HRESULT 0x8007007E）。

**两套构建的关系**：原生 `.vcxproj` 与 `CMakeLists.txt` 描述同一份源码、输出到**同一个**
`build\Release\segformer_Dll_func.dll`，所以可以互换。**改其中一处请同步改另一处**
（两个文件里都有互相指路的注释）。

已验证二者行为完全一致：同样的 89 张 val_strat 上指标**逐位相同**
（mIoU 0.7593 / silie 0.5820 / fanzhe 0.7052、假阳均值 1988.0、最大假阳 59740），
且同一张无损 PNG 下 mask **逐像素相同（0 差异）**。

**构建顺序：先 C++ 后 C#。** `SegTest.csproj` 的 post-build 会从 `build\Release\` 拷贝原生 DLL；
若还没编过 C++，该步骤会静默跳过（拷贝是条件执行的），编译不会失败，但运行时 SegTest 会报找不到 DLL。

## 9. 运行

```
run.bat --model <name> --input <图片或文件夹> --out <输出目录> [--mask]
run.bat --model <name> --eval <val_full.txt> [--out <目录>] [--mask]
run.bat --model <name> --input <单张图> --dump-logits <out.bin>
```

- `--model` 解析为 `models/segformer_<name>_b1_fp16_640.engine`（先找当前工作目录、再找 exe 同级/上级的 `models\`）
- `--engine <完整路径>` 直接指定；两者**后写的生效**，所以 `run.bat --model v2_strat --engine models\X.engine`
  会用 `X.engine`
- `--out`：叠加图 `<stem>.png`；`--mask` 时再存 `mask/<stem>.png`（0/1/2 语义图）
  > 89 张全图叠加图约 633 MB，验证时生成、事后可删，随时用 `--out` 重跑即可重建
- `--eval`：读 `val_full.txt`（`图<TAB>GT.png`），打印 IoU 与平均耗时
- `--dump-logits`：单 tile 原始 logits，用于与 `tools/parity_check.py` 做数值对齐
- `--stride`（默认 448）/ `--batch`（默认 16）/ `--limit` / `--warmup`
- `--confidence-threshold`（默认 0.5，范围 `[0,1]`）：Softmax 最大置信度低于阈值时判为背景，
  等于阈值时保留；设为 0 可恢复只做 argmax 的标签结果（NaN 仍判为背景）

---

## 10. 接入视觉软件：C# Dll + C++ Dll

链路与 `D:\CaoJiaoJie` 里的 YOLO 方案完全同构：

```
视觉软件(Halcon 脚本) → EVDll_Segformer.dll (C#) → segformer_Dll_func.dll (C++) → TensorRT
  AI_Seg1.Init(engine, ref erro)     P/Invoke          SegCreate / SegLoad
  AI_Seg1.Run(bmp, ref erro) ────────────────────────► SegForward → SegGetOverlay
```

**没有一行重复的推理逻辑**：C++ DLL 直接复用 §9 里已验证的 `seg_pipeline` / `trt_engine`。

### C++ 导出接口

| 函数 | 说明 |
|---|---|
| `void* SegCreate()` / `SegDestroy(void*)` | 不透明句柄，每个句柄一份引擎+CUDA 上下文 |
| `bool SegLoad(const char* engine, void*)` | 加载 `.engine` |
| `bool SegSetConfidenceThreshold(void*, float)` | 设置 Softmax 阈值 `[0,1]`，默认 0.5；可在加载前后调用 |
| `int SegForward(uchar* bgr, int w, int h, int stride, void*)` | 整图滑窗推理；返回 0 成功 |
| `const char* SegGetOverlay(int& w,int& h,int& step,int& ch, void*)` | 叠加图（BGR 3ch） |
| `const char* SegGetMask(...)` | 纯语义图（单通道，像素值 0/1/2） |
| `void SegFreeImage(const char*)` | 释放上面两个返回的缓冲 |
| `void SegGetStat(void*, SegStat*)` | 缺陷统计（按需计算并缓存） |
| `int SegStatSize()` / `const char* SegLastError()` | 布局自检 / 取错误原因 |

`SegStat`（按面积降序，不做任何最小面积过滤）：

```
ng                    // 是否有缺陷
px[3]                 // 各类别像素总数：0=背景 1=silie(撕裂) 2=fanzhe(翻折)
ncomp[3]              // 各类别连通域真实总数（ncomp[0] 恒 0）
nbox / boxCls[]{64} boxX/Y/W/H[] boxPx[]
                      // 前 64 个连通域的外接框与像素数；超出的丢掉最小的，
                      // 而 ncomp 报的是真实总数，所以截断不会被掩盖
```

### C# API（`EVDll_Segformer.AI_Seg1`，与 `EVDll.AI_Gpu1` 同构）

```csharp
public static void   Init(string AIPath);
public static void   Init(string AIPath, ref string ErroMessage);
public static bool   SetConfidenceThreshold(float threshold);
public static bool   SetConfidenceThreshold(float threshold, ref string ErroMessage);
public static Bitmap Run(Bitmap bitmap);                     // 叠加图
public static Bitmap Run(Bitmap bitmap, ref string ErroMessage);
public static Bitmap GetMask();                              // 纯 0/1/2 mask
public static SegStat GetStat();                             // 缺陷统计
public static void   Close();
```

脚本里与 YOLO 版对照着看：

```csharp
using EVDll_Segformer;
...
AI_Seg1.Init(@"...\segformer_sw_s4_b1_fp16_640.engine", ref erro);
if (!AI_Seg1.SetConfidenceThreshold(0.5f, ref erro)) { LogError(erro); return false; }
var image = AI_Seg1.Run(GetBitMap("采集图像.图像"), ref erro);
if (erro != "") { LogError(erro); return false; }
SetBitMap(image, "采集图像.图像");

var st = AI_Seg1.GetStat();          // 缺陷判定
if (st.ng == 1) { /* st.px[1] 撕裂像素数 / st.px[2] 翻折像素数 / st.box* 外接框 */ }
```

48 个静态槽都提供 `SetConfidenceThreshold`，在 `Init` 成功后调用一次即可持续生效。
需要同时跑多个模型或更精细控制加载/释放生命周期时，也可以用实例式的 `SegModel`
（`new SegModel(engine)` / `SetConfidenceThreshold(0.5f)` / `Run` / `Dispose`）；
`AI_Seg1` 只是它的静态薄壳，与 `AI_Gpu1` 的「一个槽一个模型」限制一致。

可直接粘到软件脚本编辑器里的完整模板见 `EVDll_Segformer\SoftwareScriptSample.cs`
（含纯语义图、缺陷统计与小面积过滤示例）。

### 送给「斑点分析」的必须是掩膜，不能是叠加图

`Run()` 返回的是 **24bpp BGR 叠加图**（原图 + 红/蓝着色），只适合显示。
软件里的斑点分析模块只接受单通道图像，喂叠加图会直接报：

```
输入图像是多通道图像,无法进行blob分析
```

`GetMask()` 返回 `Format8bppIndexed`（像素值 0/1/2），软件 `SetBitMap` 对非
`Format24bppRgb` 的 Bitmap 按 `"byte"` 写入，因此到斑点分析是**单通道**：

```csharp
SetBitMap(AI_Seg1.GetMask(), "采集图像.图像");   // 斑点分析沿用原链接即可
```

斑点分析的阈值设 `1~2` 可同时取两类；要分开统计就设 `1~1`（撕裂）与 `2~2`（翻折）。
若要同时保留叠加图显示，给掩膜或叠加图单独建一个图像变量，二者不要写进同一个变量。

> `SetBitMap` 会把宽度不是 4 的倍数的图裁到 `width - width % 4`（`EVision.Common.dll`
> 的实现），相机图宽通常是 4 的倍数，不受影响；做 ROI 时注意这一点。

### 构建与部署

```
build.bat                          :: 先出 build\Release\segformer_Dll_func.dll
EVDll_Segformer\build_csharp.bat   :: 再出 EVDll_Segformer.dll 与 SegTest.exe
```

部署时把这两个 DLL 放进软件目录（例如 `D:\CaoJiaoJie\hmxstudio\00Exe\`）即可：
`nvinfer.dll` / `nvinfer_plugin.dll` / `opencv_world480.dll` 那里**已经有了**，无需我们自带。

### 实测（89 张 val_strat）

| 指标 | PyTorch 基线 | **C#→C++ DLL** | seg_infer.exe |
|---|---|---|---|
| mIoU | 0.7593 | **0.7593** | 0.7594 |
| background IoU | 0.9908 | 0.9908 | 0.9908 |
| silie IoU | 0.5833 | 0.5820 | 0.5824 |
| fanzhe IoU | 0.7039 | 0.7052 | 0.7050 |
| 负样本假阳均值 | 2003.3 | **1988.0** | 2016.4 |
| 负样本最大假阳 | 59578 | 59740 | 59671 |
| 负样本 >100px 图 / 全干净图 | 7 / 33 | **7 / 33** | 7 / 33 |
| 耗时（含全部封送） | 864 ms | **205 ms** | 150.7 ms（纯推理） |

**包装零偏差的证明**：拿同一张**无损 PNG** 分别喂 `seg_infer.exe` 和 `SegTest.exe`，
两者产出的 mask **逐像素完全相同（0 差异）**，各类像素数也一致。
89 张全量下 DLL 与 exe 的 0.0001 mIoU 差来自 **GDI+ 与 OpenCV 的 JPEG 解码差异**——
软件实际喂的是相机内存里的 Bitmap，不存在这个环节。

### 与 YOLO 版的差异（踩过的点）

1. **`bool` 返回值必须加 `[MarshalAs(UnmanagedType.I1)]`**。C++ 的 `bool` 是 1 字节，
   而 .NET 默认按 4 字节 Win32 BOOL 封送。`AI_Gpu.cs` 里的 `ModelLoad` 没加（靠运气）；
   我们这边显式加了。
2. **mask 存成 PNG 时 GDI+ 会写成调色板 PNG**（IHDR color type=3），OpenCV 读它会展开成
   BGR（红→灰 76、蓝→灰 29），看着像「全是背景」。用 PIL 读索引值，或在 .NET 里直接读
   Bitmap 的像素值。**运行时不受影响**——Halcon 拿到的是 Bitmap 对象，不是 PNG。
3. **Softmax 置信度阈值默认 0.5**，低于阈值判为背景；`Run` 仍不带 `(conf, nms)` 参数，
   实例式 `SegModel` 可在 `Init` 后调用 `SetConfidenceThreshold` 调整，不做 NMS。
4. 启动时会断言 `Marshal.SizeOf(SegStat) == SegStatSize()`（本机为 1840 字节），
   布局漂移会立刻报错而不是静默读错数据。

---

## 11. 实现与 infer_fast_strat.py 的对齐关系

| 环节 | Python 参考 | C++ 实现 |
|---|---|---|
| 切块 | `collect_tiles` | `seg_pipeline.cpp:collectTiles` |
| 预处理 | `batch_forward` | `fillNCHW`（BGR→RGB、/255、ImageNet mean/std、NCHW） |
| 推理 | `batch_forward` | TRT `execute_async_v3` + CUDA stream |
| 上采样 | `F.interpolate` | `cv::resize(INTER_LINEAR)` |
| 累加与分类 | `sliding_predict_fast` / `logits_to_labels` | 累加 `logits` 与覆盖次数，平均后做 Softmax 置信度过滤 |
| 叠加 | `overlay` | `main.cpp:overlay`，背景(60,60,60)/silie(0,0,255)/fanzhe(255,0,0)，alpha 0.5 |

两个关键等价性：

1. **上采样**：`F.interpolate(bilinear, align_corners=False)` 与 `cv::resize(INTER_LINEAR)`
   同为半像素中心约定，映射公式一致
2. **置信度过滤**：先算 `avg_logits=acc/cov`，标签取 `argmax(avg_logits)`，
   置信度取 `max(Softmax(avg_logits))`；大于等于阈值时保留标签，否则取背景 0。
   `cov==0` 的像素也取背景 0。

实现细节取舍：
- 累加**必须串行**（`stride 448 < tile 640`，相邻块 ROI 重叠），故只并行上采样、串行累加
- 用 pinned 内存（`cudaHostAlloc`），pageable 会让 `cudaMemcpyAsync` 退化成同步拷贝（245→153 ms 的主因）
- 输出形状按引擎静态维直接乘算，不调 `inferShapes()`（该调用在本引擎上返回 false）

---

## 12. 验证方法（可复跑）

1. **引擎可加载**：启动打印 `pixel_values: -1x3x640x640`、`logits: -1x3x160x160`、profile 1/16/16
2. **数值对齐（单 tile）**：
   ```
   run.bat --model <name> --input <含缺陷的 640x640 裁图> --dump-logits trt.bin
   F:\HymsonSoftWare\MiniForge\envs\yolov10\python.exe tools\parity_check.py ^
       --img <同图> --best ..\segformer_v2\runs_strat\<name>\best --out torch.bin
   ```
   比 max\|Δ\|、相关系数、argmax 一致率
3. **端到端对齐（89 张）**：`run.bat --model <name> --eval ..\segformer_v2\dataset_strat\val_full.txt --confidence-threshold 0`
   —— 阈值 0 用于复现 §3.1 的旧版 argmax 基线；默认阈值 0.5 的指标需另行评估
4. **交叉校验**（当 C++ 与基线对不齐时，判断是 fp16 差异还是管线 bug）：
   ```
   run.bat --model <name> --eval <val_full> --out out_cpp --mask --confidence-threshold <threshold>
   F:\HymsonSoftWare\MiniForge\envs\yolov10\python.exe tools\crosscheck_eval.py ^
       --best ..\segformer_v2\runs_strat\<name>\best --val-full <val_full> --out out_py ^
       --confidence-threshold <threshold>
   ```
   两端必须使用相同阈值，然后逐像素比 `out_cpp/mask/` 与 `out_py/`；
   重点检查类别边界及置信度接近阈值的位置。

---

## 13. 未做 / 后续可选

- 未新建 HTTP 服务（按需求出独立可执行程序 + DLL）
- **未改动 `D:\CaoJiaoJie\EVDll` 与 `D:\CaoJiaoJie\gpu_Dll_func` 任何文件**（只读参考）
- 未重训，未改训练损失与反向传播；ONNX/引擎仍输出原始 logits，无需因后处理变化重新导出
- 未改历史报告及已有 `sweep_eval.json` 结果文件
- 未重新评估 222/C8/fl 三个现场批（超出「部署」范围）
- 不做最小面积过滤（按要求只报原始统计）
- 若还要更快：exe 路径 150.7 ms 中 GPU 占约 97 ms；DLL 路径多出的 ~55 ms 是
  C#/C++ 边界上的若干次全图内存拷贝（23 MB×数次），要再压需要改成零拷贝接口
- `run.bat` 目前只对 fp16 引擎做 `--model` 名字解析；用 fp32 请传
  `--engine models\segformer_<name>_b1_fp32_640.engine`
