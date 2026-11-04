# segformer_dll50 —— RTX 50 系列显卡适配版

本目录是 `..\segformer_dll`（TensorRT 8.6 版）的 RTX 50 适配工程。
**导出接口、`SegStat` 布局、DLL 文件名（`segformer_Dll_func.dll`）完全不变**，
因此 C# 层 `EVDll_Segformer` 与视觉软件无需任何改动。

---

## 1. 与 TRT 8.6 版的差异

| 项目 | `..\segformer_dll` | **本目录** |
|---|---|---|
| TensorRT | 8.6.1.6 | **10.10.0.31** |
| TRT 根目录 | `E:\Tensorrt\TensorRT-8.6.1.6` | **`D:\Rtx50Test\TensorRT-10.10.0.31`** |
| 链接库 | `nvinfer.lib` / `nvinfer_plugin.lib` | **`nvinfer_10.lib`** |
| 销毁接口 | `ctx_->destroy()` 等 | **`delete`**（TRT10 已移除 `destroy()`） |
| 推理方式 | 单 context，batch=16 | **固定 batch=1 + `numWorkers`(默认4) 路多流并发** |
| DLL 输出位置 | `..\build\Release\` | **本目录 `segformer_dll50\`** |
| 引擎 | `..\models\*.engine`（TRT8） | **`..\models50\*.engine`（TRT10，固定 batch=1，在 RTX50 机器上重建）** |

`seg_test.cpp`、`segformer_dll.h` 的 C ABI 与原工程一致；`seg_pipeline.*` 与
`trt_engine.*` 已在「batch=1 + 多流」上分叉，**不要再与 `..\segformer_dll` / `..\src`
的副本互相同步**。

### 1.1 为什么固定 batch=1、用多流并发

在 RTX 50（TRT 10.10）上实测该 SegFormer 模型存在 **batch>1 数值错误**：
同一份输入、同一引擎，batch=1 时缺陷类输出 31.9% 前景；batch=2/4/8/16
（动态 profile 或固定 batch）全部被判成背景，logits 系统性偏移（缺陷类均值
下降约 5、背景类上升约 1.3）。FP32/FP16、opset16、固定/动态 batch 均复现，
而 RTX3090 + TRT8.6 的 batch=16 完全正常。

因此本工程：

- 引擎用 **固定 batch=1** 构建（`build_engine50.bat` 已改为 `1x3x640x640`）；
- DLL 用 `numWorkers` 路（默认 4）**独立 execution context + CUDA stream** 并行处理
  滑窗 tile，主线程仍按 tile 原顺序串行累加，保证与 Python / RTX3090 结果逐像素一致；
- 并发路数可用环境变量 `SEGFORMER_WORKERS`（1~16）覆盖，显存不足会自动降级（4→2→1）。

---

## 2. 目标机器（RTX 50）前置条件

1. **TensorRT 10.10.0.31 SDK**（与 `D:\CaoJiaoJie\AIRTX50\gpu_Dll_func` 相同）
   - 默认路径：`D:\Rtx50Test\TensorRT-10.10.0.31`
   - 安装包：`D:\GoogleDownloads\TensorRT-10.10.0.31.Windows.win10.cuda-12.9.zip`
   - 若路径不同，改 `segformer_dll.vcxproj` 的 `<TrtRoot>`
2. **CUDA 12.9 运行时 + cuDNN 9**（TRT10.10 的 Windows 包按 CUDA 12.9 构建）
   - 编译只需要 CUDA 12.x 的头/`cudart.lib`（`$(CUDA_PATH)` 指向 12.1 也可以）
   - **运行时**必须能找到 `cudart64_12.dll`、`cublas64_12.dll`、`cublasLt64_12.dll`、
     `cudnn*_9.dll`。本机可复用 `D:\CaoJiaoJie\AIRTX50\Debug50\` 这一整套；
     目标机若已装 CUDA 12.9 + cuDNN 9，则把脚本里的 `CUDART` 改成对应 `bin` 目录
3. OpenCV 4.8.0（vc16）—— 默认 `E:\opencv_exe\opencv\build`
4. VS2022 v143，x64

> 本工程**不在开发机（RTX 3090）编译验证**，需整体拷到 RTX50 机器上打开。

---

## 3. 拷到 RTX50 机器

最少需要拷贝：

```
deploy_cpp\segformer_dll50\          # 整个工程目录
deploy_cpp\models\*.onnx             # 建引擎用；至少 segformer_sw_s4_b1_640.onnx
                                     # （v2_strat 的 ONNX 在 segformer_v2\runs_strat\v2_strat\）
```

目标机上的 SDK / OpenCV 路径按 §8 修改后，再编译和建引擎。

---

## 4. 编译

用 VS 打开 `segformer_dll50\segformer_dll.sln`，配置选 **Release | x64**，生成解决方案。

产物：

```
segformer_dll50\
├── segformer_Dll_func.dll     <- Release，交付给视觉软件（直接在本目录）
├── segformer_Dll_func.lib
├── segformer_Dll_func.exp
├── segformer_Dll_func.pdb
├── seg_test.exe               <- Release 自测台
├── Debug\                     <- Debug 配置的 DLL/exe 单独放这里，避免误交付
└── obj\                       <- 编译中间文件
```

也可命令行：

```bat
MSBuild segformer_dll.sln /p:Configuration=Release /p:Platform=x64 /v:minimal
```

**注意：先编 C++（本解决方案），C# 层用原 `EVDll_Segformer\build_csharp.bat` 即可。**

---

## 5. 构建 TensorRT 10 引擎（必须在 RTX50 机器上做）

> **必须构建固定 batch=1 的引擎**（脚本已固定 `1x3x640x640`）。batch>1 的引擎在
> RTX50/TRT10 上数值错误，DLL 加载后会全部推成背景。workspace 默认 1024MiB，
> 为 4 路 context 留显存。

```bat
build_engine50.bat sw_s4                    :: 从 ..\models\segformer_sw_s4_b1_640.onnx
build_engine50.bat v2_strat                 :: 从 segformer_v2\runs_strat\v2_strat\*.onnx
build_engine50.bat <model> --onnx <path>    :: 显式指定 ONNX
build_engine50.bat <model> --fp32 --force   :: FP32 / 强制重建
```

输出：`deploy_cpp\models50\segformer_<model>_b1_<fp16|fp32>_640.engine`（b1 = batch 1）

> ⚠️ `.engine` 与 TensorRT 版本 + 显卡绑定，**不能**把 RTX3090 上建的引擎拿到
> RTX50 上用，反之亦然。换卡/升级 TRT 都要重建。
>
> ⚠️ `cemian` / `zhengmian` 等模型如果 `models\` 下没有对应 ONNX，
> 需要先用 `tools\build_engine.bat` 那套导出流程生成 ONNX，再跑本脚本。

---

## 6. 运行自测

```bat
:: 默认 4 路并发
run_seg_test.bat ..\models50\segformer_sw_s4_b1_fp16_640.engine <图> <输出目录> --mask
run_seg_test.bat ..\models50\segformer_sw_s4_b1_fp16_640.engine --eval <val_full.txt>

:: 调并发路数（1~16），例如 6 路
set SEGFORMER_WORKERS=6
run_seg_test.bat ..\models50\segformer_sw_s4_b1_fp16_640.engine <图> <输出目录>
```

`run_seg_test.bat` 会把以下目录加入 PATH：

- 本目录（`segformer_Dll_func.dll` / `seg_test.exe`）
- `%TRT%\lib`、`%TRT%\bin`（`nvinfer_10.dll` 等）
- `%CUDART%`（CUDA 12.9 + cuDNN 9）
- `%OCV%\x64\vc16\bin`

VS 里按 F5 时，`segformer_dll.vcxproj.user` 的 `LocalDebuggerEnvironment`
提供同样的 PATH，并把启动程序设为 `seg_test.exe`。

---

## 7. 部署到视觉软件

把下列文件放进软件目录（例如 `D:\CaoJiaoJie\hmxstudio\00Exe\`）：

| 文件 | 说明 |
|---|---|
| `segformer_Dll_func.dll` | 本工程产物 |
| `EVDll_Segformer.dll` | C# 层，**不需要重编**（接口未变） |
| `nvinfer_10.dll` | TensorRT 10 运行时 |
| `nvinfer_plugin_10.dll` | TensorRT 插件（部分网络需要） |
| `nvinfer_lean_10.dll` / `nvinfer_dispatch_10.dll` / `nvinfer_vc_plugin_10.dll` | TensorRT 10 运行时依赖 |
| `nvinfer_builder_resource_10.dll` | 仅建引擎/反序列化部分场景需要（体积约 2GB） |
| `cudnn*_9.dll`（整套）、`cublas64_12.dll`、`cublasLt64_12.dll`、`cudart64_12.dll` | CUDA 12.9 + cuDNN 9 运行时 |
| `opencv_world480.dll` | OpenCV 4.8 |
| `segformer_<model>_b1_fp16_640.engine` | 在 RTX50 机器上构建的引擎 |

`nvinfer_10.dll` 等不能用旧版 `nvinfer.dll` 替代（文件名/ABI 都不同）。

---

## 8. 需要按机器修改的路径

| 文件 | 变量 | 当前值 |
|---|---|---|
| `segformer_dll.vcxproj` | `TrtRoot` | `D:\Rtx50Test\TensorRT-10.10.0.31` |
| `segformer_dll.vcxproj` / `seg_test.vcxproj` | `OpenCvBuild` | `E:\opencv_exe\opencv\build` |
| `segformer_dll.vcxproj.user` / `seg_test.vcxproj.user` | `CudaRuntime` | `D:\CaoJiaoJie\AIRTX50\Debug50` |
| `run_seg_test.bat` / `build_engine50.bat` | `TRT` / `CUDART` / `OCV` | 同上 |
| `build_engine50.bat` | `PY` | `F:\HymsonSoftWare\MiniForge\envs\yolov10\python.exe`（仅导出 ONNX 时用） |

---

## 9. 验证要点

1. `seg_test.exe` 能 `LoadLibrary` 本目录的 DLL，且 `SegStat` 布局自检为 1840 字节
2. 加载固定 batch=1 引擎：输入 `pixel_values` 应为 `1x3x640x640`、输出 `logits` 为 `1x3x160x160`
3. 单张图推理 `ng/px/comp/box` 与 TRT8 版一致或仅有 fp16 舍入级差异
4. 89 张 `--eval` 的 mIoU 应接近 0.759（TRT 10 选核不同，允许 ±0.001 级差异）
5. 性能：分别设 `SEGFORMER_WORKERS=1/2/4/6` 对比 ms/图；16GB 显存建议 4，若初始化时
   打印「资源不足，并发路数降级」则说明显存/context 不够，可用更小的 workspace 重建引擎
   （`--memPoolSize=workspace:1024MiB`，`build_engine50.bat` 默认已是 1024MiB）
6. 热切换/重复 `SegLoad` 后各 `Forward` 仍正常（context 由 TrtEngine 统一释放）
