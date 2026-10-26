# 极耳缺陷分割模型研究工程

本工程用于探索极耳撕裂（`silie`）与翻折（`fanzhe`）缺陷分割模型，并对 YOLOv8n-seg、Mask2Former、SegFormer 进行统一口径比较。当前工程根目录为：

```text
D:\qhchen\NewModel\ADevFolder
```

## 当前结论

在 89 张 `val_strat` 全图验证集上，SegFormer `sw_s4` 的综合效果最好：

| 模型 | mIoU | 平均耗时 |
|---|---:|---:|
| YOLOv8n-seg | 0.5205 | 19.2 ms/图 |
| SegFormer v2-fast | 0.7531 | 871.1 ms/图 |
| **SegFormer sw_s4** | **0.7593** | 871.9 ms/图 |
| **SegFormer sw_s4 + TensorRT FP16** | **0.7594** | **155.2 ms/图** |

`sw_s4` 同时保留住负样本假阳下降约 82% 的关键优势。详细指标见 `final_report.md`、部署精度与速度见 `deploy_cpp/README.md`。

## 目录结构

| 路径 | 内容 |
|---|---|
| `newsiliec9*.yaml` | YOLO 数据配置 |
| `split_*.py`、`filter_train_empty.py` | 数据划分与过滤脚本 |
| `train*.txt`、`val*.txt` | 小型训练/验证划分列表 |
| `yolo_train_strat.py` | YOLOv8n-seg 分层重切训练入口 |
| `segformer_baseline/` | SegFormer-B1 基线训练、评估与权重转换 |
| `segformer_v2/` | Focal + Dice + Lovász、Copy-Paste、超参扫描和导出脚本 |
| `mask2former_baseline/` | Mask2Former-Swin-Tiny 对比实验 |
| `deploy_cpp/` | C++ / TensorRT / C# 部署工程 |
| `best_model/sw_s4/` | 当前最佳配置与小型评估指标；权重不提交 |

## 版本控制策略

仓库只提交代码、工程配置、数据划分列表、报告和可复盘的小型指标文件。以下内容不提交，需要时由脚本生成或重新下载：

- 训练图片、标注文件、切块数据集和缓存；
- 训练输出目录、长日志、可视化结果；
- `*.pt`、`*.pth`、`*.ckpt`、`*.safetensors` 模型权重；
- `*.onnx`、`*.engine` 部署模型；
- DLL、EXE、PDB 等编译产物。

首次使用或克隆后，需要自行准备原始图片和标注，再按下面步骤恢复权重。

## 权重恢复

### SegFormer B1 预训练权重

```bash
python segformer_baseline/convert_weights.py
```

脚本会从 Hugging Face 下载 `nvidia/segformer-b1-finetuned-ade-512-512`，并转换到：

```text
segformer_baseline/models/segformer-b1-ade/
```

### Mask2Former 预训练权重

```bash
huggingface-cli download facebook/mask2former-swin-tiny-coco-instance \
  --local-dir mask2former_baseline/models/m2f-swin-tiny-coco-instance
```

### YOLO 预训练权重

`yolo_train_strat.py` 首次运行时会读取根目录下的 `yolov8n-seg.pt`；文件不存在时可由 Ultralytics 自动获取，也可手动下载后放到工程根目录。

### 最佳模型与部署引擎

最佳业务权重需重新训练生成。训练完成后构建 ONNX 和 TensorRT 引擎：

```bat
cd deploy_cpp
tools\build_engine.bat sw_s4
```

引擎与 TensorRT 版本、GPU 绑定，换机器后必须重建；完整步骤见 `deploy_cpp/SOP_ENGINE_EXPORT.md`。

## 常用入口

### 公平比较全流程

准备数据和预训练权重后运行：

```bash
python segformer_v2/run_strat_pipeline.py
```

该流程会执行分层划分、三套切块、Mask2Former 缓存、四组训练、统一评估和汇总。

### SegFormer 基线

```bash
cd segformer_baseline
python convert_weights.py   # 首次
python convert.py           # 数据切块
python train.py --tag full
python infer.py --run full
```

### 当前最佳配置 sw_s4

```bash
cd segformer_v2
python convert_strat_b.py --config config_sw_s4.yaml
python train_strat_b.py --config config_sw_s4.yaml --tag sw_s4
python sweep_eval.py --config config_sw_s4.yaml --run sw_s4
```

### C++ / TensorRT 部署

```bat
cd deploy_cpp
tools\build_engine.bat sw_s4
build.bat
run.bat --model sw_s4 --input <图片或目录> --out <输出目录>
```

更多部署、C# 调用和跨机器迁移说明见 `deploy_cpp/README.md`。

## 数据准备

当前划分列表指向工程根目录下的 `images/`。请按原始工程布局放置图片与标注，或重新运行划分和切块脚本生成自己的数据目录。训练数据和生成产物均被 `.gitignore` 排除，不会进入 GitHub 仓库。
