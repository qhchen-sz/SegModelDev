# segformer_v2_test —— cemian1103 数据预处理 / 训练 / 导出部署 调试工程（归档+模板）

> **来源**：`D:\qhchen\NewModel\hefeic1\cemian1103`（2026-11-06 拷贝整理）
>
> **定位**：归档 + 可调试用模板。代码、配置、SOP、VS Code 调试配置已就位；
> 数据不随本目录拷贝（调试数据子集约 679MB 仍在 hefeic1 原处），见 §4 数据准备。
>
> **模型**：SegFormer-B1，sw_s4 配方（bg_keep=0.12 / focal_alpha=[0.8,2.0,1.0] /
> focal+dice+lovasz 三损失 / copy-paste 0.5），3 类（background / silie / fanzhe）。

---

## 1. 全流程梳理（cemian1103 实际执行顺序）

| 阶段 | 做什么 | 脚本 | launch.json 配置 |
|---|---|---|---|
| SOP1 数据 | 现场图 + ISAT 标注（json 与图同名同目录） | — | — |
| SOP2 划分 | ISAT JSON → YOLO txt；按类别组合分层 8:2 | `scripts/isat_to_yolo_labels.py` → `scripts/split_stratified_debug.py` | `SOP2 划分…` / `SOP3 划分: split_stratified (debug)` |
| SOP3 切块 | 语义 mask → 640 滑窗切块 + bg_keep 采样；silie 小目标抠取；数据集自检 | `scripts/pipeline/convert_strat_b.py`、`scripts/preprocess_silie_crops.py`、`scripts/validate_dataset.py` | `SOP3 切块/验证…` |
| SOP4 训练 | 早停 5000 epoch / patience 100，先 1 epoch 冒烟再正式 | `scripts/pipeline/train_strat_b.py` | `SOP4 训练: 冒烟/正式 (debug)` |
| SOP5 评估 | PyTorch 全图滑窗评估；C++ 与 Python 结果对比打分 | `scripts/eval_full.py`、`scripts/score_masks.py` | `SOP5 评估…` |
| SOP6 导出 | best → ONNX → TensorRT engine（trtexec fp16） | `scripts/pipeline/export_best_onnx.py`、`scripts/export_engine.py`/`.bat` | `SOP6 导出…` |

详细命令、验收指标、常见问题见 **`SOP_cemian1103.md`**（原样拷贝自 hefeic1）。

## 2. 目录结构

```
segformer_v2_test/
├── README.md                 ← 本文件
├── SOP_cemian1103.md         ← 完整 SOP（9 章：数据→划分→切块→训练→评估→导出部署→验收）
├── config_sw_s4.yaml         ← 正式训练配置【存档：路径仍指 hefeic1，正式训练在 hefeic1 跑】
├── config_debug.yaml         ← 调试配置：所有产物重定向到 debug/，路径已改写为本目录
├── .vscode/
│   └── launch.json           ← 9 个 SOP 阶段调试配置（F5 按阶段单步调试）
├── scripts/
│   ├── isat_to_yolo_labels.py      # ISAT JSON → YOLO txt（带 --report 校验报告）
│   ├── split_stratified_debug.py   # 分层切分调试版（--images/--labels/--out-dir 参数化）
│   ├── preprocess_silie_crops.py   # 抠 silie 小目标（copy-paste 增广用）
│   ├── validate_dataset.py         # 数据集自检（--stage all）
│   ├── eval_full.py                # PyTorch 全图评估（自动把 pipeline/ 加入 sys.path）
│   ├── score_masks.py              # C++ vs Python 掩膜逐像素对比打分
│   ├── export_engine.py/.bat       # trtexec 构建 fp16 engine
│   └── pipeline/                   # sw_s4 管线冻结快照（SOURCE.txt 溯源）
│       ├── train_strat_b.py        # 训练（--config/--tag/--epochs/--no_resume）
│       ├── convert_strat_b.py      # 切块（--config）
│       ├── export_best_onnx.py     # ONNX 导出 + onnxruntime 校验
│       ├── dataset.py / losses.py / sweep_eval.py
│       └── config_strat.yaml       # 【历史快照路径，见 §5 注意 2】
└── debug/                    ← 调试产物目录（运行时生成，需自备数据，见 §4）
    ├── images/ labels/ logs/       # 输入数据子集 + 各阶段报告
    ├── dataset_strat/              # 切块产物 + val_full
    ├── runs_strat/                 # 训练/评估输出
    ├── silie_crops/ export/        # crops / ONNX
    └── train_strat.txt val_strat.txt
```

## 3. 与 hefeic1 原目录的对应关系

| 本目录 | hefeic1 原位置 | 说明 |
|---|---|---|
| `SOP_cemian1103.md` | `hefeic1/cemian1103/SOP_cemian1103.md` | 原样 |
| `config_sw_s4.yaml` | `hefeic1/cemian1103/config_sw_s4.yaml` | 原样（hefeic1 路径存档） |
| `config_debug.yaml` | `hefeic1/cemian1103/config_debug.yaml` | **路径已重写**为本目录 debug/ 布局；model_ckpt 改指 `ADevFolder/segformer_baseline/...`（2026-11-06 已恢复权重） |
| `.vscode/launch.json` | `hefeic1/.vscode/launch.json` | **路径已重写**为 `${workspaceFolder}/segformer_v2_test/...`；score_masks 的 pred/ref 也统一到 debug/runs_strat |
| `scripts/` | `hefeic1/cemian1103/scripts/` | 仅拷贝 1103 用到的 8+8 个文件；**未拷贝**旧 cemian 扩充流程：`*_more.py`×5、`add_more_labels.py`、`split_more.py`、`make_sop.py`、`smoke/train_sw_s4*.bat` |
| 数据与产物 | `hefeic1/cemian1103/{images,labels,dataset_strat,runs_strat,debug}/` | **未拷贝** |

## 4. 在本目录调试的方法

1. 用 **VS Code 打开 `D:\qhchen\NewModel\ADevFolder` 根目录**（launch.json 的
   `${workspaceFolder}` 依赖这一层）；
2. 准备调试数据子集到 `segformer_v2_test/debug/`（二选一）：
   ```bat
   mkdir segformer_v2_test\debug
   :: 方式 A：直接挂载 hefeic1 的调试子集（同 D 盘，无需管理员）
   mklink /J segformer_v2_test\debug\images D:\qhchen\NewModel\hefeic1\cemian1103\debug\images
   :: 方式 B：拷贝（约 679MB，294 对 图+json）
   xcopy /E /I D:\qhchen\NewModel\hefeic1\cemian1103\debug\images segformer_v2_test\debug\images
   ```
   注意：junction 方式下 debug 流程**只读** images，但 split/convert 的产物都写本目录 debug/，不污染 hefeic1；
3. 按 SOP 顺序 F5 各调试配置：SOP2 划分 → SOP3 切块/验证 → SOP4 冒烟(1 epoch) → SOP4 正式 → SOP5 评估 → SOP6 导出；
4. 每个阶段产物和校验报告都在 `debug/` 内，可随时删 `debug/dataset_strat`、`debug/runs_strat` 重来，不碰原始数据。

## 5. 注意事项

1. **训练/切块必须显式 `--config`**：`pipeline/` 是冻结快照，脚本内默认
   `config_strat_b.yaml` 在快照里已改名为 `config_strat.yaml`，不传 --config 会报找不到文件。
2. `pipeline/config_strat.yaml` 里的路径是更老机器的历史存档
   （`C:/Users/hymson/Desktop/newsiliec9/...`），**仅作快照完整性保留**，实际流程永远被显式
   --config 覆盖，不要拿它直接跑。
3. `config_debug.yaml` 的 `model_ckpt` 指向 `ADevFolder/segformer_baseline/models/segformer-b1-ade`
   （2026-11-06 已从 hf-mirror 恢复权重）；hefeic1 原配置指向 newsiliec9 下同名目录，两者内容相同。
4. `eval_full.py` 依赖 `pipeline/sweep_eval.py`，脚本内已自行 `sys.path.insert`，保持
   `scripts/` 与 `scripts/pipeline/` 相对结构即可，勿拆散。
5. Python 环境：`F:\HymsonSoftWare\MiniForge\envs\yolov10\python.exe`（launch.json 已写死）。
6. 正式训练/评估以 hefeic1 原目录与 `SOP_cemian1103.md` 为准；本目录是调试模板，
   改动脚本后如需回同步 hefeic1，注意两处路径差异（§3 表）。

## 6. 快速自检清单

- [ ] VS Code 打开 ADevFolder 根目录后，调试下拉能列出 9 个 SOP 配置
- [ ] debug/images 就位后，F5 `SOP2 划分` 两个配置，debug/ 下生成 labels/ 与 train/val_strat.txt
- [ ] F5 `SOP3 验证: validate_dataset --stage all` 报告无 ERROR
- [ ] F5 `SOP4 冒烟` 1 epoch 跑通（约 1–2 分钟）
- [ ] F5 `SOP6 导出` 生成 ONNX 且 onnxruntime 校验 max|Δ| 正常
