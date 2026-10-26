# -*- coding: utf-8 -*-
"""
一次性：把 nvidia/segformer-b1-finetuned-ade-512-512 的 .bin 权重转成本地 safetensors，
规避 transformers 4.57 在 torch<2.6 下拒绝加载 .bin 的限制（不升级 torch、不动 yolov10 env）。
转换后模型目录: segformer_baseline/models/segformer-b1-ade
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import torch, json, shutil
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path

REPO = 'nvidia/segformer-b1-finetuned-ade-512-512'
OUT = Path(__file__).parent / 'models' / 'segformer-b1-ade'
OUT.mkdir(parents=True, exist_ok=True)

# 1. 下载 config.json
cfg_path = hf_hub_download(REPO, 'config.json')
shutil.copy(cfg_path, OUT / 'config.json')
print('config.json ->', OUT / 'config.json')

# 2. 下载 pytorch_model.bin
bin_path = hf_hub_download(REPO, 'pytorch_model.bin')
print('downloaded', bin_path)

# 3. torch.load + 转 safetensors（torch 2.5.1 仍可用 weights_only=True 加载纯张量）
from safetensors.torch import save_file
sd = torch.load(bin_path, map_location='cpu', weights_only=True)
# 去掉非张量（如有），并 contiguous
sd = {k: v.contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
save_file(sd, str(OUT / 'model.safetensors'))
print('saved', OUT / 'model.safetensors', f'({len(sd)} tensors)')

# 4. 校验能否被 from_pretrained 本地加载
from transformers import SegformerForSemanticSegmentation
m = SegformerForSemanticSegmentation.from_pretrained(
    str(OUT), num_labels=3, ignore_mismatched_sizes=True)
print('OK: 本地加载成功，classifier head =', m.classifier_weights if hasattr(m,'classifier_weights') else '(reinit)')
print('DONE ->', OUT)
