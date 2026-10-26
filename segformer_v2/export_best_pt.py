# -*- coding: utf-8 -*-
"""把 v2 训练最优权重(best/ 目录)打包成单个 .pt 文件（类似 YOLO best.pt）。
内容: {state_dict, config, id2label, label2id, num_labels}，单文件可 torch.load。
v2 与 v2-fast 共用此权重（v2-fast 仅推理不同）。
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import torch, yaml, argparse
from pathlib import Path
from transformers import SegformerForSemanticSegmentation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None, help='best 目录，默认读 config_strat 的 v2_strat/best')
    ap.add_argument('--out', default=None, help='输出 .pt 路径')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config_strat.yaml', encoding='utf-8'))
    runs = Path(cfg['out_dir'])
    src = Path(a.src) if a.src else runs / 'v2_strat' / 'best'
    out = Path(a.out) if a.out else runs / 'v2_strat' / 'segformer_v2_strat_best.pt'
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'加载: {src}')
    model = SegformerForSemanticSegmentation.from_pretrained(src)
    pkg = {
        'state_dict': model.state_dict(),
        'config': model.config.to_dict(),
        'id2label': cfg['id2label'],
        'num_classes': cfg['num_classes'],
        'arch': 'segformer-b1',
        'note': 'v2_strat best (Focal+Dice+Lovasz, 640, copy-paste silie). v2 与 v2-fast 共用此权重。',
    }
    torch.save(pkg, out)
    sz = out.stat().st_size / 1024 / 1024
    print(f'已保存单文件: {out}  ({sz:.1f} MB)')
    print(f'加载示例:')
    print(f"  pkg = torch.load(r'{out}', map_location='cpu', weights_only=False)")
    print(f"  from transformers import SegformerConfig, SegformerForSemanticSegmentation")
    print(f"  cfg2 = SegformerConfig.from_dict(pkg['config'])")
    print(f"  model = SegformerForSemanticSegmentation(cfg2)")
    print(f"  model.load_state_dict(pkg['state_dict'])")

if __name__ == '__main__':
    main()
