# -*- coding: utf-8 -*-
"""生成超参扫描配置 config_sw_s1..s4.yaml（bg_keep × focal_alpha(bg) 解耦扫描）。
所有配置共用 canonical val_full (dataset_strat/val_full.txt) 保证同 GT。
S2 复用 dataset_strat_b(bg0.40)，S3/S4 复用 dataset_strat(bg0.12)，仅 S1 需新切块。
"""
import yaml
from pathlib import Path

HERE = Path(__file__).parent
ROOT = 'D:/qhchen/NewModel/ADevFolder'

# (id, bg_keep, alpha_bg, dataset_dir, tag)
SWEEPS = [
    ('s1', 0.25, 0.20, f'{ROOT}/segformer_v2/dataset_sw_s1', 'sw_s1'),
    ('s2', 0.40, 0.20, f'{ROOT}/segformer_v2/dataset_strat_b', 'sw_s2'),   # 复用 B 切块
    ('s3', 0.12, 0.40, f'{ROOT}/segformer_v2/dataset_strat',   'sw_s3'),   # 复用基线切块
    ('s4', 0.12, 0.80, f'{ROOT}/segformer_v2/dataset_strat',   'sw_s4'),   # 复用基线切块
]

def base_cfg(bg_keep, alpha_bg, dataset_dir):
    return {
        'project_root': ROOT,
        'train_list': f'{ROOT}/train_strat.txt',
        'val_list':   f'{ROOT}/val_strat.txt',
        'dataset_dir': dataset_dir,
        'out_dir':     f'{ROOT}/segformer_v2/runs_strat',
        'val_full_txt': f'{ROOT}/segformer_v2/dataset_strat/val_full.txt',  # canonical，同 GT
        'num_classes': 3,
        'id2label': {0: 'background', 1: 'silie', 2: 'fanzhe'},
        'tile_size': 640, 'stride': 512,
        'bg_keep_ratio': bg_keep,
        'loss_focal': 1.0, 'loss_dice': 1.0, 'loss_lovasz': 1.0,
        'focal_gamma': 2.0,
        'focal_alpha': [alpha_bg, 2.00, 1.00],
        'copy_paste_prob': 0.5,
        'silie_crops_dir': f'{ROOT}/segformer_v2/silie_crops',
        'model_ckpt': f'{ROOT}/segformer_baseline/models/segformer-b1-ade',
        'epochs': 5000, 'patience': 100, 'batch_size': 16,
        'lr': 0.00006, 'weight_decay': 0.01, 'num_workers': 4, 'seed': 42,
        'eval_every': 1, 'resume': True, 'infer_stride': 320, 'tta': True,
    }

def main():
    for sid, bg, al, ds, tag in SWEEPS:
        cfg = base_cfg(bg, al, ds)
        p = HERE / f'config_sw_{sid}.yaml'
        with open(p, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f'{p.name}: bg_keep={bg} alpha_bg={al} dataset={Path(ds).name} tag={tag}')

if __name__ == '__main__':
    main()
