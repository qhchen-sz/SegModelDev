# -*- coding: utf-8 -*-
"""冒烟测试：2 epoch 训练 + 1 张全图推理，验证全链路可跑。"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')  # 国内镜像，避免连不上 huggingface.co
import yaml, torch, numpy as np, cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
import dataset as D
import infer as I

def main():
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config.yaml', encoding='utf-8'))
    cfg['epochs'] = 2; cfg['batch_size'] = 4; cfg['num_workers'] = 2
    device = 'cuda'
    base = Path(cfg['project_root']) / 'segformer_baseline'
    data = base / 'dataset'; out = Path(cfg['out_dir']); out.mkdir(parents=True, exist_ok=True)
    nc = cfg['num_classes']
    id2label = {int(k): v for k, v in cfg['id2label'].items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    # 取前 8 个 train 块做最快冒烟
    ds = D.SegTileDataset(data / 'train', train=True, tile=cfg['tile_size'])
    ds.files = ds.files[:8]; ds.class_pixels = ds.class_pixels[:8]
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg['model_ckpt'], num_labels=nc, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(cfg['class_weights'], device=device))
    scaler = GradScaler('cuda')
    model.train()
    for ep in range(2):
        for img, mask in dl:
            img, mask = img.to(device), mask.to(device)
            with autocast('cuda'):
                lg = model(pixel_values=img).logits
                lg = F.interpolate(lg, size=mask.shape[-2:], mode='bilinear', align_corners=False)
                loss = ce(lg, mask)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
            print(f'  ep{ep} loss={loss.item():.4f}  logits={tuple(lg.shape)}')
    out_dir = out / 'best'; model.save_pretrained(out_dir)
    print('train smoke OK, saved to', out_dir)

    # 推理 1 张
    lines = I.load_val_full(str(base / 'dataset' / 'val_full.txt'))
    ip, mp = lines[0]
    img = cv2.imread(ip); gt = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
    model.eval()
    pred = I.sliding_predict(model, img, cfg['tile_size'], cfg['infer_stride'], device)
    a, b = I.iou(pred, gt, nc); ious = a / (b + 1e-9)
    ov = I.overlay(img, pred)
    cv2.imwrite(str(out / 'smoke_overlay.png'), ov)
    print(f'infer smoke OK: shape={pred.shape} unique={np.unique(pred)}')
    print(f'  IoU bg={ious[0]:.3f} silie={ious[1]:.3f} fanzhe={ious[2]:.3f} mIoU={ious.mean():.3f}')
    print('overlay ->', out / 'smoke_overlay.png')

if __name__ == '__main__':
    main()
