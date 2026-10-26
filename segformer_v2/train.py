# -*- coding: utf-8 -*-
"""
SegFormer v2 训练: Focal+Dice+Lovász 损失 + Copy-paste + 640 + 断点续训 + 中途温度检查 + 每 epoch 评估存 best。
模型: MiT-B1 + SegFormerHead (复用 segformer_baseline 本地 b1 权重)
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import random, math, yaml, csv, argparse, time, json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
from dataset import SegV2Dataset
from losses import V2Loss

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def gpu_temp():
    try: return int(os.popen('nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader').read().strip().splitlines()[0])
    except: return 0

def guard_temp(hot=85, cool=72):
    t = gpu_temp()
    if t >= hot:
        print(f'  !! GPU {t}°C >= {hot}°C, 暂停冷却...', flush=True)
        while True:
            t = gpu_temp()
            if t == 0 or t <= cool:
                print(f'  GPU 降至 {t}°C, 恢复', flush=True); break
            time.sleep(20)

def poly_lr(step, max_step, base, min_lr=0.0):
    return base * (1 - step / max_step) ** 0.9 + min_lr

@torch.no_grad()
def evaluate(model, loader, device, nc):
    model.eval()
    inter = np.zeros(nc, dtype=np.float64); union = np.zeros(nc, dtype=np.float64)
    for img, mask in loader:
        img = img.to(device, non_blocking=True, memory_format=torch.channels_last)
        mask = mask.to(device, non_blocking=True)
        out = model(pixel_values=img).logits
        out = F.interpolate(out, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        pred = out.argmax(1)
        for c in range(nc):
            inter[c] += ((pred == c) & (mask == c)).sum().item()
            union[c] += ((pred == c) | (mask == c)).sum().item()
    ious = inter / (union + 1e-9)
    return ious.mean(), ious

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v2')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--no_resume', action='store_true')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config.yaml', encoding='utf-8'))
    if a.epochs is not None: cfg['epochs'] = a.epochs
    if a.batch_size is not None: cfg['batch_size'] = a.batch_size
    resume = cfg.get('resume', True) and not a.no_resume

    set_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    base = Path(cfg['project_root']) / 'segformer_v2'
    data = base / 'dataset'
    out = Path(cfg['out_dir']) / a.tag; out.mkdir(parents=True, exist_ok=True)
    nc = cfg['num_classes']
    id2label = {int(k): v for k, v in cfg['id2label'].items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    train_ds = SegV2Dataset(data / 'train', cfg['silie_crops_dir'], train=True,
                            tile=cfg['tile_size'], cp_prob=cfg['copy_paste_prob'])
    val_ds = SegV2Dataset(data / 'val', cfg['silie_crops_dir'], train=False, tile=cfg['tile_size'])
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                          num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
                          persistent_workers=(cfg['num_workers']>0))
    val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                       num_workers=cfg['num_workers'], pin_memory=True,
                       persistent_workers=(cfg['num_workers']>0))
    print(f'[v2] train tiles={len(train_ds)} val tiles={len(val_ds)} bs={cfg["batch_size"]} epochs={cfg["epochs"]}')

    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg['model_ckpt'], num_labels=nc, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)
    if device == 'cuda':
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = GradScaler('cuda')
    criterion = V2Loss(cfg, device)

    total_steps = len(train_dl) * cfg['epochs']
    step = 0; best_miou = 0.0; start_epoch = 0
    ckpt = out / 'ckpt.pt'
    if resume and ckpt.exists():
        d = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(d['model']); optim.load_state_dict(d['optim'])
        scaler.load_state_dict(d['scaler']); start_epoch = d['epoch'] + 1
        best_miou = d['best_miou']; step = d['step']
        print(f'[v2] 续训 from epoch {start_epoch}, best={best_miou:.4f}, step={step}')
    log_csv = out / 'train_log.csv'
    if not log_csv.exists():
        with open(log_csv, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['epoch','train_loss','val_miou','iou_bg','iou_silie','iou_fanzhe','lr'])

    for epoch in range(start_epoch, cfg['epochs']):
        guard_temp()
        model.train()
        ep_loss = 0.0; t0 = time.time(); nb = 0
        for img, mask in train_dl:
            img = img.to(device, non_blocking=True, memory_format=torch.channels_last)
            mask = mask.to(device, non_blocking=True)
            lr = poly_lr(step, total_steps, cfg['lr'])
            for pg in optim.param_groups: pg['lr'] = lr
            optim.zero_grad(set_to_none=True)
            with autocast('cuda'):
                lg = model(pixel_values=img).logits
                lg = F.interpolate(lg, size=mask.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(lg, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            ep_loss += loss.item(); step += 1; nb += 1
            if nb % 50 == 0:
                guard_temp()  # 训练中中途温度检查
        ep_loss /= max(1, len(train_dl))
        miou, ious = evaluate(model, val_dl, device, nc)
        with open(log_csv, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, f'{ep_loss:.4f}', f'{miou:.4f}',
                         f'{ious[0]:.4f}', f'{ious[1]:.4f}', f'{ious[2]:.4f}', f'{lr:.2e}'])
        print(f'[v2] ep{epoch:3d} loss={ep_loss:.4f} mIoU={miou:.4f} '
              f'[bg={ious[0]:.3f} silie={ious[1]:.3f} fanzhe={ious[2]:.3f}] '
              f'lr={lr:.2e} ({time.time()-t0:.0f}s)', flush=True)
        if miou > best_miou:
            best_miou = miou
            model.save_pretrained(out / 'best')
            torch.save({'epoch': epoch, 'miou': float(miou)}, out / 'best_meta.pt')
        # 续训 checkpoint（每 epoch 存，防卡死丢进度）
        torch.save({'model': model.state_dict(), 'optim': optim.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch,
                    'best_miou': best_miou, 'step': step}, ckpt)
    (out / 'final_miou.txt').write_text(f'{best_miou:.4f}', encoding='utf-8')
    print(f'[v2] done. best mIoU={best_miou:.4f}')

if __name__ == '__main__':
    main()
