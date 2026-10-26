# -*- coding: utf-8 -*-
"""Mask2Former 训练（分层重切版）: 读 config_strat.yaml，数据来自 dataset_strat/，
加 patience 早停。绝不改动 train.py。"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import random, yaml, csv, argparse, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from pathlib import Path
from dataset import M2FTileDataset, collate

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def gpu_temp():
    try: return int(os.popen('nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader').read().strip().splitlines()[0])
    except: return 0

def guard_temp(hot=85, cool=72):
    t = gpu_temp()
    if t >= hot:
        print(f'  !! GPU 温度 {t}°C >= {hot}°C，暂停冷却...', flush=True)
        while True:
            t = gpu_temp()
            if t == 0 or t <= cool:
                print(f'  GPU 降至 {t}°C，恢复训练', flush=True); break
            time.sleep(30)

def poly_lr(step, max_step, base, min_lr=0.0):
    return base * (1 - step / max_step) ** 0.9 + min_lr

@torch.no_grad()
def evaluate(model, ds, processor, device, max_tiles=250, bs=16):
    model.eval()
    n = min(max_tiles, len(ds))
    idxs = sorted(set(list(range(0, len(ds), max(1, len(ds) // n)))[:n]))
    sub = torch.utils.data.Subset(ds, idxs)
    dl = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=4,
                    collate_fn=collate, pin_memory=True)
    inter = np.zeros(3, dtype=np.float64); union = np.zeros(3, dtype=np.float64)
    sub_pos = 0
    for batch in dl:
        pv = batch['pixel_values'].to(device, non_blocking=True)
        pm = batch['pixel_mask'].to(device, non_blocking=True)
        out = model(pixel_values=pv, pixel_mask=pm)
        H, W = pv.shape[-2], pv.shape[-1]
        res = processor.post_process_instance_segmentation(out, target_sizes=[(H, W)] * pv.shape[0],
            threshold=0.5, mask_threshold=0.5)
        for b in range(pv.shape[0]):
            pred = np.zeros((H, W), dtype=np.uint8)
            seg = res[b]['segmentation'].cpu().numpy()
            for info in res[b]['segments_info']:
                pred[seg == info['id']] = int(info['label_id']) + 1
            real_idx = idxs[sub_pos]; sub_pos += 1
            stem = Path(ds.files[real_idx]).stem
            c = torch.load(ds.cache_dir / f'{stem}.pt', weights_only=False)
            gm = np.asarray(c['masks']); gc = np.asarray(c['classes'])
            gt = np.zeros((H, W), dtype=np.uint8)
            for m, cl in zip(gm, gc):
                gt[m > 0] = int(cl) + 1
            for cc in range(3):
                inter[cc] += ((pred == cc) & (gt == cc)).sum()
                union[cc] += ((pred == cc) | (gt == cc)).sum()
    ious = inter / (union + 1e-9)
    return ious.mean(), ious

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full_strat')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config_strat.yaml', encoding='utf-8'))
    if a.epochs is not None: cfg['epochs'] = a.epochs
    if a.batch_size is not None: cfg['batch_size'] = a.batch_size
    patience = cfg.get('patience', 100)

    set_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data = Path(cfg['dataset_dir'])
    out = Path(cfg['out_dir']) / a.tag; out.mkdir(parents=True, exist_ok=True)
    (out / 'run_config.json').write_text(json.dumps({'tag': a.tag, 'lr': cfg['lr'],
        'epochs': cfg['epochs'], 'batch_size': cfg['batch_size'], 'patience': patience},
        ensure_ascii=False, indent=2), encoding='utf-8')

    num_labels = cfg['num_labels']
    id2label = {int(k): v for k, v in cfg['id2label'].items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    train_ds = M2FTileDataset(data / 'train', train=True, tile=cfg['tile_size'])
    val_ds = M2FTileDataset(data / 'val', train=False, tile=cfg['tile_size'])
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                          num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
                          collate_fn=collate, persistent_workers=(cfg['num_workers']>0))
    print(f'[tag={a.tag}] train tiles={len(train_ds)} val tiles={len(val_ds)} '
          f'lr={cfg["lr"]} bs={cfg["batch_size"]} epochs={cfg["epochs"]} patience={patience}')

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg['model_ckpt'], num_labels=num_labels, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = GradScaler('cuda')
    proc = Mask2FormerImageProcessor.from_pretrained(cfg['model_ckpt'])

    total_steps = len(train_dl) * cfg['epochs']
    step = 0; best_miou = 0.0; best_epoch = -1; start_epoch = 0
    log_csv = out / 'train_log.csv'
    ckpt = out / 'ckpt.pt'
    if ckpt.exists():
        d = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(d['model']); optim.load_state_dict(d['optim'])
        scaler.load_state_dict(d['scaler']); start_epoch = d['epoch'] + 1
        best_miou = d['best_miou']; step = d['step']; best_epoch = d.get('best_epoch', -1)
        print(f'[{a.tag}] 续训 from epoch {start_epoch}, best={best_miou:.4f}@ep{best_epoch}, step={step}', flush=True)
    if not log_csv.exists():
        with open(log_csv, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['epoch','train_loss','val_miou','iou_bg','iou_silie','iou_fanzhe','lr'])

    for epoch in range(start_epoch, cfg['epochs']):
        guard_temp()
        model.train()
        ep_loss = 0.0; t0 = time.time()
        for batch in train_dl:
            pv = batch['pixel_values'].to(device, non_blocking=True)
            pm = batch['pixel_mask'].to(device, non_blocking=True)
            ml = [m.to(device) for m in batch['mask_labels']]
            cl = [c.to(device) for c in batch['class_labels']]
            lr = poly_lr(step, total_steps, cfg['lr'])
            for pg in optim.param_groups: pg['lr'] = lr
            optim.zero_grad(set_to_none=True)
            with autocast('cuda'):
                outputs = model(pixel_values=pv, pixel_mask=pm, mask_labels=ml, class_labels=cl)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            ep_loss += loss.item(); step += 1
        ep_loss /= max(1, len(train_dl))
        miou, ious = evaluate(model, val_ds, proc, device)
        with open(log_csv, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, f'{ep_loss:.4f}', f'{miou:.4f}',
                         f'{ious[0]:.4f}', f'{ious[1]:.4f}', f'{ious[2]:.4f}', f'{lr:.2e}'])
        print(f'[{a.tag}] ep{epoch:3d} loss={ep_loss:.4f} mIoU={miou:.4f} '
              f'[bg={ious[0]:.3f} silie={ious[1]:.3f} fanzhe={ious[2]:.3f}] '
              f'lr={lr:.2e} ({time.time()-t0:.0f}s)', flush=True)
        if miou > best_miou:
            best_miou = miou; best_epoch = epoch
            model.save_pretrained(out / 'best'); proc.save_pretrained(out / 'best')
            torch.save({'epoch': epoch, 'miou': float(miou)}, out / 'best_meta.pt')
        torch.save({'model': model.state_dict(), 'optim': optim.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch,
                    'best_miou': best_miou, 'best_epoch': best_epoch, 'step': step}, ckpt)
        if patience > 0 and best_epoch >= 0 and (epoch - best_epoch) >= patience:
            print(f'[{a.tag}] early stop @ep{epoch} (no improve since ep{best_epoch}, patience={patience})', flush=True)
            break
    (out / 'final_miou.txt').write_text(f'{best_miou:.4f}', encoding='utf-8')
    print(f'[{a.tag}] done. best mIoU={best_miou:.4f}@ep{best_epoch}')

if __name__ == '__main__':
    main()
