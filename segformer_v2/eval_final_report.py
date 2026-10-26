# -*- coding: utf-8 -*-
"""三模型 (S4 / v2-fast / YOLO) 在 val_strat 验证集上的 per-class IoU + recall + precision + mIoU。
口径: S4 & v2-fast = bs16 fp16 stride448 noTTA 全图滑窗; YOLO = imgsz1200 conf0.25 iou0.5。
GT: segformer_v2/dataset_strat/val_full.txt (88 图全图 mask)。
输出: 项目根目录 final_report.md。
"""
import os
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
import time, yaml
import numpy as np, cv2, torch
import torch.nn.functional as F
from torch.amp import autocast
from transformers import SegformerForSemanticSegmentation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL_FULL = ROOT / 'segformer_v2' / 'dataset_strat' / 'val_full.txt'
OUT_MD = ROOT / 'final_report.md'
S4_BEST = ROOT / 'segformer_v2' / 'runs_strat' / 'sw_s4' / 'best'
V2_BEST = ROOT / 'segformer_v2' / 'runs_strat' / 'v2_strat' / 'best'
YOLO_BEST = ROOT / 'runs_strat' / 'train_strat' / 'weights' / 'best.pt'

MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
TILE,STRIDE,BATCH=640,448,16
CLS=['background','silie','fanzhe']
dev='cuda'

def load_val():
    out=[]
    for line in open(VAL_FULL,encoding='utf-8'):
        line=line.rstrip('\n')
        if line:
            a,b=line.split('\t'); out.append((a,b))
    return out

def collect(img):
    H,W=img.shape[:2]
    ys=list(range(0,max(1,H-TILE+1),STRIDE))
    if ys[-1]+TILE<H: ys.append(max(0,H-TILE))
    xs=list(range(0,max(1,W-TILE+1),STRIDE))
    if xs[-1]+TILE<W: xs.append(max(0,W-TILE))
    r=[]
    for y0 in ys:
        for x0 in xs:
            p=img[y0:y0+TILE,x0:x0+TILE]; ph,pw=p.shape[:2]
            if ph<TILE or pw<TILE: p=cv2.copyMakeBorder(p,0,TILE-ph,0,TILE-pw,cv2.BORDER_REFLECT)
            r.append((y0,x0,p,ph,pw))
    return r,H,W

def seg_predict(m,img):
    tiles,H,W=collect(img); acc=None; cov=np.zeros((H,W),np.float32)
    for i in range(0,len(tiles),BATCH):
        b=tiles[i:i+BATCH]
        a=[p[:,:,::-1].astype(np.float32)/255.0 for _,_,p,_,_ in b]; a=[(x-MEAN)/STD for x in a]
        t=torch.from_numpy(np.stack(a,0)).permute(0,3,1,2).to(dev).float()
        with torch.no_grad():
            with autocast('cuda',dtype=torch.float16): lg=m(pixel_values=t).logits
        lg=F.interpolate(lg.float(),size=(TILE,TILE),mode='bilinear',align_corners=False).cpu().numpy()
        if acc is None: acc=np.zeros((lg.shape[1],H,W),np.float32)
        for j,(y0,x0,_,ph,pw) in enumerate(b):
            acc[:,y0:y0+ph,x0:x0+pw]+=lg[j][:,:ph,:pw]; cov[y0:y0+ph,x0:x0+pw]+=1
    cov=np.maximum(cov,1e-6); return (acc/cov[None]).argmax(0).astype(np.uint8)

def yolo_predict(ym,img):
    H,W=img.shape[:2]; res=ym(img,imgsz=1200,conf=0.25,iou=0.5,verbose=False)[0]
    sem=np.zeros((H,W),np.uint8)
    if res.masks is not None:
        cls=res.boxes.cls.cpu().numpy().astype(int); cf=res.boxes.conf.cpu().numpy()
        for k in np.argsort(cf):
            pl=res.masks.xy[k]
            if pl is None or len(pl)<3: continue
            cv2.fillPoly(sem,[np.round(pl).astype(np.int32)],int(cls[k]))
    return sem

def evaluate(pred_fn, val, warmup_img):
    tp=np.zeros(3,np.int64); gtpx=np.zeros(3,np.int64); predpx=np.zeros(3,np.int64)
    ts=[]
    _=pred_fn(warmup_img); torch.cuda.synchronize()
    for ip,mp in val:
        img=cv2.imread(ip); gt=cv2.imread(mp,cv2.IMREAD_UNCHANGED)
        torch.cuda.synchronize(); t0=time.time(); pred=pred_fn(img); torch.cuda.synchronize()
        ts.append(time.time()-t0)
        for c in range(3):
            tp[c]+=((pred==c)&(gt==c)).sum(); gtpx[c]+=(gt==c).sum(); predpx[c]+=(pred==c).sum()
    iou=tp/np.maximum(gtpx+predpx-tp,1)
    rec=tp/np.maximum(gtpx,1)
    prec=tp/np.maximum(predpx,1)
    return iou,rec,prec,float(np.mean(ts))*1000

def main():
    val=load_val(); print(f'val_strat: {len(val)} 图')
    results={}
    # S4 + v2-fast (segformer)
    for name,path in [('S4',S4_BEST),('v2-fast',V2_BEST)]:
        print(f'评估 {name} ...',flush=True)
        m=SegformerForSemanticSegmentation.from_pretrained(str(path)).to(dev).eval()
        results[name]=evaluate(lambda im:seg_predict(m,im), val, cv2.imread(val[0][0]))
        del m; torch.cuda.empty_cache()
    # yolo
    print('评估 YOLO ...',flush=True)
    from ultralytics import YOLO
    ym=YOLO(str(YOLO_BEST))
    results['YOLO']=evaluate(lambda im:yolo_predict(ym,im), val, cv2.imread(val[0][0]))

    order=['S4','v2-fast','YOLO']
    L=['# 三模型在 val_strat 验证集上的指标对比\n',
       f'- 验证集: `val_strat.txt`（{len(val)} 张全图，含 GT）',
       '- 口径: S4 & v2-fast = bs16 fp16 stride448 noTTA 全图滑窗; YOLO = imgsz1200 conf0.25 iou0.5',
       '- 指标为像素级 per-class：IoU = TP/(TP+FP+FN)，Recall = TP/(TP+FN)，Precision = TP/(TP+FP)\n',
       '## per-class IoU\n',
       '| 类别 | S4 | v2-fast | YOLO |', '|---|---|---|---|']
    for c in range(3):
        L.append(f'| {CLS[c]} | {results["S4"][0][c]:.4f} | {results["v2-fast"][0][c]:.4f} | {results["YOLO"][0][c]:.4f} |')
    L.append(f'| **mIoU** | **{results["S4"][0].mean():.4f}** | **{results["v2-fast"][0].mean():.4f}** | **{results["YOLO"][0].mean():.4f}** |')
    L.append('\n## per-class Recall\n')
    L.append('| 类别 | S4 | v2-fast | YOLO |'); L.append('|---|---|---|---|')
    for c in range(3):
        L.append(f'| {CLS[c]} | {results["S4"][1][c]:.4f} | {results["v2-fast"][1][c]:.4f} | {results["YOLO"][1][c]:.4f} |')
    L.append(f'| **mRecall** | **{results["S4"][1].mean():.4f}** | **{results["v2-fast"][1].mean():.4f}** | **{results["YOLO"][1].mean():.4f}** |')
    L.append('\n## per-class Precision\n')
    L.append('| 类别 | S4 | v2-fast | YOLO |'); L.append('|---|---|---|---|')
    for c in range(3):
        L.append(f'| {CLS[c]} | {results["S4"][2][c]:.4f} | {results["v2-fast"][2][c]:.4f} | {results["YOLO"][2][c]:.4f} |')
    L.append(f'| **mPrecision** | **{results["S4"][2].mean():.4f}** | **{results["v2-fast"][2].mean():.4f}** | **{results["YOLO"][2].mean():.4f}** |')
    L.append('\n## 推理速度（RTX 3090，全图 ~2601×3001）\n')
    L.append('| 模型 | 平均 ms/图 |'); L.append('|---|---|')
    for n in order: L.append(f'| {n} | {results[n][3]:.1f} |')
    (OUT_MD).write_text('\n'.join(L),encoding='utf-8')
    print('\n'.join(L)); print('\n->',OUT_MD)

if __name__=='__main__':
    main()
