# -*- coding: utf-8 -*-
"""v2-fast 口径全图评估 + 背景假阳统计（对比基线 v2_strat 与 B方案 v2b_strat）。
在 val_full_strat 上做 bs16 fp16 stride448 noTTA 全图滑窗推理：
输出 整体 IoU + GT负样本(全0 mask)上的假阳统计 + 计时。
用法: python eval_strat_b.py --config config_strat.yaml --run v2_strat
      python eval_strat_b.py --config config_strat_b.yaml --run v2b_strat
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import time, yaml, argparse
import numpy as np
import cv2, torch
import torch.nn.functional as F
from torch.amp import autocast
from transformers import SegformerForSemanticSegmentation
from pathlib import Path

MEAN = np.array([0.485,0.456,0.406],dtype=np.float32); STD = np.array([0.229,0.224,0.225],dtype=np.float32)
TILE, STRIDE, BATCH = 640, 448, 16

def load_val_full(txt):
    out=[]
    for line in open(txt,encoding='utf-8'):
        line=line.rstrip('\n')
        if line: out.append(line.split('\t'))
    return out

def collect(img,tile,stride):
    H,W=img.shape[:2]
    ys=list(range(0,max(1,H-tile+1),stride))
    if ys[-1]+tile<H: ys.append(max(0,H-tile))
    xs=list(range(0,max(1,W-tile+1),stride))
    if xs[-1]+tile<W: xs.append(max(0,W-tile))
    t=[]
    for y0 in ys:
        for x0 in xs:
            p=img[y0:y0+tile,x0:x0+tile]; ph,pw=p.shape[:2]
            if ph<tile or pw<tile: p=cv2.copyMakeBorder(p,0,tile-ph,0,tile-pw,cv2.BORDER_REFLECT)
            t.append((y0,x0,p,ph,pw))
    return t,H,W

def fwd(model,ps,device):
    a=[p[:,:,::-1].astype(np.float32)/255.0 for p in ps]; a=[(x-MEAN)/STD for x in a]
    t=torch.from_numpy(np.stack(a,0)).permute(0,3,1,2).to(device).float()
    with torch.no_grad():
        with autocast('cuda',dtype=torch.float16): lg=model(pixel_values=t).logits
    return F.interpolate(lg.float(),size=(TILE,TILE),mode='bilinear',align_corners=False).cpu().numpy()

def predict(model,img,device):
    tiles,H,W=collect(img,TILE,STRIDE); acc=None; cov=np.zeros((H,W),np.float32)
    for i in range(0,len(tiles),BATCH):
        b=tiles[i:i+BATCH]; lg=fwd(model,[x[2] for x in b],device)
        if acc is None: acc=np.zeros((lg.shape[1],H,W),np.float32)
        for j,(y0,x0,_,ph,pw) in enumerate(b):
            acc[:,y0:y0+ph,x0:x0+pw]+=lg[j][:,:ph,:pw]; cov[y0:y0+ph,x0:x0+pw]+=1
    cov=np.maximum(cov,1e-6); return (acc/cov[None]).argmax(0).astype(np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', default='config_strat.yaml')
    ap.add_argument('--run', default='v2_strat')
    a=ap.parse_args()
    cfg=yaml.safe_load(open(Path(__file__).parent/a.config,encoding='utf-8'))
    device='cuda'
    out=Path(cfg['out_dir'])/a.run
    model=SegformerForSemanticSegmentation.from_pretrained(out/'best').to(device).eval()
    lines=load_val_full(str(cfg['val_full_txt']))
    inter=np.zeros(3,np.float64); union=np.zeros(3,np.float64)
    neg_fp=[]; times=[]; n_pos=0; pos_silie_i=pos_silie_u=pos_fan_i=pos_fan_u=0
    _=predict(model,cv2.imread(lines[0][0]),device); torch.cuda.synchronize()
    print(f'[{a.run}] fast eval {len(lines)} imgs ...',flush=True)
    for i,(ip,mp) in enumerate(lines):
        img=cv2.imread(ip); gt=cv2.imread(mp,cv2.IMREAD_UNCHANGED)
        torch.cuda.synchronize(); t0=time.time()
        pred=predict(model,img,device); torch.cuda.synchronize(); times.append(time.time()-t0)
        for c in range(3):
            inter[c]+=((pred==c)&(gt==c)).sum(); union[c]+=((pred==c)|(gt==c)).sum()
        neg = (gt>0).sum()==0
        if neg:
            fp=int((pred>0).sum()); neg_fp.append(fp)
        else:
            n_pos+=1
            pos_silie_i+=((pred==1)&(gt==1)).sum(); pos_silie_u+=((pred==1)|(gt==1)).sum()
            pos_fan_i+=((pred==2)&(gt==2)).sum(); pos_fan_u+=((pred==2)|(gt==2)).sum()
        if i<3 or i%20==0: print(f'  [{i+1}/{len(lines)}] {Path(ip).name} {times[-1]*1000:.0f}ms neg={neg} fp={int((pred>0).sum()) if neg else "-"}',flush=True)
    ious=inter/(union+1e-9); avg_ms=float(np.mean(times))*1000
    neg_fp=np.array(neg_fp);
    n_neg=len(neg_fp)
    fp_img_100=int((neg_fp>100).sum()); fp_img_1000=int((neg_fp>1000).sum())
    L=[]
    L.append(f'=== {a.run} fast 全图评估 (bs16 fp16 stride448 noTTA, {len(lines)} 图) ===')
    L.append(f'整体: mIoU={ious.mean():.4f}  bg={ious[0]:.4f} silie={ious[1]:.4f} fanzhe={ious[2]:.4f}')
    L.append(f'-- 仅正样本图({n_pos}张)缺陷IoU: silie={pos_silie_i/(pos_silie_u+1e-9):.4f} fanzhe={pos_fan_i/(pos_fan_u+1e-9):.4f}')
    L.append(f'-- GT负样本(全0 mask, {n_neg}张)假阳统计:')
    L.append(f'     平均假阳像素/图 = {neg_fp.mean():.1f}  中位={np.median(neg_fp):.0f}  最大={neg_fp.max():.0f}')
    L.append(f'     假阳>100px 的图: {fp_img_100}/{n_neg} ({fp_img_100/max(n_neg,1)*100:.0f}%)')
    L.append(f'     假阳>1000px 的图: {fp_img_1000}/{n_neg} ({fp_img_1000/max(n_neg,1)*100:.0f}%)')
    L.append(f'     全零假阳(完全干净) 的图: {n_neg-fp_img_100}/{n_neg}')
    L.append(f'推理: avg={avg_ms:.1f} ms/img')
    txt='\n'.join(L); print(txt)
    (out/'eval_fp_report.txt').write_text(txt,encoding='utf-8')
    print(f'\n-> {out/"eval_fp_report.txt"}')

if __name__=='__main__':
    main()
