# -*- coding: utf-8 -*-
"""扫描评估: 给定 --config --run，评估一个模型在 (a) val_strat 全图IoU+负样本假阳
(b) 3 个现场批(222/C8/fanzhesilielousha) 的缺陷输出量。结果写 runs_strat/<run>/sweep_eval.json。
口径: v2-fast (bs16 fp16 stride448 noTTA)。
"""
import os
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
import time, yaml, argparse, glob, json
import numpy as np, cv2, torch
import torch.nn.functional as F
from torch.amp import autocast
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
from postprocess import logits_to_labels, resolve_confidence_threshold

MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
TILE,STRIDE,BATCH=640,448,16
FIELD={
    '222': 'C:/Users/hymson/Desktop/222',
    'C8':  'C:/Users/hymson/Desktop/C8fanzhesilieguosha0925',
    'fl':  'C:/Users/hymson/Desktop/fanzhesilielousha0925',
}

def load_val_full(txt):
    out=[]
    for line in open(txt,encoding='utf-8'):
        line=line.rstrip('\n')
        if line: out.append(line.split('\t'))
    return out

def collect(img):
    H,W=img.shape[:2]
    ys=list(range(0,max(1,H-TILE+1),STRIDE))
    if ys[-1]+TILE<H: ys.append(max(0,H-TILE))
    xs=list(range(0,max(1,W-TILE+1),STRIDE))
    if xs[-1]+TILE<W: xs.append(max(0,W-TILE))
    return [(y0,x0,img[y0:y0+TILE,x0:x0+TILE]) for y0 in ys for x0 in xs],H,W

def predict(m,img,confidence_threshold=0.5):
    tiles,H,W=collect(img); acc=None; cov=np.zeros((H,W),np.float32)
    for i in range(0,len(tiles),BATCH):
        b=tiles[i:i+BATCH]; ps=[]
        for _,_,p in b:
            ph,pw=p.shape[:2]
            if ph<TILE or pw<TILE: p=cv2.copyMakeBorder(p,0,TILE-ph,0,TILE-pw,cv2.BORDER_REFLECT)
            ps.append(p)
        a=[p[:,:,::-1].astype(np.float32)/255.0 for p in ps]; a=[(x-MEAN)/STD for x in a]
        t=torch.from_numpy(np.stack(a,0)).permute(0,3,1,2).to('cuda').float()
        with torch.no_grad():
            with autocast('cuda',dtype=torch.float16): lg=m(pixel_values=t).logits
        lg=F.interpolate(lg.float(),size=(TILE,TILE),mode='bilinear',align_corners=False).cpu().numpy()
        if acc is None: acc=np.zeros((lg.shape[1],H,W),np.float32)
        for j,(y0,x0,p) in enumerate(b):
            ph,pw=p.shape[:2]; acc[:,y0:y0+ph,x0:x0+pw]+=lg[j][:,:ph,:pw]; cov[y0:y0+ph,x0:x0+pw]+=1
    cov=np.maximum(cov,1e-6)
    return logits_to_labels(acc/cov[None],confidence_threshold)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--confidence-threshold', type=float, default=None)
    a=ap.parse_args()
    cfg=yaml.safe_load(open(Path(__file__).parent/a.config,encoding='utf-8'))
    threshold=resolve_confidence_threshold(cfg,a.confidence_threshold)
    out=Path(cfg['out_dir'])/a.run
    m=SegformerForSemanticSegmentation.from_pretrained(out/'best').to('cuda').eval()
    res={'run':a.run,'config':a.config,'bg_keep':cfg['bg_keep_ratio'],
         'alpha_bg':cfg['focal_alpha'][0],'confidence_threshold':threshold}

    # (a) val_strat
    lines=load_val_full(str(cfg['val_full_txt']))
    inter=np.zeros(3,np.float64); union=np.zeros(3,np.float64)
    neg_fp=[]; n_pos=0; psi=psu=pfi=pfu=0; times=[]
    _=predict(m,cv2.imread(lines[0][0]),threshold); torch.cuda.synchronize()
    for ip,mp in lines:
        img=cv2.imread(ip); gt=cv2.imread(mp,cv2.IMREAD_UNCHANGED)
        torch.cuda.synchronize(); t0=time.time(); pred=predict(m,img,threshold); torch.cuda.synchronize(); times.append(time.time()-t0)
        for c in range(3):
            inter[c]+=((pred==c)&(gt==c)).sum(); union[c]+=((pred==c)|(gt==c)).sum()
        if (gt>0).sum()==0:
            neg_fp.append(int((pred>0).sum()))
        else:
            n_pos+=1
            psi+=((pred==1)&(gt==1)).sum(); psu+=((pred==1)|(gt==1)).sum()
            pfi+=((pred==2)&(gt==2)).sum(); pfu+=((pred==2)|(gt==2)).sum()
    ious=inter/(union+1e-9); neg_fp=np.array(neg_fp)
    res['val']={'mIoU':round(float(ious.mean()),4),'bg':round(float(ious[0]),4),
        'silie':round(float(ious[1]),4),'fanzhe':round(float(ious[2]),4),
        'pos_silie':round(float(psi/(psu+1e-9)),4),'pos_fanzhe':round(float(pfi/(pfu+1e-9)),4),
        'n_neg':int(len(neg_fp)),'neg_avg_fp':round(float(neg_fp.mean()),1),
        'neg_max_fp':int(neg_fp.max()),'neg_gt100':int((neg_fp>100).sum()),
        'neg_gt1000':int((neg_fp>1000).sum()),'neg_clean':int((neg_fp==0).sum()),
        'avg_ms':round(float(np.mean(times))*1000,1)}

    # (b) 3 现场批
    res['field']={}
    for name,d in FIELD.items():
        imgs=sorted(glob.glob(str(Path(d)/'*.jpg')))
        per={}; flag=0; tot=0
        for f in imgs:
            pred=predict(m,cv2.imread(f),threshold); dpx=int((pred>0).sum())
            per[Path(f).name]=dpx
            if dpx>0: flag+=1
            tot+=dpx
        res['field'][name]={'n':len(imgs),'flagged':flag,'total_px':tot,'per_image':per}
        print(f'  [{a.run}] {name}: {flag}/{len(imgs)} flagged, total {tot} px',flush=True)

    (out/'sweep_eval.json').write_text(json.dumps(res,ensure_ascii=False,indent=1),encoding='utf-8')
    v=res['val']
    print(f'[{a.run}] val mIoU={v["mIoU"]} silie={v["silie"]} fanzhe={v["fanzhe"]} | neg avg_fp={v["neg_avg_fp"]} max={v["neg_max_fp"]} gt100={v["neg_gt100"]}/{v["n_neg"]} | confidence_threshold={threshold}',flush=True)
    print(f'-> {out/"sweep_eval.json"}',flush=True)

if __name__=='__main__':
    main()
