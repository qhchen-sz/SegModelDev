# -*- coding: utf-8 -*-
"""S4 模型 (bg_keep=0.12, alpha_bg=0.80) 现场批测试: S4-fast vs YOLO。
用法: python test_s4_field.py --batch <图片文件夹>
输出到 <batch>/_results/: overlays_s4/ stats_s4.csv timing_s4.csv report_s4.md （_s4 命名，不覆盖已有 v2-fast 结果）
口径: S4 bs16 fp16 stride448 noTTA; YOLO imgsz1200 conf0.25 iou0.5。
"""
import os
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
import time, csv, glob, argparse
import numpy as np, cv2, torch
import torch.nn.functional as F
from torch.amp import autocast
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
from postprocess import logits_to_labels, resolve_confidence_threshold

S4_BEST = r"D:/qhchen/NewModel/ADevFolder/segformer_v2/runs_strat/sw_s4/best"
YOLO_BEST = r"D:/qhchen/NewModel/ADevFolder/runs_strat/train_strat/weights/best.pt"
TILE,STRIDE,BATCH,FP16=640,448,16,True
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
COLS=[(60,60,60),(0,0,255),(255,0,0)]
dev='cuda' if torch.cuda.is_available() else 'cpu'

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

def s4_predict(m,img,confidence_threshold=0.5):
    tiles,H,W=collect(img); acc=None; cov=np.zeros((H,W),np.float32)
    for i in range(0,len(tiles),BATCH):
        b=tiles[i:i+BATCH]
        a=[p[:,:,::-1].astype(np.float32)/255.0 for _,_,p,_,_ in b]; a=[(x-MEAN)/STD for x in a]
        t=torch.from_numpy(np.stack(a,0)).permute(0,3,1,2).to(dev).float()
        with torch.no_grad():
            with autocast('cuda',dtype=torch.float16) if FP16 else torch.no_grad(): lg=m(pixel_values=t).logits
        lg=F.interpolate(lg.float(),size=(TILE,TILE),mode='bilinear',align_corners=False).cpu().numpy()
        if acc is None: acc=np.zeros((lg.shape[1],H,W),np.float32)
        for j,(y0,x0,_,ph,pw) in enumerate(b):
            acc[:,y0:y0+ph,x0:x0+pw]+=lg[j][:,:ph,:pw]; cov[y0:y0+ph,x0:x0+pw]+=1
    cov=np.maximum(cov,1e-6)
    return logits_to_labels(acc/cov[None],confidence_threshold)

def yolo_predict(ym,img):
    H,W=img.shape[:2]; res=ym(img,imgsz=1200,conf=0.25,iou=0.5,verbose=False)[0]
    sem=np.zeros((H,W),np.uint8); n=0
    if res.masks is not None:
        cls=res.boxes.cls.cpu().numpy().astype(int); cf=res.boxes.conf.cpu().numpy(); polys=res.masks.xy
        for k in np.argsort(cf):
            pl=polys[k]
            if pl is None or len(pl)<3: continue
            cv2.fillPoly(sem,[np.round(pl).astype(np.int32)],int(cls[k]))
        n=len(cls)
    return sem,n

def overlay(img,pred):
    ov=img.copy()
    for c,col in enumerate(COLS):
        m=pred==c; ov[m]=(img[m]*0.5+np.array(col,np.float32)*0.5).astype(np.uint8)
    return ov

def iou2(a,b,nc=3):
    it=np.zeros(nc,np.int64); un=np.zeros(nc,np.int64)
    for c in range(nc):
        pa=a==c; pb=b==c; it[c]=(pa&pb).sum(); un[c]=(pa|pb).sum()
    return it,un

def main():
    import yaml
    ap=argparse.ArgumentParser()
    ap.add_argument('--batch',required=True)
    ap.add_argument('--confidence-threshold',type=float,default=None)
    a=ap.parse_args()
    cfg=yaml.safe_load(open(Path(__file__).parent/'config_sw_s4.yaml',encoding='utf-8'))
    threshold=resolve_confidence_threshold(cfg,a.confidence_threshold)
    IMG=Path(a.batch); OUT=IMG/'_results'; OUT.mkdir(exist_ok=True)
    OV=OUT/'overlays_s4'; OV.mkdir(exist_ok=True)
    OVY=OUT/'overlays_yolo'; OVY.mkdir(exist_ok=True)
    imgs=sorted(glob.glob(str(IMG/'*.jpg')))
    H0,W0=cv2.imread(imgs[0]).shape[:2]
    print(f'[{IMG.name}] {len(imgs)} 张, {W0}x{H0}, confidence_threshold={threshold}')
    s4=SegformerForSemanticSegmentation.from_pretrained(S4_BEST).to(dev).eval()
    from ultralytics import YOLO
    ym=YOLO(YOLO_BEST)
    _=s4_predict(s4,cv2.imread(imgs[0]),threshold); _=yolo_predict(ym,cv2.imread(imgs[0]))
    if dev=='cuda': torch.cuda.synchronize()
    rt=[]; rs=[]; I=np.zeros(3,np.int64); U=np.zeros(3,np.int64)
    s4f=yf=0
    for i,ip in enumerate(imgs):
        img=cv2.imread(ip)
        if dev=='cuda': torch.cuda.synchronize()
        t0=time.time(); p4=s4_predict(s4,img,threshold)
        if dev=='cuda': torch.cuda.synchronize()
        t4=(time.time()-t0)*1000
        if dev=='cuda': torch.cuda.synchronize()
        t0=time.time(); py,ni=yolo_predict(ym,img)
        if dev=='cuda': torch.cuda.synchronize()
        ty=(time.time()-t0)*1000
        s_s4=int((p4==1).sum()); f_s4=int((p4==2).sum())
        s_y=int((py==1).sum()); f_y=int((py==2).sum())
        if s_s4+f_s4>0: s4f+=1
        if s_y+f_y>0: yf+=1
        it,un=iou2(p4,py); I+=it; U+=un
        cv2.imwrite(str(OV/(Path(ip).stem+'.png')),overlay(img,p4))
        cv2.imwrite(str(OVY/(Path(ip).stem+'.png')),overlay(img,py))
        rt.append([Path(ip).name,f'{t4:.1f}',f'{ty:.1f}'])
        rs.append([Path(ip).name,s_s4,f_s4,s_y,f_y,ni])
        if i<3 or i%10==0: print(f'  [{i+1}/{len(imgs)}] {Path(ip).name[:24]} s4={t4:.0f}ms yolo={ty:.0f}ms s4[s{s_s4}/f{f_s4}] yolo[s{s_y}/f{f_y}]',flush=True)
    t4s=[float(r[1]) for r in rt]; tys=[float(r[2]) for r in rt]; ag=I/(U+1e-9)
    with open(OUT/'timing_s4.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['image','s4_ms','yolo_ms']); w.writerows(rt)
    with open(OUT/'stats_s4.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['image','s4_silie_px','s4_fanzhe_px','yolo_silie_px','yolo_fanzhe_px','yolo_instances']); w.writerows(rs)
    n=len(imgs)
    L=[f'# S4 模型测试报告（{IMG.name} | {n} 张全图，无 GT）\n',
       '## 说明\n',
       '- 模型: **S4** (bg_keep=0.12, focal_alpha(bg)=0.80) —— 超参扫描胜出配置，相较 v2 基线假阳 -82%、IoU 不降。',
       f'- 对比: YOLOv8n-seg。口径: S4 bs16 fp16 stride448 noTTA Softmax置信度过滤(conf{threshold}); yolo imgsz1200 conf0.25 iou0.5。',
       f'- 本批无标注, 精度指标 N/A; 叠加图 `overlays_s4/`、`overlays_yolo/`（silie红/fanzhe蓝）。\n',
       f'## 推理速度（RTX 3090，全图 {W0}×{H0}）\n',
       '| 模型 | 平均 ms/图 | 最小 | 最大 |', '|---|---|---|---|',
       f'| **S4** | {np.mean(t4s):.1f} | {min(t4s):.1f} | {max(t4s):.1f} |',
       f'| YOLOv8n-seg | {np.mean(tys):.1f} | {min(tys):.1f} | {max(tys):.1f} |',
       f'| 速度比 | {np.mean(t4s)/max(np.mean(tys),1e-9):.1f}x | | |\n',
       '## 检出与统计\n',
       f'- {n} 张中 S4 判定有缺陷: **{s4f}** 张；YOLO: **{yf}** 张',
       '| 模型 | 平均 silie 像素 | 平均 fanzhe 像素 |', '|---|---|---|',
       f'| S4 | {np.mean([r[1] for r in rs]):.0f} | {np.mean([r[2] for r in rs]):.0f} |',
       f'| YOLO | {np.mean([r[3] for r in rs]):.0f} | {np.mean([r[4] for r in rs]):.0f} |\n',
       '## S4 vs YOLO 一致性 IoU\n',
       '| 类别 | IoU |', '|---|---|',
       f'| background | {ag[0]:.4f} |', f'| silie | {ag[1]:.4f} |', f'| fanzhe | {ag[2]:.4f} |', f'| mIoU | {ag.mean():.4f} |\n',
       '## 结论\n',
       f'- 速度: S4 {np.mean(t4s):.1f}ms vs YOLO {np.mean(tys):.1f}ms, YOLO 快 {np.mean(t4s)/max(np.mean(tys),1e-9):.1f}x。',
       f'- 检出: S4 {s4f}/{n} vs YOLO {yf}/{n}。',
       '- 无 GT, 精度不可判; 逐图对比见 stats_s4.csv 与两套叠加图, 建议抽样标注复核。',
       '- 参考: runs_strat/sweep_report.md(S4 胜出依据)、项目根目录 final_report.md(有 GT 验证集指标)。']
    (OUT/'report_s4.md').write_text('\n'.join(L),encoding='utf-8')
    print('\n'.join(L)); print('\n->',OUT/'report_s4.md')

if __name__=='__main__':
    main()
