# -*- coding: utf-8 -*-
"""
在 训练集(train.txt 192张) 上评估所有模型: v1 / v2(+TTA) / v2-fast / YOLO / Mask2Former。
overlay 存 runs/train_eval/<model>/overlays/ (与 val 的 overlay 分开, 用 train 标识)。
输出 runs/train_eval/compare_report.md (train 指标 + train/val gap)。
"""
import os, sys, json, time, glob
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import numpy as np, cv2, torch, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # 工程根目录
V2 = ROOT / 'segformer_v2'
sys.path.insert(0, str(V2))
sys.path.insert(0, str(ROOT / 'mask2former_baseline'))
sys.path.insert(0, str(ROOT / 'segformer_baseline'))

CAT2ID = {'silie': 1, 'fanzhe': 2}
OUT_ROOT = V2 / 'runs' / 'train_eval'

# ---------- 生成训练集全图 GT ----------
def rasterize(objects, W, H):
    mask = np.zeros((H, W), dtype=np.uint8)
    items = []
    for o in objects:
        cat = o.get('category','').strip().lower()
        if cat in ('background','__background__'): continue
        if cat not in CAT2ID: continue
        seg = o.get('segmentation',[])
        if not seg: continue
        poly = np.array(seg, dtype=np.int32)
        items.append((float(o.get('area', cv2.contourArea(poly))), CAT2ID[cat], poly))
    items.sort(key=lambda x: x[0])
    for _, cid, poly in items:
        cv2.fillPoly(mask, [poly], cid)
    return mask

def gen_train_full(cfg):
    out_dir = V2 / 'dataset' / 'train_full'; out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / 'train_full.txt'
    if txt.exists():
        return txt
    lines = []
    for ip in open(cfg['train_list'], encoding='utf-8'):
        ip = ip.strip().replace('\\','/')
        if not ip: continue
        jp = Path(ip).with_suffix('.json').as_posix()
        if not os.path.exists(jp): continue
        d = json.load(open(jp, encoding='utf-8'))
        W, H = d['info']['width'], d['info']['height']
        mask = rasterize(d.get('objects',[]), W, H)
        fp = out_dir / (Path(ip).stem + '.png')
        cv2.imwrite(str(fp), mask)
        lines.append(f'{ip}\t{fp.as_posix()}')
    txt.write_text('\n'.join(lines), encoding='utf-8')
    print(f'生成 train_full: {len(lines)} 张 -> {out_dir}')
    return txt

def load_lines(txt):
    return [l.rstrip('\n').split('\t') for l in open(txt, encoding='utf-8') if l.strip()]

def iou(pred, gt, nc=3):
    inter = np.zeros(nc, dtype=np.float64); union = np.zeros(nc, dtype=np.float64)
    for c in range(nc):
        p = pred==c; t = gt==c
        inter[c] = (p&t).sum(); union[c] = (p|t).sum()
    return inter, union

def overlay(img, pred, alpha=0.5):
    ov = img.copy()
    for c, col in enumerate([(60,60,60),(0,0,255),(255,0,0)]):
        m = pred == c
        ov[m] = (img[m].astype(np.float32)*(1-alpha) + np.array(col,dtype=np.float32)*alpha).astype(np.uint8)
    return ov

def run_model(name, predict_fn, lines, device):
    """predict_fn(img_bgr)->pred(HxW). 通用评估+overlay。"""
    od = OUT_ROOT / name; (od / 'overlays').mkdir(parents=True, exist_ok=True)
    inter = np.zeros(3, dtype=np.float64); union = np.zeros(3, dtype=np.float64)
    times = []
    _ = predict_fn(cv2.imread(lines[0][0]))  # warmup
    torch.cuda.synchronize() if device == 'cuda' else None
    print(f'[{name}] train infer {len(lines)} images ...')
    for i, (ip, mp) in enumerate(lines):
        img = cv2.imread(ip); gt = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        torch.cuda.synchronize() if device == 'cuda' else None; t0 = time.time()
        pred = predict_fn(img)
        torch.cuda.synchronize() if device == 'cuda' else None; times.append(time.time()-t0)
        a, b = iou(pred, gt); inter += a; union += b
        cv2.imwrite(str(od / 'overlays' / f'train_{Path(ip).stem}.png'), overlay(img, pred))
        if i < 2 or i % 40 == 0:
            print(f'  [{i+1}/{len(lines)}] {Path(ip).name} {times[-1]*1000:.0f}ms')
    ious = inter / (union + 1e-9)
    avg = float(np.mean(times))*1000
    with open(od / 'eval_result.txt', 'w', encoding='utf-8') as f:
        f.write(f'background_IoU={ious[0]:.4f}\nsilie_IoU={ious[1]:.4f}\nfanzhe_IoU={ious[2]:.4f}\n'
                f'mIoU={ious.mean():.4f}\navg_inference_ms={avg:.1f}\nnum_images={len(times)}\n')
    print(f'[{name}] mIoU={ious.mean():.4f} bg={ious[0]:.3f} silie={ious[1]:.3f} fanzhe={ious[2]:.3f} {avg:.0f}ms')
    return {'mIoU': ious.mean(), 'bg': ious[0], 'silie': ious[1], 'fanzhe': ious[2], 'ms': avg}

def main():
    cfg = yaml.safe_load(open(V2 / 'config.yaml', encoding='utf-8'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_txt = gen_train_full(cfg)
    lines = load_lines(str(train_txt))
    res = {}

    # --- SegFormer v1: tile512 stride448 noTTA ---
    from transformers import SegformerForSemanticSegmentation
    import importlib.util
    spec = importlib.util.spec_from_file_location('inf', V2 / 'infer.py')
    inf = importlib.util.module_from_spec(spec); spec.loader.exec_module(inf)
    m1 = SegformerForSemanticSegmentation.from_pretrained(ROOT/'segformer_baseline'/'runs'/'full'/'best').to(device).eval()
    res['v1'] = run_model('v1', lambda img: inf.sliding_predict(m1, img, 512, 448, device, False), lines, device)
    del m1; torch.cuda.empty_cache()

    # --- SegFormer v2 (+TTA): tile640 stride320 ---
    m2 = SegformerForSemanticSegmentation.from_pretrained(V2/'runs'/'v2'/'best').to(device).eval()
    res['v2_tta'] = run_model('v2_tta', lambda img: inf.sliding_predict(m2, img, 640, 320, device, True), lines, device)
    del m2; torch.cuda.empty_cache()

    # --- SegFormer v2-fast (bs16 fp16 stride448 noTTA) ---
    spec = importlib.util.spec_from_file_location('inff', V2 / 'infer_fast.py')
    inff = importlib.util.module_from_spec(spec); spec.loader.exec_module(inff)
    m3 = SegformerForSemanticSegmentation.from_pretrained(V2/'runs'/'v2'/'best').to(device).eval()
    res['v2_fast'] = run_model('v2_fast', lambda img: inff.sliding_predict_fast(m3, img, device, True), lines, device)
    del m3; torch.cuda.empty_cache()

    # --- YOLOv8n-seg ---
    from eval_yolo import paint_semantic
    from ultralytics import YOLO
    ym = YOLO(str(ROOT / 'runs' / 'segment' / 'train_split' / 'weights' / 'best.pt'))
    def yolo_pred(img):
        r = ym(img, imgsz=1200, conf=0.25, iou=0.5, verbose=False)[0]
        return paint_semantic(r, img.shape[0], img.shape[1])
    res['yolo'] = run_model('yolo', yolo_pred, lines, device)
    del ym; torch.cuda.empty_cache()

    # --- Mask2Former ---
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    mm = Mask2FormerForUniversalSegmentation.from_pretrained(ROOT/'mask2former_baseline'/'runs'/'full'/'best').to(device).eval()
    mproc = Mask2FormerImageProcessor.from_pretrained(ROOT/'mask2former_baseline'/'models'/'m2f-swin-tiny-coco-instance')
    spec2 = importlib.util.spec_from_file_location('m2feval', ROOT/'mask2former_baseline'/'eval.py')
    m2fe = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(m2fe)
    def m2f_pred(img):
        return m2fe.sliding_predict(mm, mproc, img, 512, 448, device)
    res['m2f'] = run_model('m2f', m2f_pred, lines, device)
    del mm; torch.cuda.empty_cache()

    # --- 报告: train 指标 + train/val gap ---
    def kv(p):
        d = {}
        if not p.exists(): return d
        for l in open(p, encoding='utf-8'):
            if '=' in l:
                k,v = l.strip().split('=',1)
                try: d[k]=float(v)
                except: d[k]=v
        return d
    val_paths = {'v1': ROOT/'segformer_baseline'/'runs'/'full'/'eval_result.txt',
                 'v2_tta': V2/'runs'/'v2'/'eval_result.txt',
                 'v2_fast': V2/'runs'/'v2_fast'/'eval_result.txt',
                 'yolo': ROOT/'segformer_baseline'/'runs'/'yolo_eval'/'eval_result.txt',
                 'm2f': ROOT/'mask2former_baseline'/'runs'/'full'/'eval_result.txt'}
    L = ['# 训练集(train.txt 192张) 全模型评估 + train/val gap\n',
         '## 训练集像素级指标\n',
         '| 模型 | mIoU | silie IoU | fanzhe IoU | 推理ms |',
         '|---|---|---|---|---|']
    for k in ['v1','v2_tta','v2_fast','yolo','m2f']:
        r = res[k]
        L.append(f"| {k} | {r['mIoU']:.4f} | {r['silie']:.4f} | {r['fanzhe']:.4f} | {r['ms']:.0f} |")
    L += ['', '## train vs val gap (过拟合分析, mIoU)\n',
          '| 模型 | train mIoU | val mIoU | gap (train-val) |',
          '|---|---|---|---|']
    for k in ['v1','v2_tta','v2_fast','yolo','m2f']:
        v = kv(val_paths[k])
        vt = v.get('mIoU', 'N/A')
        vt_s = f"{vt:.4f}" if isinstance(vt,float) else vt
        L.append(f"| {k} | {res[k]['mIoU']:.4f} | {vt_s} | {res[k]['mIoU']-(vt if isinstance(vt,float) else 0):+.4f} |")
    L += ['',
          '## 说明',
          '- train 指标反映模型在训练数据上的拟合程度; gap=train-val 越大越过拟合。',
          '- overlay 存 runs/train_eval/<model>/overlays/train_*.png (train 前缀与 val 区分)。']
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / 'compare_report.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))
    print('\n报告 ->', OUT_ROOT / 'compare_report.md')

if __name__ == '__main__':
    main()
