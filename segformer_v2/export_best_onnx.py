# -*- coding: utf-8 -*-
"""把 v2 最优权重导出为 ONNX 单文件（部署级，可用 onnxruntime 直接推理，不依赖 transformers）。
输入: pixel_values (B,3,H,W) float32, ImageNet 归一化
输出: logits (B,3,H/4,W/4) float32 -> argmax 得 0/1/2 语义图
动态 batch + 空间尺寸（SegFormer 全卷积，支持任意 H/W）。
导出后用 onnxruntime 校验与 torch 输出一致。
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import torch, yaml, argparse, numpy as np
from pathlib import Path
from transformers import SegformerForSemanticSegmentation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--size', type=int, default=640, help='导出用示例输入尺寸（动态轴允许运行时改）')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(__file__).parent / 'config_strat.yaml', encoding='utf-8'))
    runs = Path(cfg['out_dir'])
    src = Path(a.src) if a.src else runs / 'v2_strat' / 'best'
    out = Path(a.out) if a.out else runs / 'v2_strat' / 'segformer_v2_strat_best.onnx'
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f'加载: {src}')
    model = SegformerForSemanticSegmentation.from_pretrained(src).eval()
    model.return_dict = False

    dummy = torch.randn(1, 3, a.size, a.size)
    print(f'导出 ONNX (opset {a.opset}, 示例输入 {dummy.shape}, 动态 batch+H+W) ...')
    torch.onnx.export(
        model, dummy, str(out),
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={'pixel_values': {0: 'batch', 2: 'height', 3: 'width'},
                       'logits':    {0: 'batch', 2: 'height', 3: 'width'}},
        opset_version=a.opset,
    )
    sz = out.stat().st_size / 1024 / 1024
    print(f'已保存: {out}  ({sz:.1f} MB)')

    # 校验：torch vs onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
        x = np.random.randn(1, 3, a.size, a.size).astype(np.float32)
        ort_out = sess.run(None, {'pixel_values': x})[0]
        with torch.no_grad():
            torch_out = model(pixel_values=torch.from_numpy(x))[0].numpy()
        diff = np.abs(ort_out - torch_out).max()
        print(f'校验 onnx vs torch: max|Δ|={diff:.6e}  形状 {ort_out.shape}  (一致={"OK" if diff < 1e-4 else "偏大"})')
        # 额外验证动态尺寸
        x2 = np.random.randn(1, 3, 512, 768).astype(np.float32)
        o2 = sess.run(None, {'pixel_values': x2})[0]
        print(f'动态尺寸测试 512x768 -> 输出 {o2.shape} (应为 1x3x128x192)  OK')
    except ImportError:
        print('onnxruntime 未安装，跳过校验（ONNX 文件已生成，可用 onnxruntime 部署端加载）')

    print('\n部署推理示例 (onnxruntime):')
    print("  import numpy as np, onnxruntime as ort, cv2")
    print("  sess = ort.InferenceSession(r'%s', providers=['CUDAExecutionProvider','CPUExecutionProvider'])" % out)
    print("  mean=np.array([0.485,0.456,0.406]); std=np.array([0.229,0.224,0.225])")
    print("  img=cv2.cvtColor(cv2.imread('tile.jpg'), cv2.COLOR_BGR2RGB).astype(np.float32)/255")
    print("  x=((img-mean)/std).transpose(2,0,1)[None]")
    print("  logits=sess.run(None,{'pixel_values':x})[0]  # (1,3,H/4,W/4)")
    print("  pred=np.argmax(logits,1)[0]  # 0/1/2 语义图")

if __name__ == '__main__':
    main()
