# -*- coding: utf-8 -*-
"""
解析 trtexec 导出的 SegFormer logits，判断引擎在真实输入下是否正常。

支持两种来源（在 50 系机器上二选一）：
  1) trtexec ... --exportOutput=out.json   ->  python parse_trtexec_output.py --input out.json
  2) trtexec ... --dumpOutput > out.txt    ->  python parse_trtexec_output.py --input out.txt

可加 --ref <另一份输出> 直接对比两份 logits（例如 30 系引擎 vs 50 系引擎）。

输出：元素数/shape、NaN/Inf、每通道 min/max/mean/std、argmax 三类像素数。
SegFormer 输出 NCHW = [numClasses=3, H=160, W=160]。
"""
import argparse
import json
import re

import numpy as np

NUM_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


def arrays_from_json(obj):
    """递归找出 JSON 里所有纯数字数组。"""
    if isinstance(obj, list):
        if obj and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            yield obj
        else:
            for v in obj:
                yield from arrays_from_json(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from arrays_from_json(v)


def load_values(path, n_expected=None):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    if text.lstrip().startswith(("{", "[")):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if obj is not None:
            arrs = [np.asarray(a, np.float64) for a in arrays_from_json(obj)]
            if not arrs:
                raise SystemExit("JSON 中没有找到数值数组: %s" % path)
            if n_expected:
                hit = [a for a in arrs if a.size == n_expected]
                return hit[-1] if hit else max(arrs, key=lambda a: a.size)
            return max(arrs, key=lambda a: a.size)

    # --dumpOutput 文本：每行一个数（夹着 [I] 日志行）
    vals = []
    for line in text.splitlines():
        s = line.strip()
        if NUM_RE.match(s):
            vals.append(float(s))
    if not vals:
        raise SystemExit("文本里没有解析到数值: %s" % path)
    return np.asarray(vals, np.float64)


def summarize(v, shape, tag=""):
    v3 = v.reshape(shape)
    print("\n[%s] 元素数=%d  shape=%s" % (tag, v3.size, "x".join(map(str, v3.shape))))
    print("  NaN=%d  Inf=%d  min=%.6g  max=%.6g  mean=%.6g  std=%.6g"
          % (np.isnan(v3).sum(), np.isinf(v3).sum(),
             np.nanmin(v3), np.nanmax(v3), np.nanmean(v3), np.nanstd(v3)))
    for c in range(v3.shape[0]):
        pc = v3[c]
        print("  ch%d: min=%9.4f max=%9.4f mean=%9.4f std=%8.4f"
              % (c, np.nanmin(pc), np.nanmax(pc), np.nanmean(pc), np.nanstd(pc)))
    lab = np.nan_to_num(v3, nan=-1e30).argmax(0)
    counts = np.bincount(lab.reshape(-1), minlength=v3.shape[0])
    total = lab.size
    print("  argmax 像素: 0(背景)=%d  1(silie)=%d  2(fanzhe)=%d"
          % tuple(counts[:3]))
    print("  前景占比 = %.4f%%" % (100.0 * (counts[1] + counts[2]) / total))
    return v3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="trtexec 输出(json 或 txt)")
    ap.add_argument("--ref", default="", help="另一份输出，用于对比")
    ap.add_argument("--shape", default="3,160,160",
                    help="logits 形状 C,H,W（默认 3,160,160）")
    args = ap.parse_args()

    shape = tuple(int(s) for s in args.shape.split(","))
    n = int(np.prod(shape))

    v = load_values(args.input, n)
    if v.size != n:
        print("[WARN] 解析到 %d 个值，期望 %d；若引擎 batch != 1 请改 --shape" % (v.size, n))
        if v.size % n == 0:
            shape = (v.size // n,) + shape
    a = summarize(v, shape, "50 系/待测")

    if args.ref:
        r = load_values(args.ref, n)
        b = summarize(r, shape, "30 系/参考")
        d = np.abs(a - b)
        print("\n[对比] max|diff|=%.6g  mean|diff|=%.6g  NaN(a)=%d NaN(b)=%d"
              % (np.nanmax(d), np.nanmean(d), np.isnan(a).sum(), np.isnan(b).sum()))
        la = np.nan_to_num(a, nan=-1e30).argmax(0)
        lb = np.nan_to_num(b, nan=-1e30).argmax(0)
        print("[对比] argmax 不一致像素 = %d / %d"
              % (int((la != lb).sum()), la.size))


if __name__ == "__main__":
    main()
