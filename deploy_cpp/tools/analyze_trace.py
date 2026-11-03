#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 segtrace CSV（SEGFORMER_PROFILE=1 时由 segformer_Dll_func.dll 输出）。

用法：
    python analyze_trace.py                       # 自动找 %TEMP% 下最新的 seg_trace_*.csv
    python analyze_trace.py D:\\trace\\seg_trace.csv
    python analyze_trace.py xxx.csv --keep-warmup # 保留第 1 帧（TRT 首次执行，通常很慢）

输出：
    1. 事件统计（count / mean / P50 / P95 / max）
    2. 每帧各段均值，以及「已覆盖段之和」与 dll.forward 的差额
    3. 按 ng=0/1 分组的耗时对比
    4. 最慢的 10 帧及其各段拆分

只依赖 Python 标准库。
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict


# 报告里固定按这个顺序展示，方便逐帧对照
ORDER = [
    "dll.collect",
    "dll.clear",
    "dll.infer",
    "dll.post.acc",
    "dll.post.argmax",
    "dll.inferImage",
    "dll.stat",
    "dll.stat.scan",
    "dll.stat.cc",
    "dll.mask.copyout",
    "dll.overlay.build",
    "dll.overlay.copyout",
    "dll.freeImage",
    "dll.forward.copyInput",
    "dll.forward",
]

# dll.forward 内部互不重叠的段（加上参数检查/copyInput，应接近 dll.forward）
FORWARD_PARTS = [
    "dll.collect",
    "dll.clear",
    "dll.infer",
    "dll.post.acc",
    "dll.post.argmax",
]

# 发生在 dll.forward 返回之后的帧外段（软件里先 RunMask，再 GetMask / GetStat）
AFTER_PARTS = [
    "dll.stat",
    "dll.mask.copyout",
    "dll.overlay.build",
    "dll.overlay.copyout",
]


def find_latest():
    bases = []
    tmp = os.environ.get("TEMP") or os.environ.get("TMP")
    if tmp:
        bases.append(tmp)
    for b in bases:
        files = glob.glob(os.path.join(b, "seg_trace_*.csv"))
        if files:
            return max(files, key=os.path.getmtime)
    return None


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for rec in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "frame": int(rec["frame"]),
                        "wall": rec["wall"],
                        "t_ms": float(rec["t_ms"]),
                        "event": rec["event"],
                        "ms": float(rec["ms"]),
                        "note": (rec.get("note") or "").strip(),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue  # 跳过损坏行，不让一条坏记录毁掉整份报告
    return rows


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def fmt_row(cols, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def main():
    ap = argparse.ArgumentParser(description="分析 segtrace CSV")
    ap.add_argument("trace", nargs="?", help="CSV 路径；省略时自动找 %%TEMP%% 下最新的文件")
    ap.add_argument("--keep-warmup", action="store_true", help="保留第 1 帧（默认排除）")
    args = ap.parse_args()

    # Windows 控制台默认 936 代码页，中文报告会乱码；切到 UTF-8 只影响本次控制台。
    if sys.platform == "win32":
        os.system("chcp 65001 >nul 2>&1")
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    path = args.trace or find_latest()
    if not path or not os.path.isfile(path):
        sys.stderr.write("找不到 trace 文件，请显式传入路径\n")
        return 2

    rows = load(path)
    if not rows:
        sys.stderr.write("trace 文件为空或格式不对：%s\n" % path)
        return 2

    meta = [r for r in rows if r["event"] == "meta"]
    events = [r for r in rows if r["event"] != "meta"]

    frames = sorted({r["frame"] for r in events if r["frame"] > 0})
    if not args.keep_warmup:
        frames = [f for f in frames if f > 1]
        keep = set(frames)
        events = [r for r in events if r["frame"] in keep]
    elif frames:
        # 保留 warmup 时也至少让第 1 帧进入统计
        pass

    if not frames:
        sys.stderr.write("没有有效帧（只有 meta 或全是第 1 帧）\n")
        return 2

    print("=" * 78)
    print("trace 文件 : %s" % path)
    print("帧数       : %d  (frame %d..%d%s)" % (
        len(frames), frames[0], frames[-1],
        "" if args.keep_warmup else "，已排除第 1 帧 warmup"))
    print("墙钟范围   : %s .. %s" % (events[0]["wall"], events[-1]["wall"]))
    for m in meta:
        print("元信息     : %s" % m["note"])
    print("=" * 78)

    # ---- 1. 事件统计 ----
    by_event = defaultdict(list)
    for r in events:
        by_event[r["event"]].append(r["ms"])

    print("\n[1] 各事件耗时（ms）")
    widths = [18, 7, 10, 10, 10, 10]
    print("  " + fmt_row(["event", "count", "mean", "p50", "p95", "max"], widths))
    print("  " + "-" * 70)
    seen = set()
    names = [n for n in ORDER if n in by_event]
    names += sorted(n for n in by_event if n not in ORDER)
    for name in names:
        v = by_event[name]
        seen.add(name)
        print("  " + fmt_row([
            name, len(v),
            "%.1f" % (sum(v) / len(v)),
            "%.1f" % percentile(v, 50),
            "%.1f" % percentile(v, 95),
            "%.1f" % max(v),
        ], widths))

    # ---- 2. 每帧分解 ----
    per_frame = defaultdict(lambda: defaultdict(float))
    for r in events:
        per_frame[r["frame"]][r["event"]] += r["ms"]

    print("\n[2] 每帧分解（均值 ms，n=%d）" % len(frames))
    part_mean = {}
    for name in FORWARD_PARTS:
        vals = [per_frame[f].get(name, 0.0) for f in frames]
        part_mean[name] = sum(vals) / len(vals)
    print("  dll.forward 内：")
    for name in FORWARD_PARTS:
        print("    %-18s %8.1f" % (name, part_mean[name]))

    covered = sum(part_mean.values())
    fwd = [per_frame[f].get("dll.forward", 0.0) for f in frames]
    fwd_mean = sum(fwd) / len(fwd) if fwd else 0.0
    infer_image = [per_frame[f].get("dll.inferImage", 0.0) for f in frames]
    ii_mean = sum(infer_image) / len(infer_image) if infer_image else 0.0
    print("    %-18s %8.1f" % ("合计", covered))
    print("  dll.inferImage = %.1f ms" % ii_mean)
    print("  dll.forward    = %.1f ms（含参数检查 / copyInput）" % fwd_mean)
    if fwd_mean > 0:
        print("  未解释差额     = %.1f ms（%.1f%%）" % (
            fwd_mean - covered, 100.0 * (fwd_mean - covered) / fwd_mean))

    after_mean = {}
    for name in AFTER_PARTS:
        vals = [per_frame[f].get(name, 0.0) for f in frames]
        after_mean[name] = sum(vals) / len(vals)
    after_total = sum(after_mean.values())
    print("  dll.forward 之后（帧外）：")
    for name in AFTER_PARTS:
        if any(per_frame[f].get(name, 0.0) for f in frames):
            print("    %-18s %8.1f" % (name, after_mean[name]))
    print("    %-18s %8.1f" % ("合计", after_total))
    print("\n  整帧合计（forward + 帧外段） = %.1f ms" % (fwd_mean + after_total))

    # ---- 3. NG 分组 ----
    ng_of = {}
    for r in events:
        if r["event"] == "dll.stat" and r["frame"] not in ng_of:
            m = re.search(r"ng=(\d)", r["note"])
            if m:
                ng_of[r["frame"]] = int(m.group(1))

    print("\n[3] 按是否 NG 分组（dll.forward / dll.stat / 整帧）")
    for want in (0, 1):
        fs = [f for f in frames if ng_of.get(f) == want]
        if not fs:
            continue
        vals = [per_frame[f].get("dll.forward", 0.0) for f in fs]
        stats = [per_frame[f].get("dll.stat", 0.0) for f in fs]
        whole = [per_frame[f].get("dll.forward", 0.0) +
                 sum(per_frame[f].get(n, 0.0) for n in AFTER_PARTS) for f in fs]
        print("  ng=%d  n=%-4d forward: mean=%.1f p50=%.1f p95=%.1f | stat: mean=%.1f "
              "max=%.1f | 整帧: mean=%.1f p95=%.1f" % (
                  want, len(fs), sum(vals) / len(vals), percentile(vals, 50),
                  percentile(vals, 95), sum(stats) / len(stats), max(stats),
                  sum(whole) / len(whole), percentile(whole, 95)))
    unknown = [f for f in frames if f not in ng_of]
    if unknown:
        print("  无 dll.stat 记录的帧：%d" % len(unknown))

    # ---- 4. 最慢的 10 帧 ----
    ranked = sorted(frames, key=lambda f: per_frame[f].get("dll.forward", 0.0),
                    reverse=True)[:10]
    print("\n[4] 最慢的 10 帧")
    cols = ["frame", "forward"] + FORWARD_PARTS + AFTER_PARTS
    widths = [7, 9] + [13] * (len(FORWARD_PARTS) + len(AFTER_PARTS))
    print("  " + fmt_row(cols, widths))
    print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for f in ranked:
        vals = [per_frame[f].get(n, 0.0) for n in FORWARD_PARTS + AFTER_PARTS]
        print("  " + fmt_row(
            [f, "%.1f" % per_frame[f].get("dll.forward", 0.0)] +
            ["%.1f" % v for v in vals], widths))

    # ---- 5. 波动概览 ----
    print("\n[5] 波动概览")
    for name in ["dll.forward", "dll.inferImage", "dll.clear", "dll.infer",
                 "dll.post.acc", "dll.stat"]:
        v = [per_frame[f].get(name, 0.0) for f in frames]
        if not any(v):
            continue
        mean = sum(v) / len(v)
        p50, p95 = percentile(v, 50), percentile(v, 95)
        ratio = (p95 / p50) if p50 > 0 else 0.0
        print("  %-16s mean=%7.1f p50=%7.1f p95=%7.1f max=%7.1f  p95/p50=%.2f" % (
            name, mean, p50, p95, max(v), ratio))

    return 0


if __name__ == "__main__":
    sys.exit(main())
