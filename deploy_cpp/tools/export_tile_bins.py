# -*- coding: utf-8 -*-
"""
导出 segformer 滑窗 tile 为 TensorRT trtexec 可直接喂入的 .bin。

与 C++ 侧严格对齐（deploy_cpp/segformer_dll/seg_pipeline.cpp）：
  collectTiles() : tile=640, stride=448, 行优先, 末尾补 H-tile / W-tile
  fillNCHW()     : BGR->RGB, /255, (x-mean)/std, float32, NCHW
  padding        : cv2.BORDER_REFLECT（只有 H/W < tile 时才会触发）

用法：
  python export_tile_bins.py --image <图片> [--outdir tiles_bin] [--all-tiles] [--batches]

默认只导出 3 个代表性 tile（左上 / 含主缺陷 / 右下背景）。
--all-tiles  导出全部 42 个 tile 的单文件 bin
--batches    额外导出按 batch(<=16) 分组的 bin，用于一次跑完整图
输出：
  <outdir>/tiles_sheet.jpg   每个 tile 缩略图 + 编号 + (y0,x0)
  <outdir>/tiles_meta.json   每个 tile 的坐标与文件名
  <outdir>/tile_XXX_yYYYY_xXXXX.bin   1x3x640x640 float32 NCHW
"""
import argparse
import json
import os

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def collect_tiles(h, w, tile, stride):
    ys = []
    stop_y = max(1, h - tile + 1)
    y = 0
    while y < stop_y:
        ys.append(y)
        y += stride
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))

    xs = []
    stop_x = max(1, w - tile + 1)
    x = 0
    while x < stop_x:
        xs.append(x)
        x += stride
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))

    tiles = []
    for y0 in ys:
        for x0 in xs:
            tiles.append((y0, x0, min(tile, h - y0), min(tile, w - x0)))
    return tiles


def fill_nchw(patch, tile):
    """BGR 8U -> NCHW float32, 与 seg_pipeline.cpp fillNCHW 逐像素一致。"""
    if patch.shape[0] != tile or patch.shape[1] != tile:
        patch = cv2.copyMakeBorder(patch, 0, tile - patch.shape[0],
                                   0, tile - patch.shape[1], cv2.BORDER_REFLECT)
    rgb = patch[:, :, ::-1].astype(np.float32) / np.float32(255.0)
    rgb = (rgb - MEAN) / STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "tiles_bin"))
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=448)
    ap.add_argument("--all-tiles", action="store_true")
    ap.add_argument("--tiles", default="",
                    help="逗号分隔的 tile 索引，如 1,8,9,10,15,16；默认导出 0,1,last")
    ap.add_argument("--batches", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("读图失败: %s" % args.image)
    h, w = img.shape[:2]
    tiles = collect_tiles(h, w, args.tile, args.stride)
    os.makedirs(args.outdir, exist_ok=True)

    print("图片 %dx%d (HxW), tile=%d stride=%d -> %d 个 tile (行优先)"
          % (h, w, args.tile, args.stride, len(tiles)))

    # 默认导出的代表性 tile：左上 / 含主缺陷 / 右下背景
    if args.all_tiles:
        picked = set(range(len(tiles)))
    elif args.tiles:
        picked = {int(s) for s in args.tiles.split(",") if s.strip()}
        bad = [i for i in picked if i < 0 or i >= len(tiles)]
        if bad:
            raise SystemExit("tile 索引越界: %s (共 %d 个)" % (bad, len(tiles)))
    else:
        picked = {0, 1, len(tiles) - 1}

    meta = []
    thumbs = []
    cols = 7
    th = 200
    for i, (y0, x0, ph, pw) in enumerate(tiles):
        patch = img[y0:y0 + ph, x0:x0 + pw]
        name = "tile_%03d_y%04d_x%04d.bin" % (i, y0, x0)
        meta.append({
            "index": i, "y0": y0, "x0": x0, "ph": ph, "pw": pw,
            "bin": name if i in picked else None,
        })
        if i in picked:
            arr = fill_nchw(patch, args.tile)
            arr.tofile(os.path.join(args.outdir, name))
            print("  [%3d] y0=%4d x0=%4d -> %s  (mean=%.4f, %.1f MB)"
                  % (i, y0, x0, name, float(arr.mean()), arr.nbytes / 1048576.0))

        thumb = cv2.resize(patch, (th, th), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, "#%d (%d,%d)" % (i, y0, x0), (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        if i in picked:
            cv2.rectangle(thumb, (0, 0), (th - 1, th - 1), (0, 0, 255), 3)
        thumbs.append(thumb)

    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * th, 3), np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * th:(c + 1) * th] = t
    cv2.imwrite(os.path.join(args.outdir, "tiles_sheet.jpg"), sheet)

    if args.batches:
        bs = 16
        for b in range(0, len(tiles), bs):
            group = tiles[b:b + bs]
            arr = np.empty((len(group), 3, args.tile, args.tile), np.float32)
            for j, (y0, x0, ph, pw) in enumerate(group):
                arr[j] = fill_nchw(img[y0:y0 + ph, x0:x0 + pw], args.tile)
            name = "batch_%02d.bin" % (b // bs)
            arr.tofile(os.path.join(args.outdir, name))
            print("  batch %s: %d 张, %.1f MB" % (name, len(group), arr.nbytes / 1048576.0))

    with open(os.path.join(args.outdir, "tiles_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"image": os.path.abspath(args.image), "H": h, "W": w,
                   "tile": args.tile, "stride": args.stride, "tiles": meta}, f,
                  ensure_ascii=False, indent=2)
    print("输出目录:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
