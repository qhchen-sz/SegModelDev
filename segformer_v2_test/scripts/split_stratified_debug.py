# -*- coding: utf-8 -*-
"""
split_stratified.py 的调试副本：切分逻辑逐字一致，仅增加 --images/--labels/--out-dir
三个参数，便于把输出重定向到 debug 目录，不触碰原始 train_strat.txt / val_strat.txt。
不传参时默认行为与 split_stratified.py 等价。
"""
import os, glob, random, argparse
from collections import defaultdict

VAL_RATIO = 0.2
SEED = 42

GROUP_NAME = {
    frozenset(): "neg(纯背景)",
    frozenset({1}): "silie-only",
    frozenset({2}): "fanzhe-only",
    frozenset({1, 2}): "both(silie+fanzhe)",
}

IMAGES_DIR = None
LABELS_DIR = None
OUT_DIR = None


def defect_classes(label_path):
    """返回该图中出现的缺陷类别集合（标签首列；0=背景不计）。"""
    cs = set()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts and parts[0].isdigit():
                c = int(parts[0])
                if c in (1, 2):
                    cs.add(c)
    return cs


def to_abs_backslash(rel):
    """xxx.jpg -> <images_dir>\\xxx.jpg（ultralytics 要反斜杠绝对路径）。"""
    return os.path.abspath(os.path.join(IMAGES_DIR, rel)).replace("/", os.sep)


def count_instances(img_list):
    """统计列表中 silie / fanzhe 的实例数与出现图数。"""
    si_inst = fa_inst = 0
    si_img = fa_img = 0
    for rel in img_list:
        stem = os.path.splitext(os.path.basename(rel))[0]
        lp = os.path.join(LABELS_DIR, stem + ".txt")
        cs = defect_classes(lp)
        s = f = 0
        with open(lp, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if parts and parts[0] == "1":
                    s += 1
                elif parts and parts[0] == "2":
                    f += 1
        si_inst += s; fa_inst += f
        if s > 0: si_img += 1
        if f > 0: fa_img += 1
    return si_inst, fa_inst, si_img, fa_img


def main():
    global IMAGES_DIR, LABELS_DIR, OUT_DIR
    ds_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(ds_root, "images"),
                    help="图片目录（默认 <数据集根>/images）")
    ap.add_argument("--labels", default=os.path.join(ds_root, "labels"),
                    help="标签目录（默认 <数据集根>/labels）")
    ap.add_argument("--out-dir", default=ds_root,
                    help="train_strat.txt / val_strat.txt 输出目录（默认 <数据集根>）")
    a = ap.parse_args()
    IMAGES_DIR, LABELS_DIR, OUT_DIR = a.images, a.labels, a.out_dir

    jpgs = sorted(os.path.relpath(p, IMAGES_DIR).replace(os.sep, "/")
                  for p in glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    print(f"共发现 {len(jpgs)} 张图片（images={IMAGES_DIR}, labels={LABELS_DIR}）")

    groups = defaultdict(list)
    for rel in jpgs:
        stem = os.path.splitext(os.path.basename(rel))[0]
        lp = os.path.join(LABELS_DIR, stem + ".txt")
        cs = defect_classes(lp)
        groups[frozenset(cs)].append(rel)

    random.seed(SEED)
    train_list, val_list = [], []
    print("\n组内分层（8:2）：")
    print(f"  {'组':<22}{'总数':>6}{'train':>8}{'val':>6}")
    for key in sorted(groups.keys(), key=lambda k: (len(k), sorted(k))):
        items = groups[key][:]
        random.shuffle(items)
        n_val = max(1, round(len(items) * VAL_RATIO)) if len(items) >= 5 else 0
        val = items[:n_val]
        train = items[n_val:]
        train_list.extend(train)
        val_list.extend(val)
        name = GROUP_NAME.get(key, str(sorted(key)))
        print(f"  {name:<20}{len(items):>6}{len(train):>8}{len(val):>6}")

    random.shuffle(train_list)
    random.shuffle(val_list)

    train_abs = [to_abs_backslash(p) for p in train_list]
    val_abs = [to_abs_backslash(p) for p in val_list]
    train_path = os.path.join(OUT_DIR, "train_strat.txt")
    val_path = os.path.join(OUT_DIR, "val_strat.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_abs) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_abs) + "\n")

    print(f"\ntrain_strat.txt: {len(train_list)} 张 -> {train_path}")
    print(f"val_strat.txt:   {len(val_list)} 张 -> {val_path}")

    # 实例/图数分布
    t_si, t_fa, t_si_img, t_fa_img = count_instances(train_list)
    v_si, v_fa, v_si_img, v_fa_img = count_instances(val_list)
    t_neg = sum(1 for r in train_list if not defect_classes(
        os.path.join(LABELS_DIR, os.path.splitext(os.path.basename(r))[0] + ".txt")))
    v_neg = sum(1 for r in val_list if not defect_classes(
        os.path.join(LABELS_DIR, os.path.splitext(os.path.basename(r))[0] + ".txt")))

    print("\n=== train_strat ===")
    print(f"  silie: {t_si} 实例 / {t_si_img} 图 | fanzhe: {t_fa} 实例 / {t_fa_img} 图 | 负样本: {t_neg} 图 ({t_neg/len(train_list)*100:.1f}%)")
    print("=== val_strat ===")
    print(f"  silie: {v_si} 实例 / {v_si_img} 图 | fanzhe: {v_fa} 实例 / {v_fa_img} 图 | 负样本: {v_neg} 图 ({v_neg/len(val_list)*100:.1f}%)")


if __name__ == "__main__":
    main()
