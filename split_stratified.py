# -*- coding: utf-8 -*-
"""
按"类别组合"分层重切（非破坏性）：全部 446 张图，按每张图出现的缺陷类别组合
分组（neg / silie-only / fanzhe-only / both），组内 8:2 分层，seed=42。
仅生成新文件 train_strat.txt / val_strat.txt，绝不改动 train.txt / val.txt。
绝对路径 + 反斜杠（兼容 ultralytics img2label_paths 的 \images\→\labels\ 替换）。
"""
import os, glob, random
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(ROOT, "images")
LABELS_DIR = os.path.join(ROOT, "labels")

VAL_RATIO = 0.2
SEED = 42

GROUP_NAME = {
    frozenset(): "neg(纯背景)",
    frozenset({1}): "silie-only",
    frozenset({2}): "fanzhe-only",
    frozenset({1, 2}): "both(silie+fanzhe)",
}


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
    """images/xxx.jpg -> C:\\...\\images\\xxx.jpg（ultralytics 要反斜杠绝对路径）。"""
    return os.path.abspath(os.path.join(ROOT, rel)).replace("/", os.sep)


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
    jpgs = sorted(os.path.relpath(p, ROOT).replace(os.sep, "/")
                  for p in glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    print(f"共发现 {len(jpgs)} 张图片")

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
    train_path = os.path.join(ROOT, "train_strat.txt")
    val_path = os.path.join(ROOT, "val_strat.txt")
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
