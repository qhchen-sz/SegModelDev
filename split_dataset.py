# -*- coding: utf-8 -*-
"""
将 images/ 下的图片按 8:2 分层划分成训练集 / 验证集。
非破坏性：不移动任何图片/标签，仅生成 train.txt / val.txt 两个列表文件。
可重复运行（幂等）。
"""
import os
import glob
import random
from collections import defaultdict

# 项目根目录（脚本所在目录）
ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(ROOT, "images")
LABELS_DIR = os.path.join(ROOT, "labels")

VAL_RATIO = 0.2
SEED = 42


def parse_classes(label_path):
    """读取一个 YOLO 标签文件，返回该图中出现的类别集合（每行首列）。"""
    classes = set()
    if not os.path.exists(label_path):
        return classes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts and parts[0].isdigit():
                classes.add(int(parts[0]))
    return classes


def main():
    # 1. 收集全部 .jpg（相对路径 images/xxx.jpg）
    jpgs = sorted(
        os.path.relpath(p, ROOT).replace(os.sep, "/")
        for p in glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
    )
    print(f"共发现 {len(jpgs)} 张图片")

    # 2. 按类别组合分层
    groups = defaultdict(list)
    missing_labels = []
    for img_rel in jpgs:
        stem = os.path.splitext(os.path.basename(img_rel))[0]
        label_path = os.path.join(LABELS_DIR, stem + ".txt")
        if not os.path.exists(label_path):
            missing_labels.append(img_rel)
            continue
        classes = parse_classes(label_path)
        # 用 frozenset 作为分层键（保证同类组合的图分在一起）
        groups[frozenset(classes)].append(img_rel)

    if missing_labels:
        print(f"[警告] {len(missing_labels)} 张图无对应标签，已剔除：")
        for m in missing_labels:
            print(f"  - {m}")

    # 3. 每组内打乱后按比例切分
    random.seed(SEED)
    train_list, val_list = [], []
    for key, items in groups.items():
        items = items[:]  # 拷贝
        random.shuffle(items)
        n_val = max(1, round(len(items) * VAL_RATIO)) if len(items) >= 5 else 0
        # 组内样本太少时全留训练集，避免少数类在 val 中消失
        val = items[:n_val]
        train = items[n_val:]
        train_list.extend(train)
        val_list.extend(val)
        print(f"  组(类别={sorted(key)}) 共{len(items)}: train={len(train)} val={len(val)}")

    # 4. 整体再各打乱一次（避免同类聚集）
    random.shuffle(train_list)
    random.shuffle(val_list)

    # 5. 写出列表文件（绝对路径，Windows 下为反斜杠，确保 ultralytics
    #    的 img2label_paths 能匹配 \images\ -> \labels\ 子串替换）
    def to_abs(rel):
        return os.path.abspath(os.path.join(ROOT, rel))

    train_abs = [to_abs(p) for p in train_list]
    val_abs = [to_abs(p) for p in val_list]
    train_path = os.path.join(ROOT, "train.txt")
    val_path = os.path.join(ROOT, "val.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_abs) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_abs) + "\n")

    print(f"\ntrain.txt: {len(train_list)} 张 -> {train_path}")
    print(f"val.txt:   {len(val_list)} 张 -> {val_path}")

    # 6. 统计各类别在 train/val 的分布
    def class_counts(img_list):
        cnt = defaultdict(int)
        for img_rel in img_list:
            stem = os.path.splitext(os.path.basename(img_rel))[0]
            for c in parse_classes(os.path.join(LABELS_DIR, stem + ".txt")):
                cnt[c] += 1
        return cnt

    tc = class_counts(train_list)
    vc = class_counts(val_list)
    print("\n各类别图片数（train / val）：")
    for c in sorted(set(tc) | set(vc)):
        print(f"  类{c}: train={tc[c]}  val={vc[c]}")

    # 7. 删除陈旧缓存，让 ultralytics 重建
    for cache in [
        os.path.join(ROOT, "labels.cache"),
        os.path.join(LABELS_DIR, "labels.cache"),
    ]:
        if os.path.exists(cache):
            os.remove(cache)
            print(f"\n已删除陈旧缓存: {cache}")


if __name__ == "__main__":
    main()
