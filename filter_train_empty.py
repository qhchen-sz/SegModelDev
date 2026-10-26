# -*- coding: utf-8 -*-
"""
从 train.txt 中剔除“空标签图”（标签文件无任何目标行）。
- val.txt 不读不写。
- 不动 images/ 与 labels/ 原始文件，只就地改写 train.txt。
- 可逆：重跑 split_dataset.py 即可恢复含空图的 358 行版本。
"""
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(ROOT, "labels")
TRAIN_TXT = os.path.join(ROOT, "train.txt")


def label_has_objects(img_path):
    """图片路径 -> 对应 labels/<stem>.txt 是否含 >=1 行有效目标。"""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABELS_DIR, stem + ".txt")
    if not os.path.exists(label_path):
        return False
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts and parts[0].isdigit():
                return True
    return False


def parse_classes(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABELS_DIR, stem + ".txt")
    classes = set()
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts and parts[0].isdigit():
                    classes.add(int(parts[0]))
    return classes


def main():
    with open(TRAIN_TXT, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    before = len(lines)
    kept = [p for p in lines if label_has_objects(p)]
    removed = before - len(kept)

    with open(TRAIN_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")

    print(f"train.txt: {before} -> {len(kept)} (剔除空标签图 {removed} 张)")

    # 各类别分布
    cnt = defaultdict(int)
    for p in kept:
        for c in parse_classes(p):
            cnt[c] += 1
    print("保留 train 各类别图片数:", dict(cnt))

    # 删陈旧缓存
    for cache in [os.path.join(ROOT, "labels.cache"),
                  os.path.join(LABELS_DIR, "labels.cache")]:
        if os.path.exists(cache):
            os.remove(cache)
            print(f"已删除缓存: {cache}")


if __name__ == "__main__":
    main()
