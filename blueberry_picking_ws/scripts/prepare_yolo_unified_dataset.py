#!/usr/bin/env python3
"""Merge wrist (class 0=berry) + fixed mono (class 1=cluster) into one YOLO dataset.

One model, two classes — deploy same weights on global + wrist nodes with class filter.

Usage:
  python3 scripts/prepare_yolo_unified_dataset.py
  python3 scripts/prepare_yolo_unified_dataset.py --fixed-images datasets/fixed_mono/images
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from glob import glob


def _collect_pairs(images_dir: str, labels_dir: str, prefix: str = '') -> list[tuple[str, str]]:
    out = []
    for img in sorted(glob(os.path.join(images_dir, '*'))):
        if not img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        stem = os.path.splitext(os.path.basename(img))[0]
        lab = os.path.join(labels_dir, f'{stem}.txt')
        if not os.path.isfile(lab):
            print(f'[skip] no label: {stem}')
            continue
        out.append((img, lab))
    return out


def _remap_label_file(src_lab: str, dst_lab: str, class_map: dict[int, int]) -> None:
    lines_out = []
    with open(src_lab, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            old_cls = int(parts[0])
            new_cls = class_map.get(old_cls, old_cls)
            parts[0] = str(new_cls)
            lines_out.append(' '.join(parts))
    with open(dst_lab, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_out) + ('\n' if lines_out else ''))


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrist-images', default=os.path.join(root, 'datasets', 'blueberry', 'images'))
    parser.add_argument('--wrist-labels', default=os.path.join(root, 'datasets', 'blueberry', 'labels'))
    parser.add_argument('--fixed-images', default=os.path.join(root, 'datasets', 'fixed_mono', 'images'))
    parser.add_argument('--fixed-labels', default=os.path.join(root, 'datasets', 'fixed_mono', 'labels'))
    parser.add_argument('--out', default=os.path.join(root, 'datasets', 'blueberry_unified'))
    parser.add_argument('--val-count', type=int, default=24)
    parser.add_argument('--fixed-val-count', type=int, default=4,
                        help='fixed_mono images held out for val (stratified)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    wrist = _collect_pairs(args.wrist_images, args.wrist_labels)
    fixed = _collect_pairs(args.fixed_images, args.fixed_labels)

    # Wrist labels: class 0 blueberry → 0 berry
    # Fixed labels: human annotate uses 0 → remap to 1 cluster at copy time
    random.seed(args.seed)
    fixed_shuf = list(fixed)
    random.shuffle(fixed_shuf)
    n_fixed_val = min(args.fixed_val_count, max(1, len(fixed_shuf) - 1), len(fixed_shuf))
    fixed_val = fixed_shuf[:n_fixed_val]
    fixed_train = fixed_shuf[n_fixed_val:]

    wrist_shuf = list(wrist)
    random.shuffle(wrist_shuf)
    n_wrist_val = min(max(0, args.val_count - n_fixed_val), len(wrist_shuf) - 1)
    wrist_val = wrist_shuf[:n_wrist_val]
    wrist_train = wrist_shuf[n_wrist_val:]

    train_set: list[tuple[str, str, str]] = [
        (img, lab, 'wrist') for img, lab in wrist_train
    ] + [(img, lab, 'fixed') for img, lab in fixed_train]
    val_set: list[tuple[str, str, str]] = [
        (img, lab, 'wrist') for img, lab in wrist_val
    ] + [(img, lab, 'fixed') for img, lab in fixed_val]
    random.shuffle(train_set)
    random.shuffle(val_set)

    merged_count = len(wrist) + len(fixed)
    if merged_count < 5:
        raise SystemExit(f'Need ≥5 labeled images total, got {merged_count} '
                         f'(wrist={len(wrist)} fixed={len(fixed)})')

    for split, items in ('train', train_set), ('val', val_set):
        img_dir = os.path.join(args.out, split, 'images')
        lab_dir = os.path.join(args.out, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        for old in glob(os.path.join(img_dir, '*')):
            os.remove(old)
        for old in glob(os.path.join(lab_dir, '*')):
            os.remove(old)

        for img, lab, source in items:
            stem = os.path.splitext(os.path.basename(img))[0]
            name = f'{source}_{stem}.png' if img.lower().endswith('.png') else f'{source}_{stem}.jpg'
            dst_img = os.path.join(img_dir, name)
            dst_lab = os.path.join(lab_dir, os.path.splitext(name)[0] + '.txt')
            shutil.copy2(img, dst_img)
            if source == 'wrist':
                _remap_label_file(lab, dst_lab, {0: 0})  # berry stays 0
            else:
                _remap_label_file(lab, dst_lab, {1: 1, 0: 1})  # force cluster=1

    yaml_path = os.path.join(args.out, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f'path: {args.out}\n')
        f.write('train: train/images\n')
        f.write('val: val/images\n')
        f.write('\nnames:\n')
        f.write('  0: berry\n')
        f.write('  1: cluster\n')

    print(f'[unified] wrist={len(wrist)} fixed={len(fixed)}')
    print(f'[unified] train={len(train_set)} (wrist {len(wrist_train)} + fixed {len(fixed_train)})')
    print(f'[unified] val={len(val_set)} (wrist {len(wrist_val)} + fixed {len(fixed_val)})')
    print(f'[unified] wrote {yaml_path}')
    print('Train: bash scripts/train_blueberry_yolo_v5.sh blueberry-unified-1 \\')
    print(f'  --data {yaml_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
