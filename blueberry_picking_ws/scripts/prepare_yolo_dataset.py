#!/usr/bin/env python3
"""Split annotated images/labels into YOLO train/val and write data.yaml."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from glob import glob


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', default=os.path.join(root, 'datasets', 'blueberry', 'images'))
    parser.add_argument('--labels', default=os.path.join(root, 'datasets', 'blueberry', 'labels'))
    parser.add_argument('--out', default=os.path.join(root, 'datasets', 'blueberry'))
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Val split ratio (default 0.1 → 10%% for ~100 images)')
    parser.add_argument('--val-count', type=int, default=0,
                        help='Exact val count; overrides --val-ratio if > 0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    images = sorted(
        p for p in glob(os.path.join(args.images, '*'))
        if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    )
    paired = []
    for img in images:
        stem = os.path.splitext(os.path.basename(img))[0]
        lab = os.path.join(args.labels, f'{stem}.txt')
        if os.path.isfile(lab):
            paired.append((img, lab))
        else:
            print(f'[skip] no label: {stem}')

    if len(paired) < 5:
        raise SystemExit(f'Need at least 5 labeled images, got {len(paired)}')

    random.seed(args.seed)
    random.shuffle(paired)
    if args.val_count > 0:
        n_val = min(args.val_count, len(paired) - 1)
    else:
        n_val = max(1, int(len(paired) * args.val_ratio))
    val_set = paired[:n_val]
    train_set = paired[n_val:]

    for split, items in ('train', train_set), ('val', val_set):
        img_dir = os.path.join(args.out, split, 'images')
        lab_dir = os.path.join(args.out, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        for old in glob(os.path.join(img_dir, '*')):
            os.remove(old)
        for old in glob(os.path.join(lab_dir, '*')):
            os.remove(old)
        for img, lab in items:
            name = os.path.basename(img)
            shutil.copy2(img, os.path.join(img_dir, name))
            shutil.copy2(lab, os.path.join(lab_dir, os.path.basename(lab)))

    yaml_path = os.path.join(args.out, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f'path: {args.out}\n')
        f.write('train: train/images\n')
        f.write('val: val/images\n')
        f.write('\nnames:\n')
        f.write('  0: blueberry\n')

    print(f'[prepare] train={len(train_set)} val={len(val_set)}')
    print(f'[prepare] wrote {yaml_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
