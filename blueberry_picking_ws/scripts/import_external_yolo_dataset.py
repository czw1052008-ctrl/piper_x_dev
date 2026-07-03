#!/usr/bin/env python3
"""Import a YOLO-format dataset (e.g. Roboflow export) into local blueberry folder.

Roboflow: Export -> YOLOv11 -> unzip to datasets/blueberry/imports/<name>/

Expected layout (any split names ok):
  <import_dir>/train/images/*.jpg
  <import_dir>/train/labels/*.txt
  <import_dir>/valid/images/ ...  (or val/)

Copies into datasets/blueberry/images + labels with prefix to avoid name clashes.
Class id is remapped to 0 (blueberry). Other classes are skipped with a warning.

Usage:
  python3 scripts/import_external_yolo_dataset.py datasets/blueberry/imports/roboflow_web --prefix web_
  bash scripts/run_local_yolo_pipeline.sh prepare
  bash scripts/run_local_yolo_pipeline.sh train
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys


def _find_splits(import_dir: str) -> list[tuple[str, str]]:
    splits: list[tuple[str, str]] = []
    for name in ('train', 'valid', 'val', 'test'):
        img_dir = os.path.join(import_dir, name, 'images')
        lab_dir = os.path.join(import_dir, name, 'labels')
        if os.path.isdir(img_dir) and os.path.isdir(lab_dir):
            splits.append((img_dir, lab_dir))
    if splits:
        return splits
    # flat layout: images/ + labels/
    img_dir = os.path.join(import_dir, 'images')
    lab_dir = os.path.join(import_dir, 'labels')
    if os.path.isdir(img_dir) and os.path.isdir(lab_dir):
        return [(img_dir, lab_dir)]
    return []


def _remap_label(src_lab: str, dst_lab: str, keep_class: int) -> int:
    """Return number of boxes written."""
    out_lines: list[str] = []
    with open(src_lab, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls != keep_class:
                continue
            out_lines.append('0 ' + ' '.join(parts[1:5]))
    with open(dst_lab, 'w', encoding='utf-8') as f:
        if out_lines:
            f.write('\n'.join(out_lines) + '\n')
    return len(out_lines)


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('import_dir', help='Unzipped YOLO export directory')
    parser.add_argument('--prefix', default='web_', help='Filename prefix (default: web_)')
    parser.add_argument('--out-images', default=os.path.join(root, 'datasets', 'blueberry', 'images'))
    parser.add_argument('--out-labels', default=os.path.join(root, 'datasets', 'blueberry', 'labels'))
    parser.add_argument('--class-id', type=int, default=0,
                        help='Only import this class id from external set (default: 0)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    import_dir = os.path.abspath(args.import_dir)
    splits = _find_splits(import_dir)
    if not splits:
        print(f'ERROR: no images/labels splits under {import_dir}', file=sys.stderr)
        print('  Export Roboflow as YOLOv11 and unzip there.', file=sys.stderr)
        return 1

    os.makedirs(args.out_images, exist_ok=True)
    os.makedirs(args.out_labels, exist_ok=True)

    copied = 0
    skipped = 0
    total_boxes = 0

    for img_dir, lab_dir in splits:
        for img_path in sorted(glob.glob(os.path.join(img_dir, '*'))):
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                continue
            stem = os.path.splitext(os.path.basename(img_path))[0]
            src_lab = os.path.join(lab_dir, f'{stem}.txt')
            if not os.path.isfile(src_lab):
                skipped += 1
                continue

            ext = os.path.splitext(img_path)[1]
            out_stem = f'{args.prefix}{stem}'
            dst_img = os.path.join(args.out_images, out_stem + ext)
            dst_lab = os.path.join(args.out_labels, out_stem + '.txt')

            if args.dry_run:
                print(f'[dry-run] {stem} -> {out_stem}')
                copied += 1
                continue

            shutil.copy2(img_path, dst_img)
            n = _remap_label(src_lab, dst_lab, args.class_id)
            total_boxes += n
            copied += 1

    print(f'[import] copied={copied} skipped_no_label={skipped} boxes={total_boxes}')
    print(f'[import] -> {args.out_images}')
    print('[import] Next: bash scripts/run_local_yolo_pipeline.sh prepare && ... train')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
