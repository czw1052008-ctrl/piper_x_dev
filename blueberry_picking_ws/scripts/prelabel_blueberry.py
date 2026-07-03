#!/usr/bin/env python3
"""Auto-generate YOLO bbox labels with open-vocab YOLOE; review in annotate_blueberry.py.

Usage:
  python3 scripts/prelabel_blueberry.py
  python3 scripts/prelabel_blueberry.py --images datasets/blueberry/images --overwrite
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'picking_perception'))

from picking_perception.yolo_berry_detector import YoloBerryDetector  # noqa: E402

CLASS_ID = 0


def _list_images(image_dir: str) -> list[str]:
    paths: list[str] = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(paths)


def _bbox_to_yolo_line(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> str:
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    cx = (x0 + x1) * 0.5 / w
    cy = (y0 + y1) * 0.5 / h
    bw = max(x1 - x0, 1) / w
    bh = max(y1 - y0, 1) / h
    return f'{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}'


def _label_path(label_dir: str, image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(label_dir, f'{stem}.txt')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', default=os.path.join(ROOT, 'datasets', 'blueberry', 'images'))
    parser.add_argument('--labels', default=os.path.join(ROOT, 'datasets', 'blueberry', 'labels'))
    parser.add_argument('--model', default='yoloe-11s-seg-pf.pt')
    parser.add_argument('--conf', type=float, default=0.20)
    parser.add_argument('--prompts', default='blueberry,blueberries')
    parser.add_argument('--prefix', default='',
                        help='Only process images whose filename starts with this prefix')
    parser.add_argument('--custom-model', action='store_true',
                        help='Use trained detect weights (e.g. runs/.../best.pt); no text prompts')
    parser.add_argument('--max-detections', type=int, default=12)
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing label files')
    parser.add_argument('--preview-dir', default='',
                        help='Optional dir to save preview images with boxes drawn')
    args = parser.parse_args()

    image_dir = os.path.abspath(args.images)
    label_dir = os.path.abspath(args.labels)
    os.makedirs(label_dir, exist_ok=True)

    paths = _list_images(image_dir)
    if args.prefix:
        paths = [p for p in paths if os.path.basename(p).startswith(args.prefix)]
    if not paths:
        print(f'ERROR: no images in {image_dir}', file=sys.stderr)
        return 1

    prompts = [p.strip() for p in args.prompts.split(',') if p.strip()]
    open_vocab = not args.custom_model
    detector = YoloBerryDetector(
        model_name=args.model,
        class_prompts=prompts,
        conf_threshold=args.conf,
        max_detections=args.max_detections,
        open_vocab=open_vocab,
    )
    if not detector.ready:
        print('ERROR: YOLO model not available (install ultralytics in conda env foundationpose)',
              file=sys.stderr)
        return 1

    preview_dir = os.path.abspath(args.preview_dir) if args.preview_dir else ''
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    written = 0
    skipped = 0
    empty = 0
    total_boxes = 0

    print(f'[prelabel] {len(paths)} images | model={args.model} conf={args.conf}')
    print(f'[prelabel] labels -> {label_dir}')
    print('[prelabel] Then review: bash scripts/run_annotate_blueberry.sh')

    for i, path in enumerate(paths, 1):
        lp = _label_path(label_dir, path)
        if os.path.isfile(lp) and not args.overwrite:
            skipped += 1
            continue

        bgr = cv2.imread(path)
        if bgr is None:
            print(f'[prelabel] WARN: cannot read {path}', file=sys.stderr)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        dets = detector.detect(rgb)

        lines = [_bbox_to_yolo_line(*det.bbox_xyxy, w, h) for det in dets]
        with open(lp, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            if lines:
                f.write('\n')

        written += 1
        total_boxes += len(lines)
        if not lines:
            empty += 1

        if preview_dir:
            vis = bgr.copy()
            for det in dets:
                x0, y0, x1, y1 = det.bbox_xyxy
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(vis, f'{det.confidence:.2f}', (x0, max(y0 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            out_name = os.path.basename(path)
            cv2.imwrite(os.path.join(preview_dir, out_name), vis)

        print(f'[prelabel] {i}/{len(paths)} {os.path.basename(path)}: {len(lines)} boxes')

    print(f'[prelabel] Done: wrote={written} skipped={skipped} empty={empty} boxes={total_boxes}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
