#!/usr/bin/env python3
"""Compare VLM pseudo-labels against human GT (YOLO format)."""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import List, Tuple


def _load_boxes(path: str, w: int, h: int) -> List[Tuple[int, int, int, int]]:
    if not os.path.isfile(path):
        return []
    boxes = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x0 = int((cx - bw / 2) * w)
            y0 = int((cy - bh / 2) * h)
            x1 = int((cx + bw / 2) * w)
            y1 = int((cy + bh / 2) * h)
            boxes.append((x0, y0, x1, y1))
    return boxes


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / float(area_a + area_b - inter)


def _match_boxes(
    gt: List[Tuple[int, int, int, int]],
    pred: List[Tuple[int, int, int, int]],
    iou_thresh: float = 0.5,
) -> Tuple[int, int, int, List[float]]:
    """Return (tp, fp, fn, matched_ious)."""
    if not gt and not pred:
        return 0, 0, 0, []
    if not gt:
        return 0, len(pred), 0, []
    if not pred:
        return 0, 0, len(gt), []

    pairs = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            pairs.append((_iou(g, p), gi, pi))
    pairs.sort(reverse=True)

    used_g, used_p = set(), set()
    matched_ious: List[float] = []
    for iou_val, gi, pi in pairs:
        if iou_val < iou_thresh:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched_ious.append(iou_val)

    tp = len(matched_ious)
    fp = len(pred) - len(used_p)
    fn = len(gt) - len(used_g)
    return tp, fp, fn, matched_ious


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', default=os.path.join(root, 'datasets', 'fixed_mono', 'images'))
    parser.add_argument('--gt', default=os.path.join(root, 'datasets', 'fixed_mono', 'labels_gt'))
    parser.add_argument('--pred', default=os.path.join(root, 'datasets', 'fixed_mono', 'labels_vlm'))
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--json-out', default='')
    args = parser.parse_args()

    import cv2

    rows = []
    total_tp = total_fp = total_fn = 0
    all_ious: List[float] = []
    count_exact = 0

    for img_path in sorted(glob(os.path.join(args.images, '*'))):
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        im = cv2.imread(img_path)
        if im is None:
            continue
        h, w = im.shape[:2]
        gt_boxes = _load_boxes(os.path.join(args.gt, f'{stem}.txt'), w, h)
        pred_boxes = _load_boxes(os.path.join(args.pred, f'{stem}.txt'), w, h)
        tp, fp, fn, ious = _match_boxes(gt_boxes, pred_boxes, args.iou)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)
        exact = len(gt_boxes) == len(pred_boxes)
        if exact:
            count_exact += 1
        rows.append({
            'stem': stem,
            'gt_n': len(gt_boxes),
            'pred_n': len(pred_boxes),
            'tp': tp, 'fp': fp, 'fn': fn,
            'mean_iou': sum(ious) / len(ious) if ious else 0.0,
            'count_match': exact,
        })

    n = len(rows)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

    summary = {
        'n_images': n,
        'iou_thresh': args.iou,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'mean_matched_iou': mean_iou,
        'count_exact_match': count_exact,
        'per_image': rows,
        'worst': sorted(rows, key=lambda r: (r['tp'], r['mean_iou']))[:5],
    }

    print(f'[eval] images={n} IoU@{args.iou}')
    print(f'  TP={total_tp} FP={total_fp} FN={total_fn}')
    print(f'  P={precision:.3f} R={recall:.3f} F1={f1:.3f} mean_IoU={mean_iou:.3f}')
    print(f'  count_exact={count_exact}/{n}')
    print('  worst:')
    for r in summary['worst']:
        print(f"    {r['stem']}: gt={r['gt_n']} pred={r['pred_n']} "
              f"tp={r['tp']} fp={r['fp']} fn={r['fn']} iou={r['mean_iou']:.2f}")

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
