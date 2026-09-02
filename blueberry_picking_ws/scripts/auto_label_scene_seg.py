#!/usr/bin/env python3
"""First-pass berry/branch/rigid/ego masks for scene_seg. Human review expected.

Berry: YOLO (if available) + dark-blue blobs.
Branch: thin dark/brown structure (not leaves, not a canopy blob).
Rigid: pot / desk / tools / monitor in the near field.
Ego is not auto-painted (gripper looks like metal); mark it in the annotator.

Writes YOLO-seg txt, PNG masks (0=bg,1=berry,2=branch,3=rigid,4=ego), and overlay.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASS_BERRY, CLASS_BRANCH, CLASS_RIGID = 0, 1, 2
PIX_BERRY, PIX_BRANCH, PIX_RIGID = 1, 2, 3
YOLO_WEIGHTS = ROOT / 'runs' / 'detect' / 'blueberry-unified-1' / 'weights' / 'best.pt'


def _load_yolo(weights: Path, conf: float):
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    if not weights.is_file():
        return None
    model = YOLO(str(weights))
    model.overrides['conf'] = conf
    return model


def _yolo_boxes(bgr: np.ndarray, model, conf: float):
    """Return (berry_xyxy, cluster_xyxy) in pixel coords."""
    empty = np.zeros((0, 4), dtype=np.float32)
    if model is None:
        return empty, empty
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        res = model.predict(rgb, conf=conf, verbose=False, imgsz=640)[0]
    except Exception:
        return empty, empty
    names = res.names or {}
    boxes = getattr(res, 'boxes', None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return empty, empty
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
    berries, clusters = [], []
    for i, box in enumerate(xyxy):
        cid = int(cls[i]) if i < len(cls) else 0
        name = str(names.get(cid, '')).lower()
        if name in ('berry', 'blueberry', 'blueberries') or cid == 0:
            if name not in ('cluster', 'plant'):
                berries.append(box)
                continue
        if name in ('cluster', 'plant') or cid == 1:
            clusters.append(box)
    b = np.array(berries, dtype=np.float32) if berries else empty
    c = np.array(clusters, dtype=np.float32) if clusters else empty
    return b, c


def _roi_from_boxes(shape, boxes: np.ndarray, pad: int) -> np.ndarray:
    h, w = shape[:2]
    roi = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w - 1, x1 + pad)
        y1 = min(h - 1, y1 + pad)
        roi[y0:y1 + 1, x0:x1 + 1] = 255
    if roi.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21))
        roi = cv2.dilate(roi, k, iterations=1)
    return roi


def _plant_roi(bgr: np.ndarray, berry_boxes: np.ndarray, cluster_boxes: np.ndarray, wrist: bool) -> np.ndarray:
    h, w = bgr.shape[:2]
    roi = _roi_from_boxes((h, w), cluster_boxes, pad=28)
    if berry_boxes.size:
        roi = cv2.bitwise_or(roi, _roi_from_boxes((h, w), berry_boxes, pad=48 if wrist else 22))
    if roi.any():
        # include stem / pot below the fruit
        ys, xs = np.where(roi > 0)
        x0, x1 = int(xs.min()), int(xs.max())
        y1 = min(h - 1, int(ys.max() + (0.22 * h if wrist else 0.18 * h)))
        y0 = max(0, int(ys.min() - 0.04 * h))
        extra = np.zeros_like(roi)
        extra[y0:y1 + 1, max(0, x0 - 12):min(w, x1 + 12)] = 255
        roi = cv2.bitwise_or(roi, extra)
        return roi
    # fallback: central column (fixed) / center (wrist)
    roi = np.zeros((h, w), dtype=np.uint8)
    if wrist:
        roi[int(0.05 * h):int(0.85 * h), int(0.28 * w):int(0.78 * w)] = 255
    else:
        roi[int(0.08 * h):int(0.78 * h), int(0.12 * w):int(0.52 * w)] = 255
    return roi


def _color_berry_in_roi(bgr: np.ndarray, roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (100, 40, 15), (140, 255, 130))
    m = cv2.bitwise_and(m, roi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    h, w = bgr.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 10 or area > 0.012 * w * h:
            continue
        peri = float(cv2.arcLength(c, True))
        if peri < 1e-3:
            continue
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < 0.35:
            continue
        cv2.drawContours(out, [c], -1, 255, -1)
    return out


def _ellipse_from_boxes(shape, boxes: np.ndarray) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        if bw * bh < 16 or bw * bh > 0.04 * w * h:
            continue
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        cv2.ellipse(
            mask, (int(cx), int(cy)),
            (max(2, int(0.48 * bw)), max(2, int(0.48 * bh))),
            0, 0, 360, 255, -1)
    return mask


def _green_leaf_mask(hsv: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv, (32, 35, 35), (92, 255, 230))


def _branch_mask(bgr: np.ndarray, berry: np.ndarray, roi: np.ndarray, wrist: bool) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]
    brown = cv2.inRange(hsv, (0, 25, 12), (30, 210, 150))
    dark = cv2.inRange(hsv, (0, 0, 8), (180, 100, 70))
    raw = cv2.bitwise_or(brown, dark)
    raw = cv2.bitwise_and(raw, roi)
    raw[berry > 0] = 0
    raw[_green_leaf_mask(hsv) > 0] = 0
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k3)
    keep = np.zeros((h, w), dtype=np.uint8)
    max_area = 0.045 * w * h
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 12 or area > max_area:
            continue
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        long, short = max(rw, rh), max(1.0, min(rw, rh))
        aspect = long / short
        if aspect >= 1.8 or short <= 12:
            cv2.drawContours(keep, [c], -1, 255, -1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    keep = cv2.dilate(keep, k2, iterations=1)
    keep[berry > 0] = 0
    keep[roi == 0] = 0
    if wrist:
        keep[int(0.70 * h):, :] = 0
    else:
        keep[:, int(0.56 * w):] = 0
    return keep


def _near_valid(depth_u16: Optional[np.ndarray], max_mm: int) -> Optional[np.ndarray]:
    if depth_u16 is None:
        return None
    d = depth_u16
    return ((d > 80) & (d < max_mm)).astype(np.uint8) * 255


def _rigid_mask(
    bgr: np.ndarray,
    berry: np.ndarray,
    branch: np.ndarray,
    plant_roi: np.ndarray,
    depth_u16: Optional[np.ndarray],
    wrist: bool,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]
    wood = cv2.inRange(hsv, (6, 18, 70), (30, 170, 235))
    white = cv2.inRange(hsv, (0, 0, 165), (180, 55, 255))
    blue_tool = cv2.inRange(hsv, (88, 70, 35), (130, 255, 255))
    yellow = cv2.inRange(hsv, (18, 80, 80), (38, 255, 255))
    metal = cv2.inRange(hsv, (0, 0, 70), (180, 40, 200))
    black_obj = cv2.inRange(hsv, (0, 0, 8), (180, 80, 55))
    raw = wood | white | blue_tool | yellow | metal
    if wrist:
        raw[int(0.62 * h):, :] = cv2.bitwise_or(raw[int(0.62 * h):, :], black_obj[int(0.62 * h):, :])
    else:
        # robot on the right of the fixed camera
        raw[:, int(0.58 * w):] = cv2.bitwise_or(raw[:, int(0.58 * w):], black_obj[:, int(0.58 * w):])
    # pot is rigid even inside plant ROI (white, lower half of ROI)
    pot = cv2.bitwise_and(white, plant_roi)
    if plant_roi.any():
        ys = np.where(plant_roi > 0)[0]
        y_cut = int(np.percentile(ys, 55))
        pot[:y_cut, :] = 0
    raw = cv2.bitwise_or(raw, pot)
    # do not paint canopy as desk
    canopy = plant_roi.copy()
    canopy[pot > 0] = 0
    raw[canopy > 0] = 0
    near = _near_valid(depth_u16, 1800 if wrist else 2500)
    if near is not None:
        raw = cv2.bitwise_and(raw, near)
        # black cables on desk if they have depth
        cab = cv2.bitwise_and(black_obj, near)
        raw = cv2.bitwise_or(raw, cab)
    raw[berry > 0] = 0
    raw[branch > 0] = 0
    raw[_green_leaf_mask(hsv) > 0] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    keep = np.zeros((h, w), dtype=np.uint8)
    min_area = 180 if wrist else 400
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        cv2.drawContours(keep, [c], -1, 255, -1)
    keep[berry > 0] = 0
    keep[branch > 0] = 0
    return keep


def _to_polys(mask: np.ndarray, class_id: int, min_area: float, simplify: float) -> List[str]:
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lines: List[str] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        eps = max(1.0, simplify * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float64)
        coords = []
        for x, y in pts:
            coords.append(f'{float(np.clip(x / w, 0, 1)):.6f}')
            coords.append(f'{float(np.clip(y / h, 0, 1)):.6f}')
        lines.append(f'{class_id} ' + ' '.join(coords))
    return lines


def label_one(
    bgr: np.ndarray,
    depth_u16: Optional[np.ndarray],
    *,
    yolo,
    yolo_conf: float,
    wrist: bool,
) -> Tuple[np.ndarray, List[str]]:
    berry_boxes, cluster_boxes = _yolo_boxes(bgr, yolo, yolo_conf)
    roi = _plant_roi(bgr, berry_boxes, cluster_boxes, wrist=wrist)
    berry = _color_berry_in_roi(bgr, roi)
    if berry_boxes.size:
        berry = cv2.bitwise_or(berry, cv2.bitwise_and(_ellipse_from_boxes(bgr.shape, berry_boxes), roi))
    branch = _branch_mask(bgr, berry, roi, wrist=wrist)
    rigid = _rigid_mask(bgr, berry, branch, roi, depth_u16, wrist=wrist)
    h, w = bgr.shape[:2]
    pix = np.zeros((h, w), dtype=np.uint8)
    pix[rigid > 0] = PIX_RIGID
    pix[branch > 0] = PIX_BRANCH
    pix[berry > 0] = PIX_BERRY
    lines: List[str] = []
    lines += _to_polys(berry, CLASS_BERRY, min_area=10, simplify=0.008)
    # keep branch polygons denser
    lines += _to_polys(branch, CLASS_BRANCH, min_area=12, simplify=0.004)
    lines += _to_polys(rigid, CLASS_RIGID, min_area=150, simplify=0.012)
    return pix, lines


def overlay(bgr: np.ndarray, pix: np.ndarray) -> np.ndarray:
    vis = bgr.copy()
    tint = np.zeros_like(bgr)
    tint[pix == PIX_BERRY] = (255, 0, 255)
    tint[pix == PIX_BRANCH] = (0, 220, 0)
    tint[pix == PIX_RIGID] = (0, 0, 255)
    tint[pix == 4] = (220, 200, 40)
    return cv2.addWeighted(vis, 0.62, tint, 0.38, 0)


def _is_wrist(stem: str) -> bool:
    return 'wrist' in stem.lower()


def _find_depth(color_path: Path) -> Optional[np.ndarray]:
    stem = color_path.stem
    for cand in (
        color_path.with_name(stem.replace('_color', '_depth') + color_path.suffix),
        color_path.with_name(stem.replace('color', 'depth') + color_path.suffix),
    ):
        if cand.is_file():
            d = cv2.imread(str(cand), cv2.IMREAD_UNCHANGED)
            if d is not None and d.ndim == 2:
                return d
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', default='', help='folder of *color*.png (default: scene_seg/raw)')
    parser.add_argument('--raw', default=str(ROOT / 'datasets' / 'scene_seg' / 'raw'))
    parser.add_argument('--train', default=str(ROOT / 'datasets' / 'scene_seg' / 'train'))
    parser.add_argument('--val', default=str(ROOT / 'datasets' / 'scene_seg' / 'val'))
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--overlay', default=str(ROOT / 'datasets' / 'scene_seg' / 'overlays'))
    parser.add_argument('--no-yolo', action='store_true')
    parser.add_argument('--conf', type=float, default=0.18)
    parser.add_argument('--weights', default=str(YOLO_WEIGHTS))
    args = parser.parse_args()

    img_root = Path(args.images) if args.images else Path(args.raw)
    paths = sorted(
        p for p in img_root.rglob('*')
        if p.suffix.lower() == '.png'
        and 'color' in p.stem
        and '_vis' not in p.stem
        and p.parent.name != 'overlays'
        and (img_root.name == '_preview' or p.parent.name != '_preview')
    )
    if not paths:
        print(f'no color images in {img_root}', file=sys.stderr)
        return 2

    yolo = None if args.no_yolo else _load_yolo(Path(args.weights), args.conf)
    print(f'[auto_label] images={len(paths)} yolo={"yes" if yolo else "no"}')

    overlay_dir = Path(args.overlay)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    train_img = Path(args.train) / 'images'
    train_lab = Path(args.train) / 'labels'
    val_img = Path(args.val) / 'images'
    val_lab = Path(args.val) / 'labels'
    for d in (train_img, train_lab, val_img, val_lab):
        d.mkdir(parents=True, exist_ok=True)

    n = len(paths)
    val_n = max(1, int(round(n * args.val_frac))) if n >= 6 else 0
    # last val_n after sort → chronological mix of cameras
    val_set = set(p.name for p in paths[-val_n:]) if val_n else set()

    counts = {0: 0, 1: 0, 2: 0}
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        depth = _find_depth(path)
        pix, lines = label_one(
            bgr, depth, yolo=yolo, yolo_conf=args.conf, wrist=_is_wrist(path.stem))
        for ln in lines:
            counts[int(ln.split()[0])] += 1
        split_img = val_img if path.name in val_set else train_img
        split_lab = val_lab if path.name in val_set else train_lab
        dst = split_img / path.name
        if path.resolve() != dst.resolve():
            cv2.imwrite(str(dst), bgr)
        (split_lab / f'{path.stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
        cv2.imwrite(str(split_lab / f'{path.stem}_mask.png'), pix)
        vis = overlay(bgr, pix)
        cv2.imwrite(str(overlay_dir / f'{path.stem}.jpg'), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f'  {path.name}  berry={sum(1 for l in lines if l.startswith("0 "))} '
              f'branch={sum(1 for l in lines if l.startswith("1 "))} '
              f'rigid={sum(1 for l in lines if l.startswith("2 "))}')

    print(f'[auto_label] polygons berry={counts[0]} branch={counts[1]} rigid={counts[2]}')
    print(f'[auto_label] overlays → {overlay_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
