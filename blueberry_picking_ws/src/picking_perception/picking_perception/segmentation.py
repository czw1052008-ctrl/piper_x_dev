"""HSV segmentation utilities for blueberry detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import cv2
import numpy as np


def segment_blueberry_hsv(rgb: np.ndarray) -> np.ndarray:
    """Return (H,W) uint8 mask with blueberry-colored regions as 1."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Real berries are often dark / low-saturation under indoor backlight.
    lo = np.array([95, 20, 20])
    hi = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lo, hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return (mask > 0).astype(np.uint8)


@dataclass
class BerryBlob:
    """One connected component that passes blueberry shape heuristics."""

    label_id: int
    x0: int
    y0: int
    x1: int
    y1: int
    area_px: int
    circularity: float


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def iter_berry_blobs(
    mask: np.ndarray,
    *,
    min_area: int = 80,
    max_area: int = 12000,
    min_circularity: float = 0.45,
    max_aspect: float = 2.5,
) -> Iterator[BerryBlob]:
    """Yield blob boxes that look like round berries (filters stems/leaves/clusters)."""
    num_labels, labels = cv2.connectedComponents(mask)
    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        area = int(xs.size)
        if area < min_area or area > max_area:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        w, h = max(x1 - x0, 1), max(y1 - y0, 1)
        aspect = max(w / h, h / w)
        if aspect > max_aspect:
            continue
        component = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        circ = _circularity(cv2.contourArea(contours[0]), cv2.arcLength(contours[0], True))
        if circ < min_circularity:
            continue
        yield BerryBlob(label_id, x0, y0, x1, y1, area, circ)


def filter_berry_mask(
    mask: np.ndarray,
    *,
    min_area: int = 80,
    max_area: int = 12000,
    min_circularity: float = 0.45,
    max_aspect: float = 2.5,
) -> np.ndarray:
    """Keep only blob pixels that pass berry shape filters."""
    out = np.zeros_like(mask)
    num_labels, labels = cv2.connectedComponents(mask)
    for blob in iter_berry_blobs(
        mask,
        min_area=min_area,
        max_area=max_area,
        min_circularity=min_circularity,
        max_aspect=max_aspect,
    ):
        out[labels == blob.label_id] = 1
    return out
