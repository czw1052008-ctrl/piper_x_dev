"""Open-vocabulary YOLO (YOLOE) blueberry detector — no custom training."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class YoloDetection:
    mask: np.ndarray
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


def _mask_circularity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    area = float(cv2.contourArea(contours[0]))
    peri = float(cv2.arcLength(contours[0], True))
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (peri * peri))


class YoloBerryDetector:
    """YOLOE text-prompt detector; falls back to bbox ellipses if seg unavailable."""

    def __init__(
        self,
        model_name: str = 'yoloe-11s-seg-pf.pt',
        class_prompts: Optional[List[str]] = None,
        conf_threshold: float = 0.22,
        min_area_px: int = 120,
        max_area_frac: float = 0.04,
        max_aspect_ratio: float = 2.0,
        min_circularity: float = 0.45,
        max_detections: int = 6,
        border_margin_frac: float = 0.02,
        open_vocab: bool = True,
        device: str = '',
    ) -> None:
        self._ready = False
        self._model = None
        self._model_name = model_name
        self._open_vocab = open_vocab
        self._prompts = class_prompts or ['blueberry', 'blueberries']
        self._conf = conf_threshold
        self._min_area = min_area_px
        self._max_area_frac = max_area_frac
        self._max_aspect = max_aspect_ratio
        self._min_circ = min_circularity
        self._max_dets = max_detections
        self._border_margin = border_margin_frac
        self._device = device

        try:
            from ultralytics import YOLO  # noqa: WPS433

            self._model = YOLO(model_name)
            self._ready = True
            mode = 'open-vocab' if open_vocab else 'custom'
            logger.info('YoloBerryDetector ready model=%s mode=%s', model_name, mode)
        except Exception as exc:
            logger.warning('YoloBerryDetector unavailable: %s', exc)

    @property
    def ready(self) -> bool:
        return self._ready

    def _passes_shape_filter(
        self, mask: np.ndarray, bbox: Tuple[int, int, int, int], h: int, w: int,
    ) -> bool:
        x0, y0, x1, y1 = bbox
        bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
        aspect = max(bw / bh, bh / bw)
        if aspect > self._max_aspect:
            return False
        if _mask_circularity(mask) < self._min_circ:
            return False
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        mx, my = w * self._border_margin, h * self._border_margin
        if cx < mx or cy < my or cx > w - mx or cy > h - my:
            return False
        return True

    def detect(self, rgb: np.ndarray) -> List[YoloDetection]:
        if not self._ready or self._model is None:
            return []

        h, w = rgb.shape[:2]
        max_area = int(h * w * self._max_area_frac)
        predict_kw = dict(conf=self._conf, verbose=False, device=self._device or None)
        if self._open_vocab:
            predict_kw['prompts'] = self._prompts
        results = self._model.predict(rgb, **predict_kw)
        if not results:
            return []

        res = results[0]
        candidates: List[YoloDetection] = []

        if res.masks is not None and len(res.masks):
            for i, mask_tensor in enumerate(res.masks.data):
                conf = float(res.boxes.conf[i]) if res.boxes is not None else 0.5
                xyxy = res.boxes.xyxy[i].cpu().numpy().astype(int).tolist() if res.boxes is not None else [0, 0, w, h]
                bbox = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                mask = mask_tensor.cpu().numpy()
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_u8 = (mask > 0.5).astype(np.uint8)
                area = int(mask_u8.sum())
                if area < self._min_area or area > max_area:
                    continue
                if not self._passes_shape_filter(mask_u8, bbox, h, w):
                    continue
                candidates.append(YoloDetection(mask_u8, conf, bbox))
        elif res.boxes is not None:
            for box in res.boxes:
                conf = float(box.conf)
                x0, y0, x1, y1 = [int(v) for v in box.xyxy[0].tolist()]
                bbox = (x0, y0, x1, y1)
                bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
                area = bw * bh
                if area < self._min_area or area > max_area:
                    continue
                mask = np.zeros((h, w), dtype=np.uint8)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                cv2.ellipse(mask, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, 1, -1)
                if not self._passes_shape_filter(mask, bbox, h, w):
                    continue
                candidates.append(YoloDetection(mask, conf, bbox))

        candidates.sort(key=lambda d: d.confidence, reverse=True)
        return candidates[: self._max_dets]

    def combined_mask(self, rgb: np.ndarray) -> Tuple[np.ndarray, List[YoloDetection]]:
        """Binary mask for debug; detections kept separate for FP."""
        dets = self.detect(rgb)
        if not dets:
            return np.zeros(rgb.shape[:2], dtype=np.uint8), dets
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for det in dets:
            mask = np.maximum(mask, det.mask)
        return mask, dets
