"""Open-vocabulary YOLO (YOLOE) blueberry detector — no custom training."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class YoloDetection:
    mask: np.ndarray
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass
class TrackedYoloDetection(YoloDetection):
    track_id: int = -1


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
        target_class_ids: Optional[List[int]] = None,
        device: str = '',
        tracker_config: str = 'botsort.yaml',
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
        self._target_class_ids = (
            None if target_class_ids is None else [int(c) for c in target_class_ids]
        )
        self._device = device
        self._tracker_config = tracker_config

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
        *,
        min_circularity: Optional[float] = None,
        border_margin_frac: Optional[float] = None,
        max_aspect_ratio: Optional[float] = None,
    ) -> bool:
        x0, y0, x1, y1 = bbox
        bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
        max_aspect = self._max_aspect if max_aspect_ratio is None else float(max_aspect_ratio)
        min_circ = self._min_circ if min_circularity is None else float(min_circularity)
        border = self._border_margin if border_margin_frac is None else float(border_margin_frac)
        aspect = max(bw / bh, bh / bw)
        if aspect > max_aspect:
            return False
        if _mask_circularity(mask) < min_circ:
            return False
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        mx, my = w * border, h * border
        if cx < mx or cy < my or cx > w - mx or cy > h - my:
            return False
        return True

    def detect(
        self,
        rgb: np.ndarray,
        *,
        conf: Optional[float] = None,
        min_area_px: Optional[int] = None,
        max_detections: Optional[int] = None,
        min_circularity: Optional[float] = None,
        border_margin_frac: Optional[float] = None,
        max_aspect_ratio: Optional[float] = None,
    ) -> List[YoloDetection]:
        if not self._ready or self._model is None:
            return []

        h, w = rgb.shape[:2]
        predict_kw = dict(
            conf=float(self._conf if conf is None else conf),
            verbose=False,
            device=self._device or None,
        )
        if self._open_vocab:
            predict_kw['prompts'] = self._prompts
        results = self._model.predict(rgb, **predict_kw)
        if not results:
            return []

        tracked = self._parse_result_detections(
            results[0], h, w, with_track_id=False,
            min_area_px=min_area_px,
            min_circularity=min_circularity,
            border_margin_frac=border_margin_frac,
            max_aspect_ratio=max_aspect_ratio,
        )
        tracked.sort(key=lambda d: d.confidence, reverse=True)
        cap = self._max_dets if max_detections is None else int(max_detections)
        return tracked[: max(1, cap)]

    def detect_near_uv(
        self,
        rgb: np.ndarray,
        uv: Sequence[float],
        *,
        half_px: int = 96,
        conf: float = 0.05,
        min_area_px: int = 40,
    ) -> List[YoloDetection]:
        """YOLO on a crop around ``uv``; boxes/masks mapped back to full image.

        Lock-local path: no full-frame top-N. Crop is letterboxed to the
        network size so a ~15 px berry is large enough to fire.
        """
        if not self._ready or self._model is None:
            return []
        h, w = rgb.shape[:2]
        u = int(round(float(uv[0])))
        v = int(round(float(uv[1])))
        half = int(max(32, half_px))
        x0 = max(0, u - half)
        y0 = max(0, v - half)
        x1 = min(w, u + half)
        y1 = min(h, v + half)
        if (x1 - x0) < 32 or (y1 - y0) < 32:
            return []
        crop = rgb[y0:y1, x0:x1]
        dets = self.detect(
            crop,
            conf=float(conf),
            min_area_px=int(min_area_px),
            max_detections=20,
            min_circularity=0.25,
            border_margin_frac=0.0,
        )
        out: List[YoloDetection] = []
        for det in dets:
            bx0, by0, bx1, by1 = det.bbox_xyxy
            full_mask = np.zeros((h, w), dtype=np.uint8)
            mh, mw = int(det.mask.shape[0]), int(det.mask.shape[1])
            full_mask[y0:y0 + mh, x0:x0 + mw] = det.mask[:mh, :mw]
            out.append(YoloDetection(
                mask=full_mask,
                confidence=float(det.confidence),
                bbox_xyxy=(
                    int(bx0) + x0, int(by0) + y0,
                    int(bx1) + x0, int(by1) + y0,
                ),
            ))
        return out

    def reset_tracker(self) -> None:
        """Clear BoT-SORT state (FSM clear_lock / new REFINING pin).

        Only call tracker.reset() — do NOT empty predictor.trackers; ultralytics
        assumes trackers[0] exists on the next track() frame.
        """
        if self._model is None:
            return
        predictor = getattr(self._model, 'predictor', None)
        if predictor is None:
            return
        trackers = getattr(predictor, 'trackers', None)
        if not trackers:
            return
        for tr in trackers:
            if hasattr(tr, 'reset'):
                try:
                    tr.reset()
                except Exception:
                    pass

    def _parse_result_detections(
        self,
        res,
        h: int,
        w: int,
        *,
        with_track_id: bool,
        min_area_px: Optional[int] = None,
        min_circularity: Optional[float] = None,
        border_margin_frac: Optional[float] = None,
        max_aspect_ratio: Optional[float] = None,
    ) -> List[TrackedYoloDetection]:
        max_area = int(h * w * self._max_area_frac)
        min_area = self._min_area if min_area_px is None else int(min_area_px)
        candidates: List[TrackedYoloDetection] = []
        shape_kw = dict(
            min_circularity=min_circularity,
            border_margin_frac=border_margin_frac,
            max_aspect_ratio=max_aspect_ratio,
        )

        if res.masks is not None and len(res.masks):
            for i, mask_tensor in enumerate(res.masks.data):
                if res.boxes is not None and self._target_class_ids is not None:
                    cls_id = int(res.boxes.cls[i].item())
                    if cls_id not in self._target_class_ids:
                        continue
                conf = float(res.boxes.conf[i]) if res.boxes is not None else 0.5
                xyxy = (
                    res.boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    if res.boxes is not None else [0, 0, w, h])
                bbox = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                tid = -1
                if with_track_id and res.boxes is not None and res.boxes.id is not None:
                    tid = int(res.boxes.id[i].item())
                mask = mask_tensor.cpu().numpy()
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_u8 = (mask > 0.5).astype(np.uint8)
                area = int(mask_u8.sum())
                if area < min_area or area > max_area:
                    continue
                if not self._passes_shape_filter(mask_u8, bbox, h, w, **shape_kw):
                    continue
                candidates.append(TrackedYoloDetection(mask_u8, conf, bbox, tid))
        elif res.boxes is not None:
            for bi, box in enumerate(res.boxes):
                if self._target_class_ids is not None:
                    cls_id = int(box.cls.item())
                    if cls_id not in self._target_class_ids:
                        continue
                conf = float(box.conf)
                x0, y0, x1, y1 = [int(v) for v in box.xyxy[0].tolist()]
                bbox = (x0, y0, x1, y1)
                tid = -1
                if with_track_id and res.boxes.id is not None:
                    tid = int(res.boxes.id[bi].item())
                bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
                area = bw * bh
                if area < min_area or area > max_area:
                    continue
                mask = np.zeros((h, w), dtype=np.uint8)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                cv2.ellipse(mask, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, 1, -1)
                if not self._passes_shape_filter(mask, bbox, h, w, **shape_kw):
                    continue
                candidates.append(TrackedYoloDetection(mask, conf, bbox, tid))

        return candidates

    def track(
        self, rgb: np.ndarray, *, conf: Optional[float] = None,
    ) -> List[TrackedYoloDetection]:
        """YOLO + BoT-SORT; requires consecutive frames (persist=True)."""
        if not self._ready or self._model is None:
            return []

        h, w = rgb.shape[:2]
        track_kw = dict(
            conf=float(self._conf if conf is None else conf),
            verbose=False,
            device=self._device or None,
            persist=True,
            tracker=self._tracker_config,
        )
        if self._open_vocab:
            track_kw['prompts'] = self._prompts
        results = self._model.track(rgb, **track_kw)
        if not results:
            return []

        candidates = self._parse_result_detections(results[0], h, w, with_track_id=True)
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
