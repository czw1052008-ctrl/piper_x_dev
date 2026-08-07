"""REFINING depth helpers: probe-tri mono chord scale (scheme G)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from picking_perception.berry_tracker import BerryTracker
from picking_perception.yolo_berry_detector import YoloBerryDetector

_YOLO: Optional[YoloBerryDetector] = None


def chord_scale_berry(
    berry_base: Tuple[float, float, float],
    tcp_base: Tuple[float, float, float],
    scale: float,
) -> Tuple[float, float, float]:
    b = np.asarray(berry_base, dtype=np.float64)
    t = np.asarray(tcp_base, dtype=np.float64)
    out = t + (b - t) * float(scale)
    return float(out[0]), float(out[1]), float(out[2])


def apply_mono_chord_scale(
    berry_base: Tuple[float, float, float],
    tcp_base: Tuple[float, float, float],
    *,
    z_cam_m: float,
    z_mono_m: float,
    scale_min: float = 0.45,
    scale_max: float = 1.05,
) -> Tuple[Optional[Tuple[float, float, float]], Dict]:
    """Scale frozen tri anchor along cup→berry chord by z_mono / z_cam."""
    if z_cam_m <= 1e-4 or z_mono_m <= 1e-4:
        return None, {'rejected': 'invalid_z'}
    scale = float(z_mono_m / z_cam_m)
    meta: Dict = {
        'scale': scale,
        'z_mono_m': float(z_mono_m),
        'z_cam_m': float(z_cam_m),
    }
    if scale < scale_min or scale > scale_max:
        meta['rejected'] = 'scale_out_of_range'
        return None, meta
    berry_new = chord_scale_berry(berry_base, tcp_base, scale)
    meta['berry_base'] = list(berry_new)
    meta['dist_cup_m'] = float(np.linalg.norm(np.asarray(berry_new) - np.asarray(tcp_base)))
    return berry_new, meta


def _yolo_detector(conf_threshold: float = 0.15) -> YoloBerryDetector:
    global _YOLO
    if _YOLO is None or not _YOLO.ready:
        _YOLO = YoloBerryDetector(conf_threshold=conf_threshold)
    return _YOLO


def _mask_center(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def z_mono_near_reproject_uv(
    rgb: np.ndarray,
    u: float,
    v: float,
    *,
    focal_px: float = 550.0,
    berry_diameter_m: float = 0.015,
    max_pick_pix: float = 120.0,
    conf_threshold: float = 0.15,
) -> Tuple[Optional[float], Dict]:
    """YOLO on wrist RGB → min-mono in mask near reproject UV."""
    det = YoloBerryDetector(conf_threshold=conf_threshold)
    if not det.ready:
        return None, {'rejected': 'yolo_unavailable'}
    h, w = rgb.shape[:2]
    k = np.array([[focal_px, 0, w * 0.5], [0, focal_px, h * 0.5], [0, 0, 1.0]], dtype=np.float64)
    tracker = BerryTracker(berry_diameter_m=berry_diameter_m, depth_min_m=0.03)
    dets = det.detect(rgb)
    if not dets:
        return None, {'rejected': 'no_yolo_det', 'n_dets': 0}

    ranked: List[Tuple[float, object]] = []
    for d in dets:
        cu, cv = _mask_center(d.mask)
        ranked.append((math.hypot(cu - float(u), cv - float(v)), d))
    ranked.sort(key=lambda x: x[0])
    pix_off, best = ranked[0]
    if pix_off > max_pick_pix:
        return None, {
            'rejected': 'yolo_too_far',
            'pix_off_px': pix_off,
            'n_dets': len(dets),
        }

    z_mono = tracker.mono_depth_near_uv(best.mask, float(u), float(v), k)
    cu, cv = _mask_center(best.mask)
    return float(z_mono), {
        'z_mono_m': float(z_mono),
        'pix_off_px': float(pix_off),
        'berry_uv': [cu, cv],
        'reproject_uv': [float(u), float(v)],
        'yolo_conf': float(best.confidence),
        'n_dets': len(dets),
        'conf_threshold': conf_threshold,
    }


def z_mono_near_reproject_from_sources(
    sources: List[Tuple[str, np.ndarray]],
    u: float,
    v: float,
    *,
    focal_px: float = 550.0,
    berry_diameter_m: float = 0.015,
    max_pick_pix: float = 120.0,
) -> Tuple[Optional[float], Dict]:
    """Try YOLO min-mono on multiple RGB buffers (wrist, fine_viz, …)."""
    last_meta: Dict = {'rejected': 'no_sources'}
    for name, rgb in sources:
        if rgb is None:
            continue
        for conf in (0.12, 0.15):
            zm, meta = z_mono_near_reproject_uv(
                rgb, u, v,
                focal_px=focal_px,
                berry_diameter_m=berry_diameter_m,
                max_pick_pix=max_pick_pix,
                conf_threshold=conf,
            )
            meta['rgb_source'] = name
            last_meta = meta
            if zm is not None and zm > 1e-4:
                return float(zm), meta
    return None, last_meta
