#!/usr/bin/env python3
"""Offline: detect() + reproject-UV assoc + lock-ray scale on QA wrist frames.

  python3 scripts/pbvs_ray_scale_offline.py \\
    --sessions 20260813_132800,20260813_132305,20260812_190355,20260812_183547
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
PERC = ROOT / 'src' / 'picking_perception'
if str(PERC) not in sys.path:
    sys.path.insert(0, str(PERC))

from pbvs_ray_scale import (  # noqa: E402
    assoc_max_px,
    associate_nearest_uv,
    lateral_baseline_m,
    ray_from_uv,
    recover_K_from_lock,
    reproject,
    scale_lock_ray,
)
from wrist_cam_extrinsic import load_mount_rpy, T_base_cam  # noqa: E402

QA_ROOT = ROOT / 'log' / 'real_robot' / 'qa'
DEFAULT_WEIGHTS = ROOT / 'runs' / 'detect' / 'blueberry-7' / 'weights' / 'best.pt'


def _joints_rad(row: Dict) -> List[float]:
    j = row.get('joints_deg') or {}
    return [math.radians(float(j.get(k, 0.0))) for k in ('j1', 'j2', 'j3', 'j4', 'j5', 'j6')]


def _load_stream(qa: Path) -> List[Dict]:
    rows: List[Dict] = []
    p = qa / 'pbvs_stream.jsonl'
    if not p.is_file():
        return rows
    with p.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _lock_frame_idx(rows: List[Dict]) -> int:
    for d in rows:
        if d.get('track_id') is not None and d.get('berry_base_locked') is not None:
            return int(d['frame'])
    return 0


def _run_session(
    sess: str,
    detector,
    mount: Tuple[float, float, float, float, float, float],
    *,
    min_lat_m: float,
    max_px: float,
    out_dir: Path,
) -> Dict[str, Any]:
    qa = QA_ROOT / sess
    lock = json.loads((qa / 'refine_fruit_lock.json').read_text(encoding='utf-8'))
    rows = _load_stream(qa)
    P0 = np.asarray(lock['berry_base'], dtype=np.float64)
    uv0 = np.asarray(lock['berry_uv'], dtype=np.float64)
    berry_cam = np.asarray(lock['berry_cam'], dtype=np.float64)
    K = recover_K_from_lock(uv0, berry_cam)
    tx, ty, tz, rx, ry, rz = mount
    f0 = _lock_frame_idx(rows)
    lock_row = next((r for r in rows if int(r['frame']) == f0), rows[0] if rows else None)
    if lock_row is None:
        return {'session': sess, 'ok': False, 'reason': 'no_stream'}
    T0 = T_base_cam(_joints_rad(lock_row), tx, ty, tz, rx, ry, rz)
    o0, d0 = ray_from_uv(T0, uv0, K)
    uv_chk = reproject(T0, P0, K)
    reproj_err = (
        float(np.linalg.norm(uv_chk - uv0)) if uv_chk is not None else None)

    frames_dir = qa / 'pbvs_frames'
    picks: List[Dict] = []
    chosen: Optional[Dict] = None
    for row in rows:
        fi = int(row['frame'])
        if fi < f0:
            continue
        wrist = frames_dir / f'frame_{fi:05d}_wrist.jpg'
        if not wrist.is_file():
            continue
        T = T_base_cam(_joints_rad(row), tx, ty, tz, rx, ry, rz)
        lat, along = lateral_baseline_m(o0, d0, T[:3, 3])
        # Lock + first moving frames + once lat is interesting.
        keep = True
        if not keep:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(wrist)), cv2.COLOR_BGR2RGB)
        pred = reproject(T, P0, K)
        pc = np.linalg.inv(T) @ np.array([P0[0], P0[1], P0[2], 1.0])
        zc = float(pc[2]) if float(pc[2]) > 1e-6 else 0.25
        max_px_i = min(20.0, float(max_px), assoc_max_px(zc, K[0]))
        cand_uv = []
        cand_meta = []
        h_img, w_img = rgb.shape[:2]
        in_img = (
            pred is not None
            and 8.0 <= float(pred[0]) < w_img - 8.0
            and 8.0 <= float(pred[1]) < h_img - 8.0)
        dets = (
            detector.detect_near_uv(rgb, pred, half_px=96, conf=0.05, min_area_px=40)
            if in_img else [])
        for det in dets:
            ys, xs = np.where(det.mask > 0)
            if xs.size:
                uv = (float(xs.mean()), float(ys.mean()))
            else:
                x0, y0, x1, y1 = det.bbox_xyxy
                uv = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
            cand_uv.append(uv)
            cand_meta.append({
                'uv': [float(uv[0]), float(uv[1])],
                'conf': float(det.confidence),
                'bbox': [int(v) for v in det.bbox_xyxy],
            })
        assoc = None
        duv = None
        if pred is not None and cand_uv:
            idx = associate_nearest_uv(pred, cand_uv, max_px=max_px_i)
            if idx is not None:
                assoc = cand_meta[idx]
                duv = float(np.linalg.norm(np.asarray(assoc['uv']) - pred))
        rec = {
            'frame': fi,
            't_rel_s': row.get('t_rel_s'),
            'lat_m': lat,
            'along_m': along,
            'pred_uv': pred.tolist() if pred is not None else None,
            'n_det': len(dets),
            'assoc': assoc,
            'assoc_duv_px': duv,
            'scale': None,
        }
        if assoc is not None and lat >= min_lat_m:
            rec['scale'] = scale_lock_ray(
                o0, d0, T, assoc['uv'], K, P0,
                min_lat_m=min_lat_m, max_dp_m=0.012)
            if rec['scale'].get('ok') and rec['scale'].get('apply') and chosen is None:
                chosen = rec
        picks.append(rec)
        if fi > f0 + 40 and lat > 0.09:
            break

    summary = {
        'session': sess,
        'lock_frame': f0,
        'lock_uv': uv0.tolist(),
        'P0': P0.tolist(),
        'K': list(K),
        'reproj_lock_err_px': reproj_err,
        'r_px': float(math.hypot(uv0[0] - K[2], uv0[1] - K[3])),
        'n_frames': len(picks),
        'chosen': chosen,
        'frames': picks,
    }
    if chosen and chosen.get('scale'):
        sc = chosen['scale']
        summary['ok'] = bool(sc.get('ok') and sc.get('apply'))
        summary['reason'] = sc.get('reason')
        summary['dp_mm'] = (sc.get('dp_m') or 0.0) * 1000.0
        summary['s0_mm'] = (sc.get('s0_m') or 0.0) * 1000.0
        summary['s1_mm'] = (sc.get('s1_m') or 0.0) * 1000.0
        summary['lat_mm'] = chosen['lat_m'] * 1000.0
        summary['assoc_duv_px'] = chosen.get('assoc_duv_px')
    else:
        # Best attempted scale (even if rejected).
        tried = [p for p in picks if p.get('scale')]
        summary['ok'] = False
        if tried:
            last = tried[-1]
            summary['reason'] = (last['scale'] or {}).get('reason') or 'no_apply'
            summary['dp_mm'] = ((last['scale'] or {}).get('dp_m') or 0.0) * 1000.0
            summary['lat_mm'] = last['lat_m'] * 1000.0
            summary['assoc_duv_px'] = last.get('assoc_duv_px')
        else:
            assoc_n = sum(1 for p in picks if p.get('assoc'))
            summary['reason'] = f'no_assoc_or_baseline (assoc_frames={assoc_n})'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{sess}.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sessions',
        default='20260813_142025,20260813_132800,20260813_132305,'
                '20260813_131304,20260812_190355,20260812_183547,'
                '20260812_143141,20260812_214509',
    )
    parser.add_argument('--model', default=str(DEFAULT_WEIGHTS))
    parser.add_argument('--min-lat-m', type=float, default=0.030)
    parser.add_argument('--max-px', type=float, default=22.0)
    parser.add_argument(
        '--out',
        default=str(ROOT / 'log' / 'real_robot' / 'qa' / '_ray_scale_offline'),
    )
    args = parser.parse_args()

    from picking_perception.yolo_berry_detector import YoloBerryDetector

    detector = YoloBerryDetector(
        model_name=args.model,
        class_prompts=['blueberry'],
        conf_threshold=0.10,
        min_area_px=120,
        max_area_frac=0.18,
        min_circularity=0.45,
        max_aspect_ratio=2.0,
        max_detections=8,
        open_vocab=False,
    )
    if not detector.ready:
        print('ERROR: YOLO not ready', file=sys.stderr)
        return 1
    mount = load_mount_rpy()
    out_dir = Path(args.out)
    sessions = [s.strip() for s in args.sessions.split(',') if s.strip()]
    print(f"{'session':<16} {'rpx':>4} {'lat':>6} {'duv':>5} {'Δs':>7} {'|ΔP|':>6}  result")
    for sess in sessions:
        r = _run_session(
            sess, detector, mount,
            min_lat_m=float(args.min_lat_m),
            max_px=float(args.max_px),
            out_dir=out_dir,
        )
        rpx = r.get('r_px')
        lat = r.get('lat_mm')
        duv = r.get('assoc_duv_px')
        s0 = r.get('s0_mm')
        s1 = r.get('s1_mm')
        ds = (s1 - s0) if (s0 is not None and s1 is not None) else None
        dp = r.get('dp_mm')
        print(
            f"{sess:<16} {rpx if rpx is not None else -1:4.0f} "
            f"{(lat if lat is not None else -1):6.1f} "
            f"{(duv if duv is not None else -1):5.1f} "
            f"{(ds if ds is not None else float('nan')):7.1f} "
            f"{(dp if dp is not None else -1):6.1f}  "
            f"{'APPLY' if r.get('ok') else 'SKIP'} {r.get('reason')}"
        )
    print(f'\nwrote {out_dir}')
    return 0


if __name__ == '__main__':
    os.chdir(str(ROOT))
    raise SystemExit(main())
