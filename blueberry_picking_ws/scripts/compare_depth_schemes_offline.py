#!/usr/bin/env python3
"""Offline compare depth/contact schemes on saved QA session (no robot).

Compares probe-tri reproject vs post-center mono refresh vs surface anchor
against a ruler ground-truth dist_cup (default 0.25 m for session 20260806_155054).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src' / 'picking_perception'))

from piper_position_ik import _joint_transform, _JOINT_FIXED  # noqa: E402
from picking_perception.berry_tracker import BerryTracker  # noqa: E402
from picking_perception.yolo_berry_detector import YoloBerryDetector  # noqa: E402

# URDF: link6 → camera_wrist_color_optical_frame
_T_LINK6_CAM = np.eye(4)
_T_LINK6_CAM[:3, 3] = [0.0, -0.08, -0.04]

FOCAL_PX = 550.0
BERRY_DIAM_M = 0.015
CONTACT_OFFSET_M = 0.04
GROUND_TRUTH_DIST_M = 0.25


def fk_T_base_link6(q: List[float]) -> np.ndarray:
    T = np.eye(4)
    for i, (xyz, rpy, axis) in enumerate(_JOINT_FIXED):
        qi = 0.0 if i == 5 else float(q[i])
        T = T @ _joint_transform(xyz, rpy, axis, qi)
    return T


def fk_T_base_cam(q: List[float]) -> np.ndarray:
    return fk_T_base_link6(q) @ _T_LINK6_CAM


def cam_xyz_to_base(T: np.ndarray, cam_xyz: Tuple[float, float, float]) -> np.ndarray:
    v = T @ np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0], dtype=np.float64)
    return v[:3]


def base_xyz_to_cam(T: np.ndarray, base_xyz: Tuple[float, float, float]) -> np.ndarray:
    Tinv = np.linalg.inv(T)
    v = Tinv @ np.array([base_xyz[0], base_xyz[1], base_xyz[2], 1.0], dtype=np.float64)
    return v[:3]


def uv_to_cam_xyz(u: float, v: float, z: float, w: int, h: int, f: float) -> Tuple[float, float, float]:
    cx, cy = w * 0.5, h * 0.5
    x = (u - cx) * z / f
    y = (v - cy) * z / f
    return float(x), float(y), float(z)


def calibrate_T_base_cam(q: List[float], probe_obs: Dict) -> np.ndarray:
    """FK rotation + bias-corrected translation from last probe obs."""
    T_fk = fk_T_base_cam(q)
    q_probe = [float(x) for x in probe_obs['joints_rad']]
    T_probe_fk = fk_T_base_cam(q_probe)
    bias = np.array(probe_obs['T_base_cam_t'], dtype=np.float64) - T_probe_fk[:3, 3]
    T = T_fk.copy()
    T[:3, 3] = T_fk[:3, 3] + bias
    return T


def chord_scale_berry(
    berry: np.ndarray,
    tcp: np.ndarray,
    scale: float,
) -> np.ndarray:
    return tcp + (berry - tcp) * scale


def dist_cup(berry: np.ndarray, tcp: np.ndarray) -> float:
    return float(np.linalg.norm(berry - tcp))


def travel(dist: float, contact: float = CONTACT_OFFSET_M) -> float:
    return max(0.0, dist - contact)


def mask_center(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def detect_berries(rgb_path: Path) -> Tuple[np.ndarray, List]:
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    det = YoloBerryDetector(conf_threshold=0.15)
    if not det._ready:
        raise RuntimeError('YOLO unavailable')
    dets = det.detect(rgb)
    return rgb, dets


def pick_nearest(dets, u0: float, v0: float):
    best, best_d = None, 1e18
    for d in dets:
        u, v = mask_center(d.mask)
        dd = math.hypot(u - u0, v - v0)
        if dd < best_d:
            best_d = dd
            best = d
    return best, best_d


def scheme_row(
    name: str,
    berry: np.ndarray,
    tcp: np.ndarray,
    *,
    gt: float,
    extra: Optional[Dict] = None,
) -> Dict:
    d = dist_cup(berry, tcp)
    t = travel(d)
    return {
        'scheme': name,
        'berry_base': berry.tolist(),
        'dist_cup_m': d,
        'travel_to_contact_m': t,
        'err_vs_ruler_cm': (d - gt) * 100.0,
        'abs_err_cm': abs(d - gt) * 100.0,
        **(extra or {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--qa-dir',
        type=Path,
        default=ROOT / 'log/real_robot/qa/20260806_155054',
    )
    ap.add_argument('--ground-truth-m', type=float, default=GROUND_TRUTH_DIST_M)
    ap.add_argument('--out-json', type=Path, default=None)
    args = ap.parse_args()

    qa = args.qa_dir
    gt = float(args.ground_truth_m)

    with open(qa / 'center_depth_qa.json', encoding='utf-8') as f:
        qa_json = json.load(f)
    with open(qa / 'mono_probe_center.json', encoding='utf-8') as f:
        probe_json = json.load(f)
    with open(qa / 'servo_01.json', encoding='utf-8') as f:
        servo_json = json.load(f)

    tcp = np.array(qa_json['tcp_base'], dtype=np.float64)
    berry_tri = np.array(qa_json['berry_base'], dtype=np.float64)
    berry_uv_reproj = qa_json['berry_uv']
    z_cam_tri = float(qa_json['z_cam_m'])

    # Center pose joints from last motion frame
    mf = servo_json['motion_frames'][-1]
    jd = mf['joints_deg']
    q = [
        math.radians(jd['j1']),
        math.radians(jd['j2']),
        math.radians(jd['j3']),
        0.0,
        math.radians(jd['j5']),
        0.0,
    ]
    T = calibrate_T_base_cam(q, probe_json['obs'][-1])

    # Verify calibrated cam origin vs probe obs (sanity)
    probe_t = probe_json['obs'][-1]['T_base_cam_t']
    fk_t = T[:3, 3]
    fk_err_mm = np.linalg.norm(fk_t - np.array(probe_t)) * 1000.0

    w, h = 640, 480
    f = FOCAL_PX
    tracker = BerryTracker(berry_diameter_m=BERRY_DIAM_M, depth_min_m=0.03)
    k = np.array([[f, 0, w * 0.5], [0, f, h * 0.5], [0, 0, 1.0]], dtype=np.float64)

    wrist_png = qa / 'center_depth_qa_wrist.png'
    rgb, dets = detect_berries(wrist_png)

    results: List[Dict] = []
    results.append(scheme_row(
        '0_baseline_probe_tri_reproject',
        berry_tri,
        tcp,
        gt=gt,
        extra={
            'depth_source': qa_json['depth_source'],
            'z_cam_m': z_cam_tri,
            'reproject_pix_off_px': qa_json.get('reproject_pix_off_px'),
        },
    ))

    # Scheme A: post-center mono refresh at optical center
    optical = (w * 0.5, h * 0.5)
    det_a, pix_a = pick_nearest(dets, optical[0], optical[1])
    if det_a is not None:
        u_a, v_a = mask_center(det_a.mask)
        z_mono_a = tracker.mono_depth_from_mask(det_a.mask, k)
        cam_a = uv_to_cam_xyz(u_a, v_a, z_mono_a, w, h, f)
        berry_a = cam_xyz_to_base(T, cam_a)
        results.append(scheme_row(
            'A_mono_refresh_optical_center',
            berry_a,
            tcp,
            gt=gt,
            extra={
                'z_mono_m': z_mono_a,
                'berry_uv': [u_a, v_a],
                'pix_off_optical_px': pix_a,
                'yolo_conf': float(det_a.confidence),
                'n_dets': len(dets),
            },
        ))
    else:
        results.append({'scheme': 'A_mono_refresh_optical_center', 'error': 'no YOLO det'})

    # Scheme B: surface anchor at reproject UV — mono depth on ray through reproject pixel
    u_r, v_r = float(berry_uv_reproj[0]), float(berry_uv_reproj[1])
    det_b, pix_b = pick_nearest(dets, u_r, v_r)
    if det_b is not None:
        z_mono_b = tracker.mono_depth_from_mask(det_b.mask, k)
        cam_b = uv_to_cam_xyz(u_r, v_r, z_mono_b, w, h, f)
        berry_b = cam_xyz_to_base(T, cam_b)
        results.append(scheme_row(
            'B_surface_mono_at_reproject_uv',
            berry_b,
            tcp,
            gt=gt,
            extra={
                'z_mono_m': z_mono_b,
                'berry_uv': [u_r, v_r],
                'pix_off_reproj_px': pix_b,
                'yolo_conf': float(det_b.confidence),
            },
        ))

        # B2: min z_mono in mask neighborhood near reproject (surface proxy without depth cam)
        ys, xs = np.where(det_b.mask > 0)
        if xs.size > 0:
            dists = np.hypot(xs - u_r, ys - v_r)
            near = dists <= max(25.0, math.sqrt(xs.size / math.pi) * 0.6)
            if near.any():
                sub = np.zeros_like(det_b.mask)
                sub[ys[near], xs[near]] = det_b.mask[ys[near], xs[near]]
                z_near = tracker.mono_depth_from_mask(sub, k)
                cam_b2 = uv_to_cam_xyz(u_r, v_r, z_near, w, h, f)
                berry_b2 = cam_xyz_to_base(T, cam_b2)
                results.append(scheme_row(
                    'B2_surface_near_reproject_min_mono',
                    berry_b2,
                    tcp,
                    gt=gt,
                    extra={'z_mono_m': z_near, 'berry_uv': [u_r, v_r]},
                ))
    else:
        results.append({'scheme': 'B_surface_mono_at_reproject_uv', 'error': 'no YOLO det'})

    # Scheme A+B: optical-center z_mono + reproject UV direction
    if det_a is not None:
        z_ab = tracker.mono_depth_from_mask(det_a.mask, k)
        cam_ab = uv_to_cam_xyz(u_r, v_r, z_ab, w, h, f)
        berry_ab = cam_xyz_to_base(T, cam_ab)
        results.append(scheme_row(
            'A+B_reproject_uv_with_optical_z_mono',
            berry_ab,
            tcp,
            gt=gt,
            extra={'z_mono_m': z_ab, 'berry_uv': [u_r, v_r]},
        ))

    # Scheme C: scale tri anchor along cup→berry chord by mono/tri z ratio at center
    if z_cam_tri > 1e-4 and det_a is not None:
        z_mono_a = tracker.mono_depth_from_mask(det_a.mask, k)
        s = z_mono_a / z_cam_tri
        berry_c = tcp + (berry_tri - tcp) * s
        results.append(scheme_row(
            'C_scale_tri_by_z_mono_over_z_tri',
            berry_c,
            tcp,
            gt=gt,
            extra={'scale': s, 'z_mono_m': z_mono_a, 'z_cam_tri_m': z_cam_tri},
        ))

    # Scheme D: ray surface pull — move tri point toward cup by berry radius along view ray
    cam_tri = base_xyz_to_cam(T, tuple(berry_tri))
    ray = cam_tri / (np.linalg.norm(cam_tri) + 1e-12)
    berry_radius = BERRY_DIAM_M * 0.5
    cam_surf = cam_tri - ray * berry_radius
    berry_d = cam_xyz_to_base(T, tuple(cam_surf))
    results.append(scheme_row(
        'D_tri_minus_radius_along_ray',
        berry_d,
        tcp,
        gt=gt,
        extra={'pull_m': berry_radius},
    ))

    # FK-free: scale frozen tri dist by z_mono / z_cam (same pixel ray depth ratio)
    z_mono_full = tracker.mono_depth_from_mask(det_a.mask, k) if det_a else None
    if det_a is not None:
        z_m = z_mono_full
        if z_cam_tri > 1e-4:
            s_opt = z_m / z_cam_tri
            berry_e = chord_scale_berry(berry_tri, tcp, s_opt)
            results.append(scheme_row(
                'E_chord_scale_z_mono_optical_over_z_tri',
                berry_e,
                tcp,
                gt=gt,
                extra={'scale': s_opt, 'z_mono_m': z_m},
            ))
        if det_b is not None:
            z_br = tracker.mono_depth_from_mask(det_b.mask, k)
            s_rp = z_br / z_cam_tri
            berry_f = chord_scale_berry(berry_tri, tcp, s_rp)
            results.append(scheme_row(
                'F_chord_scale_z_mono_reproj_over_z_tri',
                berry_f,
                tcp,
                gt=gt,
                extra={'scale': s_rp, 'z_mono_m': z_br},
            ))
            # B2 near-pixel min mono + chord scale (no FK)
            ys, xs = np.where(det_b.mask > 0)
            if xs.size > 0:
                dists = np.hypot(xs - u_r, ys - v_r)
                near = dists <= max(25.0, math.sqrt(xs.size / math.pi) * 0.6)
                if near.any():
                    sub = np.zeros_like(det_b.mask)
                    sub[ys[near], xs[near]] = det_b.mask[ys[near], xs[near]]
                    z_near = tracker.mono_depth_from_mask(sub, k)
                    s_n = z_near / z_cam_tri
                    berry_g = chord_scale_berry(berry_tri, tcp, s_n)
                    results.append(scheme_row(
                        'G_chord_scale_B2_near_mono_over_z_tri',
                        berry_g,
                        tcp,
                        gt=gt,
                        extra={'scale': s_n, 'z_mono_m': z_near},
                    ))

    # Rank valid schemes
    ranked = sorted(
        [r for r in results if 'dist_cup_m' in r],
        key=lambda r: r['abs_err_cm'],
    )

    print(f'QA session: {qa.name}')
    print(f'Ruler ground truth dist_cup: {gt*100:.1f} cm  (contact offset {CONTACT_OFFSET_M*100:.0f} cm)')
    print(f'FK cam origin vs last probe t: {fk_err_mm:.1f} mm')
    print(f'YOLO dets on center_depth_qa_wrist: {len(dets)}')
    print()
    print(f'{"scheme":<42} {"dist":>7} {"travel":>7} {"err":>8} {"|err|":>7}')
    print('-' * 78)
    for r in ranked:
        print(
            f'{r["scheme"]:<42} '
            f'{r["dist_cup_m"]*100:6.1f}cm '
            f'{r["travel_to_contact_m"]*100:6.1f}cm '
            f'{r["err_vs_ruler_cm"]:+7.1f}cm '
            f'{r["abs_err_cm"]:6.1f}cm',
        )
    if ranked:
        best = ranked[0]
        print()
        print(f'BEST: {best["scheme"]}  dist={best["dist_cup_m"]*100:.1f}cm  '
              f'(ruler {gt*100:.1f}cm, Δ{best["err_vs_ruler_cm"]:+.1f}cm)')

    out = {
        'qa_dir': str(qa),
        'ground_truth_dist_m': gt,
        'fk_cam_origin_err_mm': fk_err_mm,
        'n_yolo_dets': len(dets),
        'ranked': ranked,
        'all': results,
    }
    out_path = args.out_json or (qa / 'depth_scheme_compare.json')
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\nWrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
