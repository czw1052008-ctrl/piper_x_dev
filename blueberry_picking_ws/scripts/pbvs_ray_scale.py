#!/usr/bin/env python3
"""Lock-ray 1-DOF range correction from a second live UV (no BoT-SORT).

Keep the lock-frame ray (YOLO UV identity). Replace only the scale using a
later camera pose + live bbox center. Project triangulation onto the lock ray
so a neighbor swap shows up as skew, not a new 3D point.

Used by offline QA replay and PBVS oneshot residual retarget.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

K4 = Tuple[float, float, float, float]


def ray_from_uv(
    T_base_cam: np.ndarray,
    uv: Sequence[float],
    K: K4,
) -> Tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = K
    u, v = float(uv[0]), float(uv[1])
    r = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    r = r / (np.linalg.norm(r) + 1e-12)
    o = np.asarray(T_base_cam[:3, 3], dtype=np.float64).copy()
    d = T_base_cam[:3, :3] @ r
    d = d / (np.linalg.norm(d) + 1e-12)
    return o, d


def reproject(
    T_base_cam: np.ndarray,
    P: np.ndarray,
    K: K4,
) -> Optional[np.ndarray]:
    Tinv = np.linalg.inv(np.asarray(T_base_cam, dtype=np.float64))
    pc = Tinv @ np.array([float(P[0]), float(P[1]), float(P[2]), 1.0], dtype=np.float64)
    if float(pc[2]) <= 1e-6:
        return None
    fx, fy, cx, cy = K
    return np.array(
        [fx * float(pc[0]) / float(pc[2]) + cx,
         fy * float(pc[1]) / float(pc[2]) + cy],
        dtype=np.float64,
    )


def triangulate_rays(
    o0: np.ndarray, d0: np.ndarray, o1: np.ndarray, d1: np.ndarray,
) -> Optional[Tuple[np.ndarray, float]]:
    d0 = d0 / (np.linalg.norm(d0) + 1e-12)
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    w0 = o0 - o1
    a = float(np.dot(d0, d0))
    b = float(np.dot(d0, d1))
    c = float(np.dot(d1, d1))
    d = float(np.dot(d0, w0))
    e = float(np.dot(d1, w0))
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return None
    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    p0 = o0 + t * d0
    p1 = o1 + s * d1
    return 0.5 * (p0 + p1), float(np.linalg.norm(p0 - p1))


def lateral_baseline_m(
    o_lock: np.ndarray, d_lock: np.ndarray, o_now: np.ndarray,
) -> Tuple[float, float]:
    """Return (lateral_m, along_ray_m) of camera origin motion vs lock ray."""
    move = np.asarray(o_now, dtype=np.float64) - np.asarray(o_lock, dtype=np.float64)
    d = d_lock / (np.linalg.norm(d_lock) + 1e-12)
    along = float(np.dot(move, d))
    lat = float(np.linalg.norm(move - along * d))
    return lat, along


def assoc_max_px(z_cam_m: float, fx: float, *, radius_m: float = 0.008) -> float:
    """Max UV error for the same berry (~8 mm at current range)."""
    z = max(0.08, float(z_cam_m))
    return float(np.clip(radius_m * float(fx) / z, 8.0, 22.0))


def associate_nearest_uv(
    pred_uv: Sequence[float],
    cand_uvs: Sequence[Sequence[float]],
    *,
    max_px: float = 16.0,
    margin_px: float = 6.0,
) -> Optional[int]:
    """Index of nearest candidate within max_px, or None if ambiguous/far."""
    if not cand_uvs:
        return None
    pred = np.asarray(pred_uv, dtype=np.float64).reshape(2)
    dists = [
        float(np.linalg.norm(np.asarray(uv, dtype=np.float64).reshape(2) - pred))
        for uv in cand_uvs
    ]
    order = sorted(range(len(dists)), key=lambda i: dists[i])
    best = order[0]
    if dists[best] > float(max_px):
        return None
    if len(order) >= 2 and (dists[order[1]] - dists[best]) < float(margin_px):
        if dists[best] > 8.0:
            return None
    return int(best)


def scale_lock_ray(
    o0: np.ndarray,
    d0: np.ndarray,
    T1: np.ndarray,
    uv1: Sequence[float],
    K: K4,
    P0: np.ndarray,
    *,
    min_lat_m: float = 0.030,
    max_gap_m: float = 0.010,
    min_dp_m: float = 0.0,
    max_dp_m: float = 0.012,
    o1: Optional[np.ndarray] = None,
) -> Dict:
    """1-DOF: triangulate lock ray × second UV, then project onto lock ray."""
    d0u = d0 / (np.linalg.norm(d0) + 1e-12)
    o1_now = np.asarray(T1[:3, 3], dtype=np.float64) if o1 is None else np.asarray(o1)
    lat, along = lateral_baseline_m(o0, d0u, o1_now)
    out: Dict = {
        'ok': False,
        'reason': '',
        'lat_m': float(lat),
        'along_m': float(along),
        'P0': np.asarray(P0, dtype=np.float64).reshape(3).tolist(),
        'P1': None,
        'dp_m': None,
        'gap_m': None,
        's0_m': None,
        's1_m': None,
    }
    if lat < float(min_lat_m):
        out['reason'] = f'lat {lat*1000:.1f}mm < {min_lat_m*1000:.0f}mm'
        return out
    o1r, d1 = ray_from_uv(T1, uv1, K)
    tri = triangulate_rays(o0, d0u, o1r, d1)
    if tri is None:
        out['reason'] = 'parallel_rays'
        return out
    P_tri, gap = tri
    s1 = float(np.dot(P_tri - o0, d0u))
    P1 = o0 + s1 * d0u
    P0a = np.asarray(P0, dtype=np.float64).reshape(3)
    s0 = float(np.dot(P0a - o0, d0u))
    dp = float(np.linalg.norm(P1 - P0a))
    out.update({
        'P1': P1.tolist(),
        'P_tri': P_tri.tolist(),
        'dp_m': dp,
        'gap_m': float(gap),
        's0_m': s0,
        's1_m': s1,
    })
    if gap > float(max_gap_m):
        out['reason'] = f'skew {gap*1000:.1f}mm > {max_gap_m*1000:.0f}mm'
        return out
    if dp > float(max_dp_m):
        out['reason'] = f'|ΔP| {dp*1000:.1f}mm > {max_dp_m*1000:.0f}mm'
        return out
    if dp < float(min_dp_m):
        out['reason'] = f'|ΔP| {dp*1000:.1f}mm < {min_dp_m*1000:.0f}mm (keep P0)'
        out['ok'] = True
        out['apply'] = False
        return out
    out['ok'] = True
    out['apply'] = True
    out['reason'] = 'ok'
    return out


def recover_K_from_lock(
    uv: Sequence[float],
    berry_cam: Sequence[float],
    *,
    cx: float = 321.0,
    cy: float = 244.0,
) -> K4:
    """fx,fy from lock UV + berry_cam, using overlay principal point."""
    u, v = float(uv[0]), float(uv[1])
    x, y, z = (float(berry_cam[0]), float(berry_cam[1]), float(berry_cam[2]))
    fx = (u - cx) * z / x if abs(x) > 1e-9 else 372.0
    fy = (v - cy) * z / y if abs(y) > 1e-9 else fx
    return float(fx), float(fy), float(cx), float(cy)


def _self_test() -> None:
    K = (372.0, 372.0, 321.0, 244.0)
    T0 = np.eye(4)
    T0[:3, 3] = [0.0, 0.0, 0.0]
    z0 = 0.254
    uv0 = (447.0, 178.0)
    o0, d0 = ray_from_uv(T0, uv0, K)
    P0 = o0 + z0 / d0[2] * d0
    # True fruit 5mm farther along the same ray.
    Ptrue = o0 + (z0 + 0.005) / d0[2] * d0
    T1 = np.eye(4)
    T1[:3, 3] = [0.04, 0.0, 0.02]
    uv1 = reproject(T1, Ptrue, K)
    assert uv1 is not None
    r = scale_lock_ray(o0, d0, T1, uv1, K, P0, min_lat_m=0.02)
    assert r['ok'] and r.get('apply'), r
    err = float(np.linalg.norm(np.asarray(r['P1']) - Ptrue))
    assert err < 0.001, err
    # Coast UV (= reprojection of P0) must not "correct".
    uv_coast = reproject(T1, P0, K)
    assert uv_coast is not None
    r2 = scale_lock_ray(o0, d0, T1, uv_coast, K, P0, min_lat_m=0.02, min_dp_m=0.002)
    assert r2['ok'] and not r2.get('apply'), r2


if __name__ == '__main__':
    _self_test()
    print('pbvs_ray_scale self-test ok')
