#!/usr/bin/env python3
"""Fit a local contact-plane normal from wrist depth near a berry bbox center.

Used at PBVS lock (frozen) and for offline QA replay. No ROS.
Normal convention: unit vector in the same frame as ``points``, flipped to
point toward ``toward`` (cup / camera). Approach/press travel along -n.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Apparent berry radius used only for pixel-ring size (not contact standoff).
_BERRY_R_M = 0.008
_MIN_INLIERS = 4
_DZ_MAX_M = 0.008
_RING_FRAC = 0.30
_N_RING = 8


def sample_uv_ring(
    u: float,
    v: float,
    radius_px: float,
    n_ring: int = _N_RING,
) -> List[Tuple[float, float]]:
    pts = [(float(u), float(v))]
    r = max(2.0, float(radius_px))
    for k in range(int(n_ring)):
        a = 2.0 * math.pi * k / float(n_ring)
        pts.append((float(u) + r * math.cos(a), float(v) + r * math.sin(a)))
    return pts


def pixel_ring_radius_px(
    z_m: float,
    fx: float,
    berry_r_m: float = _BERRY_R_M,
    frac: float = _RING_FRAC,
) -> float:
    z = max(float(z_m), 1e-3)
    diam_px = 2.0 * float(fx) * float(berry_r_m) / z
    return max(3.0, frac * diam_px)


def depth_at_uv(depth_m: np.ndarray, u: float, v: float) -> Optional[float]:
    h, w = depth_m.shape[:2]
    iu = int(round(u))
    iv = int(round(v))
    if iu < 0 or iv < 0 or iu >= w or iv >= h:
        return None
    z = float(depth_m[iv, iu])
    if not math.isfinite(z) or z < 0.05 or z > 2.0:
        return None
    return z


def unproject_cam(
    u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    return np.array(
        [(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def fit_plane_pca(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (centroid, unit normal) or None."""
    if points.shape[0] < _MIN_INLIERS:
        return None
    c = points.mean(axis=0)
    x = points - c
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    n = vt[-1].astype(np.float64)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return None
    return c, n / nn


def flip_toward(n: np.ndarray, toward: np.ndarray) -> np.ndarray:
    if float(np.dot(n, toward)) < 0.0:
        return -n
    return n


def clamp_normal_to_axis(
    n: np.ndarray, axis: np.ndarray, max_deg: float,
) -> Tuple[np.ndarray, float, bool]:
    """Clamp n toward ``axis`` (both unit). Returns (n_clamped, angle_deg, clamped)."""
    a = np.asarray(axis, dtype=np.float64).flatten()[:3]
    na = float(np.linalg.norm(a))
    if na < 1e-9:
        return n, 0.0, False
    a = a / na
    nn = np.asarray(n, dtype=np.float64).flatten()[:3]
    nn = nn / max(float(np.linalg.norm(nn)), 1e-12)
    c = float(np.clip(np.dot(nn, a), -1.0, 1.0))
    ang = math.degrees(math.acos(c))
    max_deg = float(max_deg)
    if ang <= max_deg + 1e-6:
        return nn, ang, False
    # Rotate n toward axis to max_deg.
    axis_w = np.cross(nn, a)
    nw = float(np.linalg.norm(axis_w))
    if nw < 1e-9:
        return a, ang, True
    axis_w = axis_w / nw
    # Rodrigues: rotate nn toward a by (ang - max_deg)
    delta = math.radians(ang - max_deg)
    k = axis_w
    n2 = (nn * math.cos(delta)
          + np.cross(k, nn) * math.sin(delta)
          + k * float(np.dot(k, nn)) * (1.0 - math.cos(delta)))
    n2 = n2 / max(float(np.linalg.norm(n2)), 1e-12)
    return n2, ang, True


def fit_contact_surface(
    depth_m: np.ndarray,
    u: float,
    v: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    T_base_cam: np.ndarray,
    toward_base: np.ndarray,
    z_hint_m: Optional[float] = None,
    radius_px: Optional[float] = None,
    dz_max_m: float = _DZ_MAX_M,
    min_inliers: int = _MIN_INLIERS,
) -> Dict[str, Any]:
    """Fit contact plane in base_link.

    ``T_base_cam`` is 4x4 cam→base. ``toward_base`` flips n to point at cup.
    """
    out: Dict[str, Any] = {
        'ok': False,
        'reason': '',
        'n_base': None,
        'contact_base': None,
        'n_inliers': 0,
        'samples': [],
        'radius_px': None,
        'angle_to_toward_deg': None,
    }
    z0 = depth_at_uv(depth_m, u, v)
    if z0 is None:
        z0 = float(z_hint_m) if z_hint_m is not None and z_hint_m > 0.05 else None
    if z0 is None:
        out['reason'] = 'no_center_depth'
        return out
    rpx = float(radius_px) if radius_px is not None else pixel_ring_radius_px(z0, fx)
    out['radius_px'] = rpx
    uvs = sample_uv_ring(u, v, rpx)
    samples: List[Dict[str, Any]] = []
    pts_cam: List[np.ndarray] = []
    for uu, vv in uvs:
        z = depth_at_uv(depth_m, uu, vv)
        rec = {'u': uu, 'v': vv, 'z_m': z, 'inlier': False}
        if z is None or abs(z - z0) > float(dz_max_m):
            samples.append(rec)
            continue
        rec['inlier'] = True
        samples.append(rec)
        pts_cam.append(unproject_cam(uu, vv, z, fx, fy, cx, cy))
    out['samples'] = samples
    out['n_inliers'] = len(pts_cam)
    if len(pts_cam) < int(min_inliers):
        out['reason'] = f'too_few_inliers:{len(pts_cam)}'
        return out
    pc = np.vstack(pts_cam)
    fitted = fit_plane_pca(pc)
    if fitted is None:
        out['reason'] = 'pca_failed'
        return out
    c_cam, n_cam = fitted
    R = T_base_cam[:3, :3]
    t = T_base_cam[:3, 3]
    c_base = R @ c_cam + t
    n_base = R @ n_cam
    n_base = n_base / max(float(np.linalg.norm(n_base)), 1e-12)
    toward = np.asarray(toward_base, dtype=np.float64).flatten()[:3]
    tn = float(np.linalg.norm(toward))
    if tn > 1e-9:
        toward = toward / tn
        n_base = flip_toward(n_base, toward)
        out['angle_to_toward_deg'] = math.degrees(
            math.acos(float(np.clip(np.dot(n_base, toward), -1.0, 1.0))))
    out['ok'] = True
    out['n_base'] = n_base.tolist()
    out['contact_base'] = c_base.tolist()
    out['reason'] = 'ok'
    return out


def synthetic_sphere_depth(
    h: int,
    w: int,
    *,
    u0: float,
    v0: float,
    z0: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    radius_m: float = _BERRY_R_M,
) -> np.ndarray:
    """Front-facing sphere depth (camera looking +Z). Center pixel (u0,v0) at z0."""
    # Sphere center along the center ray, behind the surface by radius.
    ray = unproject_cam(u0, v0, 1.0, fx, fy, cx, cy)
    ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
    p_surf = ray * float(z0)
    c = p_surf + ray * float(radius_m)  # center farther than surface
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dx = (uu - cx) / fx
    dy = (vv - cy) / fy
    # Ray dir (dx, dy, 1), intersect sphere |o + t d - c|^2 = r^2, o=0.
    d = np.stack([dx, dy, np.ones_like(dx)], axis=-1)
    dn = np.linalg.norm(d, axis=-1, keepdims=True)
    d = d / np.maximum(dn, 1e-12)
    oc = -c
    b = 2.0 * np.sum(d * oc, axis=-1)
    cc = float(np.dot(oc, oc) - radius_m * radius_m)
    disc = b * b - 4.0 * cc
    depth = np.full((h, w), np.nan, dtype=np.float32)
    hit = disc >= 0.0
    sqrt_d = np.sqrt(np.maximum(disc, 0.0))
    t0 = (-b - sqrt_d) / 2.0
    t1 = (-b + sqrt_d) / 2.0
    t = np.where((t0 > 1e-4) & (t0 <= t1), t0, t1)
    ok = hit & (t > 1e-4)
    depth[ok] = (t[ok] * d[ok, 2]).astype(np.float32)
    return depth


def _overlay_png(
    rgb_path: str,
    out_path: str,
    samples: Sequence[Dict[str, Any]],
    u: float,
    v: float,
    n_cam_uv: Optional[Tuple[float, float]] = None,
) -> None:
    try:
        import cv2
    except ImportError:
        return
    im = cv2.imread(rgb_path)
    if im is None:
        return
    for s in samples:
        uu, vv = int(round(s['u'])), int(round(s['v']))
        col = (0, 220, 0) if s.get('inlier') else (0, 0, 220)
        cv2.circle(im, (uu, vv), 4, col, 1)
    cv2.circle(im, (int(round(u)), int(round(v))), 6, (0, 255, 255), 2)
    if n_cam_uv is not None:
        cv2.arrowedLine(
            im,
            (int(round(u)), int(round(v))),
            (int(round(n_cam_uv[0])), int(round(n_cam_uv[1]))),
            (255, 180, 0), 2, tipLength=0.2)
    cv2.imwrite(out_path, im)


def replay_qa_session(qa_dir: str, *, fx: float = 488.0) -> Dict[str, Any]:
    """Offline fit for a QA session. Uses synthetic sphere if no depth raster."""
    lock_path = os.path.join(qa_dir, 'refine_fruit_lock.json')
    with open(lock_path, 'r', encoding='utf-8') as f:
        lock = json.load(f)
    uv = lock.get('berry_uv') or [None, None]
    cam = lock.get('berry_cam')
    base = lock.get('berry_base')
    cup = lock.get('tcp_base') or lock.get('cup_open')  # lock json has tcp; cup in overlay
    if uv[0] is None or cam is None or base is None:
        raise SystemExit(f'missing uv/cam/base in {lock_path}')
    u, v = float(uv[0]), float(uv[1])
    z = float(cam[2])
    fy = fx
    # Infer cx,cy so unproject(u,v,z) matches berry_cam xy.
    cx = u - float(cam[0]) * fx / z
    cy = v - float(cam[1]) * fy / z
    h, w = 480, 640
    wrist_png = os.path.join(qa_dir, 'lock_wait_000_wrist.png')
    if os.path.isfile(wrist_png):
        try:
            import cv2
            im = cv2.imread(wrist_png)
            if im is not None:
                h, w = im.shape[:2]
        except ImportError:
            pass
    depth = synthetic_sphere_depth(
        h, w, u0=u, v0=v, z0=z, fx=fx, fy=fy, cx=cx, cy=cy)
    # Cam→base from berry_cam / berry_base: use a translation-only approx plus
    # identity R is wrong. Reconstruct T so cam origin maps consistently:
    # p_base = R p_cam + t, we only have one point. Use lock joints? Skip R:
    # fit in cam, then rotate n into base by aligning cam Z to berry_cam.
    p_cam = np.asarray(cam, dtype=np.float64)
    p_base = np.asarray(base, dtype=np.float64)
    z_axis = p_cam / max(float(np.linalg.norm(p_cam)), 1e-12)
    # Build R_base_cam: cam Z → p_cam direction in... we need cam axes in base.
    # Approximate: t = p_base - R p_cam with R = I is bad.
    # Use: R such that cam optical axis (0,0,1) maps to p_base-tcp if available.
    tcp = np.asarray(lock.get('tcp_base') or base, dtype=np.float64)
    # Camera is near tcp (wrist). Optical +Z ≈ berry_base - cam_origin.
    # cam_origin_base ≈ p_base - R p_cam. If R maps cam Z to (p_base - origin)/| |
    # Simpler offline: fit n in camera, map n_base ≈ (p_cam-style): 
    # n_cam toward -optical (camera looks +Z at fruit, outward n ≈ -Z_cam).
    T = np.eye(4, dtype=np.float64)
    # Place cam origin so p_cam unprojects to p_base: origin = p_base - p_cam
    # with R=I (cam axes // base) — only for QA magnitude check.
    T[:3, 3] = p_base - p_cam
    toward = tcp - p_base
    if float(np.linalg.norm(toward)) < 1e-6:
        toward = -p_cam
    fit = fit_contact_surface(
        depth, u, v,
        fx=fx, fy=fy, cx=cx, cy=cy,
        T_base_cam=T,
        toward_base=toward,
        z_hint_m=z,
    )
    fit['session'] = os.path.basename(os.path.abspath(qa_dir))
    fit['source'] = 'synthetic_sphere_from_lock'
    fit['lock_uv'] = [u, v]
    fit['lock_berry_base'] = list(p_base)
    out_json = os.path.join(qa_dir, 'pbvs_surface_fit.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(fit, f, indent=2)
    if os.path.isfile(wrist_png):
        _overlay_png(
            wrist_png,
            os.path.join(qa_dir, 'pbvs_surface_fit_overlay.png'),
            fit.get('samples') or [],
            u, v)
    return fit


def _self_test() -> None:
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    u0, v0, z0 = 320.0, 240.0, 0.22
    depth = synthetic_sphere_depth(
        480, 640, u0=u0, v0=v0, z0=z0, fx=fx, fy=fy, cx=cx, cy=cy)
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.3, 0.25]
    toward = np.array([0.0, 0.0, -1.0])  # toward camera (-Z)
    fit = fit_contact_surface(
        depth, u0, v0, fx=fx, fy=fy, cx=cx, cy=cy,
        T_base_cam=T, toward_base=toward, z_hint_m=z0)
    assert fit['ok'], fit
    n = np.array(fit['n_base'])
    # Outward n should point toward camera (-Z in this setup).
    assert float(n[2]) < -0.7, n
    assert int(fit['n_inliers']) >= _MIN_INLIERS
    n2, ang, clamped = clamp_normal_to_axis(
        np.array([0.0, 0.2, -1.0]), np.array([0.0, 0.0, -1.0]), 20.0)
    assert ang > 5.0
    assert float(np.linalg.norm(n2)) > 0.99
    print('berry_surface_fit self-test OK',
          f'n={np.round(n, 3)} inliers={fit["n_inliers"]} clamp_ang={ang:.1f}')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qa-dir', action='append', default=[],
                   help='QA session dir (repeatable). Writes pbvs_surface_fit.json')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test or not args.qa_dir:
        _self_test()
        if not args.qa_dir:
            return 0
    for d in args.qa_dir:
        fit = replay_qa_session(d)
        print(f'{d}: ok={fit.get("ok")} inliers={fit.get("n_inliers")} '
              f'n={fit.get("n_base")} reason={fit.get("reason")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
