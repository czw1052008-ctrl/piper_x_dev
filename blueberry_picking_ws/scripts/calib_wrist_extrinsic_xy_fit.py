#!/usr/bin/env python3
"""Fit wrist CAMERA_MOUNT TX/TY (+ optional full RPY) from multi-view labeled bboxes.

  # TX/TY only (legacy)
  python3 scripts/calib_wrist_extrinsic_xy_fit.py \\
    --session-dir log/real_robot/calib_extrinsic_xy/<session> --apply

  # TX/TY + roll/pitch/yaw joint fit
  python3 scripts/calib_wrist_extrinsic_xy_fit.py \\
    --session-dir log/real_robot/calib_extrinsic_xy/<session> --fit-rpy --apply
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from wrist_cam_extrinsic import (  # noqa: E402
    DEFAULT_ENV,
    depth_to_base,
    load_mount_rpy,
    point_to_ray_dist_m,
    project_gt_uv,
    ray_base,
    T_base_cam,
)


def _roi_depth_m(depth: np.ndarray, bbox: Sequence[int]) -> Optional[float]:
    x0, y0, x1, y1 = bbox
    h, w = depth.shape[:2]
    x0 = max(0, min(w - 1, int(x0)))
    x1 = max(x0 + 1, min(w, int(x1)))
    y0 = max(0, min(h - 1, int(y0)))
    y1 = max(y0 + 1, min(h, int(y1)))
    roi = depth[y0:y1, x0:x1]
    valid = roi[(roi >= 0.08) & (roi <= 1.5) & np.isfinite(roi)]
    if valid.size < 8:
        return None
    return float(np.median(valid))


def _bbox_center(bbox: Sequence[int]) -> Tuple[float, float]:
    return 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])


def _dist3(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _labeled_views(manifest: Dict[str, Any], exclude: List[str]) -> List[Dict[str, Any]]:
    out = []
    for v in manifest['views']:
        if v['id'] in exclude or v.get('skipped'):
            continue
        bb = v.get('bbox_xyxy')
        if bb and len(bb) == 4:
            out.append(v)
    return out


def _load_view_depth_m(session_dir: Path, v: Dict[str, Any]) -> Optional[float]:
    name = v.get('depth')
    bb = v.get('bbox_xyxy')
    if not name or not bb:
        return None
    path = session_dir / str(name)
    if not path.is_file():
        return None
    try:
        depth = np.load(str(path))
    except Exception:
        return None
    return _roi_depth_m(depth, bb)


def _eval_params(
    views: List[Dict[str, Any]], gt: Sequence[float], K: Tuple[float, float, float, float],
    tx: float, ty: float, tz: float,
    rx: float, ry: float = 0.0, rz: float = 0.0,
    depth_by_view: Optional[Dict[str, float]] = None,
    session_dir: Optional[Path] = None,
    sphere_r_m: float = 0.0,
) -> List[Dict[str, Any]]:
    rows = []
    depth_by_view = dict(depth_by_view or {})
    for v in views:
        if v['id'] not in depth_by_view and session_dir is not None:
            z_auto = _load_view_depth_m(session_dir, v)
            if z_auto is not None:
                depth_by_view[v['id']] = z_auto
        uv = _bbox_center(v['bbox_xyxy'])
        joints = v['joints_rad']
        T = T_base_cam(joints, tx, ty, tz, rx, ry, rz)
        o, d = ray_base(T, uv, K)
        ray_mm = point_to_ray_dist_m(gt, o, d) * 1000
        uv_proj = project_gt_uv(gt, joints, K, tx, ty, tz, rx, ry, rz)
        uv_err = math.hypot(uv[0] - uv_proj[0], uv[1] - uv_proj[1])
        row: Dict[str, Any] = {
            'id': v['id'],
            'target_angle_deg': v.get('j1_delta_deg'),
            'uv_label': list(uv),
            'uv_gt_project': list(uv_proj),
            'uv_err_px': round(uv_err, 2),
            'ray_dist_mm': round(ray_mm, 2),
        }
        z = depth_by_view.get(v['id'])
        if z is not None and z > 0:
            base = depth_to_base(
                uv, z, joints, K, tx, ty, tz, rx, ry, rz, sphere_r_m=sphere_r_m)
            base_raw = depth_to_base(
                uv, z, joints, K, tx, ty, tz, rx, ry, rz, sphere_r_m=0.0)
            row['z_depth_m'] = z
            row['berry_base_depth'] = base.tolist()
            row['berry_base_depth_raw'] = base_raw.tolist()
            row['depth_err_mm'] = round(_dist3(base, gt) * 1000, 2)
            row['depth_err_raw_mm'] = round(_dist3(base_raw, gt) * 1000, 2)
            row['depth_delta_mm'] = [
                round((base[i] - gt[i]) * 1000, 2) for i in range(3)]
        rows.append(row)
    return rows


def _ray_residuals(
    views: List[Dict[str, Any]], gt: Sequence[float], K: Tuple[float, float, float, float],
    tx: float, ty: float, tz: float, rx: float, ry: float, rz: float,
) -> np.ndarray:
    return np.array([
        point_to_ray_dist_m(gt, *ray_base(
            T_base_cam(v['joints_rad'], tx, ty, tz, rx, ry, rz),
            _bbox_center(v['bbox_xyxy']), K))
        for v in views
    ], dtype=np.float64)


def _fit_ty_tx(
    views: List[Dict[str, Any]], gt: Sequence[float], K: Tuple[float, float, float, float],
    tz: float, rx: float, ry: float, rz: float, tx0: float, ty0: float,
) -> Tuple[float, float, Dict[str, Any]]:
    from scipy.optimize import least_squares

    def residual(p: np.ndarray) -> np.ndarray:
        tx, ty = float(p[0]), float(p[1])
        return _ray_residuals(views, gt, K, tx, ty, tz, rx, ry, rz)

    sol = least_squares(residual, x0=np.array([tx0, ty0]), method='lm')
    tx_f, ty_f = float(sol.x[0]), float(sol.x[1])
    return tx_f, ty_f, {
        'success': bool(sol.success),
        'cost': float(sol.cost),
        'tx_before': tx0, 'ty_before': ty0,
        'tx_after': tx_f, 'ty_after': ty_f,
        'delta_tx_mm': (tx_f - tx0) * 1000,
        'delta_ty_mm': (ty_f - ty0) * 1000,
    }


def _fit_tx_ty_rpy(
    views: List[Dict[str, Any]], gt: Sequence[float], K: Tuple[float, float, float, float],
    tz: float, tx0: float, ty0: float, rx0: float, ry0: float, rz0: float,
    angle_bound_deg: float = 12.0, trans_bound_mm: float = 25.0,
) -> Tuple[float, float, float, float, float, Dict[str, Any]]:
    from scipy.optimize import least_squares

    x0 = np.array([tx0, ty0, rx0, ry0, rz0], dtype=np.float64)
    lo = x0.copy()
    hi = x0.copy()
    lo[0] -= trans_bound_mm / 1000.0
    hi[0] += trans_bound_mm / 1000.0
    lo[1] -= trans_bound_mm / 1000.0
    hi[1] += trans_bound_mm / 1000.0
    b = math.radians(angle_bound_deg)
    lo[2:] = x0[2:] - b
    hi[2:] = x0[2:] + b

    def residual(p: np.ndarray) -> np.ndarray:
        tx, ty, rx, ry, rz = [float(x) for x in p]
        return _ray_residuals(views, gt, K, tx, ty, tz, rx, ry, rz)

    sol = least_squares(residual, x0=x0, bounds=(lo, hi), method='trf', loss='soft_l1')
    tx_f, ty_f, rx_f, ry_f, rz_f = [float(x) for x in sol.x]
    return tx_f, ty_f, rx_f, ry_f, rz_f, {
        'success': bool(sol.success),
        'cost': float(sol.cost),
        'tx_before': tx0, 'ty_before': ty0,
        'rpy_before_rad': [rx0, ry0, rz0],
        'tx_after': tx_f, 'ty_after': ty_f,
        'rpy_after_rad': [rx_f, ry_f, rz_f],
        'delta_tx_mm': (tx_f - tx0) * 1000,
        'delta_ty_mm': (ty_f - ty0) * 1000,
        'delta_roll_deg': math.degrees(rx_f - rx0),
        'delta_pitch_deg': math.degrees(ry_f - ry0),
        'delta_yaw_deg': math.degrees(rz_f - rz0),
    }


def _summary(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return {'mean': -1, 'max': -1, 'rms': -1}
    return {
        'mean': round(float(np.mean(vals)), 2),
        'max': round(float(np.max(vals)), 2),
        'rms': round(float(math.sqrt(np.mean(np.square(vals)))), 2),
    }


def _apply_env(
    tx: float, ty: float, tz: float,
    rx: float, ry: float = 0.0, rz: float = 0.0,
) -> None:
    env_path = DEFAULT_ENV
    text = env_path.read_text(encoding='utf-8')
    rpy = f'{rx:.6f},{ry:.6f},{rz:.6f}'
    text = re.sub(r'(?m)^CAMERA_MOUNT_TX=.*$', f'CAMERA_MOUNT_TX={tx}', text)
    text = re.sub(r'(?m)^CAMERA_MOUNT_TY=.*$', f'CAMERA_MOUNT_TY={ty}', text)
    text = re.sub(r'(?m)^PICK_CAMERA_MOUNT_Y=.*$', f'PICK_CAMERA_MOUNT_Y={ty}', text)
    text = re.sub(r'(?m)^CAMERA_MOUNT_RPY=.*$', f'CAMERA_MOUNT_RPY={rpy}', text)
    env_path.write_text(text, encoding='utf-8')
    yaml_path = ROOT / 'src' / 'picking_description' / 'calibration' / 'wrist_camera_to_ee.yaml'
    xacro_path = ROOT / 'src' / 'picking_description' / 'urdf' / 'wrist_camera.urdf.xacro'
    if yaml_path.is_file():
        try:
            from scipy.spatial.transform import Rotation
            qx, qy, qz, qw = Rotation.from_euler('xyz', [rx, ry, rz]).as_quat()
        except Exception:
            qx = qy = qz = 0.0
            qw = 1.0
        yaml_path.write_text(
            '# Eye-in-hand: Piper X flange (link6) -> Gemini 305 optical frame.\n'
            f'# Multi-view fit via scripts/calib_wrist_extrinsic_xy_fit.py --fit-rpy\n'
            'frame_id: link6\n'
            'child_frame_id: camera_wrist_link\n'
            f'translation: [{tx}, {ty}, {tz}]\n'
            f'rotation: [{qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}]\n'
            f'rpy_rad: [{rx:.6f}, {ry:.6f}, {rz:.6f}]\n'
            f'rpy_deg: [{math.degrees(rx):.2f}, {math.degrees(ry):.2f}, {math.degrees(rz):.2f}]\n',
            encoding='utf-8')
        if xacro_path.is_file():
            text_x = xacro_path.read_text(encoding='utf-8')
            pat = re.compile(
                r'(<joint name="camera_wrist_joint"[^>]*>.*?<origin xyz=")[^"]+(" rpy=")[^"]+("/>)',
                re.DOTALL,
            )
            repl = rf'\g<1>{tx} {ty} {tz}\g<2>{rx:.6f} {ry:.6f} {rz:.6f}\g<3>'
            new, n = pat.subn(repl, text_x, count=1)
            if n == 1:
                xacro_path.write_text(new, encoding='utf-8')
    print(f'applied TX={tx:.5f} TY={ty:.5f} TZ={tz} '
          f'RPY(deg)=({math.degrees(rx):.2f},{math.degrees(ry):.2f},{math.degrees(rz):.2f})')


def run(args: argparse.Namespace) -> int:
    session_dir = Path(args.session_dir)
    if not session_dir.is_dir():
        session_dir = ROOT / args.session_dir
    exclude = [x.strip() for x in args.exclude_views.split(',') if x.strip()]
    manifest = json.loads((session_dir / 'manifest.json').read_text(encoding='utf-8'))
    views = _labeled_views(manifest, exclude)
    if args.verify_only:
        if not views:
            print('need >=1 labeled view to verify', file=sys.stderr)
            return 2
    elif len(views) < 3:
        print(f'need >=3 labeled views, got {len(views)}', file=sys.stderr)
        return 2

    gt = manifest['gt_contact_base']
    K_raw = manifest['intrinsics_fx_fy_cx_cy']
    K = tuple(float(x) for x in K_raw)  # type: ignore
    tx0, ty0, tz, rx0, ry0, rz0 = load_mount_rpy()
    sphere_r_m = float(args.sphere_r_mm) / 1000.0

    before = _eval_params(
        views, gt, K, tx0, ty0, tz, rx0, ry0, rz0,
        session_dir=session_dir, sphere_r_m=sphere_r_m)
    if args.verify_only:
        report = {
            'session': manifest['session'],
            'mode': 'verify_only',
            'gt_contact_base': gt,
            'n_views': len(views),
            'mount': {'tx': tx0, 'ty': ty0, 'tz': tz,
                      'roll_rad': rx0, 'pitch_rad': ry0, 'yaw_rad': rz0},
            'sphere_r_mm': args.sphere_r_mm,
            'ray_dist_mm': _summary(before, 'ray_dist_mm'),
            'uv_err_px': _summary(before, 'uv_err_px'),
            'depth_err_mm': _summary(before, 'depth_err_mm'),
            'depth_err_raw_mm': _summary(before, 'depth_err_raw_mm'),
            'per_view': before,
        }
        out_json = session_dir / 'multiview_verify.json'
        out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'session={manifest["session"]} views={len(views)}  VERIFY current mount')
        print(f'TX={tx0:.5f} TY={ty0:.5f} TZ={tz} '
              f'RPY(deg)=({math.degrees(rx0):.2f},{math.degrees(ry0):.2f},{math.degrees(rz0):.2f})')
        print(f'ray mean/max={report["ray_dist_mm"]["mean"]:.1f}/{report["ray_dist_mm"]["max"]:.1f} mm  '
              f'uv mean/max={report["uv_err_px"]["mean"]:.1f}/{report["uv_err_px"]["max"]:.1f} px  '
              f'depth3D(+r) mean/max={report["depth_err_mm"]["mean"]:.1f}/'
              f'{report["depth_err_mm"]["max"]:.1f} mm')
        fail = 0
        for r in before:
            ray_bad = r['ray_dist_mm'] > args.fail_ray_mm
            uv_bad = r['uv_err_px'] > args.fail_uv_px
            d_err = r.get('depth_err_mm')
            depth_bad = d_err is not None and d_err > args.fail_depth_mm
            flag = ' ⚠' if (ray_bad or uv_bad or depth_bad) else ' OK'
            if ray_bad or uv_bad or depth_bad:
                fail += 1
            ztxt = '' if d_err is None else f' depth3D={d_err:.1f}mm'
            print(f'  {r["id"]} @{r.get("target_angle_deg")}deg: '
                  f'ray={r["ray_dist_mm"]:.1f}mm uv={r["uv_err_px"]:.1f}px{ztxt}{flag}')
        print(f'json={out_json}')
        return 1 if fail else 0

    if args.fit_rpy:
        tx_f, ty_f, rx_f, ry_f, rz_f, fit_info = _fit_tx_ty_rpy(
            views, gt, K, tz, tx0, ty0, rx0, ry0, rz0,
            angle_bound_deg=args.angle_bound_deg,
            trans_bound_mm=args.trans_bound_mm)
        after = _eval_params(
            views, gt, K, tx_f, ty_f, tz, rx_f, ry_f, rz_f,
            session_dir=session_dir, sphere_r_m=sphere_r_m)
        report = {
            'session': manifest['session'],
            'mode': 'fit_rpy',
            'gt_contact_base': gt,
            'n_views': len(views),
            'sphere_r_mm': args.sphere_r_mm,
            'mount_before': {'tx': tx0, 'ty': ty0, 'tz': tz,
                             'roll_rad': rx0, 'pitch_rad': ry0, 'yaw_rad': rz0},
            'mount_after': {'tx': tx_f, 'ty': ty_f, 'tz': tz,
                            'roll_rad': rx_f, 'pitch_rad': ry_f, 'yaw_rad': rz_f},
            'fit': fit_info,
            'before': {'ray_dist_mm': _summary(before, 'ray_dist_mm'),
                       'uv_err_px': _summary(before, 'uv_err_px'),
                       'depth_err_mm': _summary(before, 'depth_err_mm'),
                       'per_view': before},
            'after': {'ray_dist_mm': _summary(after, 'ray_dist_mm'),
                      'uv_err_px': _summary(after, 'uv_err_px'),
                      'depth_err_mm': _summary(after, 'depth_err_mm'),
                      'per_view': after},
        }
        out_json = session_dir / 'extrinsic_rpy_fit.json'
        out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'session={manifest["session"]} views={len(views)}  FIT TX/TY/RPY')
        fi = fit_info
        print(f'TX {tx0:.4f}→{tx_f:.4f}  TY {ty0:.4f}→{ty_f:.4f}  '
              f'(Δ {fi["delta_tx_mm"]:+.1f}/{fi["delta_ty_mm"]:+.1f} mm)')
        print(f'RPY deg {tuple(round(math.degrees(x),2) for x in (rx0,ry0,rz0))}'
              f' → {tuple(round(math.degrees(x),2) for x in (rx_f,ry_f,rz_f))}  '
              f'(Δ {fi["delta_roll_deg"]:+.2f}/{fi["delta_pitch_deg"]:+.2f}/{fi["delta_yaw_deg"]:+.2f})')
        print(f'ray mean: {report["before"]["ray_dist_mm"]["mean"]:.1f}→'
              f'{report["after"]["ray_dist_mm"]["mean"]:.1f} mm  '
              f'max: {report["before"]["ray_dist_mm"]["max"]:.1f}→'
              f'{report["after"]["ray_dist_mm"]["max"]:.1f} mm')
        for r in after:
            flag = ' ⚠' if r['ray_dist_mm'] > args.fail_ray_mm else ''
            d_err = r.get('depth_err_mm')
            ztxt = '' if d_err is None else f' depth3D={d_err:.1f}mm'
            print(f'  {r["id"]} @{r.get("target_angle_deg")}deg: '
                  f'ray={r["ray_dist_mm"]:.1f}mm uv={r["uv_err_px"]:.1f}px{ztxt}{flag}')
        print(f'json={out_json}')
        if args.apply:
            _apply_env(tx_f, ty_f, tz, rx_f, ry_f, rz_f)
        return 0

    tx_f, ty_f, fit_info = _fit_ty_tx(views, gt, K, tz, rx0, ry0, rz0, tx0, ty0)
    after = _eval_params(
        views, gt, K, tx_f, ty_f, tz, rx0, ry0, rz0,
        session_dir=session_dir, sphere_r_m=sphere_r_m)

    report = {
        'session': manifest['session'],
        'mode': 'fit_xy',
        'gt_contact_base': gt,
        'n_views': len(views),
        'mount_before': {'tx': tx0, 'ty': ty0, 'tz': tz,
                         'roll_rad': rx0, 'pitch_rad': ry0, 'yaw_rad': rz0},
        'mount_after': {'tx': tx_f, 'ty': ty_f, 'tz': tz,
                        'roll_rad': rx0, 'pitch_rad': ry0, 'yaw_rad': rz0},
        'fit': fit_info,
        'before': {'ray_dist_mm': _summary(before, 'ray_dist_mm'),
                   'uv_err_px': _summary(before, 'uv_err_px'),
                   'depth_err_mm': _summary(before, 'depth_err_mm'),
                   'per_view': before},
        'after': {'ray_dist_mm': _summary(after, 'ray_dist_mm'),
                  'uv_err_px': _summary(after, 'uv_err_px'),
                  'depth_err_mm': _summary(after, 'depth_err_mm'),
                  'per_view': after},
    }
    out_json = session_dir / 'extrinsic_xy_fit.json'
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print(f'session={manifest["session"]} views={len(views)}')
    print(f'TX {tx0:.4f}→{tx_f:.4f}  TY {ty0:.4f}→{ty_f:.4f}  '
          f'(Δ {fit_info["delta_tx_mm"]:+.1f}/{fit_info["delta_ty_mm"]:+.1f} mm)')
    print(f'ray dist: {report["before"]["ray_dist_mm"]["mean"]:.1f}→'
          f'{report["after"]["ray_dist_mm"]["mean"]:.1f} mm (mean)')
    for r in after:
        flag = ' ⚠' if r['ray_dist_mm'] > args.fail_ray_mm else ''
        d_err = r.get('depth_err_mm')
        ztxt = '' if d_err is None else f' depth3D={d_err:.1f}mm'
        print(f'  {r["id"]} @{r.get("target_angle_deg")}deg: '
              f'ray={r["ray_dist_mm"]:.1f}mm uv={r["uv_err_px"]:.1f}px{ztxt}{flag}')
    print(f'json={out_json}')
    if args.apply:
        _apply_env(tx_f, ty_f, tz, rx0, ry0, rz0)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-dir', required=True)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--verify-only', action='store_true',
                   help='Do not refit; report 3D/UV error of current CAMERA_MOUNT vs GT')
    p.add_argument('--fit-rpy', action='store_true',
                   help='Joint fit TX/TY + roll/pitch/yaw (TZ fixed)')
    p.add_argument('--sphere-r-mm', type=float, default=7.5,
                   help='Sphere radius added to ROI depth for center correction')
    p.add_argument('--angle-bound-deg', type=float, default=12.0)
    p.add_argument('--trans-bound-mm', type=float, default=25.0)
    p.add_argument('--exclude-views', default='',
                   help='Comma-separated view ids to skip, e.g. view_06')
    p.add_argument('--fail-ray-mm', type=float, default=8.0)
    p.add_argument('--fail-uv-px', type=float, default=25.0)
    p.add_argument('--fail-depth-mm', type=float, default=12.0)
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
