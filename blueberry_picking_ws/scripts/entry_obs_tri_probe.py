#!/usr/bin/env python3
"""Entry pose: single-view depth vs two-view triangulation vs GT contact.

  python3 scripts/entry_obs_tri_probe.py \\
    --gt 0.097 0.337 0.254 \\
    --session-dir 20260812_152447 --restore --probe-step-m 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from goto_cup_base_xyz import send_traj  # noqa: E402
from piper_position_ik import fk_link6, fk_link6_T, position_ik_keep_orient  # noqa: E402

TIP_OFF = np.array([0.0, 0.01883, 0.06152], dtype=np.float64)
T_LINK6_CAM = np.eye(4, dtype=np.float64)
T_LINK6_CAM[:3, :3] = np.array([
    [1.0, 0.0, 0.0],
    [0.0, math.cos(-0.389557), -math.sin(-0.389557)],
    [0.0, math.sin(-0.389557), math.cos(-0.389557)],
])
T_LINK6_CAM[1, 3] = -0.07
T_LINK6_CAM[2, 3] = 0.04
ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def _dist3(a, b) -> float:
    return float(math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3))))


def _load_bbox(session_dir: Path) -> Tuple[int, int, int, int]:
    data = json.loads((session_dir / 'target_bbox_decision.json').read_text(encoding='utf-8'))
    bb = data['bbox_xyxy']
    return tuple(int(v) for v in bb)


def _roi_depth_m(depth: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[float]:
    x0, y0, x1, y1 = bbox
    h, w = depth.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    roi = depth[y0:y1, x0:x1]
    valid = roi[(roi >= 0.08) & (roi <= 1.5) & np.isfinite(roi)]
    if valid.size < 8:
        return None
    return float(np.median(valid))


def _lookup_T_base_cam(joints: List[float]) -> np.ndarray:
    return fk_link6_T(joints) @ T_LINK6_CAM


def _cam_to_base(cam_xyz: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    v = T_b_c @ np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0], dtype=np.float64)
    return v[:3]


def _triangulate_rays(
    o0: np.ndarray, d0: np.ndarray, o1: np.ndarray, d1: np.ndarray,
) -> Optional[np.ndarray]:
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
    return 0.5 * (p0 + p1)


def _ray_base(T_b_c: np.ndarray, uv: Tuple[float, float],
              K: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = K
    u, v = uv
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam) + 1e-12
    R = T_b_c[:3, :3]
    o = T_b_c[:3, 3].copy()
    d = R @ ray_cam
    return o, d


def _restore_entry(traj_s: float, settle_s: float) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / 'refine_entry_pose.py'), 'restore',
         '--traj-s', str(traj_s), '--settle-s', str(settle_s)],
        cwd=str(ROOT), check=False)


def _read_joints(node, state: Dict[str, Any], timeout_s: float = 4.0) -> Optional[List[float]]:
    t0 = time.time()
    while state.get('joints') is None and time.time() - t0 < timeout_s:
        import rclpy
        rclpy.spin_once(node, timeout_sec=0.1)
    return state.get('joints')


def _orbit_target_joints(
    joints: List[float], berry: np.ndarray, step_m: float, sign: float = 1.0,
) -> Tuple[List[float], Dict[str, float]]:
    """Match FSM mono_probe_orbit: rotate camera about vertical through berry."""
    T = _lookup_T_base_cam(joints)
    cam_o = T[:3, 3].copy()
    bx, by, bz = float(berry[0]), float(berry[1]), float(berry[2])
    vx, vy, vz = float(cam_o[0] - bx), float(cam_o[1] - by), float(cam_o[2] - bz)
    r_xy = math.hypot(vx, vy)
    _, p_link6 = fk_link6(joints)
    ex, ey, ez = float(p_link6[0]), float(p_link6[1]), float(p_link6[2])

    if r_xy < 0.05:
        raise RuntimeError(f'orbit radius too small r_xy={r_xy:.3f}m')
    alpha = sign * step_m / r_xy
    ca, sa = math.cos(alpha), math.sin(alpha)
    vx2 = ca * vx - sa * vy
    vy2 = sa * vx + ca * vy
    cam_new = np.array([bx + vx2, by + vy2, bz + vz], dtype=np.float64)
    d_base = cam_new - cam_o
    tgt = (ex + d_base[0], ey + d_base[1], ez + d_base[2])
    yaw_frac = 0.25
    dj1 = yaw_frac * alpha
    seed = list(joints)
    seed[0] = float(joints[0] + dj1)
    qi = position_ik_keep_orient(tgt, seed)
    if qi is None:
        raise RuntimeError('orbit IK failed')
    target = list(qi)
    target[0] = float(joints[0] + dj1)
    meta = {
        'orbit_alpha_deg': math.degrees(alpha),
        'dj1_deg': math.degrees(dj1),
        'cam_travel_m': float(np.linalg.norm(d_base)),
        'baseline_m': float(np.linalg.norm(d_base)),
    }
    return target, meta


def _capture_view(
    node, state: Dict[str, Any], bbox: Tuple[int, int, int, int], wait_s: float,
    *, uv_override: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    import rclpy
    t0 = time.time()
    while time.time() - t0 < wait_s:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (
            state.get('depth') is not None
            and state.get('K') is not None
            and state.get('joints') is not None
        ):
            break
    else:
        raise RuntimeError('timeout waiting depth/K/joints')

    fx, fy, cx, cy = state['K']
    cu = 0.5 * (bbox[0] + bbox[2])
    cv = 0.5 * (bbox[1] + bbox[3])
    uv = uv_override or (cu, cv)
    # Depth ROI around UV (berry may shift after probe move).
    half = max(8, (bbox[2] - bbox[0]) // 2 + 4)
    roi_bbox = (
        int(round(uv[0] - half)), int(round(uv[1] - half)),
        int(round(uv[0] + half)), int(round(uv[1] + half)),
    )
    z = _roi_depth_m(state['depth'], roi_bbox)
    if z is None:
        raise RuntimeError('no valid ROI depth')
    T_b_c = _lookup_T_base_cam(state['joints'])
    cam = np.array([(cu - cx) / fx * z, (cv - cy) / fy * z, z], dtype=np.float64)
    base = _cam_to_base(cam, T_b_c)
    o, d = _ray_base(T_b_c, uv, state['K'])
    return {
        'uv': [float(uv[0]), float(uv[1])],
        'bbox_xyxy': list(bbox),
        'depth_roi_bbox': list(roi_bbox),
        'z_depth_roi_m': z,
        'berry_base_depth': base.tolist(),
        'T_base_cam': T_b_c.tolist(),
        'cam_origin': o.tolist(),
        'ray_dir': d.tolist(),
        'joints_rad': list(state['joints']),
    }


def _err_record(base: List[float], gt: Tuple[float, float, float]) -> Dict[str, Any]:
    dmm = [round((base[i] - gt[i]) * 1000, 2) for i in range(3)]
    return {
        'berry_base': base,
        'err_mm': _dist3(base, gt) * 1000,
        'delta_mm': dmm,
    }


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = float(area_a + area_b - inter)
    return inter / union if union > 1e-6 else 0.0


def _pick_fine_uv(state: Dict[str, Any], bbox: Tuple[int, int, int, int],
                    wait_s: float, node) -> Optional[Tuple[float, float]]:
    """Best IoU fine detection vs user bbox."""
    import rclpy
    t0 = time.time()
    best_iou, best_uv = 0.0, None
    while time.time() - t0 < wait_s:
        rclpy.spin_once(node, timeout_sec=0.1)
        fine = state.get('fine')
        if fine is None or not fine.berries:
            continue
        for b in fine.berries:
            mode = str(getattr(b, 'depth_mode', '') or '')
            if mode == 'base_coast':
                continue
            try:
                u = float(b.image_u)
                v = float(b.image_v)
            except (TypeError, ValueError):
                continue
            if u < 0 or v < 0:
                continue
            side = max(24, bbox[2] - bbox[0])
            det_bb = (int(u - side // 2), int(v - side // 2),
                      int(u + side // 2), int(v + side // 2))
            iou = _bbox_iou(bbox, det_bb)
            if iou > best_iou:
                best_iou, best_uv = iou, (u, v)
    if best_uv is not None and best_iou >= 0.05:
        return best_uv
    return None


def _project_gt_uv(gt: Tuple[float, float, float], T_b_c: np.ndarray,
                   K: Tuple[float, float, float, float]) -> Tuple[float, float]:
    T_c_b = np.linalg.inv(T_b_c)
    p = T_c_b @ np.array([gt[0], gt[1], gt[2], 1.0], dtype=np.float64)
    if p[2] <= 1e-4:
        raise RuntimeError('GT behind camera')
    fx, fy, cx, cy = K
    return float(fx * p[0] / p[2] + cx), float(fy * p[1] / p[2] + cy)


def run(args: argparse.Namespace) -> int:
    import rclpy
    from picking_msgs.msg import DetectedBerryArray
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, JointState

    gt = (float(args.gt[0]), float(args.gt[1]), float(args.gt[2]))
    session_name = Path(args.session_dir).name
    session_dir = ROOT / 'log' / 'real_robot' / 'qa' / session_name
    if not session_dir.is_dir():
        print(f'missing session {session_dir}', file=sys.stderr)
        return 2
    bbox = _load_bbox(session_dir)
    berry_gt = np.array(gt, dtype=np.float64)

    if args.restore:
        _restore_entry(args.traj_s, args.settle_s)

    rclpy.init()
    node = Node('entry_obs_tri_probe')
    state: Dict[str, Any] = {'joints': None, 'K': None, 'depth': None}

    def on_js(msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(j in name_to_pos for j in ARM_JOINTS):
            state['joints'] = [float(name_to_pos[j]) for j in ARM_JOINTS]

    def on_info(msg: CameraInfo) -> None:
        k = msg.k
        if len(k) >= 9:
            state['K'] = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))

    def on_depth(msg: Image) -> None:
        try:
            enc = msg.encoding
            if enc in ('32FC1', '32FC'):
                state['depth'] = np.frombuffer(
                    msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
            elif enc in ('16UC1', 'mono16'):
                dmm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                state['depth'] = dmm.astype(np.float32) / 1000.0
        except Exception:
            pass

    def on_fine(msg: DetectedBerryArray) -> None:
        state['fine'] = msg

    node.create_subscription(JointState, '/feedback/joint_states', on_js, 10)
    node.create_subscription(CameraInfo, '/camera_wrist/color/camera_info', on_info, 10)
    node.create_subscription(Image, '/camera_wrist/depth/image_raw', on_depth,
                             qos_profile_sensor_data)
    node.create_subscription(DetectedBerryArray, '/perception/fine/berries', on_fine, 10)

    print(f'session={session_name} bbox={bbox} gt={gt}')
    view0 = _capture_view(node, state, bbox, float(args.wait_s))
    e0 = _err_record(view0['berry_base_depth'], gt)
    print(
        f'view0 depth base={tuple(round(x, 4) for x in view0["berry_base_depth"])} '
        f'err={e0["err_mm"]:.1f}mm delta={e0["delta_mm"]} z={view0["z_depth_roi_m"]:.3f}')

    q0 = list(state['joints'])
    target, orbit_meta = _orbit_target_joints(
        q0, berry_gt, float(args.probe_step_m), sign=1.0)
    print(
        f'orbit: alpha={orbit_meta["orbit_alpha_deg"]:.2f}deg '
        f'baseline≈{orbit_meta["baseline_m"]*1000:.1f}mm')

    print(f'moving probe view ({args.traj_s:.1f}s)...')
    if not send_traj(target, float(args.traj_s)):
        print('ERROR: probe move failed', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 3
    time.sleep(float(args.settle_s))

    state = {'joints': None, 'K': state.get('K'), 'depth': None, 'fine': None}
    uv1 = _pick_fine_uv(state, bbox, float(args.wait_s), node)
    uv1_src = 'fine_iou'
    if uv1 is None:
        T1 = _lookup_T_base_cam(state['joints'])
        uv1 = _project_gt_uv(gt, T1, state['K'])
        uv1_src = 'gt_reproject'
    view1 = _capture_view(node, state, bbox, 2.0, uv_override=uv1)
    view1['uv_source'] = uv1_src
    e1 = _err_record(view1['berry_base_depth'], gt)
    print(
        f'view1 depth base={tuple(round(x, 4) for x in view1["berry_base_depth"])} '
        f'err={e1["err_mm"]:.1f}mm delta={e1["delta_mm"]} z={view1["z_depth_roi_m"]:.3f}')

    T0 = np.array(view0['T_base_cam'], dtype=np.float64)
    T1 = np.array(view1['T_base_cam'], dtype=np.float64)
    baseline = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
    o0, d0 = _ray_base(T0, tuple(view0['uv']), state['K'])
    o1, d1 = _ray_base(T1, tuple(view1['uv']), state['K'])
    xyz_tri = _triangulate_rays(o0, d0, o1, d1)
    tri_ok = xyz_tri is not None and baseline >= 0.008
    etri = None
    if xyz_tri is not None:
        etri = _err_record(xyz_tri.tolist(), gt)
        print(
            f'tri base={tuple(round(x, 4) for x in xyz_tri.tolist())} '
            f'err={etri["err_mm"]:.1f}mm delta={etri["delta_mm"]} '
            f'baseline={baseline*1000:.1f}mm')

    # Error decomposition for view0 depth
    cam0 = T0[:3, 3]
    vec_gt = berry_gt - cam0
    z_true = float(np.linalg.norm(vec_gt))
    z_meas = float(view0['z_depth_roi_m'])
    # Scale-only correction along ray
    u0, v0 = view0['uv']
    fx, fy, cx, cy = state['K']
    ray_cam = np.array([(u0 - cx) / fx, (v0 - cy) / fy, 1.0], dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam)
    scaled = _cam_to_base(ray_cam * z_true, T0)
    escale = _err_record(scaled.tolist(), gt)

    # XY-only: keep measured Z, use GT XY projection? Better: report XY vs Z components
    delta = np.array(view0['berry_base_depth']) - berry_gt
    ray_world = T0[:3, :3] @ ray_cam
    ray_world /= np.linalg.norm(ray_world)
    along = float(np.dot(delta, ray_world))
    perp = delta - along * ray_world

    analysis = {
        'view0_depth_err_mm': e0['err_mm'],
        'view1_depth_err_mm': e1['err_mm'],
        'tri_err_mm': etri['err_mm'] if etri else None,
        'tri_improvement_mm': (
            e0['err_mm'] - etri['err_mm']) if etri else None,
        'baseline_mm': baseline * 1000,
        'z_depth_bias_mm': (z_meas - z_true) * 1000,
        'z_true_cam_m': z_true,
        'along_ray_mm': along * 1000,
        'perp_ray_mm': float(np.linalg.norm(perp)) * 1000,
        'perp_xy_mm': float(np.linalg.norm(perp[:2])) * 1000,
        'if_z_scaled_to_gt_range_err_mm': escale['err_mm'],
        'uv_shift_px': [
            round(view1['uv'][0] - view0['uv'][0], 1),
            round(view1['uv'][1] - view0['uv'][1], 1),
        ],
        'view1_uv_source': view1.get('uv_source'),
        'interpretation': (
            'Large perp_xy with small z_bias → extrinsic TX/TY or bbox center bias; '
            'tri helps depth scale but not pure XY extrinsic error at short baseline.'
        ),
    }

    node.destroy_node()
    rclpy.shutdown()

    payload = {
        'session': session_name,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'gt_contact_base': list(gt),
        'user_bbox_xyxy': list(bbox),
        'view0': view0 | {'err': e0},
        'view1': view1 | {'err': e1},
        'triangulation': {
            'ok': tri_ok,
            'baseline_m': baseline,
            'berry_base_tri': xyz_tri.tolist() if xyz_tri is not None else None,
            'err': etri,
            'orbit_meta': orbit_meta,
        },
        'error_analysis': analysis,
    }
    out_json = session_dir / 'entry_obs_tri_probe.json'
    out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('--- analysis ---')
    for k, v in analysis.items():
        if k != 'interpretation':
            print(f'  {k}: {v}')
    print(f'  note: {analysis["interpretation"]}')
    print(f'json={out_json}')

    if args.restore_after:
        _restore_entry(args.traj_s, args.settle_s)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt', nargs=3, type=float, required=True)
    p.add_argument('--session-dir', required=True)
    p.add_argument('--restore', action='store_true')
    p.add_argument('--restore-after', action='store_true', default=True)
    p.add_argument('--no-restore-after', action='store_false', dest='restore_after')
    p.add_argument('--probe-step-m', type=float, default=0.05)
    p.add_argument('--wait-s', type=float, default=5.0)
    p.add_argument('--traj-s', type=float, default=5.0)
    p.add_argument('--settle-s', type=float, default=2.0)
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
