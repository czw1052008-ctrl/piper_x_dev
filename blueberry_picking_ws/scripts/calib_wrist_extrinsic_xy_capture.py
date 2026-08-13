#!/usr/bin/env python3
"""Capture multi-view wrist RGB-D + joints for extrinsic / 3D-pose verification.

After each j1 move, waits until live joints match the target, then snapshots
RGB + depth + joints together (fixes previous image/joint desync).

  python3 scripts/calib_wrist_extrinsic_xy_capture.py \\
    --gt 0.097 0.337 0.254 --restore --j1-deltas-deg=0,-8,8,-16,16,-22,22
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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from goto_cup_base_xyz import send_traj  # noqa: E402
from wrist_cam_extrinsic import load_mount_from_env, project_gt_uv  # noqa: E402

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def _restore_entry(traj_s: float, settle_s: float) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / 'refine_entry_pose.py'), 'restore',
         '--traj-s', str(traj_s), '--settle-s', str(settle_s)],
        cwd=str(ROOT), check=False)


def _j1_target(entry_joints: List[float], delta_deg: float) -> List[float]:
    q = list(entry_joints)
    q[0] = float(entry_joints[0] + math.radians(delta_deg))
    q[5] = 0.0
    return q


def _max_dj_deg(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b), 6)
    return max(abs(math.degrees(float(a[i]) - float(b[i]))) for i in range(n))


def _decode_depth(msg) -> Optional[np.ndarray]:
    enc = msg.encoding
    if enc in ('32FC1', '32FC'):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height, msg.width).copy()
    if enc in ('16UC1', 'mono16'):
        dmm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return dmm.astype(np.float32) / 1000.0
    return None


def _wait_snapshot(
    node, state: Dict[str, Any], target: Optional[List[float]],
    timeout_s: float, max_dj_deg: float = 0.8,
) -> bool:
    """Spin until joints match target (if given), then grab fresh RGB+depth."""
    import rclpy
    t0 = time.time()
    matched_t: Optional[float] = None
    while time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.05)
        js = state.get('joints')
        if target is not None:
            if js is None or _max_dj_deg(js, target) > max_dj_deg:
                matched_t = None
                continue
            if matched_t is None:
                matched_t = time.time()
            # Hold 0.4s at target so RGB/depth catch up.
            if time.time() - matched_t < 0.4:
                continue
        if (
            state.get('rgb') is not None
            and state.get('depth') is not None
            and state.get('K') is not None
            and js is not None
        ):
            if target is None or _max_dj_deg(js, target) <= max_dj_deg:
                return True
    return False


def run(args: argparse.Namespace) -> int:
    import cv2  # type: ignore
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, JointState

    gt = [float(args.gt[0]), float(args.gt[1]), float(args.gt[2])]
    deltas = [float(x.strip()) for x in args.j1_deltas_deg.split(',') if x.strip()]
    if not deltas:
        print('no j1 deltas', file=sys.stderr)
        return 2
    deltas = sorted(deltas, key=lambda d: (abs(d) > 1e-6, abs(d), d))
    tx, ty, tz, rx = load_mount_from_env()

    if args.restore:
        _restore_entry(args.traj_s, args.settle_s)

    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = ROOT / 'log' / 'real_robot' / 'calib_extrinsic_xy' / session
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Node('calib_xy_capture')
    state: Dict[str, Any] = {
        'rgb': None, 'depth': None, 'joints': None, 'K': None, 'wh': None,
    }

    def on_rgb(msg: Image) -> None:
        try:
            from cv_bridge import CvBridge
            state['rgb'] = CvBridge().imgmsg_to_cv2(msg, 'rgb8')
        except Exception:
            pass

    def on_depth(msg: Image) -> None:
        try:
            state['depth'] = _decode_depth(msg)
        except Exception:
            pass

    def on_js(msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(j in name_to_pos for j in ARM_JOINTS):
            state['joints'] = [float(name_to_pos[j]) for j in ARM_JOINTS]

    def on_info(msg: CameraInfo) -> None:
        k = msg.k
        if len(k) >= 9:
            state['K'] = [float(k[0]), float(k[4]), float(k[2]), float(k[5])]
            state['wh'] = [int(msg.width), int(msg.height)]

    node.create_subscription(Image, '/camera_wrist/color/image_raw', on_rgb,
                           qos_profile_sensor_data)
    node.create_subscription(Image, '/camera_wrist/depth/image_raw', on_depth,
                           qos_profile_sensor_data)
    node.create_subscription(JointState, '/feedback/joint_states', on_js, 10)
    node.create_subscription(CameraInfo, '/camera_wrist/color/camera_info', on_info, 10)

    if not _wait_snapshot(node, state, None, 8.0):
        print('ERROR: no wrist RGB/depth/joints', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 3

    entry_joints = list(state['joints'])
    views: List[Dict[str, Any]] = []

    for i, delta_deg in enumerate(deltas):
        target_joints = _j1_target(entry_joints, delta_deg)
        print(f'view_{i:02d}: j1 Δ{delta_deg:+.0f}°  target_j1='
              f'{math.degrees(target_joints[0]):.1f}°')
        if abs(delta_deg) > 1e-6:
            if not send_traj(target_joints, float(args.traj_s)):
                print(f'WARN: traj rejected at view {i}', file=sys.stderr)
            state['rgb'] = None
            state['depth'] = None
        if not _wait_snapshot(node, state, target_joints, float(args.traj_s) + 6.0):
            print(f'ERROR: joints did not reach target for view {i} '
                  f'(have j1={math.degrees(state["joints"][0]) if state.get("joints") else float("nan"):.1f}°)',
                  file=sys.stderr)
            continue

        joints_now = list(state['joints'])
        dj = _max_dj_deg(joints_now, target_joints)
        img_name = f'view_{i:02d}_wrist.png'
        depth_name = f'view_{i:02d}_depth.npy'
        cv2.imwrite(str(out_dir / img_name),
                    cv2.cvtColor(state['rgb'], cv2.COLOR_RGB2BGR))
        np.save(str(out_dir / depth_name), state['depth'])

        overlay = state['rgb'].copy()
        uv_gt = None
        try:
            uv_gt = project_gt_uv(gt, joints_now, tuple(state['K']), tx, ty, tz, rx)
            u, v = int(round(uv_gt[0])), int(round(uv_gt[1]))
            cv2.drawMarker(overlay, (u, v), (0, 255, 80), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(overlay, f'GT {delta_deg:+.0f}deg', (u + 8, max(16, v - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1, cv2.LINE_AA)
        except Exception as exc:
            print(f'  GT project failed: {exc}')
        cv2.imwrite(str(out_dir / f'view_{i:02d}_gt_overlay.png'),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        views.append({
            'id': f'view_{i:02d}',
            'image': img_name,
            'depth': depth_name,
            'j1_delta_deg': delta_deg,
            'joints_rad': joints_now,
            'max_dj_to_target_deg': round(dj, 3),
            'gt_uv_project': list(uv_gt) if uv_gt else None,
            'bbox_xyxy': None,
            'labeled': False,
        })
        print(f'  saved {img_name} j1={math.degrees(joints_now[0]):.1f}° '
              f'|Δq|={dj:.2f}deg  gt_uv={None if uv_gt is None else (round(uv_gt[0],1), round(uv_gt[1],1))}')

        # Return to entry before next yaw so each move is from the same seed.
        if i < len(deltas) - 1 and abs(delta_deg) > 1e-6:
            send_traj(entry_joints, float(args.traj_s))
            state['rgb'] = None
            state['depth'] = None
            _wait_snapshot(node, state, entry_joints, float(args.traj_s) + 6.0)

    manifest = {
        'session': session,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'gt_contact_base': gt,
        'intrinsics_fx_fy_cx_cy': state.get('K'),
        'image_size_wh': state.get('wh'),
        'entry_joints_rad': entry_joints,
        'j1_deltas_deg': deltas,
        'mount_used': {'tx': tx, 'ty': ty, 'tz': tz, 'rx_rad': rx},
        'views': views,
        'label_instructions': (
            f'python3 scripts/label_calib_multi_bbox.py '
            f'--session-dir log/real_robot/calib_extrinsic_xy/{session}'
        ),
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')

    node.destroy_node()
    rclpy.shutdown()

    print('restoring refine entry ...')
    _restore_entry(args.traj_s, args.settle_s)

    print(f'session={session}')
    print(f'dir={out_dir}')
    print(f'views={len(views)}')
    print('下一步: python3 scripts/label_calib_multi_bbox.py '
          f'--session-dir log/real_robot/calib_extrinsic_xy/{session}')
    return 0 if views else 4


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt', nargs=3, type=float, required=True)
    p.add_argument('--restore', action='store_true')
    p.add_argument(
        '--j1-deltas-deg', default='0,-8,8,-16,16,-22,22',
        help='j1 yaw deltas from refine entry (comma-separated, no spaces)')
    p.add_argument('--traj-s', type=float, default=5.0)
    p.add_argument('--settle-s', type=float, default=2.0)
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
