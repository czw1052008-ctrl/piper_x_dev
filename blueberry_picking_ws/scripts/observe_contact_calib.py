#!/usr/bin/env python3
"""Entry-pose wrist observation vs known contact GT in base_link.

Usage:
  python3 scripts/observe_contact_calib.py \\
    --gt 0.097 0.337 0.254 \\
    --restore --restart-detector

Writes log/real_robot/qa/<session>/entry_obs_calib.json + wrist snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

from piper_position_ik import fk_link6_T, tip_xyz  # noqa: E402

from wrist_cam_extrinsic import T_base_cam, load_mount_rpy  # noqa: E402

TIP_OFF = np.array([0.0, 0.01883, 0.06152], dtype=np.float64)


def _berry_base(msg) -> Optional[Tuple[float, float, float]]:
    p = msg.pose.pose.position
    return float(p.x), float(p.y), float(p.z)


def _dist3(a, b) -> float:
    return float(math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))))


def _restart_fine_detector() -> None:
    subprocess.run(
        ['pkill', '-f', 'fine_detector_node'], check=False)
    time.sleep(2)
    log = ROOT / 'log' / 'real_robot' / 'fine_detector.log'
    env = os.environ.copy()
    env['PATH'] = '/usr/bin:/bin:' + env.get('PATH', '')
    with open(log, 'a', encoding='utf-8') as f:
        subprocess.Popen(
            ['bash', str(SCRIPTS / 'run_fine_detector_node.sh'),
             '-p', 'depth_pose_source:=depth'],
            cwd=str(ROOT),
            stdout=f, stderr=subprocess.STDOUT,
            env=env,
        )
    time.sleep(6)


def _load_bbox_decision(path: Path) -> Optional[Tuple[int, int, int, int]]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    bb = data.get('bbox_xyxy')
    if not bb or len(bb) != 4:
        return None
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


def _cam_to_base(cam_xyz: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    v = T_b_c @ np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0], dtype=np.float64)
    return v[:3]


def _lookup_T_base_cam(joints: List[float]) -> np.ndarray:
    tx, ty, tz, rx, ry, rz = load_mount_rpy()
    return T_base_cam(joints, tx, ty, tz, rx, ry, rz)


def _restore_entry(traj_s: float, settle_s: float) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / 'refine_entry_pose.py'), 'restore',
         '--traj-s', str(traj_s), '--settle-s', str(settle_s)],
        cwd=str(ROOT), check=False)


def run(args: argparse.Namespace) -> int:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from picking_msgs.msg import DetectedBerryArray
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, JointState

    gt = (float(args.gt[0]), float(args.gt[1]), float(args.gt[2]))
    if args.restore:
        _restore_entry(args.traj_s, args.settle_s)
    if args.restart_detector:
        _restart_fine_detector()

    bbox: Optional[Tuple[int, int, int, int]] = None
    if args.bbox:
        bbox = tuple(int(v) for v in args.bbox)
    elif args.bbox_decision:
        bbox = _load_bbox_decision(Path(args.bbox_decision))
    elif args.session_dir:
        bbox = _load_bbox_decision(
            Path(args.session_dir) / 'target_bbox_decision.json')

    session = args.session_dir or datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = ROOT / 'log' / 'real_robot' / 'qa' / session
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Node('observe_contact_calib')
    state: Dict[str, Any] = {
        'fine': None, 'joints': None, 'tcp': None, 'K': None,
        'wrist_rgb': None, 'fine_viz': None, 'depth': None,
    }

    def on_fine(msg: DetectedBerryArray) -> None:
        state['fine'] = msg

    def on_js(msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(j in name_to_pos for j in (
                'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')):
            state['joints'] = [
                float(name_to_pos[f'joint{i}']) for i in range(1, 7)]

    def on_tcp(msg: PoseStamped) -> None:
        state['tcp'] = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )

    def on_info(msg: CameraInfo) -> None:
        k = msg.k
        if len(k) >= 9:
            state['K'] = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))

    def on_wrist(msg: Image) -> None:
        try:
            from cv_bridge import CvBridge
            state['wrist_rgb'] = CvBridge().imgmsg_to_cv2(msg, 'rgb8')
        except Exception:
            pass

    def on_viz(msg: Image) -> None:
        try:
            from cv_bridge import CvBridge
            state['fine_viz'] = CvBridge().imgmsg_to_cv2(msg, 'rgb8')
        except Exception:
            pass

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

    node.create_subscription(DetectedBerryArray, '/perception/fine/berries', on_fine, 10)
    node.create_subscription(JointState, '/feedback/joint_states', on_js, 10)
    node.create_subscription(PoseStamped, '/feedback/tcp_pose', on_tcp, 10)
    node.create_subscription(CameraInfo, '/camera_wrist/color/camera_info', on_info, 10)
    node.create_subscription(Image, '/camera_wrist/color/image_raw', on_wrist,
                             qos_profile_sensor_data)
    node.create_subscription(Image, '/perception/fine/detection_viz', on_viz,
                             qos_profile_sensor_data)
    node.create_subscription(Image, '/camera_wrist/depth/image_raw', on_depth,
                             qos_profile_sensor_data)

    roi_samples: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    deadline = time.time() + float(args.wait_s)
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (
            bbox is not None
            and state.get('depth') is not None
            and state.get('K') is not None
            and state.get('joints') is not None
        ):
            z_roi = _roi_depth_m(state['depth'], bbox)
            if z_roi is not None:
                fx, fy, cx, cy = state['K']
                cu = 0.5 * (bbox[0] + bbox[2])
                cv = 0.5 * (bbox[1] + bbox[3])
                cam = np.array([
                    (cu - cx) / fx * z_roi,
                    (cv - cy) / fy * z_roi,
                    z_roi,
                ], dtype=np.float64)
                base = _cam_to_base(cam, _lookup_T_base_cam(state['joints']))
                roi_samples.append({
                    't': time.time(),
                    'bbox_xyxy': list(bbox),
                    'center_uv': [cu, cv],
                    'z_depth_roi_m': z_roi,
                    'berry_base': base.tolist(),
                    'err_to_gt_m': _dist3(base, gt),
                    'delta_mm': [round((base[i] - gt[i]) * 1000, 2) for i in range(3)],
                })
        fine = state['fine']
        if fine is None or not fine.berries:
            continue
        frame: Dict[str, Any] = {
            't': time.time(),
            'berries': [],
        }
        for b in fine.berries:
            base = _berry_base(b)
            if base is None:
                continue
            mode = str(b.depth_mode or '')
            zd = float(b.z_depth_m)
            zm = float(b.z_mono_m)
            err = _dist3(base, gt)
            frame['berries'].append({
                'track_id': int(b.track_id),
                'confidence': float(b.confidence),
                'depth_mode': mode,
                'z_depth_m': zd if zd > 0 else None,
                'z_mono_m': zm if zm > 0 else None,
                'berry_base': list(base),
                'berry_uv': [float(b.image_u), float(b.image_v)],
                'err_to_gt_m': err,
                'delta_mm': [round((base[i] - gt[i]) * 1000, 2) for i in range(3)],
            })
        if frame['berries']:
            samples.append(frame)
        time.sleep(0.12)

    node.destroy_node()
    rclpy.shutdown()

    if not samples and not roi_samples:
        print('ERROR: no fine detections and no ROI depth samples', file=sys.stderr)
        return 2

    # Pick berry closest to GT among depth-trusted frames.
    best: Optional[Dict[str, Any]] = None
    best_err = float('inf')
    for frame in samples:
        for b in frame['berries']:
            mode = b['depth_mode']
            if mode not in ('depth_raw', 'depth_coast'):
                continue
            if b.get('z_depth_m') is None:
                continue
            if b['err_to_gt_m'] < best_err:
                best_err = b['err_to_gt_m']
                best = dict(b)
                best['frame_t'] = frame['t']

    cup_open = None
    tcp = state.get('tcp')
    joints = state.get('joints')
    if joints is not None:
        T = fk_link6_T(joints)
        cup_open = tip_xyz(joints, tip_offset_link6=TIP_OFF).tolist()
        if tcp is None:
            tcp = T[:3, 3].tolist()

    import cv2  # type: ignore
    if state.get('wrist_rgb') is not None:
        vis = state['wrist_rgb'].copy()
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 80, 255), 2)
            cv2.putText(vis, 'USER BBOX', (x0, max(12, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 255), 1, cv2.LINE_AA)
        cv2.imwrite(
            str(out_dir / 'entry_obs_wrist.png'),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    if state.get('fine_viz') is not None:
        cv2.imwrite(
            str(out_dir / 'entry_obs_fine_viz.png'),
            cv2.cvtColor(state['fine_viz'], cv2.COLOR_RGB2BGR))

    payload = {
        'session': session,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'gt_contact_base': list(gt),
        'n_samples': len(samples),
        'n_frames_with_berries': len(samples),
        'tcp_base': list(tcp) if tcp is not None else None,
        'cup_open_base': cup_open,
        'intrinsics_fx_fy_cx_cy': list(state['K']) if state['K'] else None,
        'best_depth_berry': best,
        'user_bbox_xyxy': list(bbox) if bbox else None,
        'user_bbox_roi_median': (
            roi_samples[-1] if roi_samples else None),
        'all_depth_berries_last_frame': [
            b for b in samples[-1]['berries']
            if b['depth_mode'] in ('depth_raw', 'depth_coast')
        ] if samples else [],
        'policy': 'depth_pose_source=depth; user_bbox uses ROI depth median',
    }
    out_json = out_dir / 'entry_obs_calib.json'
    out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print(f'session={session}')
    print(f'gt_contact_base={gt}')
    if cup_open is not None:
        print(f'cup_open_entry={tuple(round(x, 4) for x in cup_open)}')
    if roi_samples:
        r = roi_samples[-1]
        print(
            f'user_bbox ROI depth base={tuple(round(x, 4) for x in r["berry_base"])}')
        print(
            f'  err={r["err_to_gt_m"]*1000:.1f}mm delta_mm={r["delta_mm"]} '
            f'z_roi={r["z_depth_roi_m"]:.3f}')
    if best:
        print(
            f'best_depth tid={best["track_id"]} mode={best["depth_mode"]} '
            f'base={tuple(round(x, 4) for x in best["berry_base"])}')
        print(
            f'  err={best["err_to_gt_m"]*1000:.1f}mm '
            f'delta_mm={best["delta_mm"]} '
            f'z_depth={best.get("z_depth_m")} z_mono={best.get("z_mono_m")}')
    else:
        print('WARNING: no depth_raw/coast berry in samples')
    print(f'json={out_json}')
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt', nargs=3, type=float, required=True,
                   metavar=('X', 'Y', 'Z'), help='Known contact point base_link (m)')
    p.add_argument('--restore', action='store_true', help='Move arm to refine entry pose')
    p.add_argument('--restart-detector', action='store_true',
                   help='Restart fine_detector with depth_pose_source=depth')
    p.add_argument('--wait-s', type=float, default=8.0, help='Observation window (s)')
    p.add_argument('--session-dir', default='',
                   help='Reuse QA session dir (reads target_bbox_decision.json)')
    p.add_argument('--bbox', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'),
                   help='Manual target bbox in wrist pixels')
    p.add_argument('--bbox-decision', default='',
                   help='Path to target_bbox_decision.json')
    p.add_argument('--traj-s', type=float, default=6.0)
    p.add_argument('--settle-s', type=float, default=2.0)
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
