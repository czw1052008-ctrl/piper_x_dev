#!/usr/bin/env python3
"""At REFINING entry pose: lock = nearest fine berry by cup(TCP)↔berry 3D distance."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_POSE = Path(__file__).resolve().parent.parent / 'log' / 'real_robot' / 'refine_entry_pose.json'
OUT_DIR = Path(__file__).resolve().parent.parent / 'log' / 'real_robot' / 'refine_tune_viz'
BASE_FRAME = 'base_link'
WRIST_CAM = 'camera_wrist_color_optical_frame'
FOCAL_PX = 550.0


@dataclass
class BerryRank:
    idx: int
    berry: object
    base: Tuple[float, float, float]
    dist_cup: float
    z_cam: Optional[float]
    pix_err: Optional[float]


def _matrix_from_tf(transform) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[0, 3], T[1, 3], T[2, 3] = t.x, t.y, t.z
    return T


def berry_base_xyz(tf_buffer, berry) -> Optional[Tuple[float, float, float]]:
    import rclpy
    p = berry.pose.pose.position
    bf = berry.pose.header.frame_id or berry.header.frame_id or BASE_FRAME
    if bf == BASE_FRAME or bf.endswith('/' + BASE_FRAME):
        return float(p.x), float(p.y), float(p.z)
    try:
        tf = tf_buffer.lookup_transform(BASE_FRAME, bf, rclpy.time.Time())
    except Exception:
        return float(p.x), float(p.y), float(p.z)
    v = _matrix_from_tf(tf) @ np.array([p.x, p.y, p.z, 1.0], dtype=np.float64)
    return float(v[0]), float(v[1]), float(v[2])


def berry_cam_xyz(tf_buffer, berry) -> Optional[Tuple[float, float, float]]:
    import rclpy
    p = berry.pose.pose.position
    bf = berry.pose.header.frame_id or berry.header.frame_id or BASE_FRAME
    try:
        tf = tf_buffer.lookup_transform(WRIST_CAM, bf, rclpy.time.Time())
    except Exception:
        try:
            tf = tf_buffer.lookup_transform(WRIST_CAM, BASE_FRAME, rclpy.time.Time())
        except Exception:
            return None
    v = _matrix_from_tf(tf) @ np.array([float(p.x), float(p.y), float(p.z), 1.0], dtype=np.float64)
    return float(v[0]), float(v[1]), float(v[2])


def rank_by_cup_dist(berries, tf_buffer, tcp_xyz: Tuple[float, float, float]) -> List[BerryRank]:
    tx, ty, tz = tcp_xyz
    ranked: List[BerryRank] = []
    for i, b in enumerate(berries):
        base = berry_base_xyz(tf_buffer, b)
        if base is None:
            continue
        dist = math.sqrt(
            (base[0] - tx) ** 2 + (base[1] - ty) ** 2 + (base[2] - tz) ** 2)
        cam = berry_cam_xyz(tf_buffer, b)
        z_cam = cam[2] if cam is not None else None
        pix_err = None
        if cam is not None and cam[2] > 1e-4:
            pix_err = math.hypot(cam[0], cam[1]) / cam[2] * FOCAL_PX
        ranked.append(BerryRank(i, b, base, dist, z_cam, pix_err))
    ranked.sort(key=lambda r: r.dist_cup)
    return ranked


def imgmsg_to_rgb(msg):
    from cv_bridge import CvBridge
    bridge = CvBridge()
    enc = msg.encoding.lower()
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8' if 'rgb' in enc else 'passthrough')
    if img.ndim == 2:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def cam_uv_on_image(cam_xyz: Tuple[float, float, float], wrist_shape) -> Tuple[int, int]:
    h, w = wrist_shape[:2]
    cx, cy, cz = cam_xyz
    du = int(FOCAL_PX * cx / cz) + w // 2
    dv = int(FOCAL_PX * cy / cz) + h // 2
    return du, dv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT_DIR / 'entry_lock_inspect.png')
    parser.add_argument('--wait-s', type=float, default=8.0)
    parser.add_argument('--top-n', type=int, default=5, help='show top-N by cup distance in overlay')
    args = parser.parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from picking_msgs.msg import DetectedBerryArray
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = Node('refine_entry_lock_viz')
    tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
    TransformListener(tf_buffer, node, spin_thread=True)

    state = {
        'fine': None, 'wrist': None, 'viz': None, 'tcp': None, 'fine_t': 0.0,
    }

    def on_fine(msg: DetectedBerryArray) -> None:
        state['fine'] = msg
        state['fine_t'] = time.time()

    def on_wrist(msg: Image) -> None:
        state['wrist'] = msg

    def on_viz(msg: Image) -> None:
        state['viz'] = msg

    def on_tcp(msg: PoseStamped) -> None:
        state['tcp'] = msg

    node.create_subscription(DetectedBerryArray, '/perception/fine/berries', on_fine, 10)
    node.create_subscription(Image, '/camera_wrist/color/image_raw', on_wrist, qos_profile_sensor_data)
    node.create_subscription(Image, '/perception/fine/detection_viz', on_viz, qos_profile_sensor_data)
    node.create_subscription(PoseStamped, '/feedback/tcp_pose', on_tcp, 10)

    t0 = time.time()
    while time.time() - t0 < args.wait_s:
        rclpy.spin_once(node, timeout_sec=0.2)
        if state['fine'] and state['wrist'] and state['tcp']:
            break

    import cv2

    fine = state['fine']
    if fine is None or not fine.berries:
        print('no /perception/fine/berries — is fine_detector running?', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 3
    if state['tcp'] is None:
        print('no /feedback/tcp_pose', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 4

    tp = state['tcp'].pose.position
    tcp_xyz = (float(tp.x), float(tp.y), float(tp.z))
    ranked = rank_by_cup_dist(list(fine.berries), tf_buffer, tcp_xyz)
    if not ranked:
        print('could not compute berry positions', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 5

    lock = ranked[0]
    p = lock.berry.pose.pose.position
    lines: List[str] = [
        f'REFINING entry lock rule: nearest cup↔berry (3D)',
        f'fine n={len(fine.berries)} age={time.time() - state["fine_t"]:.2f}s',
        f'tcp=({tcp_xyz[0]:.3f},{tcp_xyz[1]:.3f},{tcp_xyz[2]:.3f})',
        f'LOCK [#{lock.idx}] T{int(lock.berry.track_id)} conf={lock.berry.confidence:.2f} '
        f'base=({p.x:.3f},{p.y:.3f},{p.z:.3f})',
        f'cup↔berry={lock.dist_cup:.3f}m',
    ]
    if lock.z_cam is not None:
        lines.append(f'z_cam={lock.z_cam:.3f}m  pix_err≈{lock.pix_err:.0f}px')
    lines.append('rank (cup dist):')
    for rank, r in enumerate(ranked[: args.top_n], start=1):
        bp = r.berry.pose.pose.position
        mark = ' ← LOCK' if r is lock else ''
        ztxt = f' z={r.z_cam:.3f}' if r.z_cam is not None else ''
        lines.append(
            f'  {rank}. [#{r.idx}] T{int(r.berry.track_id)} conf={r.berry.confidence:.2f} '
            f'cup={r.dist_cup:.3f}m{ztxt}{mark}')

    vis = None
    if state['wrist'] is not None:
        vis = imgmsg_to_rgb(state['wrist']).copy()
        h, w = vis.shape[:2]
        cv2.drawMarker(vis, (w // 2, h // 2), (80, 255, 120), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(vis, 'cam center', (w // 2 + 8, h // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 120), 1, cv2.LINE_AA)
        for rank, r in enumerate(ranked[: args.top_n], start=1):
            cam = berry_cam_xyz(tf_buffer, r.berry)
            if cam is None or cam[2] <= 1e-4:
                continue
            uv = cam_uv_on_image(cam, vis.shape)
            if r is lock:
                color = (255, 80, 255)
                cv2.circle(vis, uv, 18, color, 3)
                label = f'LOCK T{int(r.berry.track_id)} {r.dist_cup:.2f}m'
            else:
                color = (255, 200, 80)
                cv2.circle(vis, uv, 10, color, 2)
                label = f'T{int(r.berry.track_id)} {r.dist_cup:.2f}m'
            cv2.putText(vis, label, (uv[0] + 10, uv[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if r is lock:
                cv2.line(vis, (w // 2, h // 2), uv, (255, 180, 80), 1, cv2.LINE_AA)

    panels = []
    if vis is not None:
        panels.append(('wrist RGB (rank by cup dist)', vis))
    if state['viz'] is not None:
        try:
            panels.append(('fine detection_viz', imgmsg_to_rgb(state['viz'])))
        except Exception:
            pass

    if not panels:
        print('no images received', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 6

    target_h = 480
    resized = []
    for _, img in panels:
        scale = target_h / float(img.shape[0])
        nw = max(1, int(img.shape[1] * scale))
        resized.append(cv2.resize(img, (nw, target_h)))
    gap = np.ones((target_h, 10, 3), dtype=np.uint8) * 48
    composite = resized[0]
    for img in resized[1:]:
        composite = np.hstack([composite, gap, img])

    y = 24
    for line in lines:
        cv2.putText(composite, line[:72], (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 120), 1, cv2.LINE_AA)
        y += 22

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.out), bgr)
    latest = args.out.parent / 'latest_entry_lock_inspect.png'
    cv2.imwrite(str(latest), bgr)

    print('\n'.join(lines))
    print(f'→ {args.out}')
    print(f'→ {latest}')

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
