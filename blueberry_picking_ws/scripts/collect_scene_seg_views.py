#!/usr/bin/env python3
"""Capture fixed+wrist RGB-D at several wrist viewpoints for scene_seg.

Moves only j1/j2/j3/j5 around the current (or --start) pose, then restores.
Does not start reach FSM or occupancy.

Usage:
  python3 scripts/collect_scene_seg_views.py
  python3 scripts/collect_scene_seg_views.py --no-move   # current pose only
"""

from __future__ import annotations

import math
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
DEFAULT_OUT = os.path.join(ROOT, 'datasets', 'scene_seg', 'raw')
ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def _decode_color(msg):
    enc = msg.encoding.lower()
    h, w = int(msg.height), int(msg.width)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == 'rgb8':
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()
    if enc == 'bgr8':
        return raw.reshape(h, w, 3).copy()
    return None


def _decode_depth_u16(msg):
    h, w = int(msg.height), int(msg.width)
    if msg.encoding in ('16UC1', 'mono16'):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).copy()
    if msg.encoding in ('32FC1', '32FC'):
        d = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        return np.clip(np.nan_to_num(d, nan=0.0) * 1000.0, 0, 65535).astype(np.uint16)
    return None


def _read_joints(timeout_s: float = 4.0) -> Optional[List[float]]:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    node = Node('collect_scene_seg_read_q')
    got = {'q': None}

    def cb(msg: JointState) -> None:
        m = dict(zip(msg.name, msg.position))
        if all(j in m for j in ARM_JOINTS):
            got['q'] = [float(m[j]) for j in ARM_JOINTS]

    node.create_subscription(JointState, '/feedback/joint_states', cb, 10)
    node.create_subscription(JointState, '/joint_states', cb, 10)
    t0 = time.time()
    while time.time() - t0 < timeout_s and got['q'] is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    return got['q']


def _send_traj(target: Sequence[float], traj_s: float = 3.5, settle_s: float = 0.8) -> bool:
    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectoryPoint

    node = Node('collect_scene_seg_traj')
    client = ActionClient(node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
    if not client.wait_for_server(timeout_sec=8.0):
        print('follow_joint_trajectory not ready', file=sys.stderr)
        node.destroy_node()
        return False
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINTS)
    pt = JointTrajectoryPoint()
    pt.positions = [float(v) for v in target[:6]]
    pt.time_from_start.sec = int(traj_s)
    nsec = int((traj_s - int(traj_s)) * 1e9)
    pt.time_from_start.nanosec = nsec
    goal.trajectory.points = [pt]
    send = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send, timeout_sec=10.0)
    gh = send.result()
    if gh is None or not gh.accepted:
        print('trajectory rejected', file=sys.stderr)
        node.destroy_node()
        return False
    result = gh.get_result_async()
    deadline = time.time() + traj_s + 8.0
    while not result.done() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    time.sleep(float(settle_s))
    return True


def _enable_arm() -> None:
    import rclpy
    from std_srvs.srv import SetBool
    from rclpy.node import Node

    node = Node('collect_scene_seg_enable')
    cli = node.create_client(SetBool, '/enable_agx_arm')
    if cli.wait_for_service(timeout_sec=4.0):
        req = SetBool.Request()
        req.data = True
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=6.0)
        print('[collect] enable_agx_arm', fut.result())
    else:
        print('[collect] WARN: /enable_agx_arm not available')
    node.destroy_node()


class Grabber:
    def __init__(self, node) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self.node = node
        self.got = {}
        node.create_subscription(
            Image, '/camera_fixed/color/image_raw',
            lambda m: self.got.__setitem__('fc', m), qos_profile_sensor_data)
        node.create_subscription(
            Image, '/camera_fixed/depth/image_raw',
            lambda m: self.got.__setitem__('fd', m), qos_profile_sensor_data)
        node.create_subscription(
            Image, '/camera_wrist/color/image_raw',
            lambda m: self.got.__setitem__('wc', m), qos_profile_sensor_data)
        node.create_subscription(
            Image, '/camera_wrist/depth/image_raw',
            lambda m: self.got.__setitem__('wd', m), qos_profile_sensor_data)

    def wait(self, timeout_s: float = 4.0) -> bool:
        import rclpy
        self.got.clear()
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if all(k in self.got for k in ('fc', 'fd', 'wc', 'wd')):
                return True
        return all(k in self.got for k in ('fc', 'fd', 'wc', 'wd'))

    def save(self, out_dir: str, tag: str) -> Optional[str]:
        import cv2
        if not all(k in self.got for k in ('fc', 'fd', 'wc', 'wd')):
            return None
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        pairs = (
            ('fixed_color', _decode_color(self.got['fc'])),
            ('fixed_depth', _decode_depth_u16(self.got['fd'])),
            ('wrist_color', _decode_color(self.got['wc'])),
            ('wrist_depth', _decode_depth_u16(self.got['wd'])),
        )
        for name, img in pairs:
            if img is None:
                print(f'  skip {name}: decode failed')
                return None
            path = os.path.join(out_dir, f'{stamp}_{tag}_{name}.png')
            cv2.imwrite(path, img)
        return stamp


def _viewpoints(q0: Sequence[float]) -> List[Tuple[str, List[float]]]:
    from piper_position_ik import clamp_joints

    q0 = [float(v) for v in q0[:6]]
    views: List[Tuple[str, List[float]]] = [('entry', list(q0))]

    def add(name: str, dq: Sequence[float]) -> None:
        q = clamp_joints([q0[i] + float(dq[i]) for i in range(6)])
        q[5] = 0.0
        views.append((name, q))

    r = math.radians
    add('j5_down', [0, 0, 0, 0, r(-18), 0])
    add('j5_up', [0, 0, 0, 0, r(16), 0])
    add('j1_left', [r(-14), 0, 0, 0, 0, 0])
    add('j1_right', [r(14), 0, 0, 0, 0, 0])
    add('j4_roll_p', [0, 0, 0, r(12), 0, 0])
    add('j4_roll_n', [0, 0, 0, r(-12), 0, 0])
    add('mid', [0, r(28), r(-20), 0, 0, 0])
    add('mid_j5_down', [0, r(28), r(-20), 0, r(-20), 0])
    add('mid_j5_up', [0, r(28), r(-20), 0, r(14), 0])
    add('mid_j1_l', [r(-12), r(28), r(-20), 0, 0, 0])
    add('mid_j1_r', [r(12), r(28), r(-20), 0, 0, 0])
    add('mid_j4_p', [0, r(28), r(-20), r(10), r(-8), 0])
    add('near', [0, r(48), r(-34), 0, r(-8), 0])
    add('near_j5_down', [0, r(48), r(-34), 0, r(-22), 0])
    add('near_j5_up', [0, r(48), r(-34), 0, r(12), 0])
    add('near_j1_l', [r(-14), r(48), r(-34), 0, r(-6), 0])
    add('near_j1_r', [r(14), r(48), r(-34), 0, r(-6), 0])
    add('near_j1_l2', [r(-22), r(45), r(-32), 0, r(-10), 0])
    add('high_lookdown', [0, r(38), r(-18), 0, r(-28), 0])
    add('side_high', [r(18), r(40), r(-26), r(8), r(-12), 0])
    add('side_low', [r(-16), r(36), r(-24), r(-8), r(-16), 0])
    return views


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--no-move', action='store_true')
    parser.add_argument('--traj-s', type=float, default=3.2)
    parser.add_argument('--max-views', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import rclpy
    from rclpy.node import Node

    rclpy.init()
    try:
        _enable_arm()
        q0 = _read_joints()
        if q0 is None:
            print('no joint_states', file=sys.stderr)
            return 3
        print('[collect] start q_deg=', [round(math.degrees(v), 1) for v in q0])
        grab_node = Node('collect_scene_seg_grab')
        grab = Grabber(grab_node)
        views = _viewpoints(q0)
        if args.no_move:
            views = views[:1]
        if args.max_views > 0:
            views = views[: args.max_views]
        saved = 0
        for i, (name, q) in enumerate(views):
            if not args.no_move:
                print(f'[collect] {i+1}/{len(views)} {name} q_deg={[round(math.degrees(v),1) for v in q]}')
                if not _send_traj(q, traj_s=args.traj_s, settle_s=0.7):
                    print(f'  WARN: traj failed, skip {name}')
                    continue
            else:
                print(f'[collect] capture only {name}')
            if not grab.wait(5.0):
                print('  WARN: cameras not ready')
                continue
            stamp = grab.save(args.out, name)
            if stamp:
                print(f'  saved {stamp}_{name}')
                saved += 1
        if not args.no_move:
            print('[collect] restore start pose')
            _send_traj(q0, traj_s=max(args.traj_s, 3.5), settle_s=0.8)
        grab_node.destroy_node()
        print(f'[collect] frames={saved} → {args.out}')
        return 0 if saved else 4
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
