#!/usr/bin/env python3
"""Persist / restore REFINING entry pose (ALIGN coarse_ok joints)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
DEFAULT_PATH = Path(__file__).resolve().parent.parent / 'log' / 'real_robot' / 'refine_entry_pose.json'


def load_pose(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pose(path: Path, joints_rad: Sequence[float], *, session: str = '', source: str = '') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'session': session,
        'source': source,
        'joints_rad': [float(v) for v in joints_rad[:6]],
        'joints_deg': {
            name: math.degrees(float(joints_rad[i])) if i < len(joints_rad) else 0.0
            for i, name in enumerate(ARM_JOINTS)
        },
    }
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    tmp.replace(path)


def read_current_joints(timeout_s: float = 4.0) -> Optional[List[float]]:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node('refine_pose_read')
    got: dict = {'js': None}

    def cb(msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(j in name_to_pos for j in ARM_JOINTS):
            got['js'] = [float(name_to_pos[j]) for j in ARM_JOINTS]

    node.create_subscription(JointState, '/feedback/joint_states', cb, 10)
    t0 = time.time()
    while got['js'] is None and time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.2)
    out = got['js']
    node.destroy_node()
    rclpy.shutdown()
    return out


def max_joint_delta_deg(current: Sequence[float], target: Sequence[float]) -> float:
    n = min(len(current), len(target), 6)
    return max(abs(math.degrees(float(current[i]) - float(target[i]))) for i in range(n))


def cmd_check(args: argparse.Namespace) -> int:
    if not args.path.is_file():
        print(f'missing {args.path}', file=sys.stderr)
        return 2
    target = load_pose(args.path)['joints_rad']
    current = read_current_joints()
    if current is None:
        print('no joint_states', file=sys.stderr)
        return 3
    d = max_joint_delta_deg(current, target)
    print(f'max_delta_deg={d:.2f}')
    return 0 if d <= args.max_deg else 1


def cmd_restore(args: argparse.Namespace) -> int:
    if not args.path.is_file():
        print(f'missing {args.path}', file=sys.stderr)
        return 2
    target = load_pose(args.path)['joints_rad']
    current = read_current_joints()
    if current is not None:
        d = max_joint_delta_deg(current, target)
        print(f'before restore max_delta_deg={d:.2f}')
        if d <= args.skip_if_within_deg:
            print('already at refine entry pose — skip motion')
            return 0

    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectoryPoint

    rclpy.init()
    node = Node('refine_pose_restore')
    client = ActionClient(node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
    if not client.wait_for_server(timeout_sec=8.0):
        print('follow_joint_trajectory server not ready', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 4

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINTS)
    pt = JointTrajectoryPoint()
    pt.positions = [float(v) for v in target[:6]]
    pt.time_from_start.sec = int(args.traj_s)
    goal.trajectory.points = [pt]

    send = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send, timeout_sec=10.0)
    gh = send.result()
    if gh is None or not gh.accepted:
        print('trajectory rejected', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 5
    result = gh.get_result_async()
    deadline = time.time() + args.traj_s + 6.0
    while not result.done() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    time.sleep(float(args.settle_s))
    current = read_current_joints()
    if current is None:
        return 6
    d = max_joint_delta_deg(current, target)
    print(f'after restore max_delta_deg={d:.2f}')
    return 0 if d <= args.max_deg else 7


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, default=DEFAULT_PATH)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_check = sub.add_parser('check', help='exit 0 if arm within max-deg of saved pose')
    p_check.add_argument('--max-deg', type=float, default=8.0)
    p_check.set_defaults(func=cmd_check)

    p_restore = sub.add_parser('restore', help='move arm to saved refine entry pose')
    p_restore.add_argument('--traj-s', type=float, default=4.0)
    p_restore.add_argument('--settle-s', type=float, default=1.0)
    p_restore.add_argument('--max-deg', type=float, default=8.0)
    p_restore.add_argument('--skip-if-within-deg', type=float, default=3.0)
    p_restore.set_defaults(func=cmd_restore)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
