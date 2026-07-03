#!/usr/bin/env python3
"""Append current /joint_states as a patrol pose (after teleop tuning).

Usage:
  bash scripts/record_scan_pose.sh              # append pose1, pose2, ...
  bash scripts/record_scan_pose.sh center_view  # named pose
"""

from __future__ import annotations

import argparse
import os
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
_WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_OUT = os.path.join(_WS, 'config', 'pick_scan_recorded.txt')


class _OnceJoints(Node):
    def __init__(self) -> None:
        super().__init__('record_scan_pose')
        self._positions: dict[str, float] = {}
        self._got = False
        self.create_subscription(JointState, '/joint_states', self._cb, 10)

    def _cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self._positions[name] = float(pos)
        if all(j in self._positions for j in ARM_JOINTS):
            self._got = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label', nargs='?', default='',
                        help='Optional pose name (comment in file)')
    parser.add_argument('-o', '--output', default=DEFAULT_OUT)
    parser.add_argument('--timeout', type=float, default=10.0)
    args = parser.parse_args()

    rclpy.init()
    node = _OnceJoints()
    deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    try:
        while rclpy.ok() and not node._got:
            if node.get_clock().now().nanoseconds > deadline:
                print('[record] ERROR: no /joint_states within timeout', file=sys.stderr)
                print('  Start bringup first.', file=sys.stderr)
                return 1
            rclpy.spin_once(node, timeout_sec=0.1)
        joints = [node._positions[j] for j in ARM_JOINTS]
    finally:
        node.destroy_node()
        rclpy.shutdown()

    line = ' '.join(f'{v:.4f}' for v in joints)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Count existing poses for auto label
    label = args.label.strip()
    if not label:
        n = 0
        if os.path.isfile(args.output):
            with open(args.output, encoding='utf-8') as fh:
                for raw in fh:
                    s = raw.strip()
                    if s and not s.startswith('#') and len(s.split()) >= 6:
                        n += 1
        label = f'pose{n + 1}'

    with open(args.output, 'a', encoding='utf-8') as fh:
        fh.write(f'# {label}\n')
        fh.write(f'{line}\n')

    print(f'[record] {label}: {line}')
    print(f'[record] appended -> {args.output}')
    print('[record] Pick loop uses this file automatically when it has poses.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
