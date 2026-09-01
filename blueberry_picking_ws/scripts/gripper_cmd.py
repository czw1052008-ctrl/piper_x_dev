#!/usr/bin/env python3
"""Publish AGX gripper width command on /control/joint_states.

  python3 scripts/gripper_cmd.py --width 0.06
  python3 scripts/gripper_cmd.py --open          # profile pre_grasp_width
  python3 scripts/gripper_cmd.py --close         # profile grasp_width
  python3 scripts/gripper_cmd.py --profile agx_gripper_v1 --open
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from end_effector_profile import load_profile  # noqa: E402


def _pub_width(width_m: float, force_n: float) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node('gripper_cmd')
    pub = node.create_publisher(JointState, '/control/joint_states', 10)
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.05)
    msg = JointState()
    msg.name = ['gripper']
    msg.position = [float(width_m)]
    msg.effort = [float(force_n)]
    for _ in range(3):
        pub.publish(msg)
        time.sleep(0.05)
    node.destroy_node()
    rclpy.shutdown()
    print(f'gripper cmd width={width_m:.4f} m force={force_n:.2f} N')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', default='agx_gripper_v1')
    p.add_argument('--width', type=float, default=None, help='Absolute width (m)')
    p.add_argument('--open', action='store_true', help='Use pre_grasp_width_m')
    p.add_argument('--close', action='store_true', help='Use grasp_width_m')
    p.add_argument('--force', type=float, default=None)
    p.add_argument('--settle-s', type=float, default=0.8)
    args = p.parse_args()
    prof = load_profile(args.profile)
    force = float(args.force if args.force is not None else prof.grasp_force_n)
    if args.width is not None:
        w = float(args.width)
    elif args.open:
        w = float(prof.pre_grasp_width_m)
    elif args.close:
        w = float(prof.grasp_width_m)
    else:
        p.error('need --width / --open / --close')
        return 2
    _pub_width(w, force)
    time.sleep(max(0.0, float(args.settle_s)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
