#!/usr/bin/env python3
"""P0 tip-touch smoke: CLOSED gripper fingertip → contact XYZ (model approach segment).

Does NOT open/press-in/close — only moves tip to contact with fingers shut.
Uses /planning/tool_trajectory_4s → trajectory_executor (--drive-arm required).

  # Terminal A (if not already):
  bash scripts/run_trajectory_executor.sh --end-effector agx_gripper_v1 --drive-arm

  # Terminal B:
  python3 scripts/tip_touch_smoke.py --gt 0 0.35 0.255 --restore --drive
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from end_effector_profile import load_profile  # noqa: E402
from piper_position_ik import tip_approach_axis, tip_xyz  # noqa: E402
from tool_trajectory_utils import build_tool_trajectory_4s  # noqa: E402

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def _read_joints(timeout_s: float = 5.0) -> Optional[List[float]]:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node('tip_touch_js')
    got: dict = {'js': None}

    def cb(msg: JointState) -> None:
        m = dict(zip(msg.name, msg.position))
        if all(j in m for j in ARM_JOINTS):
            got['js'] = [float(m[j]) for j in ARM_JOINTS]

    node.create_subscription(JointState, '/feedback/joint_states', cb, 10)
    node.create_subscription(JointState, '/joint_states', cb, 10)
    t0 = time.time()
    while got['js'] is None and time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
    out = got['js']
    node.destroy_node()
    rclpy.shutdown()
    return out


def _gripper_close(profile_name: str) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / 'gripper_cmd.py'),
         '--profile', profile_name, '--close', '--settle-s', '1.0'],
        cwd=str(ROOT), check=False)


def run(args: argparse.Namespace) -> int:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Header
    from picking_msgs.msg import ExecutorStatus, IkJointTrajectory, ToolTrajectory4s

    prof = load_profile(args.end_effector)
    tip_off = list(prof.tip_offset_link6)
    gt = np.array([float(v) for v in args.gt], dtype=float)

    print('=== tip_touch_smoke (CLOSED fingertip → contact) ===')
    print(f'profile={prof.name} tip_off={tip_off}')
    print(f'gt_contact={gt.tolist()}')
    print('NOTE: open/press-in/grasp = rules later; this run is model approach only.')

    if args.close_gripper:
        print('closing gripper ...')
        _gripper_close(args.end_effector)

    if args.restore:
        print('restoring entry ...')
        subprocess.run(
            [sys.executable, str(SCRIPTS / 'refine_entry_pose.py'), 'restore',
             '--traj-s', str(args.traj_s), '--settle-s', '1.5'],
            cwd=str(ROOT), check=False)

    q = _read_joints()
    if q is None:
        print('ERROR: no joint_states', file=sys.stderr)
        return 2
    tip0 = tip_xyz(q, tip_offset_link6=tip_off)
    axis = tip_approach_axis(q)
    # Prefer look-at approach if tip→gt is well defined
    to = gt - tip0
    dist = float(np.linalg.norm(to))
    if dist > 1e-3:
        axis = to / dist
    print(f'tip0={np.round(tip0, 4).tolist()} dist={dist * 1000:.1f}mm')
    print(f'approach={np.round(axis, 4).tolist()}')

    if not args.drive:
        print('dry-run only (pass --drive to publish tool_trajectory_4s)')
        print('Ensure: bash scripts/run_trajectory_executor.sh '
              f'--end-effector {args.end_effector} --drive-arm')
        return 0

    rclpy.init()
    node = Node('tip_touch_smoke')
    got = {'ik': None, 'st': None}

    def on_ik(msg: IkJointTrajectory) -> None:
        got['ik'] = msg

    def on_st(msg: ExecutorStatus) -> None:
        got['st'] = msg
        print(f'  status={msg.status} detail={msg.detail} '
              f'tip_err={msg.tip_err_m * 1000:.1f}mm')

    node.create_subscription(IkJointTrajectory, '/planning/ik_joint_trajectory', on_ik, 10)
    node.create_subscription(ExecutorStatus, '/planning/executor_status', on_st, 10)
    pub = node.create_publisher(ToolTrajectory4s, '/planning/tool_trajectory_4s', 10)

    t0 = time.time()
    while time.time() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() >= 1:
            break
    if pub.get_subscription_count() < 1:
        print('ERROR: no subscriber on /planning/tool_trajectory_4s '
              '(start trajectory_executor --drive-arm)', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 3

    hdr = Header()
    hdr.stamp = node.get_clock().now().to_msg()
    hdr.frame_id = 'base_link'
    msg = build_tool_trajectory_4s(
        tip0, gt, axis,
        header=hdr,
        seq=int(time.time()) % 100000,
        move_duration_s=float(args.move_s),
    )
    print(f'publishing tool_traj seq={msg.seq} n={len(msg.waypoints)} '
          f'exec_until={msg.execute_until_index}')
    t_pub = time.time()
    while time.time() - t_pub < 1.5:
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)

    t1 = time.time()
    while got['ik'] is None and time.time() - t1 < float(args.move_s) + 6.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    # settle then measure tip
    time.sleep(1.0)
    node.destroy_node()
    rclpy.shutdown()

    q2 = _read_joints()
    if q2 is None:
        print('WARN: no joints after move')
        return 4
    tip1 = tip_xyz(q2, tip_offset_link6=tip_off)
    err = float(np.linalg.norm(tip1 - gt))
    print(f'tip_after={np.round(tip1, 4).tolist()}')
    print(f'tip→gt err={err * 1000:.1f} mm')
    print(f'j deg={[round(math.degrees(x), 1) for x in q2]}')
    if got['ik'] is not None:
        print(f'ik_ok={got["ik"].ok} n_wp={len(got["ik"].waypoints)}')
    ok = err < float(args.tol_m)
    print('RESULT', 'PASS' if ok else 'CHECK (tip not within tol — inspect physically)')
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt', nargs=3, type=float, default=[0.0, 0.35, 0.255],
                   metavar=('X', 'Y', 'Z'))
    p.add_argument('--end-effector', default='agx_gripper_v1')
    p.add_argument('--restore', action='store_true')
    p.add_argument('--close-gripper', action='store_true', default=True)
    p.add_argument('--no-close-gripper', action='store_false', dest='close_gripper')
    p.add_argument('--drive', action='store_true',
                   help='Publish tool_trajectory (requires executor --drive-arm)')
    p.add_argument('--move-s', type=float, default=3.0)
    p.add_argument('--traj-s', type=float, default=6.0)
    p.add_argument('--tol-m', type=float, default=0.015,
                   help='Pass if |tip-gt| < tol after move')
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
