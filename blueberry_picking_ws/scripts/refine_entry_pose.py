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
    """Prefer /feedback/joint_states; fall back to /joint_states (MoveIt).

    After drag-teach, agx feedback sometimes stops while ros2_control
    /joint_states keeps publishing the same physical pose.
    """
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node('refine_pose_read')
    got: dict = {'feedback': None, 'joint_states': None}

    def _extract(msg: JointState) -> Optional[List[float]]:
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(j in name_to_pos for j in ARM_JOINTS):
            return [float(name_to_pos[j]) for j in ARM_JOINTS]
        return None

    def on_feedback(msg: JointState) -> None:
        q = _extract(msg)
        if q is not None:
            got['feedback'] = q

    def on_js(msg: JointState) -> None:
        q = _extract(msg)
        if q is not None:
            got['joint_states'] = q

    node.create_subscription(JointState, '/feedback/joint_states', on_feedback, 10)
    node.create_subscription(JointState, '/joint_states', on_js, 10)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.2)
        if got['feedback'] is not None:
            break
    out = got['feedback'] if got['feedback'] is not None else got['joint_states']
    node.destroy_node()
    rclpy.shutdown()
    return out


def max_joint_delta_deg(current: Sequence[float], target: Sequence[float],
                        *, ignore_j6: bool = False) -> float:
    n = min(len(current), len(target), 6)
    indices = range(5) if ignore_j6 else range(n)
    return max(abs(math.degrees(float(current[i]) - float(target[i]))) for i in indices)


def per_joint_delta_deg(current: Sequence[float], target: Sequence[float]) -> List[float]:
    n = min(len(current), len(target), 6)
    return [math.degrees(float(current[i]) - float(target[i])) for i in range(n)]


def cmd_save(args: argparse.Namespace) -> int:
    current = read_current_joints(timeout_s=float(args.timeout_s))
    if current is None:
        print('no joint_states — is the arm stack up? '
              '(source /opt/ros/humble/setup.bash first)', file=sys.stderr)
        return 3
    session = args.session or time.strftime('%Y%m%d_%H%M%S')
    source = args.source or 'manual save'
    save_pose(args.path, current, session=session, source=source)
    deg = [math.degrees(float(v)) for v in current[:6]]
    print(f'saved → {args.path}')
    print(f'  source={source}  session={session}')
    print('  joints_deg: ' + ', '.join(
        f'{ARM_JOINTS[i]}={deg[i]:.2f}°' for i in range(6)))
    if abs(deg[5]) > 2.0:
        print(f'  WARN: j6={deg[5]:.2f}° (prefer ~0° for entry)')
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    if not args.path.is_file():
        print(f'missing {args.path}', file=sys.stderr)
        return 2
    target = load_pose(args.path)['joints_rad']
    current = read_current_joints()
    if current is None:
        print('no joint_states', file=sys.stderr)
        return 3
    deltas = per_joint_delta_deg(current, target)
    for i, nm in enumerate(ARM_JOINTS):
        print(f'  {nm}: Δ={deltas[i]:+.2f}°')
    d = max_joint_delta_deg(current, target, ignore_j6=args.ignore_j6)
    tag = 'j1-j5' if args.ignore_j6 else 'all'
    print(f'max_delta_deg ({tag})={d:.2f}')
    if args.ignore_j6 and abs(deltas[5]) > 2.0:
        print(f'  (j6={math.degrees(float(current[5])):.2f}° — URDF-locked, not gating entry)')
    return 0 if d <= args.max_deg else 1


def _send_joint_trajectory(
    target: Sequence[float],
    *,
    traj_s: float = 4.0,
    settle_s: float = 1.0,
    node_name: str = 'refine_pose_move',
) -> bool:
    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectoryPoint

    rclpy.init()
    node = Node(node_name)
    client = ActionClient(node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
    if not client.wait_for_server(timeout_sec=8.0):
        print('follow_joint_trajectory server not ready', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return False

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINTS)
    pt = JointTrajectoryPoint()
    pt.positions = [float(v) for v in target[:6]]
    pt.time_from_start.sec = int(traj_s)
    goal.trajectory.points = [pt]

    send = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send, timeout_sec=10.0)
    gh = send.result()
    if gh is None or not gh.accepted:
        print('trajectory rejected', file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return False
    result = gh.get_result_async()
    deadline = time.time() + traj_s + 6.0
    while not result.done() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    time.sleep(float(settle_s))
    return True


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

    if not _send_joint_trajectory(
        target,
        traj_s=args.traj_s,
        settle_s=args.settle_s,
    ):
        return 5

    current = read_current_joints()
    if current is None:
        return 6
    d_all = max_joint_delta_deg(current, target)
    d_j5 = max_joint_delta_deg(current, target, ignore_j6=True)
    print(f'after restore max_delta_deg={d_all:.2f} (j1-j5={d_j5:.2f})')
    if d_j5 <= args.max_deg:
        return 0
    if args.fix_j6:
        j6_delta = abs(math.degrees(float(current[5]) - float(target[5])))
        if j6_delta > 1.0 and d_j5 <= args.max_deg:
            print(f'j1-j5 within {d_j5:.2f}° but j6 off {j6_delta:.2f}° — fix j6 → 0')
            return cmd_fix_j6(args, current=current, target=target)
    return 7


def cmd_fix_j6(args: argparse.Namespace, *, current: Optional[Sequence[float]] = None,
               target: Optional[Sequence[float]] = None) -> int:
    """Move only j6 to saved entry (0 rad); keep j1-j5 at saved target."""
    if not args.path.is_file():
        print(f'missing {args.path}', file=sys.stderr)
        return 2
    target = list((target or load_pose(args.path)['joints_rad'])[:6])
    target[5] = 0.0
    current = current or read_current_joints()
    if current is not None:
        j6_delta = abs(math.degrees(float(current[5]) - target[5]))
        print(f'before fix-j6: j6={math.degrees(float(current[5])):.2f}° target=0.0° (Δ={j6_delta:.2f}°)')
        if j6_delta <= 1.0:
            print('j6 already at 0 — skip')
            return 0

    traj_s = float(getattr(args, 'traj_s', 2.0))
    settle_s = float(getattr(args, 'settle_s', 0.8))
    ok = _send_joint_trajectory(
        target, traj_s=traj_s, settle_s=settle_s, node_name='refine_pose_fix_j6')
    if not ok:
        ok = _send_move_j(target)
    if not ok:
        return 5

    current = read_current_joints()
    if current is None:
        return 6
    j6_delta = abs(math.degrees(float(current[5])))
    d_j5 = max_joint_delta_deg(current, target, ignore_j6=True)
    print(f'after fix-j6: j6={math.degrees(float(current[5])):.2f}° j1-j5={d_j5:.2f}°')
    if j6_delta <= 2.0:
        return 0
    print('WARN: j6 still off after traj + move_j (URDF lock — continue if j1-j5 ok)')
    return 0 if d_j5 <= args.max_deg else 7


def _send_move_j(target: Sequence[float]) -> bool:
    """Direct joint command fallback when trajectory action ignores j6."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node('refine_pose_move_j')
    pub = node.create_publisher(JointState, '/control/move_j', 10)
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.1)
    msg = JointState()
    msg.name = list(ARM_JOINTS)
    msg.position = [float(v) for v in target[:6]]
    pub.publish(msg)
    time.sleep(0.3)
    pub.publish(msg)
    node.destroy_node()
    rclpy.shutdown()
    time.sleep(2.0)
    print('sent /control/move_j (j6→0 fallback)')
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, default=DEFAULT_PATH)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_check = sub.add_parser('check', help='exit 0 if arm within max-deg of saved pose')
    p_check.add_argument('--max-deg', type=float, default=8.0)
    p_check.add_argument('--ignore-j6', action='store_true', default=True,
                         help='j6 is URDF-locked; only check j1-j5 (default)')
    p_check.add_argument('--include-j6', action='store_false', dest='ignore_j6')
    p_check.set_defaults(func=cmd_check)

    p_restore = sub.add_parser('restore', help='move arm to saved refine entry pose')
    p_restore.add_argument('--traj-s', type=float, default=4.0)
    p_restore.add_argument('--settle-s', type=float, default=1.0)
    p_restore.add_argument('--max-deg', type=float, default=8.0)
    p_restore.add_argument('--skip-if-within-deg', type=float, default=3.0)
    p_restore.add_argument('--fix-j6', action='store_true', default=True,
                           help='if j1-j5 ok but j6 stuck, send second traj for j6→0')
    p_restore.add_argument('--no-fix-j6', action='store_false', dest='fix_j6')
    p_restore.set_defaults(func=cmd_restore)

    p_j6 = sub.add_parser('fix-j6', help='move j6 to 0 rad (entry); j1-j5 → saved target')
    p_j6.add_argument('--traj-s', type=float, default=2.0)
    p_j6.add_argument('--settle-s', type=float, default=0.8)
    p_j6.add_argument('--max-deg', type=float, default=8.0)
    p_j6.set_defaults(func=cmd_fix_j6)

    p_save = sub.add_parser('save', help='read current joints and write refine_entry_pose.json')
    p_save.add_argument('--source', default='manual save',
                        help='label stored in JSON (e.g. "drag teach entry")')
    p_save.add_argument('--session', default='',
                        help='session id (default: timestamp YYYYMMDD_HHMMSS)')
    p_save.add_argument('--timeout-s', type=float, default=4.0,
                        help='wait for /feedback/joint_states')
    p_save.set_defaults(func=cmd_save)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
