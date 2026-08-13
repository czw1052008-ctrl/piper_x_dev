#!/usr/bin/env python3
"""Open-loop: move cup tip (= opening) to a base_link XYZ (keep seed orientation).

Cup model:
  tip = opening = link6 + R_link6 @ TIP_OFFSET
  TIP_OFFSET from base_link GT (see constant below). No separate rim.

IK targets link6 position (position_ik_keep_orient), then FollowJointTrajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from piper_position_ik import (  # noqa: E402
    fk_link6,
    position_ik_keep_orient,
    position_ik_keep_orient_chunked,
)

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
# Tip (= cup opening) in link6. Fit from GT 20260812 with table→base Z:
#   base_link is +12mm above desktop; user Z was table-relative.
#   tip_table (0,0.335,0.284) → tip_base (0,0.335,0.272) @ tip075 pose.
TIP_OFFSET = np.array([0.0, 0.01883, 0.06152], dtype=np.float64)
CUP_OFF = TIP_OFFSET.copy()
TABLE_TO_BASE_Z = 0.012  # base_link origin height above desktop (m)


def cup_open_xyz(q: Sequence[float]) -> np.ndarray:
    R, p = fk_link6(list(q))
    return p + R @ CUP_OFF


def tip_xyz(q: Sequence[float]) -> np.ndarray:
    return cup_open_xyz(q)


def link6_for_cup(cup_tgt: np.ndarray, R: np.ndarray) -> np.ndarray:
    return cup_tgt - R @ CUP_OFF


def read_joints(timeout_s: float = 5.0) -> Optional[List[float]]:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    if not rclpy.ok():
        rclpy.init()
    node = Node('goto_cup_js')
    got: dict = {'js': None}

    def cb(msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(j in name_to_pos for j in ARM_JOINTS):
            got['js'] = [float(name_to_pos[j]) for j in ARM_JOINTS]

    node.create_subscription(JointState, '/feedback/joint_states', cb, 10)
    t0 = time.time()
    while got['js'] is None and time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
    out = got['js']
    node.destroy_node()
    return out


def send_traj(q: Sequence[float], traj_s: float) -> bool:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    if not rclpy.ok():
        rclpy.init()
    node = Node('goto_cup_traj')
    client = ActionClient(
        node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
    if not client.wait_for_server(timeout_sec=10.0):
        print('ERROR: follow_joint_trajectory not available', file=sys.stderr)
        node.destroy_node()
        return False
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINTS)
    pt = JointTrajectoryPoint()
    pt.positions = [float(v) for v in list(q)[:6]]
    dt = float(max(0.5, traj_s))
    pt.time_from_start.sec = int(dt)
    pt.time_from_start.nanosec = int((dt - int(dt)) * 1e9)
    goal.trajectory.points = [pt]
    fut = client.send_goal_async(goal)
    t0 = time.time()
    while not fut.done() and time.time() - t0 < 10.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not fut.done():
        print('ERROR: goal send timeout', file=sys.stderr)
        node.destroy_node()
        return False
    gh = fut.result()
    if gh is None or not gh.accepted:
        print('ERROR: goal rejected', file=sys.stderr)
        node.destroy_node()
        return False
    res_fut = gh.get_result_async()
    deadline = time.time() + dt + 8.0
    while not res_fut.done() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    ok = res_fut.done()
    node.destroy_node()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', type=float, default=0.0)
    ap.add_argument('--y', type=float, default=0.35)
    ap.add_argument('--z', type=float, default=0.29)
    ap.add_argument(
        '--z-frame', choices=('base', 'table'), default='base',
        help='base=base_link Z; table=Z above desktop (subtracts 12mm to base)')
    ap.add_argument('--traj-s', type=float, default=6.0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument(
        '--out', type=str,
        default=str(ROOT.parent / 'log' / 'real_robot' / 'goto_cup_base_xyz.json'))
    args = ap.parse_args()

    cup_tgt = np.array([args.x, args.y, args.z], dtype=np.float64)
    z_frame = str(args.z_frame)
    if z_frame == 'table':
        cup_tgt = cup_tgt.copy()
        cup_tgt[2] = float(cup_tgt[2]) - TABLE_TO_BASE_Z
        print(
            f'z_frame=table: target_table=[{args.x}, {args.y}, {args.z}] '
            f'→ target_base={np.round(cup_tgt, 4).tolist()} '
            f'(base_link +{TABLE_TO_BASE_Z*1000:.0f}mm above desk)')
    q0 = read_joints()
    if q0 is None:
        print('ERROR: no /feedback/joint_states', file=sys.stderr)
        return 2

    R0, p0 = fk_link6(q0)
    cup0 = cup_open_xyz(q0)
    tip0 = tip_xyz(q0)

    print(f'tip(=cup)_offset_link6 = {CUP_OFF.tolist()}')
    print(f'q0_deg = {[round(math.degrees(v), 2) for v in q0]}')
    print(f'cup_now  = {np.round(cup0, 4).tolist()}')
    print(f'tip_now  = {np.round(tip0, 4).tolist()}')
    print(f'link6_now= {np.round(p0, 4).tolist()}')
    print(f'cup_tgt  = {np.round(cup_tgt, 4).tolist()}')

    q_sol = list(q0)
    method = 'iter_keep_orient'
    last_err = float('inf')
    for it in range(8):
        R_i, _ = fk_link6(q_sol)
        link6_goal = link6_for_cup(cup_tgt, R_i)
        qi = position_ik_keep_orient(tuple(link6_goal.tolist()), q_sol)
        step_method = 'keep_orient'
        if qi is None:
            qi = position_ik_keep_orient_chunked(tuple(link6_goal.tolist()), q_sol)
            step_method = 'keep_orient_chunked'
        if qi is None:
            print(f'ERROR: IK failed at iter {it}', file=sys.stderr)
            return 3
        cup_i = cup_open_xyz(qi)
        err_i = float(np.linalg.norm(cup_i - cup_tgt))
        print(
            f'  iter{it}: {step_method} cup={np.round(cup_i, 4).tolist()} '
            f'|err|={err_i*1000:.1f}mm')
        q_sol = list(qi)
        method = f'iter_{step_method}'
        if err_i < 0.002 or abs(last_err - err_i) < 5e-4:
            break
        last_err = err_i

    R_p, p_p = fk_link6(q_sol)
    cup_p = cup_open_xyz(q_sol)
    tip_p = tip_xyz(q_sol)
    err_p = cup_p - cup_tgt
    link6_goal = link6_for_cup(cup_tgt, R_p)
    print(f'IK method={method}')
    print(f'q_sol_deg = {[round(math.degrees(v), 2) for v in q_sol]}')
    print(f'cup_pred = {np.round(cup_p, 4).tolist()}  err_mm={np.round(err_p*1000, 1).tolist()} '
          f'|err|={np.linalg.norm(err_p)*1000:.1f}mm')
    print(f'tip_pred = {np.round(tip_p, 4).tolist()}')
    print(f'link6_pred={np.round(p_p, 4).tolist()}')

    payload = {
        'cup_target_base': cup_tgt.tolist(),
        'tip_offset_link6': TIP_OFFSET.tolist(),
        'cup_equals_tip': True,
        'q0_rad': q0,
        'cup_before': cup0.tolist(),
        'tip_before': tip0.tolist(),
        'link6_before': p0.tolist(),
        'link6_goal': link6_goal.tolist(),
        'ik_method': method,
        'q_sol_rad': q_sol,
        'cup_pred': cup_p.tolist(),
        'tip_pred': tip_p.tolist(),
        'cup_pred_err_m': err_p.tolist(),
        'executed': False,
    }

    if args.dry_run:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'dry-run only → {args.out}')
        return 0

    print(f'sending traj {args.traj_s:.1f}s ...')
    ok = send_traj(q_sol, args.traj_s)
    if not ok:
        print('ERROR: traj failed/incomplete', file=sys.stderr)
        return 4
    time.sleep(0.4)
    q1 = read_joints()
    if q1 is None:
        print('ERROR: no joints after move', file=sys.stderr)
        return 5
    R1, p1 = fk_link6(q1)
    cup1 = cup_open_xyz(q1)
    tip1 = tip_xyz(q1)
    err1 = cup1 - cup_tgt
    print('--- AFTER ---')
    print(f'q1_deg = {[round(math.degrees(v), 2) for v in q1]}')
    print(f'cup_after = {np.round(cup1, 4).tolist()}  '
          f'err_mm={np.round(err1*1000, 1).tolist()} '
          f'|err|={np.linalg.norm(err1)*1000:.1f}mm')
    print(f'tip_after = {np.round(tip1, 4).tolist()}')
    print(f'link6_after={np.round(p1, 4).tolist()}')
    print(f'Δq_max_deg={max(abs(math.degrees(q1[i]-q0[i])) for i in range(6)):.2f}')

    payload.update({
        'executed': True,
        'q_after_rad': q1,
        'cup_after': cup1.tolist(),
        'tip_after': tip1.tolist(),
        'link6_after': p1.tolist(),
        'cup_after_err_m': err1.tolist(),
        'cup_after_err_norm_mm': float(np.linalg.norm(err1) * 1000),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'wrote {args.out}')
    print(f'READY — measure real tip/cup vs target {cup_tgt.tolist()}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
