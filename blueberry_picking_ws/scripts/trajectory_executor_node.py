#!/usr/bin/env python3
"""P0b trajectory executor: tool_trajectory_4s → tool_pose_ik → joint_cmd.

Subscribes:
  /planning/tool_trajectory_4s
  /feedback/joint_states (fallback /joint_states)

Publishes:
  /planning/ik_joint_trajectory
  /planning/joint_cmd
  /planning/executor_status

Optional arm drive via FollowJointTrajectory (--drive-arm).
"""

from __future__ import annotations

import argparse
import math
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from end_effector_profile import load_profile
from piper_position_ik import tip_approach_axis, tip_xyz, tool_pose_ik
from picking_msgs.msg import (
    ExecutorStatus,
    IkJointTrajectory,
    JointWaypoint,
    ToolTrajectory4s,
)

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def _duration_to_s(d: Duration) -> float:
    return float(d.sec) + 1e-9 * float(d.nanosec)


def _s_to_duration(t_s: float) -> Duration:
    t = max(0.0, float(t_s))
    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    out = Duration()
    out.sec = sec
    out.nanosec = nanosec
    return out


class TrajectoryExecutorNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('trajectory_executor')
        self._args = args
        self._profile = load_profile(args.end_effector)
        self._tip_off = self._profile.tip_offset_link6
        self._joints: Optional[List[float]] = None
        self._seq = 0
        self._last_tool_seq = -1
        self._busy_until = 0.0
        self._arm_goal_handle = None

        self.create_subscription(
            ToolTrajectory4s, '/planning/tool_trajectory_4s', self._on_tool_traj, 10)
        self.create_subscription(
            JointState, '/feedback/joint_states', self._on_joints, 10)
        self.create_subscription(
            JointState, '/joint_states', self._on_joints, 10)

        self._pub_ik = self.create_publisher(IkJointTrajectory, '/planning/ik_joint_trajectory', 10)
        self._pub_cmd = self.create_publisher(JointTrajectory, '/planning/joint_cmd', 10)
        self._pub_status = self.create_publisher(ExecutorStatus, '/planning/executor_status', 10)

        self._arm = None
        if bool(args.drive_arm):
            self._arm = ActionClient(
                self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
            self.get_logger().info('trajectory_executor drive-arm ON')
        else:
            self.get_logger().info('trajectory_executor shadow mode (no FollowJointTrajectory)')

        self.get_logger().info(
            f'trajectory_executor end_effector={args.end_effector} '
            f'tip_off={list(self._tip_off)} max_dq={args.max_joint_jump_rad:.3f}')

    def _on_joints(self, msg: JointState) -> None:
        name_to_pos = {n: float(p) for n, p in zip(msg.name, msg.position)}
        if all(j in name_to_pos for j in ARM_JOINTS):
            self._joints = [float(name_to_pos[j]) for j in ARM_JOINTS]

    def _publish_status(
        self,
        *,
        status: str,
        detail: str,
        tool_traj_seq: int,
        ik_traj_seq: int = 0,
        tip_err_m: float = float('nan'),
        axis_err_rad: float = float('nan'),
        max_joint_err_rad: float = float('nan'),
    ) -> None:
        self._seq += 1
        msg = ExecutorStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.seq = int(self._seq)
        msg.tool_traj_seq = int(tool_traj_seq)
        msg.ik_traj_seq = int(ik_traj_seq)
        msg.status = str(status)
        msg.detail = str(detail)[:240]
        msg.tip_err_m = float(tip_err_m)
        msg.axis_err_rad = float(axis_err_rad)
        msg.max_joint_err_rad = float(max_joint_err_rad)
        self._pub_status.publish(msg)

    def _on_tool_traj(self, msg: ToolTrajectory4s) -> None:
        if int(msg.seq) == int(self._last_tool_seq):
            return
        now = time.time()
        if now < self._busy_until:
            self.get_logger().debug(
                f'skip tool_traj seq={msg.seq}: busy for {self._busy_until - now:.2f}s')
            return
        if self._joints is None:
            self._publish_status(
                status='idle', detail='no joint_states yet', tool_traj_seq=int(msg.seq))
            return
        if not msg.waypoints:
            self._publish_status(
                status='ik_fail', detail='empty waypoints', tool_traj_seq=int(msg.seq))
            return

        self._last_tool_seq = int(msg.seq)
        exec_i = int(msg.execute_until_index)
        exec_i = max(0, min(len(msg.waypoints) - 1, exec_i))
        wps = list(msg.waypoints[: exec_i + 1])

        tip_off = self._tip_off
        seed = list(self._joints)
        ik_wps: List[JointWaypoint] = []
        q_path: List[List[float]] = []
        n_ok = 0
        n_fail = 0
        fail_reason = ''

        # Optional w0 blend: if FK tip far from w0, seed from current tip as first sample.
        tip_now = tip_xyz(seed, tip_offset_link6=tip_off)
        w0 = wps[0]
        w0_tip = np.asarray(w0.position, dtype=float).reshape(3)
        blend_m = float(self._args.w0_blend_m)
        if float(np.linalg.norm(w0_tip - tip_now)) > blend_m:
            axis0 = np.asarray(w0.approach_axis, dtype=float).reshape(3)
            if float(np.linalg.norm(axis0)) < 1e-9:
                axis0 = tip_approach_axis(seed)
            # Insert current tip as synthetic start (time 0); shift times slightly.
            jw = JointWaypoint()
            jw.time_from_start = _s_to_duration(0.0)
            jw.q_rad = [float(v) for v in seed]
            jw.tip_xyz = [float(v) for v in tip_now]
            jw.tip_approach = [float(v) for v in tip_approach_axis(seed)]
            jw.ik_ok = True
            jw.ik_reason = 'w0_blend_current'
            ik_wps.append(jw)
            q_path.append(list(seed))
            n_ok += 1

        for wi, wp in enumerate(wps):
            tip = [float(v) for v in wp.position]
            axis = [float(v) for v in wp.approach_axis]
            if float(np.linalg.norm(axis)) < 1e-9:
                q_sol = None
                reason = 'zero_approach_axis'
            else:
                q_sol = tool_pose_ik(
                    tip, axis, seed,
                    tip_offset_link6=tip_off,
                    tol_m=float(self._args.ik_tol_m),
                    tol_dir_rad=float(self._args.ik_tol_dir_rad),
                )
                reason = 'ok' if q_sol is not None else 'tool_pose_ik_fail'

            jw = JointWaypoint()
            jw.time_from_start = wp.time_from_start
            if q_sol is None:
                jw.q_rad = [float(v) for v in seed]
                jw.tip_xyz = [float('nan')] * 3
                jw.tip_approach = [float('nan')] * 3
                jw.ik_ok = False
                jw.ik_reason = reason
                ik_wps.append(jw)
                n_fail += 1
                fail_reason = reason
                break

            # Jump gate vs previous solved q.
            if q_path:
                dq = max(abs(q_sol[i] - q_path[-1][i]) for i in range(6))
                if dq > float(self._args.max_joint_jump_rad):
                    jw.q_rad = [float(v) for v in q_sol]
                    tip_fk = tip_xyz(q_sol, tip_offset_link6=tip_off)
                    jw.tip_xyz = [float(v) for v in tip_fk]
                    jw.tip_approach = [float(v) for v in tip_approach_axis(q_sol)]
                    jw.ik_ok = False
                    jw.ik_reason = f'jump_reject dq={dq:.3f}'
                    ik_wps.append(jw)
                    n_fail += 1
                    fail_reason = jw.ik_reason
                    self._emit_ik(msg, ik_wps, ok=False, n_ok=n_ok, n_fail=n_fail,
                                  detail=fail_reason)
                    self._publish_status(
                        status='jump_reject',
                        detail=fail_reason,
                        tool_traj_seq=int(msg.seq),
                        max_joint_err_rad=float(dq),
                    )
                    return

            tip_fk = tip_xyz(q_sol, tip_offset_link6=tip_off)
            jw.q_rad = [float(v) for v in q_sol]
            jw.tip_xyz = [float(v) for v in tip_fk]
            jw.tip_approach = [float(v) for v in tip_approach_axis(q_sol)]
            jw.ik_ok = True
            jw.ik_reason = reason
            ik_wps.append(jw)
            q_path.append(list(q_sol))
            seed = list(q_sol)
            n_ok += 1

        if n_fail > 0 or not q_path:
            self._emit_ik(msg, ik_wps, ok=False, n_ok=n_ok, n_fail=n_fail,
                          detail=fail_reason or 'ik_fail')
            self._publish_status(
                status='ik_fail',
                detail=fail_reason or 'ik_fail',
                tool_traj_seq=int(msg.seq),
            )
            return

        tip_err = float(np.linalg.norm(
            tip_xyz(q_path[-1], tip_offset_link6=tip_off)
            - np.asarray(wps[-1].position, dtype=float).reshape(3)))
        axis_des = np.asarray(wps[-1].approach_axis, dtype=float).reshape(3)
        an = float(np.linalg.norm(axis_des))
        if an > 1e-9:
            axis_des = axis_des / an
            axis_err = float(np.arccos(np.clip(
                np.dot(tip_approach_axis(q_path[-1]), axis_des), -1.0, 1.0)))
        else:
            axis_err = float('nan')

        ik_seq = self._emit_ik(msg, ik_wps, ok=True, n_ok=n_ok, n_fail=0, detail='ok')
        cmd = self._build_joint_cmd(ik_wps)
        self._pub_cmd.publish(cmd)

        horizon = _duration_to_s(ik_wps[-1].time_from_start)
        if horizon < 1e-3:
            horizon = float(msg.dt_s) * max(1, exec_i)
        self._busy_until = time.time() + horizon + float(self._args.settle_s)

        if self._arm is not None:
            self._send_arm(cmd)

        self._publish_status(
            status='ok',
            detail=f'n={n_ok} horizon={horizon:.2f}s',
            tool_traj_seq=int(msg.seq),
            ik_traj_seq=int(ik_seq),
            tip_err_m=tip_err,
            axis_err_rad=axis_err,
        )
        self.get_logger().info(
            f'exec tool_seq={msg.seq} n_wp={n_ok} tip_err={tip_err*1e3:.1f}mm '
            f'axis_err={axis_err:.3f}rad drive={self._arm is not None}')

    def _emit_ik(
        self,
        tool: ToolTrajectory4s,
        ik_wps: List[JointWaypoint],
        *,
        ok: bool,
        n_ok: int,
        n_fail: int,
        detail: str,
    ) -> int:
        self._seq += 1
        out = IkJointTrajectory()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = tool.header.frame_id or 'base_link'
        out.seq = int(self._seq)
        out.tool_traj_seq = int(tool.seq)
        out.horizon_s = float(tool.horizon_s)
        out.dt_s = float(tool.dt_s)
        out.execute_until_index = int(tool.execute_until_index)
        out.joint_names = list(ARM_JOINTS)
        out.waypoints = ik_wps
        out.ok = bool(ok)
        out.n_ok = int(n_ok)
        out.n_fail = int(n_fail)
        out.detail = str(detail)[:240]
        self._pub_ik.publish(out)
        return int(out.seq)

    def _build_joint_cmd(self, ik_wps: List[JointWaypoint]) -> JointTrajectory:
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.header.frame_id = 'base_link'
        traj.joint_names = list(ARM_JOINTS)
        points: List[JointTrajectoryPoint] = []
        for wp in ik_wps:
            if not wp.ik_ok:
                break
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in wp.q_rad]
            pt.time_from_start = wp.time_from_start
            # Ensure strictly increasing times for controller.
            if points:
                t_prev = _duration_to_s(points[-1].time_from_start)
                t_cur = _duration_to_s(pt.time_from_start)
                if t_cur <= t_prev:
                    pt.time_from_start = _s_to_duration(t_prev + 0.05)
            points.append(pt)
        if not points and self._joints is not None:
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in self._joints]
            pt.time_from_start = _s_to_duration(0.1)
            points = [pt]
        traj.points = points
        return traj

    def _send_arm(self, traj: JointTrajectory) -> None:
        assert self._arm is not None
        if not self._arm.server_is_ready():
            self.get_logger().warn('arm FollowJointTrajectory not ready')
            return
        if self._arm_goal_handle is not None:
            try:
                self._arm_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self._arm_goal_handle = None
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        fut = self._arm.send_goal_async(goal)

        def _on_sent(f) -> None:
            try:
                gh = f.result()
            except Exception:
                return
            if gh is None or not gh.accepted:
                self.get_logger().warn('arm traj rejected')
                return
            prev = self._arm_goal_handle
            self._arm_goal_handle = gh
            if prev is not None and prev != gh:
                try:
                    prev.cancel_goal_async()
                except Exception:
                    pass

        fut.add_done_callback(_on_sent)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end-effector', default='suction_cup_v1')
    parser.add_argument('--drive-arm', action='store_true', default=False,
                        help='Send FollowJointTrajectory (else shadow publish only)')
    parser.add_argument('--max-joint-jump-rad', type=float, default=0.55,
                        help='Reject consecutive IK Δq above this (rad)')
    parser.add_argument('--ik-tol-m', type=float, default=2.5e-3)
    parser.add_argument('--ik-tol-dir-rad', type=float, default=0.12)
    parser.add_argument('--w0-blend-m', type=float, default=0.03,
                        help='If FK tip vs w0 exceeds this, prepend current q')
    parser.add_argument('--settle-s', type=float, default=0.3)
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    node = TrajectoryExecutorNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
