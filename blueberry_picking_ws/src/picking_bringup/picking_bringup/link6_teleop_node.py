#!/usr/bin/env python3
"""Keyboard teleop: link6 velocity jog → Piper 5-DOF IK → arm_controller.

Piper X is effectively 5-DOF for absolute poses (see piper_position_ik).
MoveIt 6-DOF GetPositionIK with fixed orientation fails for most base-frame
translations (A/D/Q/E). Linear jog uses position_ik_keep_orient; angular keys
nudge j4/j5/j1 in joint space.
"""

from __future__ import annotations

import copy
import math
import os
import select
import sys
import termios
import tty
from typing import Dict, List, Optional, Tuple

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener

# Workspace analytic IK (5-DOF).
_SCRIPT_CANDIDATES = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'scripts')),
    '/home/user/codes/piper_x_dev/blueberry_picking_ws/scripts',
]
for _p in _SCRIPT_CANDIDATES:
    if os.path.isfile(os.path.join(_p, 'piper_position_ik.py')) and _p not in sys.path:
        sys.path.insert(0, _p)
        break
from piper_position_ik import clamp_joints, position_ik, position_ik_keep_orient  # type: ignore

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

# Unit direction per key: (lin_x, lin_y, lin_z, roll, pitch, yaw)
KEY_DIRS: Dict[str, Tuple[float, float, float, float, float, float]] = {
    'q': (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    'e': (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    'a': (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    'd': (0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    'w': (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    's': (0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
    'o': (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    'k': (0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    'i': (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    'j': (0.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    'u': (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    'h': (0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
}

KEY_HELP = """
link6 teleop — hold to move, release to stop (Piper 5-DOF IK):
  W/S  up/down (+Z)   A/D  left/right (+Y)   Q/E  forward/back (+X)  [base_link]
  O/K  j4 roll   I/J  j5 pitch   U/H  j1 yaw   (joint nudge)
  [ ]  speed  R   home (MoveIt plan)  Space vibrate toggle
  ?    help   Esc  quit
"""


def quat_multiply(q1: Tuple[float, float, float, float],
                  q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_rotate(q: Tuple[float, float, float, float],
                vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    qx, qy, qz, qw = q
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def quat_from_axis_angle(axis: Tuple[float, float, float], angle: float) -> Tuple[float, float, float, float]:
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    ax, ay, az = ax / n, ay / n, az / n
    s = math.sin(angle * 0.5)
    return ax * s, ay * s, az * s, math.cos(angle * 0.5)


def _normalize3(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-9:
        return 0.0, 0.0, 0.0
    return v[0] / n, v[1] / n, v[2] / n


class Link6TeleopNode(Node):
    def __init__(self) -> None:
        super().__init__('link6_teleop')
        self.declare_parameter('planning_group', 'arm')
        self.declare_parameter('tip_link', 'link6')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('linear_speed_m_s', 0.06)
        self.declare_parameter('angular_speed_deg_s', 20.0)
        self.declare_parameter('control_rate_hz', 40.0)
        self.declare_parameter('key_release_timeout_sec', 0.12)
        self.declare_parameter('trajectory_time_sec', 0.05)
        self.declare_parameter('avoid_collisions', False)
        self.declare_parameter('ik_service', '/compute_ik')
        self.declare_parameter('vibration_topic', '/vibration_motor_controller/commands')
        self.declare_parameter('move_group_action', '/move_action')
        self.declare_parameter('home_joint_positions', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('home_velocity_scale', 0.15)
        self.declare_parameter('linear_jog_frame', 'base_link')

        self._linear_speed = float(self.get_parameter('linear_speed_m_s').value)
        self._angular_speed = math.radians(float(self.get_parameter('angular_speed_deg_s').value))
        self._traj_time = float(self.get_parameter('trajectory_time_sec').value)
        self._key_release_ns = int(
            float(self.get_parameter('key_release_timeout_sec').value) * 1e9)
        self._avoid_collisions = bool(self.get_parameter('avoid_collisions').value)
        self._group = self.get_parameter('planning_group').value
        self._tip = self.get_parameter('tip_link').value
        self._base = self.get_parameter('base_frame').value
        self._linear_jog_frame = self.get_parameter('linear_jog_frame').value
        self._home_joints = [
            float(v) for v in self.get_parameter('home_joint_positions').value]
        self._home_velocity_scale = float(
            self.get_parameter('home_velocity_scale').value)

        self._joint_positions: Dict[str, float] = {}
        self._target: Optional[PoseStamped] = None
        self._key_last_seen: Dict[str, Time] = {}
        self._arm_goal_handle = None
        self._home_goal_handle = None
        self._home_in_progress = False
        self._was_jogging = False
        self._vibrate_on = False
        self._running = True
        self._last_log_time = self.get_clock().now()
        self._stdin_fd = sys.stdin.fileno()
        self._stdin_old = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)

        self._pub_cmd = self.create_publisher(PoseStamped, '/teleop/link6_command', 10)
        self._pub_state = self.create_publisher(PoseStamped, '/teleop/link6_state', 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._arm_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self._move_group_client = ActionClient(
            self, MoveGroup, self.get_parameter('move_group_action').value)

        from std_msgs.msg import Float64MultiArray
        self._Float64MultiArray = Float64MultiArray
        self._vib_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter('vibration_topic').value, 10)

        rate = max(10.0, float(self.get_parameter('control_rate_hz').value))
        period = 1.0 / rate
        self._control_period = period
        self.create_timer(period, self._control_tick)
        self.create_timer(0.05, self._vibration_tick)

        self.get_logger().info('link6_teleop waiting for arm_controller (5-DOF local IK)...')
        self._arm_client.wait_for_server(timeout_sec=60.0)
        self.get_logger().info(
            f'hold-to-jog: linear={self._linear_speed:.3f} m/s '
            f'angular={math.degrees(self._angular_speed):.1f} deg/s '
            f'rate={rate:.0f} Hz linear_frame={self._linear_jog_frame} ik=piper_5dof')
        self.get_logger().info(KEY_HELP.strip())

    def destroy_node(self) -> bool:
        termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old)
        return super().destroy_node()

    def _on_joints(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name in ARM_JOINTS:
                self._joint_positions[name] = pos

    def _current_link6_pose(self) -> Optional[PoseStamped]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base, self._tip, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception:
            return None
        ps = PoseStamped()
        ps.header.frame_id = self._base
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = tf.transform.translation.x
        ps.pose.position.y = tf.transform.translation.y
        ps.pose.position.z = tf.transform.translation.z
        ps.pose.orientation = tf.transform.rotation
        return ps

    def _ensure_target(self) -> Optional[PoseStamped]:
        if self._target is None:
            self._target = self._current_link6_pose()
        return self._target

    def _vibration_tick(self) -> None:
        msg = self._Float64MultiArray()
        msg.data = [0.0015 if self._vibrate_on else 0.0]
        self._vib_pub.publish(msg)

    def _poll_keyboard(self) -> None:
        while self._running and rclpy.ok():
            if not select.select([sys.stdin], [], [], 0)[0]:
                break
            ch = sys.stdin.read(1)
            if ch in ('\x1b', '\x03'):
                self._running = False
                rclpy.shutdown()
                return
            self._handle_key(ch.lower())

    def _active_motion_keys(self) -> List[str]:
        now = self.get_clock().now()
        active: List[str] = []
        expired: List[str] = []
        for key, seen in self._key_last_seen.items():
            if (now - seen).nanoseconds > self._key_release_ns:
                expired.append(key)
            else:
                active.append(key)
        for key in expired:
            del self._key_last_seen[key]
        return active

    def _motion_twist(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        lin = [0.0, 0.0, 0.0]
        ang = [0.0, 0.0, 0.0]
        for key in self._active_motion_keys():
            d = KEY_DIRS.get(key)
            if d is None:
                continue
            lin[0] += d[0]
            lin[1] += d[1]
            lin[2] += d[2]
            ang[0] += d[3]
            ang[1] += d[4]
            ang[2] += d[5]

        lin_u = _normalize3((lin[0], lin[1], lin[2]))
        ang_u = _normalize3((ang[0], ang[1], ang[2]))
        dpos = (
            lin_u[0] * self._linear_speed * self._control_period,
            lin_u[1] * self._linear_speed * self._control_period,
            lin_u[2] * self._linear_speed * self._control_period,
        )
        rpy = (
            ang_u[0] * self._angular_speed * self._control_period,
            ang_u[1] * self._angular_speed * self._control_period,
            ang_u[2] * self._angular_speed * self._control_period,
        )
        if not (any(abs(v) > 1e-9 for v in dpos) or any(abs(v) > 1e-9 for v in rpy)):
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        return dpos, rpy

    def _integrate_target(self, dpos: Tuple[float, float, float],
                          rpy_delta: Tuple[float, float, float]) -> Optional[PoseStamped]:
        target = self._ensure_target()
        if target is None:
            return None

        target = copy.deepcopy(target)
        q = (
            target.pose.orientation.x,
            target.pose.orientation.y,
            target.pose.orientation.z,
            target.pose.orientation.w,
        )
        dx, dy, dz = dpos
        if self._linear_jog_frame == 'link6':
            dx, dy, dz = quat_rotate(q, dpos[0], dpos[1], dpos[2])
        target.pose.position.x += dx
        target.pose.position.y += dy
        target.pose.position.z += dz

        for axis, angle in zip([(1, 0, 0), (0, 1, 0), (0, 0, 1)], rpy_delta):
            if abs(angle) > 1e-9:
                dq = quat_from_axis_angle(axis, angle)
                q = quat_multiply(q, dq)
        target.pose.orientation.x, target.pose.orientation.y, target.pose.orientation.z, target.pose.orientation.w = q
        target.header.stamp = self.get_clock().now().to_msg()
        return target

    def _request_ik(self, target: PoseStamped) -> None:
        """Solve Cartesian XYZ with Piper 5-DOF keep-orient IK (j6=0)."""
        seed = [float(self._joint_positions.get(n, 0.0)) for n in ARM_JOINTS]
        xyz = (
            float(target.pose.position.x),
            float(target.pose.position.y),
            float(target.pose.position.z),
        )
        joints = position_ik_keep_orient(xyz, seed)
        if joints is None:
            joints = position_ik(xyz, seed)
        if joints is None:
            self.get_logger().warn(
                f'5-DOF IK failed for xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})',
                throttle_duration_sec=1.0)
            live = self._current_link6_pose()
            if live is not None:
                self._target = live
            return

        self._send_trajectory(joints)
        self._target = copy.deepcopy(target)
        now = self.get_clock().now()
        if (now - self._last_log_time).nanoseconds > int(1e9):
            self._last_log_time = now
            p = target.pose.position
            self.get_logger().info(f'link6 -> ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')

    def _nudge_joints(self, rpy_delta: Tuple[float, float, float]) -> None:
        """Angular keys → joint-space nudge (roll=j4, pitch=j5, yaw=j1)."""
        seed = [float(self._joint_positions.get(n, 0.0)) for n in ARM_JOINTS]
        roll, pitch, yaw = rpy_delta
        seed[3] = float(seed[3] + roll)
        seed[4] = float(seed[4] + pitch)
        seed[0] = float(seed[0] + yaw)
        seed[5] = 0.0
        self._send_trajectory(clamp_joints(seed))
        live = self._current_link6_pose()
        if live is not None:
            self._target = live

    def _send_trajectory(self, positions: List[float]) -> None:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(self._traj_time * 1e9)
        goal.trajectory.points = [pt]
        send_future = self._arm_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception:
            return
        if not goal_handle.accepted:
            return
        if self._arm_goal_handle is not None:
            try:
                self._arm_goal_handle.cancel_goal_async()
            except Exception:
                pass
        self._arm_goal_handle = goal_handle

    def _stop_arm(self) -> None:
        if self._arm_goal_handle is not None:
            try:
                self._arm_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self._arm_goal_handle = None
        hold = [self._joint_positions.get(n, 0.0) for n in ARM_JOINTS]
        if any(n in self._joint_positions for n in ARM_JOINTS):
            self._send_trajectory(hold)
        self._target = self._current_link6_pose()

    def _control_tick(self) -> None:
        if not self._running or not rclpy.ok():
            return

        self._poll_keyboard()
        if self._home_in_progress:
            return

        ps = self._current_link6_pose()
        if ps is not None:
            self._pub_state.publish(ps)

        dpos, rpy = self._motion_twist()
        moving = any(abs(v) > 1e-12 for v in dpos) or any(abs(v) > 1e-12 for v in rpy)

        if not moving:
            if self._was_jogging:
                self._stop_arm()
                self.get_logger().info('jog stop')
            self._was_jogging = False
            return

        self._was_jogging = True
        # Angular-only → joint nudge (5-DOF cannot track absolute orientation).
        lin_mag = sum(abs(v) for v in dpos)
        ang_mag = sum(abs(v) for v in rpy)
        if lin_mag < 1e-12 and ang_mag > 1e-12:
            self._nudge_joints(rpy)
            return

        target = self._integrate_target(dpos, rpy)
        if target is None:
            self.get_logger().warn('link6 TF not ready', throttle_duration_sec=2.0)
            return

        self._pub_cmd.publish(target)
        self._request_ik(target)

    def _build_home_move_goal(self) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self._group
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = self._home_velocity_scale
        req.max_acceleration_scaling_factor = self._home_velocity_scale

        constraints = Constraints()
        for joint_name, position in zip(ARM_JOINTS, self._home_joints):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = position
            jc.tolerance_above = 0.02
            jc.tolerance_below = 0.02
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints = [constraints]
        goal.request = req

        opts = PlanningOptions()
        opts.plan_only = False
        opts.replan = True
        opts.replan_attempts = 3
        goal.planning_options = opts
        return goal

    def _finish_home(self, success: bool, message: str) -> None:
        self._home_in_progress = False
        self._home_goal_handle = None
        if success:
            self._target = self._current_link6_pose()
            if self._target is not None:
                p = self._target.pose.position
                self.get_logger().info(
                    f'{message} link6=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
            else:
                self.get_logger().info(message)
        else:
            self.get_logger().warn(message)

    def _on_home_result(self, future) -> None:
        try:
            result = future.result().result
        except Exception as exc:
            self._finish_home(False, f'home move failed: {exc}')
            return
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self._finish_home(
                False, f'home move failed code={result.error_code.val}')
            return
        self._finish_home(True, 'home move completed')

    def _on_home_goal_sent(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._finish_home(False, f'home goal send failed: {exc}')
            return
        if not goal_handle.accepted:
            self._finish_home(False, 'home move goal rejected by move_group')
            return
        self._home_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_home_result)

    def _go_home(self) -> None:
        self._key_last_seen.clear()
        self._stop_arm()
        self._target = None

        if self._home_in_progress:
            self.get_logger().info('home move already in progress')
            return
        if not self._move_group_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn(
                'move_group action not available (is move_group running?)')
            return

        self._home_in_progress = True
        self.get_logger().info(
            f'planning home joints={[round(v, 3) for v in self._home_joints]} '
            f'vel_scale={self._home_velocity_scale:.2f}')
        send_future = self._move_group_client.send_goal_async(
            self._build_home_move_goal())
        send_future.add_done_callback(self._on_home_goal_sent)

    def _handle_key(self, key: str) -> None:
        if key in KEY_DIRS:
            self._key_last_seen[key] = self.get_clock().now()
        elif key == ' ':
            self._vibrate_on = not self._vibrate_on
            self.get_logger().info(f'vibration {"ON" if self._vibrate_on else "OFF"}')
        elif key == 'r':
            self._go_home()
        elif key == '?':
            self.get_logger().info(KEY_HELP.strip())
        elif key == ']':
            self._linear_speed = min(0.15, self._linear_speed * 1.25)
            self._angular_speed = min(math.radians(60.0), self._angular_speed * 1.25)
            self.get_logger().info(
                f'speed linear={self._linear_speed:.3f} m/s '
                f'angular={math.degrees(self._angular_speed):.1f} deg/s')
        elif key == '[':
            self._linear_speed = max(0.01, self._linear_speed / 1.25)
            self._angular_speed = max(math.radians(5.0), self._angular_speed / 1.25)
            self.get_logger().info(
                f'speed linear={self._linear_speed:.3f} m/s '
                f'angular={math.degrees(self._angular_speed):.1f} deg/s')


def main() -> None:
    rclpy.init()
    node = Link6TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
