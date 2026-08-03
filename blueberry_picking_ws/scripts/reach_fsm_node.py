#!/usr/bin/env python3
"""Topic-driven reach FSM: lock (fixed cam) → fine 3D (wrist) → plan → touch → confirm reset.

Cmds on /reach/cmd (std_msgs/String):
  start | confirm_reset | abort | clear_lock

Status on /reach/status. Reached edge on /reach/reached (Bool).
See docs/REACH_PIPELINE.md.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    PlanningOptions,
    PositionConstraint,
)
from picking_msgs.msg import DetectedBerry, DetectedBerryArray, SuctionGraspPlan
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

STATES = (
    'IDLE', 'LOCKING', 'REFINING', 'PLANNING', 'APPROACHING',
    'REACHED', 'WAIT_CONFIRM', 'RESETTING', 'ERROR',
)


def _quat_cup_toward_berry(ax: float, ay: float, az: float) -> Tuple[float, float, float, float]:
    """Orientation with tool +Z along approach (ax,ay,az)."""
    z = [ax, ay, az]
    n = math.sqrt(z[0] ** 2 + z[1] ** 2 + z[2] ** 2) or 1.0
    z = [z[0] / n, z[1] / n, z[2] / n]
    # pick a reference not parallel to z
    ref = [0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0]
    x = [
        ref[1] * z[2] - ref[2] * z[1],
        ref[2] * z[0] - ref[0] * z[2],
        ref[0] * z[1] - ref[1] * z[0],
    ]
    xn = math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) or 1.0
    x = [x[0] / xn, x[1] / xn, x[2] / xn]
    y = [
        z[1] * x[2] - z[2] * x[1],
        z[2] * x[0] - z[0] * x[2],
        z[0] * x[1] - z[1] * x[0],
    ]
    # rotation matrix columns = x,y,z → quaternion
    m00, m01, m02 = x[0], y[0], z[0]
    m10, m11, m12 = x[1], y[1], z[1]
    m20, m21, m22 = x[2], y[2], z[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


class ReachFsmNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('reach_fsm_node')
        self._args = args
        self._state = 'IDLE'
        self._cmd_queue: List[str] = []
        self._global: Optional[DetectedBerryArray] = None
        self._fine: Optional[DetectedBerryArray] = None
        self._locked: Optional[DetectedBerry] = None
        self._plan: Optional[SuctionGraspPlan] = None
        self._joints = [0.0] * 6
        self._refine_deadline = 0.0
        self._error_msg = ''
        self._busy = False

        self._status_pub = self.create_publisher(String, '/reach/status', 10)
        self._reached_pub = self.create_publisher(Bool, '/reach/reached', 10)
        self._lock_pub = self.create_publisher(DetectedBerry, '/perception/target_lock', 10)
        self._plan_pub = self.create_publisher(SuctionGraspPlan, '/reach/plan', 10)

        self.create_subscription(String, '/reach/cmd', self._on_cmd, 10)
        self.create_subscription(DetectedBerryArray, '/perception/global/berries', self._on_global, 10)
        self.create_subscription(DetectedBerryArray, '/perception/fine/berries', self._on_fine, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._move = ActionClient(self, MoveGroup, args.move_action)

        self.create_timer(0.1, self._tick)
        self._publish_status()
        self.get_logger().info('reach_fsm ready — publish start to /reach/cmd')

    def _on_cmd(self, msg: String) -> None:
        cmd = msg.data.strip().lower()
        if cmd:
            self._cmd_queue.append(cmd)
            self.get_logger().info(f'cmd: {cmd}')

    def _on_global(self, msg: DetectedBerryArray) -> None:
        self._global = msg

    def _on_fine(self, msg: DetectedBerryArray) -> None:
        self._fine = msg

    def _on_joints(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(j in name_to_pos for j in ARM_JOINTS):
            self._joints = [float(name_to_pos[j]) for j in ARM_JOINTS]

    def _set_state(self, state: str, err: str = '') -> None:
        if state not in STATES:
            state = 'ERROR'
        self._state = state
        self._error_msg = err
        self._publish_status()
        self.get_logger().info(f'state → {state}' + (f' ({err})' if err else ''))

    def _publish_status(self) -> None:
        msg = String()
        msg.data = self._state if not self._error_msg else f'{self._state}:{self._error_msg}'
        self._status_pub.publish(msg)

    def _pop_cmd(self) -> Optional[str]:
        return self._cmd_queue.pop(0) if self._cmd_queue else None

    def _tick(self) -> None:
        if self._busy:
            return
        cmd = self._pop_cmd()
        if cmd == 'abort':
            self._locked = None
            self._plan = None
            self._set_state('IDLE')
            return
        if cmd == 'clear_lock':
            self._locked = None
            if self._state in ('LOCKING', 'REFINING', 'PLANNING'):
                self._set_state('IDLE')
            return

        if self._state == 'IDLE':
            if cmd == 'start':
                self._locked = None
                self._plan = None
                self._reached_pub.publish(Bool(data=False))
                self._set_state('LOCKING')
            return

        if self._state == 'LOCKING':
            if self._try_lock_global():
                self._refine_deadline = time.time() + self._args.refine_timeout_s
                self._set_state('REFINING')
            return

        if self._state == 'REFINING':
            berry = self._pick_fine_berry()
            if berry is not None:
                self._locked = berry  # update with fine 3D
                self._lock_pub.publish(berry)
                self._set_state('PLANNING')
            elif time.time() > self._refine_deadline:
                self._set_state('ERROR', 'refine timeout — no fine berry')
            return

        if self._state == 'PLANNING':
            if self._locked is None:
                self._set_state('ERROR', 'no locked berry')
                return
            plan = self._build_plan(self._locked)
            if plan is None:
                self._set_state('ERROR', 'plan failed')
                return
            self._plan = plan
            self._plan_pub.publish(plan)
            self._set_state('APPROACHING')
            return

        if self._state == 'APPROACHING':
            self._busy = True
            try:
                ok = self._execute_approach()
            finally:
                self._busy = False
            if ok:
                self._reached_pub.publish(Bool(data=True))
                self._set_state('REACHED')
                self._set_state('WAIT_CONFIRM')
            else:
                self._set_state('ERROR', 'approach motion failed')
            return

        if self._state == 'WAIT_CONFIRM':
            if cmd == 'confirm_reset':
                self._set_state('RESETTING')
            return

        if self._state == 'RESETTING':
            self._busy = True
            try:
                ok = self._go_home()
            finally:
                self._busy = False
            if ok:
                self._locked = None
                self._plan = None
                self._set_state('IDLE')
            else:
                self._set_state('ERROR', 'reset home failed')
            return

        if self._state == 'ERROR':
            if cmd == 'start':
                self._error_msg = ''
                self._set_state('IDLE')
                self._cmd_queue.insert(0, 'start')
            elif cmd == 'confirm_reset':
                self._error_msg = ''
                self._set_state('RESETTING')

    def _try_lock_global(self) -> bool:
        if self._global is None or not self._global.berries:
            return False
        # Highest confidence, then closest to camera origin proxy (smaller |z| in base if available)
        berries = list(self._global.berries)
        berries.sort(key=lambda b: (-float(b.confidence), abs(float(b.pose.pose.position.z))))
        self._locked = berries[0]
        self._lock_pub.publish(self._locked)
        p = self._locked.pose.pose.position
        self.get_logger().info(
            f'locked global berry conf={self._locked.confidence:.2f} '
            f'pos=({p.x:.3f},{p.y:.3f},{p.z:.3f}) frame={self._locked.pose.header.frame_id}')
        return True

    def _pick_fine_berry(self) -> Optional[DetectedBerry]:
        if self._fine is None or not self._fine.berries:
            return None
        berries = list(self._fine.berries)
        if self._locked is not None:
            lx = self._locked.pose.pose.position.x
            ly = self._locked.pose.pose.position.y
            lz = self._locked.pose.pose.position.z

            def d2(b: DetectedBerry) -> float:
                dx = b.pose.pose.position.x - lx
                dy = b.pose.pose.position.y - ly
                dz = b.pose.pose.position.z - lz
                return dx * dx + dy * dy + dz * dz

            berries.sort(key=d2)
        return berries[0]

    def _current_ee_pose(self) -> Optional[PoseStamped]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._args.base_frame, self._args.ee_link, rclpy.time.Time())
        except Exception:
            return None
        ps = PoseStamped()
        ps.header.frame_id = self._args.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        t = tf.transform.translation
        q = tf.transform.rotation
        ps.pose.position.x = t.x
        ps.pose.position.y = t.y
        ps.pose.position.z = t.z
        ps.pose.orientation = q
        return ps

    def _build_plan(self, berry: DetectedBerry) -> Optional[SuctionGraspPlan]:
        bx = float(berry.pose.pose.position.x)
        by = float(berry.pose.pose.position.y)
        bz = float(berry.pose.pose.position.z)
        ee = self._current_ee_pose()
        if ee is not None:
            ax = bx - ee.pose.position.x
            ay = by - ee.pose.position.y
            az = bz - ee.pose.position.z
            orient = ee.pose.orientation
        else:
            ax, ay, az = bx, by, bz
            orient = None
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-6:
            return None
        ax, ay, az = ax / n, ay / n, az / n
        if orient is None:
            qx, qy, qz, qw = _quat_cup_toward_berry(ax, ay, az)
        else:
            qx, qy, qz, qw = orient.x, orient.y, orient.z, orient.w

        cup = self._args.cup_contact_offset
        pre_d = self._args.pre_grasp_offset
        post_d = self._args.post_grasp_offset
        stamp = self.get_clock().now().to_msg()
        plan = SuctionGraspPlan()
        plan.header.frame_id = self._args.base_frame
        plan.header.stamp = stamp

        def _fill(ps: PoseStamped, extra: float) -> None:
            reach = cup + extra
            ps.header.frame_id = self._args.base_frame
            ps.header.stamp = stamp
            ps.pose.position.x = bx - ax * reach
            ps.pose.position.y = by - ay * reach
            ps.pose.position.z = bz - az * reach
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw

        _fill(plan.pre_grasp, pre_d)
        _fill(plan.grasp, 0.0)
        _fill(plan.post_grasp, post_d)
        return plan

    def _build_position_goal(self, target: PoseStamped) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self._args.move_group
        req.num_planning_attempts = 10
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = self._args.velocity_scale
        req.max_acceleration_scaling_factor = self._args.velocity_scale
        pc = PositionConstraint()
        pc.header = target.header
        pc.link_name = self._args.ee_link
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [self._args.position_tolerance]
        pc.constraint_region.primitives = [region]
        pc.constraint_region.primitive_poses = [target.pose]
        pc.weight = 1.0
        req.goal_constraints = [Constraints(position_constraints=[pc])]
        goal.request = req
        opts = PlanningOptions()
        opts.plan_only = False
        opts.replan = True
        opts.replan_attempts = 3
        goal.planning_options = opts
        return goal

    def _build_joint_goal(self, joints: List[float]) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self._args.move_group
        req.num_planning_attempts = 10
        req.allowed_planning_time = 8.0
        req.max_velocity_scaling_factor = self._args.velocity_scale
        req.max_acceleration_scaling_factor = self._args.velocity_scale
        constraints = Constraints()
        for joint_name, position in zip(ARM_JOINTS, joints):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(position)
            jc.tolerance_above = 0.03
            jc.tolerance_below = 0.03
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

    def _execute_move_goal(self, goal: MoveGroup.Goal, label: str) -> bool:
        if not self._move.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('move_action unavailable')
            return False
        send_future = self._move.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=30.0)
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error(f'{label}: goal rejected')
            return False
        result_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
        result = result_future.result()
        if result is None:
            self.get_logger().error(f'{label}: no result')
            return False
        code = result.result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'{label}: MoveIt error {code}')
            return False
        self.get_logger().info(f'{label}: OK')
        return True

    def _execute_approach(self) -> bool:
        if self._plan is None:
            return False
        if self._args.dry_run:
            self.get_logger().info('dry-run: skip MoveIt, pretend reached')
            return True
        if not self._execute_move_goal(self._build_position_goal(self._plan.pre_grasp), 'pre'):
            return False
        return self._execute_move_goal(self._build_position_goal(self._plan.grasp), 'contact')

    def _go_home(self) -> bool:
        if self._args.dry_run:
            return True
        return self._execute_move_goal(self._build_joint_goal(list(self._args.home_joints)), 'home')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--move-action', default='/move_action')
    parser.add_argument('--move-group', default='arm')
    parser.add_argument('--ee-link', default=os.environ.get('PICK_EE_LINK', 'tcp_link'))
    parser.add_argument('--base-frame', default='base_link')
    parser.add_argument('--cup-contact-offset', type=float,
                        default=float(os.environ.get('PICK_CUP_CONTACT_OFFSET', '0.04')))
    parser.add_argument('--pre-grasp-offset', type=float, default=0.12)
    parser.add_argument('--post-grasp-offset', type=float, default=0.10)
    parser.add_argument('--velocity-scale', type=float, default=0.12)
    parser.add_argument('--position-tolerance', type=float, default=0.025)
    parser.add_argument('--refine-timeout-s', type=float, default=20.0)
    parser.add_argument('--home-joints', type=float, nargs=6, default=HOME)
    parser.add_argument('--dry-run', action='store_true',
                        help='Skip MoveIt; still exercise topic FSM')
    args = parser.parse_args()

    rclpy.init()
    node = ReachFsmNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
