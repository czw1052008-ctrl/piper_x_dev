#!/usr/bin/env python3
"""Topic-driven reach FSM: lock → align → fine 3D → plan → touch → confirm reset.

Cmds on /reach/cmd (std_msgs/String):
  start | confirm_reset | abort | clear_lock

Status on /reach/status. Reached edge on /reach/reached (Bool).
See docs/REACH_PIPELINE.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetPositionIK
from picking_msgs.msg import DetectedBerry, DetectedBerryArray, SuctionGraspPlan
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from cup_axis_plan import (  # noqa: E402
    angle_between,
    blend_pose,
    build_cup_axis_plan,
    facing_z_from_quat,
    plant_base_yaw,
    yaw_face_standoff_pose,
)
from align_judge import (  # noqa: E402
    ALIGN_ACTIONS,
    apply_action_to_joints,
    apply_joint_command,
    decide_action,
    is_target_visible,
    norm_angle,
    verify_step,
)

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

STATES = (
    'IDLE', 'LOCKING', 'ALIGNING', 'REFINING', 'PLANNING', 'APPROACHING',
    'REACHED', 'WAIT_CONFIRM', 'RESETTING', 'ERROR',
)


def _matrix_from_tf(transform) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _project_cam_point(
    x: float, y: float, z: float, width: int, height: int, focal: float,
) -> Optional[Tuple[float, float]]:
    if z <= 1e-6:
        return None
    cx = width * 0.5
    cy = height * 0.5
    u = cx + focal * x / z
    v = cy + focal * y / z
    return (float(u), float(v))


class ReachFsmNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('reach_fsm_node')
        self._args = args
        self._state = 'IDLE'
        self._cmd_queue: List[str] = []
        self._global: Optional[DetectedBerryArray] = None
        self._fine: Optional[DetectedBerryArray] = None
        self._fine_recv_t = 0.0
        self._locked: Optional[DetectedBerry] = None
        self._plan: Optional[SuctionGraspPlan] = None
        self._joints = [0.0] * 6
        self._refine_deadline = 0.0
        self._error_msg = ''
        self._cb_group = ReentrantCallbackGroup()
        # Async MoveIt (approach/home) or IK align
        self._move_phase: Optional[str] = None
        self._goal_fut = None
        self._result_fut = None
        self._approach_step = 0  # 0=pre, 1=contact
        self._ik_fut = None
        self._traj_goal_fut = None
        self._traj_result_fut = None
        self._align_deadline = 0.0
        self._move_lock = threading.Lock()
        self._qa_rgb_fixed = None
        self._qa_rgb_wrist = None
        self._qa_rgb_global_viz = None
        self._qa_rgb_fine_viz = None
        self._qa_session = ''
        self._align_cycle_idx = 0
        self._align_step_idx = 0
        self._align_failed_actions: List[str] = []
        self._align_before_obs: Optional[Dict[str, float]] = None
        self._align_last_action: Optional[str] = None
        self._align_last_decision: Optional[Dict[str, object]] = None
        self._align_needs_verify = False
        self._align_rollback_action: Optional[str] = None
        self._align_wait_log_t = 0.0
        self._align_request_written = False
        self._align_wait_phase: str = 'command'  # command | judge
        self._align_restore_joints: Optional[List[float]] = None
        self._align_settle_until = 0.0
        self._align_settle_kind: Optional[str] = None
        self._align_arm_prepared = False
        self._align_arm_enable_ok = False
        self._align_enable_fut = None
        self._align_enable_deadline = 0.0
        self._align_no_motion_streak = 0
        self._align_commanded_joints: Optional[List[float]] = None

        self._status_pub = self.create_publisher(String, '/reach/status', 10)
        self._reached_pub = self.create_publisher(Bool, '/reach/reached', 10)
        self._lock_pub = self.create_publisher(DetectedBerry, '/perception/target_lock', 10)
        self._plan_pub = self.create_publisher(SuctionGraspPlan, '/reach/plan', 10)

        self.create_subscription(String, '/reach/cmd', self._on_cmd, 10, callback_group=self._cb_group)
        self.create_subscription(
            DetectedBerryArray, '/perception/global/berries', self._on_global, 10,
            callback_group=self._cb_group)
        self.create_subscription(
            DetectedBerryArray, '/perception/fine/berries', self._on_fine, 10,
            callback_group=self._cb_group)
        self.create_subscription(
            JointState, '/feedback/joint_states', self._on_joints, 10,
            callback_group=self._cb_group)
        self.create_subscription(
            JointState, '/joint_states', self._on_joints, 10, callback_group=self._cb_group)
        self.create_subscription(
            Image, '/camera_fixed/image_raw', self._on_fixed_img, qos_profile_sensor_data,
            callback_group=self._cb_group)
        self.create_subscription(
            Image, '/camera_wrist/color/image_raw', self._on_wrist_img, qos_profile_sensor_data,
            callback_group=self._cb_group)
        self.create_subscription(
            Image, '/perception/global/detection_viz', self._on_global_viz, qos_profile_sensor_data,
            callback_group=self._cb_group)
        self.create_subscription(
            Image, '/perception/fine/detection_viz', self._on_fine_viz, qos_profile_sensor_data,
            callback_group=self._cb_group)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._move = ActionClient(self, MoveGroup, args.move_action, callback_group=self._cb_group)
        self._ik = self.create_client(GetPositionIK, '/compute_ik', callback_group=self._cb_group)
        self._arm = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory',
            callback_group=self._cb_group)
        self._move_j_pub = self.create_publisher(JointState, '/control/move_j', 10)
        self._arm_enable = self.create_client(SetBool, '/enable_agx_arm', callback_group=self._cb_group)
        self._control_enable = self.create_client(SetBool, '/control_enable', callback_group=self._cb_group)
        self._arm_goal_handle = None
        os.makedirs(self._args.qa_dir, exist_ok=True)

        self.create_timer(0.1, self._tick, callback_group=self._cb_group)
        self._publish_status()
        self.get_logger().info(
            f'reach_fsm ready — ALIGNING via observe/judge/step loop '
            f'(judge_mode={self._args.align_judge_mode}); '
            f'qa_dir={self._args.qa_dir}')

    def _on_cmd(self, msg: String) -> None:
        cmd = msg.data.strip().lower()
        if cmd:
            self._cmd_queue.append(cmd)
            self.get_logger().info(f'cmd: {cmd}')

    def _on_global(self, msg: DetectedBerryArray) -> None:
        self._global = msg

    def _on_fine(self, msg: DetectedBerryArray) -> None:
        self._fine = msg
        self._fine_recv_t = time.time()

    def _on_joints(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(j in name_to_pos for j in ARM_JOINTS):
            self._joints = [float(name_to_pos[j]) for j in ARM_JOINTS]

    def _image_to_rgb(self, msg: Image) -> Optional[np.ndarray]:
        enc = msg.encoding.lower()
        try:
            if enc in ('rgb8',):
                return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
            if enc in ('bgr8',):
                bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                return bgr[:, :, ::-1].copy()
            if enc in ('yuv422_yuy2', 'yuyv', 'yuyv422'):
                import cv2  # type: ignore
                yuyv = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 2)
                return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
        except Exception:
            return None
        return None

    def _on_fixed_img(self, msg: Image) -> None:
        rgb = self._image_to_rgb(msg)
        if rgb is not None:
            self._qa_rgb_fixed = rgb

    def _on_wrist_img(self, msg: Image) -> None:
        rgb = self._image_to_rgb(msg)
        if rgb is not None:
            self._qa_rgb_wrist = rgb

    def _on_global_viz(self, msg: Image) -> None:
        rgb = self._image_to_rgb(msg)
        if rgb is not None:
            self._qa_rgb_global_viz = rgb

    def _on_fine_viz(self, msg: Image) -> None:
        rgb = self._image_to_rgb(msg)
        if rgb is not None:
            self._qa_rgb_fine_viz = rgb

    def _snap_qa(self, tag: str) -> None:
        """Save raw camera frames for visual A/B (not algorithm overlays as truth)."""
        import cv2  # type: ignore
        if not self._qa_session:
            self._qa_session = datetime.now().strftime('%Y%m%d_%H%M%S')
        d = os.path.join(self._args.qa_dir, self._qa_session)
        os.makedirs(d, exist_ok=True)
        mapping = {
            f'{tag}_fixed.png': self._qa_rgb_fixed,
            f'{tag}_wrist.png': self._qa_rgb_wrist,
            f'{tag}_global_viz.png': self._qa_rgb_global_viz,
            f'{tag}_fine_viz.png': self._qa_rgb_fine_viz,
        }
        saved = []
        for name, rgb in mapping.items():
            if rgb is None:
                continue
            path = os.path.join(d, name)
            cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            saved.append(name)
        self.get_logger().info(f'QA snap {self._qa_session}/{tag}: {saved or "no frames yet"}')

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

    def _cancel_move(self) -> None:
        self._move_phase = None
        self._goal_fut = None
        self._result_fut = None
        self._ik_fut = None
        self._traj_goal_fut = None
        self._traj_result_fut = None
        self._approach_step = 0

    def _reset_align_loop(self) -> None:
        self._align_cycle_idx = 0
        self._align_step_idx = 0
        self._align_failed_actions = []
        self._align_before_obs = None
        self._align_last_action = None
        self._align_last_decision = None
        self._align_needs_verify = False
        self._align_rollback_action = None
        self._align_wait_log_t = 0.0
        self._align_request_written = False
        self._align_wait_phase = 'command'
        self._align_restore_joints = None
        self._align_settle_until = 0.0
        self._align_settle_kind = None
        self._align_arm_prepared = False
        self._align_arm_enable_ok = False
        self._align_enable_fut = None
        self._align_enable_deadline = time.time() + 15.0
        self._align_no_motion_streak = 0
        self._align_commanded_joints = None
        # Force a fresh /perception/fine/berries before claiming wrist lock.
        self._fine = None
        self._fine_recv_t = 0.0

    def _tick(self) -> None:
        cmd = self._pop_cmd()
        if cmd == 'abort':
            self._locked = None
            self._plan = None
            self._cancel_move()
            self._reset_align_loop()
            self._set_state('IDLE')
            return
        if cmd == 'clear_lock':
            self._locked = None
            if self._state in ('LOCKING', 'ALIGNING', 'REFINING', 'PLANNING'):
                self._cancel_move()
                self._reset_align_loop()
                self._set_state('IDLE')
            return

        # Progress in-flight MoveIt / IK-align without blocking the executor.
        if self._move_phase is not None:
            if self._move_phase in ('align', 'align_step', 'align_rollback', 'align_settle'):
                self._poll_align_motion()
            else:
                self._poll_move()
            return

        if self._state == 'IDLE':
            if cmd == 'start':
                self._locked = None
                self._plan = None
                self._cancel_move()
                self._reset_align_loop()
                self._qa_session = datetime.now().strftime('%Y%m%d_%H%M%S')
                self._reached_pub.publish(Bool(data=False))
                self._set_state('LOCKING')
            return

        if self._state == 'LOCKING':
            if self._try_lock_global():
                self._reset_align_loop()
                self._set_state('ALIGNING')
            return

        if self._state == 'ALIGNING':
            if self._args.dry_run:
                self.get_logger().info('dry-run: skip ALIGNING loop')
                self._refine_deadline = time.time() + self._args.refine_timeout_s
                self._set_state('REFINING')
                return
            if not self._tick_aligning():
                self._set_state('ERROR', 'align loop failed')
            return

        if self._state == 'REFINING':
            berry = self._pick_fine_berry()
            if berry is not None:
                self._locked = berry
                self._lock_pub.publish(berry)
                self._set_state('PLANNING')
            elif time.time() > self._refine_deadline:
                if self._locked is not None:
                    self.get_logger().warn(
                        'refine timeout — planning from global/coarse lock')
                    self._set_state('PLANNING')
                else:
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
            if self._args.dry_run:
                self.get_logger().info('dry-run: skip MoveIt, pretend reached')
                self._reached_pub.publish(Bool(data=True))
                self._set_state('REACHED')
                self._set_state('WAIT_CONFIRM')
                return
            if self._plan is None:
                self._set_state('ERROR', 'no plan')
                return
            self._approach_step = 0
            self._start_move(self._build_pose_goal(self._plan.pre_grasp, with_orient=False), 'pre')
            return

        if self._state == 'WAIT_CONFIRM':
            if cmd == 'confirm_reset':
                self._set_state('RESETTING')
            return

        if self._state == 'RESETTING':
            if self._args.dry_run:
                self._locked = None
                self._plan = None
                self._set_state('IDLE')
                return
            self._start_move(self._build_joint_goal(list(self._args.home_joints)), 'home')
            return

        if self._state == 'ERROR':
            if cmd == 'start':
                self._error_msg = ''
                self._set_state('IDLE')
                self._cmd_queue.insert(0, 'start')
            elif cmd == 'confirm_reset':
                self._error_msg = ''
                self._set_state('RESETTING')

    def _prepare_arm_for_align(self) -> bool:
        if self._align_arm_enable_ok:
            return True
        if self._align_enable_fut is None:
            for client, label in (
                (self._arm_enable, 'enable_agx_arm'),
                (self._control_enable, 'control_enable'),
            ):
                if not client.service_is_ready():
                    self.get_logger().warn(f'ALIGNING: {label} service not ready')
                    continue
                req = SetBool.Request()
                req.data = True
                if label == 'enable_agx_arm':
                    self._align_enable_fut = client.call_async(req)
                else:
                    client.call_async(req)
            if self._align_enable_fut is None:
                return False
            self.get_logger().info('ALIGNING: waiting for enable_agx_arm response')
            return False
        if not self._align_enable_fut.done():
            return False
        res = self._align_enable_fut.result()
        self._align_enable_fut = None
        if res is None or not res.success:
            msg = res.message if res is not None else 'no response'
            self.get_logger().error(f'ALIGNING: enable_agx_arm failed: {msg}')
            return False
        self._align_arm_enable_ok = True
        self.get_logger().info('ALIGNING: arm enabled, control gate open')
        return True

    def _tick_aligning(self) -> bool:
        if self._locked is None:
            return False
        if not self._prepare_arm_for_align():
            if time.time() > self._align_enable_deadline:
                self._set_state(
                    'ERROR',
                    'enable_agx_arm failed — check e-stop, teaching mode, or re-bringup')
                return False
            return True
        if self._align_visible_in_wrist():
            self.get_logger().info('ALIGNING done: fine/wrist detector already sees target')
            self._enter_fine_after_coarse('wrist detector already sees target')
            return True
        if self._align_step_idx >= self._args.align_max_steps:
            self.get_logger().warn('ALIGNING exhausted max steps without coarse_ok')
            self._enter_fine_after_coarse('exhausted max align steps')
            return True

        obs = self._collect_align_observation()
        phase = self._align_wait_phase
        if self._args.align_judge_mode == 'file':
            tag = 'judge' if phase == 'judge' else 'request'
            req_name = f'align_{self._align_step_idx:02d}_{tag}.json'
            if not self._align_request_written:
                self._snap_qa(f'align_{self._align_step_idx:02d}_{tag}')
                self._write_align_observation(
                    req_name, obs, extra={'phase': phase, 'hint': self._align_agent_hint(phase)})
                self._align_request_written = True
                return True
            decision = self._load_align_decision_from_file(obs, phase=phase)
            if decision is None:
                return True
        else:
            decision = decide_action(
                obs,
                failed_actions=self._align_failed_actions,
                yaw_deadband_deg=self._args.align_yaw_deadband_deg,
                pitch_target_rad=self._args.align_pitch_target_rad,
                fine_visible_conf=self._args.align_fine_visible_conf,
                ee_angle_trigger_deg=self._args.align_ee_angle_trigger_deg,
                phase=phase,
            )

        action = str(decision.get('action', '')).strip().lower()
        if action in ('coarse_ok', 'done'):
            self.get_logger().info(
                f'ALIGNING {phase}: {action} — {decision.get("reason")}; '
                f'fine_visible={obs["fine_visible"]:.0f}')
            self._enter_fine_after_coarse(str(decision.get('reason', action)))
            return True

        # Position command (set_joints or legacy named action with estimated targets)
        if phase == 'judge':
            # New correction after a held pose: advance step index for QA files.
            self._align_step_idx += 1
            self._align_cycle_idx += 1
        self._align_before_obs = obs
        self._align_last_action = action
        self._align_last_decision = decision
        self._align_needs_verify = True
        self._align_request_written = False
        label = 'set_joints' if action == 'set_joints' else action
        self._snap_qa(f'align_{self._align_step_idx:02d}_before_{label}')
        self._write_align_observation(
            f'align_{self._align_step_idx:02d}_before_{label}.json', obs, decision)
        return self._start_align_step(decision, rollback=False)

    def _start_move(self, goal: MoveGroup.Goal, label: str) -> None:
        if not self._move.wait_for_server(timeout_sec=0.0):
            # Non-blocking check; retry next tick if not ready yet.
            if not self._move.server_is_ready():
                self.get_logger().warn(f'{label}: move_action not ready')
                self._set_state('ERROR', f'{label}: move_action unavailable')
                return
        self._move_phase = label
        self._goal_fut = self._move.send_goal_async(goal)
        self._result_fut = None
        self.get_logger().info(f'{label}: goal sent')

    def _poll_move(self) -> None:
        label = self._move_phase or '?'
        if self._goal_fut is not None:
            if not self._goal_fut.done():
                return
            gh = self._goal_fut.result()
            self._goal_fut = None
            if gh is None or not gh.accepted:
                self._cancel_move()
                self._set_state('ERROR', f'{label} goal rejected')
                return
            self._result_fut = gh.get_result_async()
            return

        if self._result_fut is not None:
            if not self._result_fut.done():
                return
            result = self._result_fut.result()
            self._result_fut = None
            self._move_phase = None
            if result is None:
                self._set_state('ERROR', f'{label}: no result')
                return
            code = result.result.error_code.val
            if code != MoveItErrorCodes.SUCCESS:
                self._set_state('ERROR', f'{label}: MoveIt error {code}')
                return
            self.get_logger().info(f'{label}: OK')
            self._on_move_ok(label)

    def _on_move_ok(self, label: str) -> None:
        if label == 'align':
            self._snap_qa('after_align')
            if self._args.align_only:
                self.get_logger().info('align-only: coarse look-at done')
                self._set_state('WAIT_CONFIRM')
            else:
                self._refine_deadline = time.time() + self._args.refine_timeout_s
                self._set_state('REFINING')
        elif label == 'pre':
            self._approach_step = 1
            assert self._plan is not None
            self._start_move(self._build_pose_goal(self._plan.grasp, with_orient=False), 'contact')
        elif label == 'contact':
            self._reached_pub.publish(Bool(data=True))
            self._set_state('REACHED')
            self._set_state('WAIT_CONFIRM')
        elif label == 'home':
            self._locked = None
            self._plan = None
            self._reset_align_loop()
            self._set_state('IDLE')

    def _enter_fine_after_coarse(self, reason: str) -> None:
        self.get_logger().info(f'ALIGNING → fine control ({reason})')
        self._refine_deadline = time.time() + self._args.refine_timeout_s
        if self._args.align_only:
            self._set_state('WAIT_CONFIRM')
        else:
            self._set_state('REFINING')

    def _align_agent_hint(self, phase: str) -> str:
        if phase == 'judge':
            return (
                'Pose is held after position move. Read fixed (primary) + wrist. '
                'If arm roughly faces the plant, write action=coarse_ok (enter wrist fine). '
                'If not, write action=set_joints with estimated joints_deg or delta_deg '
                '(absolute preferred). Do NOT use fixed-step whole_arm_* oscillation.'
            )
        return (
            'Estimate HOW MUCH to move from fixed mono (plant vs arm tip). '
            'Write action=set_joints with joints_deg absolute targets '
            '(joint1/joint2/joint3/joint5 in degrees), or delta_deg relative. '
            'If already roughly facing plant, action=coarse_ok. '
            'Hint fields plant_yaw_deg / joint*_deg are numeric aids only — trust the image.'
        )

    def _align_visible_in_wrist(self) -> bool:
        return is_target_visible(
            self._collect_align_observation(),
            fine_visible_conf=self._args.align_fine_visible_conf,
        )

    def _project_base_point_to_fixed_image(
        self, xyz: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float]]:
        if self._qa_rgb_fixed is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                'camera_fixed_optical_frame', self._args.base_frame, rclpy.time.Time())
        except Exception:
            return None
        T = _matrix_from_tf(tf)
        p = T @ np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
        h, w = self._qa_rgb_fixed.shape[:2]
        return _project_cam_point(
            float(p[0]), float(p[1]), float(p[2]),
            w, h, float(self._args.align_fixed_focal_px))

    def _collect_align_observation(self) -> Dict[str, float]:
        obs: Dict[str, float] = {
            'fine_visible': 0.0,
            'fine_confidence': 0.0,
            'joint1': float(self._joints[0]),
            'joint2': float(self._joints[1]),
            'joint3': float(self._joints[2]),
            'joint4': float(self._joints[3]),
            'joint5': float(self._joints[4]),
            'joint6': float(self._joints[5]),
            'joint1_deg': math.degrees(self._joints[0]),
            'joint2_deg': math.degrees(self._joints[1]),
            'joint3_deg': math.degrees(self._joints[2]),
            'joint5_deg': math.degrees(self._joints[4]),
            'plant_yaw': 0.0,
            'plant_yaw_deg': 0.0,
            'yaw_error': 0.0,
            'yaw_error_deg': 0.0,
            'ee_target_angle_deg': 180.0,
            'lock_distance_xy': 0.0,
            'fixed_has_view': 0.0,
            'fixed_dx_px': 0.0,
            'fixed_dy_px': 0.0,
        }
        fine = self._pick_fine_berry()
        if fine is not None:
            obs['fine_visible'] = 1.0
            obs['fine_confidence'] = float(fine.confidence)
        if self._locked is None:
            return obs
        p = self._locked.pose.pose.position
        plant = (float(p.x), float(p.y), float(p.z))
        plant_yaw = plant_base_yaw(plant)
        obs['plant_yaw'] = plant_yaw
        obs['plant_yaw_deg'] = math.degrees(plant_yaw)
        obs['yaw_error'] = norm_angle(plant_yaw - obs['joint1'])
        obs['yaw_error_deg'] = math.degrees(obs['yaw_error'])
        obs['lock_distance_xy'] = math.hypot(plant[0], plant[1])
        ee = self._current_ee_pose()
        plant_px = self._project_base_point_to_fixed_image(plant)
        if ee is not None:
            ee_xyz = (ee.pose.position.x, ee.pose.position.y, ee.pose.position.z)
            o = ee.pose.orientation
            cur_z = facing_z_from_quat(o.x, o.y, o.z, o.w)
            target_dir = (plant[0] - ee_xyz[0], plant[1] - ee_xyz[1], plant[2] - ee_xyz[2])
            obs['ee_target_angle_deg'] = math.degrees(angle_between(cur_z, target_dir))
            ee_px = self._project_base_point_to_fixed_image(ee_xyz)
            if plant_px is not None and ee_px is not None:
                obs['fixed_has_view'] = 1.0
                obs['fixed_dx_px'] = plant_px[0] - ee_px[0]
                obs['fixed_dy_px'] = plant_px[1] - ee_px[1]
        return obs

    def _align_decision_file_path(self) -> str:
        d = os.path.join(self._args.qa_dir, self._qa_session or 'latest')
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, 'align_decision.json')

    def _consume_align_decision_file(self, path: str) -> None:
        used = path + f'.used_{self._align_step_idx:02d}_{self._align_wait_phase}'
        try:
            os.replace(path, used)
        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass

    def _load_align_decision_from_file(
        self, obs: Dict[str, float], *, phase: str,
    ) -> Optional[Dict[str, object]]:
        del obs
        path = self._align_decision_file_path()
        if not os.path.exists(path):
            now = time.time()
            if now - self._align_wait_log_t > 2.0:
                self._align_wait_log_t = now
                self.get_logger().info(
                    f'ALIGNING waiting for decision file phase={phase}: {path} '
                    f'(set_joints+joints_deg|delta_deg, or coarse_ok)')
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            self.get_logger().warn(f'ALIGNING decision file invalid: {exc}')
            self._consume_align_decision_file(path)
            return {'action': 'coarse_ok', 'reason': 'invalid decision file', 'confidence': 0.0}

        if int(data.get('step_idx', -1)) != int(self._align_step_idx):
            return None
        dec_phase = str(data.get('phase', phase)).strip().lower()
        if dec_phase and dec_phase != phase:
            # Wrong phase for current wait — keep waiting (do not consume).
            return None

        action = str(data.get('action', '')).strip().lower()
        allowed = set(ALIGN_ACTIONS) | {'restore_joints'}
        if action not in allowed:
            self.get_logger().warn(f'ALIGNING decision action unsupported: {action!r}')
            self._consume_align_decision_file(path)
            return {'action': 'coarse_ok', 'reason': 'unsupported decision action', 'confidence': 0.0}

        decision: Dict[str, object] = {
            'action': action,
            'reason': str(data.get('reason', 'external decision')),
            'confidence': float(data.get('confidence', 0.5)),
            'phase': phase,
            'provider': str(data.get('provider', 'agent')),
        }
        if isinstance(data.get('joints_deg'), dict):
            decision['joints_deg'] = data['joints_deg']
        if isinstance(data.get('delta_deg'), dict):
            decision['delta_deg'] = data['delta_deg']
        self._consume_align_decision_file(path)
        return decision

    def _write_align_observation(
        self, name: str, obs: Dict[str, float], decision: Optional[Dict[str, object]] = None,
        verdict: Optional[Dict[str, object]] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        d = os.path.join(self._args.qa_dir, self._qa_session or 'latest')
        os.makedirs(d, exist_ok=True)
        payload: Dict[str, object] = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'state': self._state,
            'step_idx': self._align_step_idx,
            'cycle_idx': self._align_cycle_idx,
            'judge_mode': self._args.align_judge_mode,
            'phase': self._align_wait_phase,
            'failed_actions': list(self._align_failed_actions),
            'observation': obs,
        }
        if extra:
            payload.update(extra)
        if decision is not None:
            payload['decision'] = decision
        if verdict is not None:
            payload['verdict'] = verdict
        path = os.path.join(d, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

    def _align_target_joints_from_decision(
        self, decision: Dict[str, object], obs: Dict[str, float], *, rollback: bool,
    ) -> List[float]:
        action = str(decision.get('action', '')).strip().lower()
        if rollback and action == 'restore_joints' and self._align_restore_joints is not None:
            return apply_joint_command(
                self._joints, {'action': 'restore_joints'},
                restore_joints=self._align_restore_joints)
        if action == 'set_joints' or 'joints_deg' in decision or 'delta_deg' in decision:
            return apply_joint_command(
                self._joints, decision,
                joint1_limit_deg=self._args.align_joint1_limit_deg,
                joint5_min_rad=self._args.align_joint5_min_rad,
                joint5_max_rad=self._args.align_joint5_max_rad,
            )
        return apply_action_to_joints(
            action, self._joints, obs,
            yaw_step_deg=self._args.align_yaw_step_deg,
            pitch_step_deg=self._args.align_pitch_step_deg,
            joint1_limit_deg=self._args.align_joint1_limit_deg,
            joint5_min_rad=self._args.align_joint5_min_rad,
            joint5_max_rad=self._args.align_joint5_max_rad,
            face_plant_frac=self._args.align_face_plant_frac,
        )

    def _ensure_arm_control_enabled(self) -> None:
        if not self._align_arm_enable_ok:
            self._prepare_arm_for_align()

    def _start_align_step(self, decision: Dict[str, object], *, rollback: bool) -> bool:
        with self._move_lock:
            if self._move_phase in ('align_step', 'align_rollback', 'align_settle'):
                return True
            self._ensure_arm_control_enabled()
            if not self._align_arm_enable_ok:
                self.get_logger().warn('ALIGNING: skip motion — arm not enabled')
                return False
            obs = self._align_before_obs or self._collect_align_observation()
            action = str(decision.get('action', '')).strip().lower()
            if not rollback:
                self._align_restore_joints = list(self._joints)
            target = self._align_target_joints_from_decision(decision, obs, rollback=rollback)
            self._align_commanded_joints = list(target)
            label = 'rollback' if rollback else 'step'
            self.get_logger().info(
                f'ALIGNING {label}: action={action} via '
                f'{"move_j" if self._args.align_use_movej else "traj"} '
                f'j1 {math.degrees(self._joints[0]):.1f}->{math.degrees(target[0]):.1f}deg '
                f'j2 {math.degrees(self._joints[1]):.1f}->{math.degrees(target[1]):.1f}deg '
                f'j3 {math.degrees(self._joints[2]):.1f}->{math.degrees(target[2]):.1f}deg '
                f'j5 {math.degrees(self._joints[4]):.1f}->{math.degrees(target[4]):.1f}deg')
            dt = float(self._args.align_traj_s)
            self._align_deadline = time.time() + max(self._args.align_timeout_s, dt + 2.0)
            if self._args.align_use_movej:
                msg = JointState()
                msg.name = list(ARM_JOINTS)
                msg.position = [float(v) for v in target]
                self._move_j_pub.publish(msg)
                self._move_phase = 'align_settle'
                self._align_settle_kind = 'rollback' if rollback else 'step'
                self._align_settle_until = time.time() + dt + self._args.align_settle_s
                return True
            if not self._arm.server_is_ready():
                self.get_logger().warn('align: follow_joint_trajectory not ready')
                return False
            if self._arm_goal_handle is not None:
                try:
                    self._arm_goal_handle.cancel_goal_async()
                except Exception:
                    pass
                self._arm_goal_handle = None
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = list(ARM_JOINTS)
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in target]
            pt.time_from_start.sec = int(dt)
            pt.time_from_start.nanosec = int((dt - int(dt)) * 1e9)
            goal.trajectory.points = [pt]
            self._move_phase = 'align_rollback' if rollback else 'align_step'
            self._traj_goal_fut = self._arm.send_goal_async(goal)
            self._traj_result_fut = None
            return True

    def _poll_align_motion(self) -> None:
        if self._move_phase == 'align_settle':
            if time.time() < self._align_settle_until:
                return
            kind = self._align_settle_kind
            self._move_phase = None
            self._align_settle_kind = None
            if kind == 'rollback':
                self._on_align_rollback_done()
            else:
                self._on_align_hold_for_judge()
            return
        if time.time() > self._align_deadline:
            self._cancel_move()
            self._set_state('ERROR', 'align timeout')
            return
        if self._traj_goal_fut is not None:
            if not self._traj_goal_fut.done():
                return
            gh = self._traj_goal_fut.result()
            self._traj_goal_fut = None
            if gh is None or not gh.accepted:
                self._cancel_move()
                self._set_state('ERROR', 'align traj rejected')
                return
            self._arm_goal_handle = gh
            self._traj_result_fut = gh.get_result_async()
            return
        if self._traj_result_fut is not None:
            if not self._traj_result_fut.done():
                return
            phase = self._move_phase
            self._traj_result_fut = None
            self._move_phase = None
            if phase == 'align_rollback':
                self._on_align_rollback_done()
            else:
                self._align_settle_until = time.time() + self._args.align_settle_s
                self._align_settle_kind = 'step'
                self._move_phase = 'align_settle'

    def _on_align_hold_for_judge(self) -> None:
        """Arrive at commanded pose, hold, then ask agent whether coarse facing is OK."""
        if self._align_before_obs is None or self._align_last_action is None:
            self._set_state('ERROR', 'align hold missing baseline')
            return
        label = self._align_last_action
        self._snap_qa(f'align_{self._align_step_idx:02d}_after_{label}')
        after = self._collect_align_observation()
        before = self._align_before_obs
        _, verdict = verify_step(
            before, after,
            accept_score_margin=self._args.align_accept_score_margin,
            min_joint_move_deg=self._args.align_min_joint_move_deg,
            min_ee_angle_gain_deg=self._args.align_min_ee_angle_gain_deg,
        )
        verdict['action'] = label
        verdict['held_for_agent_judge'] = True
        if self._align_commanded_joints is not None:
            verdict['commanded_j1_deg'] = math.degrees(self._align_commanded_joints[0])
            verdict['commanded_j2_deg'] = math.degrees(self._align_commanded_joints[1])
            verdict['commanded_j3_deg'] = math.degrees(self._align_commanded_joints[2])
            verdict['commanded_j5_deg'] = math.degrees(self._align_commanded_joints[4])
            verdict['feedback_j1_after_deg'] = math.degrees(after['joint1'])
            verdict['feedback_j5_after_deg'] = math.degrees(after['joint5'])
        self._write_align_observation(
            f'align_{self._align_step_idx:02d}_after_{label}.json',
            after, decision=self._align_last_decision, verdict=verdict)

        moved = bool(verdict.get('moved', False))
        if not moved:
            self._align_no_motion_streak += 1
            self.get_logger().warn(
                f'ALIGNING: no joint motion detected ({self._align_no_motion_streak}/3)')
            if self._align_no_motion_streak >= 3:
                self._set_state(
                    'ERROR',
                    'align: arm not moving after 3 steps (enable_agx_arm / control_enable?)')
                return
        else:
            self._align_no_motion_streak = 0

        self.get_logger().info(
            f'ALIGNING hold after {label}: j1={math.degrees(after["joint1"]):.1f}deg '
            f'yaw_err={after.get("yaw_error_deg", 0.0):.1f}deg — waiting agent coarse judge')
        # Stay on same step_idx; switch to judge phase (no auto-rollback).
        self._align_wait_phase = 'judge'
        self._align_request_written = False
        self._align_needs_verify = False

    def _on_align_rollback_done(self) -> None:
        failed = self._align_last_action
        if failed and failed not in self._align_failed_actions:
            self._align_failed_actions.append(failed)
        self._snap_qa(f'align_{self._align_step_idx:02d}_rollback_{failed or "unknown"}')
        self.get_logger().info(
            f'ALIGNING rollback done for action={failed}; failed_actions={self._align_failed_actions}')
        self._align_step_idx += 1
        self._align_cycle_idx += 1
        self._align_before_obs = None
        self._align_last_action = None
        self._align_last_decision = None
        self._align_needs_verify = False
        self._align_rollback_action = None
        self._align_wait_phase = 'command'
        self._align_request_written = False

    def _start_align_ik(self) -> bool:
        target = self._align_look_at_pose()
        if target is None:
            return False
        if not self._ik.service_is_ready():
            self.get_logger().warn('align: /compute_ik not ready')
            return False
        if not self._arm.server_is_ready():
            self.get_logger().warn('align: follow_joint_trajectory not ready')
            return False

        req = GetPositionIK.Request()
        req.ik_request.group_name = self._args.move_group
        req.ik_request.ik_link_name = self._args.ee_link
        req.ik_request.avoid_collisions = False
        req.ik_request.pose_stamped = target
        rs = RobotState()
        js = JointState()
        js.name = list(ARM_JOINTS)
        js.position = list(self._joints)
        rs.joint_state = js
        req.ik_request.robot_state = rs

        self._move_phase = 'align'
        self._ik_fut = self._ik.call_async(req)
        self._traj_goal_fut = None
        self._traj_result_fut = None
        self._align_deadline = time.time() + max(self._args.align_timeout_s, 2.0)
        self.get_logger().info('align: IK requested')
        return True

    def _poll_align_ik(self) -> None:
        if time.time() > self._align_deadline:
            self._cancel_move()
            self._set_state('ERROR', 'align timeout')
            return

        if self._ik_fut is not None:
            if not self._ik_fut.done():
                return
            try:
                res = self._ik_fut.result()
            except Exception as exc:
                self._cancel_move()
                self._set_state('ERROR', f'align IK call failed: {exc}')
                return
            self._ik_fut = None
            if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
                code = res.error_code.val if res is not None else -1
                self._cancel_move()
                self._set_state('ERROR', f'align IK failed code={code}')
                return
            name_to_pos = dict(zip(
                res.solution.joint_state.name, res.solution.joint_state.position))
            joints = [float(name_to_pos[n]) for n in ARM_JOINTS if n in name_to_pos]
            if len(joints) != len(ARM_JOINTS):
                self._cancel_move()
                self._set_state('ERROR', 'align IK missing joints')
                return
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = list(ARM_JOINTS)
            pt = JointTrajectoryPoint()
            pt.positions = joints
            dt = float(self._args.align_traj_s)
            pt.time_from_start.sec = int(dt)
            pt.time_from_start.nanosec = int((dt - int(dt)) * 1e9)
            goal.trajectory.points = [pt]
            self._traj_goal_fut = self._arm.send_goal_async(goal)
            self.get_logger().info('align: trajectory sent')
            return

        if self._traj_goal_fut is not None:
            if not self._traj_goal_fut.done():
                return
            gh = self._traj_goal_fut.result()
            self._traj_goal_fut = None
            if gh is None or not gh.accepted:
                self._cancel_move()
                self._set_state('ERROR', 'align traj rejected')
                return
            self._traj_result_fut = gh.get_result_async()
            return

        if self._traj_result_fut is not None:
            if not self._traj_result_fut.done():
                return
            self._traj_result_fut = None
            self._move_phase = None
            self.get_logger().info('align: OK')
            self._on_move_ok('align')

    def _align_look_at_pose(self) -> Optional[PoseStamped]:
        if self._locked is None:
            return None
        p = self._locked.pose.pose.position
        ee = self._current_ee_pose()
        if ee is None:
            self.get_logger().error('align: no EE TF')
            return None
        ee_xyz = (ee.pose.position.x, ee.pose.position.y, ee.pose.position.z)
        o = ee.pose.orientation
        cur_z = facing_z_from_quat(o.x, o.y, o.z, o.w)
        plant = (p.x, p.y, p.z)
        target_dir = (p.x - ee_xyz[0], p.y - ee_xyz[1], p.z - ee_xyz[2])
        ang = math.degrees(angle_between(cur_z, target_dir))
        yaw_deg = math.degrees(plant_base_yaw(plant))
        full = yaw_face_standoff_pose(
            plant,
            standoff_m=self._args.align_standoff_m,
            eye_height_above_plant_m=self._args.align_eye_height_m,
        )
        if full is None:
            self.get_logger().error('align: degenerate yaw-face standoff')
            return None
        # Small step toward a pose that sits on base→plant ray (forces joint1 yaw).
        pose = blend_pose(
            ee_xyz, (o.x, o.y, o.z, o.w), full,
            position_blend=self._args.align_blend,
            orient_frac=self._args.align_orient_frac,
        )
        self.get_logger().info(
            f'ALIGNING yaw_face plant_yaw_deg={yaw_deg:.1f} '
            f'pos_blend={self._args.align_blend:.2f} orient_frac={self._args.align_orient_frac:.2f} '
            f'angle_to_target_deg={ang:.1f} '
            f'p_lock=({p.x:.3f},{p.y:.3f},{p.z:.3f}) '
            f'target=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}) '
            f'full=({full[0]:.3f},{full[1]:.3f},{full[2]:.3f}) ee={self._args.ee_link}')
        return self._pose_xyzq(pose[:3], pose[3:])

    def _try_lock_global(self) -> bool:
        if self._global is None or not self._global.berries:
            return False
        berries = list(self._global.berries)
        berries.sort(key=lambda b: (-float(b.confidence), abs(float(b.pose.pose.position.z))))
        self._locked = berries[0]
        self._lock_pub.publish(self._locked)
        p = self._locked.pose.pose.position
        self.get_logger().info(
            f'locked global berry conf={self._locked.confidence:.2f} '
            f'pos=({p.x:.3f},{p.y:.3f},{p.z:.3f}) frame={self._locked.pose.header.frame_id}')
        return True

    def _fine_msg_age_s(self) -> float:
        if self._fine is None or self._fine_recv_t <= 0.0:
            return float('inf')
        return max(0.0, time.time() - self._fine_recv_t)

    def _fine_is_fresh(self) -> bool:
        """Reject stale /perception/fine/berries so ALIGNING cannot false-done."""
        if self._fine is None:
            return False
        max_age = float(self._args.fine_max_age_s)
        if max_age <= 0.0:
            return True
        return self._fine_msg_age_s() <= max_age

    def _pick_fine_berry(self) -> Optional[DetectedBerry]:
        if self._fine is None or not self._fine.berries:
            return None
        if not self._fine_is_fresh():
            # Drop stale cache so later ticks stay blind until a fresh msg arrives.
            self._fine = None
            return None
        berries = list(self._fine.berries)
        max_d = self._args.fine_assoc_max_m
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
            if max_d > 0 and math.sqrt(d2(berries[0])) > max_d:
                return None
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

    def _pose_xyzq(self, xyz: Tuple[float, float, float], quat_xyzw) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = self._args.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
        if quat_xyzw is None:
            ps.pose.orientation.w = 1.0
        else:
            ps.pose.orientation.x = float(quat_xyzw[0])
            ps.pose.orientation.y = float(quat_xyzw[1])
            ps.pose.orientation.z = float(quat_xyzw[2])
            ps.pose.orientation.w = float(quat_xyzw[3])
        return ps

    def _build_plan(self, berry: DetectedBerry) -> Optional[SuctionGraspPlan]:
        bx = float(berry.pose.pose.position.x)
        by = float(berry.pose.pose.position.y)
        bz = float(berry.pose.pose.position.z)
        ee = self._current_ee_pose()
        ee_xyz = None
        ee_quat = None
        if ee is not None:
            ee_xyz = (ee.pose.position.x, ee.pose.position.y, ee.pose.position.z)
            o = ee.pose.orientation
            ee_quat = (o.x, o.y, o.z, o.w)
        waypoints = build_cup_axis_plan(
            (bx, by, bz), ee_xyz, ee_quat,
            self._args.cup_contact_offset,
            self._args.pre_grasp_offset,
            self._args.post_grasp_offset,
        )
        if waypoints is None:
            return None
        stamp = self.get_clock().now().to_msg()
        plan = SuctionGraspPlan()
        plan.header.frame_id = self._args.base_frame
        plan.header.stamp = stamp

        def _fill(ps: PoseStamped, w) -> None:
            ps.header.frame_id = self._args.base_frame
            ps.header.stamp = stamp
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = w[0], w[1], w[2]
            ps.pose.orientation.x = w[3]
            ps.pose.orientation.y = w[4]
            ps.pose.orientation.z = w[5]
            ps.pose.orientation.w = w[6]

        _fill(plan.pre_grasp, waypoints[0])
        _fill(plan.grasp, waypoints[1])
        _fill(plan.post_grasp, waypoints[2])
        return plan

    def _build_pose_goal(self, target: PoseStamped, *, with_orient: bool = True) -> MoveGroup.Goal:
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
        constraints = Constraints(position_constraints=[pc])
        if with_orient:
            oc = OrientationConstraint()
            oc.header = target.header
            oc.link_name = self._args.ee_link
            oc.orientation = target.pose.orientation
            oc.absolute_x_axis_tolerance = self._args.orient_tolerance
            oc.absolute_y_axis_tolerance = self._args.orient_tolerance
            oc.absolute_z_axis_tolerance = self._args.orient_tolerance
            oc.weight = 1.0
            constraints.orientation_constraints = [oc]
        req.goal_constraints = [constraints]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--move-action', default='/move_action')
    parser.add_argument('--move-group', default='arm')
    parser.add_argument('--ee-link', default=os.environ.get('PICK_EE_LINK', 'link6'))
    parser.add_argument('--base-frame', default='base_link')
    parser.add_argument('--cup-contact-offset', type=float,
                        default=float(os.environ.get('PICK_CUP_CONTACT_OFFSET', '0.04')))
    parser.add_argument('--pre-grasp-offset', type=float, default=0.12)
    parser.add_argument('--post-grasp-offset', type=float, default=0.10)
    parser.add_argument('--align-standoff-m', type=float, default=0.28)
    parser.add_argument('--align-eye-height-m', type=float, default=0.08,
                        help='EE height above plant for yaw-face standoff (look slightly down)')
    parser.add_argument('--align-blend', type=float, default=0.45,
                        help='Fraction of XY move toward yaw-face standoff (forces joint1)')
    parser.add_argument('--align-orient-frac', type=float, default=0.55,
                        help='Slerp fraction toward look-at orientation')
    parser.add_argument('--align-rotate-in-place', action='store_true', default=False,
                        help='Deprecated: in-place orient-only (does not face plant)')
    parser.add_argument('--no-align-rotate-in-place', action='store_false',
                        dest='align_rotate_in_place')
    parser.add_argument('--align-traj-s', type=float, default=3.5,
                        help='FollowJointTrajectory duration for align step')
    parser.add_argument('--align-timeout-s', type=float, default=12.0)
    parser.add_argument('--align-only', action='store_true',
                        help='Stop after coarse ALIGNING (QA); do not refine/approach')
    parser.add_argument('--align-judge-mode', choices=('heuristic', 'file'), default='heuristic',
                        help='Action judge source during ALIGNING; file mode waits for qa_dir/session/align_decision.json')
    parser.add_argument('--align-max-steps', type=int, default=8,
                        help='Max accepted-or-rolled-back ALIGNING attempts before exit')
    parser.add_argument('--align-yaw-step-deg', type=float, default=18.0,
                        help='Small-step base yaw delta used by ALIGNING')
    parser.add_argument('--align-pitch-step-deg', type=float, default=12.0,
                        help='Small-step wrist pitch delta used by ALIGNING')
    parser.add_argument('--align-face-plant-frac', type=float, default=0.55,
                        help='Fraction of joint1 move toward plant yaw for face_plant_yaw')
    parser.add_argument('--align-settle-s', type=float, default=1.0,
                        help='Wait after align motion before wrist A/B verify')
    parser.add_argument('--align-use-movej', action='store_true', default=False,
                        help='Use /control/move_j for ALIGNING steps')
    parser.add_argument('--align-use-traj', dest='align_use_movej', action='store_false',
                        help='Use follow_joint_trajectory for ALIGNING (default, same as teleop)')
    parser.add_argument('--align-ee-angle-trigger-deg', type=float, default=20.0,
                        help='When wrist blind and EE angle exceeds this, use face_plant_yaw')
    parser.add_argument('--align-min-joint-move-deg', type=float, default=0.8,
                        help='Minimum joint motion to count as a real step')
    parser.add_argument('--align-min-ee-angle-gain-deg', type=float, default=3.0,
                        help='Accept step if EE-to-target angle improves by at least this much')
    parser.add_argument('--align-yaw-deadband-deg', type=float, default=8.0,
                        help='If plant yaw error exceeds this, prefer base yaw step')
    parser.add_argument('--align-pitch-target-rad', type=float, default=-0.75,
                        help='Target joint5 value encouraging the wrist camera to look lower')
    parser.add_argument('--align-fixed-focal-px', type=float, default=500.0,
                        help='Fixed camera focal length in pixels for rough image-space ALIGNING judge')
    parser.add_argument('--align-fine-visible-conf', type=float, default=0.55,
                        help='Fine detector confidence threshold to treat wrist target as visible')
    parser.add_argument('--align-accept-score-margin', type=float, default=0.15,
                        help='Minimum observation score gain required to accept a step')
    parser.add_argument('--align-joint1-limit-deg', type=float, default=170.0,
                        help='Soft clamp for joint1 command during ALIGNING')
    parser.add_argument('--align-joint5-min-rad', type=float, default=-1.70,
                        help='Lower clamp for joint5 command during ALIGNING')
    parser.add_argument('--align-joint5-max-rad', type=float, default=1.20,
                        help='Upper clamp for joint5 command during ALIGNING')
    parser.add_argument('--qa-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'log', 'real_robot', 'qa'),
        help='Directory for before/after align screenshots')
    parser.add_argument('--velocity-scale', type=float, default=0.12)
    parser.add_argument('--position-tolerance', type=float, default=0.025)
    parser.add_argument('--orient-tolerance', type=float, default=0.35)
    parser.add_argument('--fine-assoc-max-m', type=float, default=0.35,
                        help='Max 3D distance global→fine association; 0=disable')
    parser.add_argument('--fine-max-age-s', type=float, default=0.5,
                        help='Ignore /perception/fine/berries older than this (wall-clock recv age); '
                             '0 disables freshness gate')
    parser.add_argument('--refine-timeout-s', type=float, default=20.0)
    parser.add_argument('--home-joints', type=float, nargs=6, default=HOME)
    parser.add_argument('--dry-run', action='store_true',
                        help='Skip MoveIt; still exercise topic FSM')
    args = parser.parse_args()

    rclpy.init()
    node = ReachFsmNode(args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
