#!/usr/bin/env python3
"""Topic-driven reach FSM: agent region lock → align → wrist visual servo → confirm.

Cmds on /reach/cmd (std_msgs/String):
  start | start_refine | next_fruit | confirm_reset | abort | clear_lock

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
from typing import Dict, List, Optional, Sequence, Tuple

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
from picking_msgs.msg import (
    DetectedBerry,
    DetectedBerryArray,
    ExecutorStatus,
    SuctionGraspPlan,
    ToolTrajectory4s,
)
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, Header, String
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
    look_at_pose,
    partial_look_at_pose,
    plant_base_yaw,
    quat_slerp,
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
from piper_position_ik import (  # noqa: E402
    aim_uv_ik,
    approach_axis_ik,
    approach_axis_ik_chunked,
    cartesian_velocity_step,
    cup_axis_ik,
    cup_axis_ik_chunked,
    fk_link6,
    fk_link6_T,
    fk_xyz,
    position_ik_keep_orient,
    position_ik_keep_orient_chunked,
    tip_xyz,
)
from refine_depth_util import (  # noqa: E402
    apply_mono_chord_scale,
    z_mono_near_reproject_from_sources,
)
from berry_surface_fit import (  # noqa: E402
    fit_contact_surface,
)

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# PLANNING/APPROACHING kept for status compatibility; happy path skips open-loop cup_axis.
STATES = (
    'IDLE', 'LOCKING', 'ALIGNING', 'REFINING', 'REFINING_WAIT_NEAR',
    'PLANNING', 'APPROACHING',
    'REACHED', 'WAIT_CONFIRM', 'RESETTING', 'ERROR',
)


def _np_zeros_rgb(h: int = 224, w: int = 224) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


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
    *,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    if z <= 1e-6:
        return None
    fx_ = float(focal if fx is None else fx)
    fy_ = float(focal if fy is None else fy)
    cx_ = float(width * 0.5 if cx is None else cx)
    cy_ = float(height * 0.5 if cy is None else cy)
    u = cx_ + fx_ * x / z
    v = cy_ + fy_ * y / z
    return (float(u), float(v))


class ReachFsmNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('reach_fsm_node')
        self._args = args
        # pbvs-vlm-reach-v2: enable continuous PBVS refining loop.
        self._use_pbvs: bool = getattr(args, 'use_pbvs', False)
        self._pbvs_via_executor: bool = bool(getattr(args, 'pbvs_via_executor', False))
        self._publish_tool_traj: bool = bool(getattr(args, 'publish_tool_traj', True))
        self._pbvs_tool_traj_seq = 0
        self._pbvs_executor_wait = False
        self._pbvs_executor_motion_until = 0.0
        self._last_executor_status: Optional[ExecutorStatus] = None
        # Goal 4: data collection for BC training.
        self._data_collector = None
        if getattr(args, 'collect_data', False):
            import sys as _sys, os as _os
            _scripts_dir = _os.path.dirname(_os.path.abspath(__file__))
            if _scripts_dir not in _sys.path:
                _sys.path.insert(0, _scripts_dir)
            from align_data_collector import AlignDataCollector
            self._data_collector = AlignDataCollector(
                save_dir=getattr(args, 'collect_data_dir', 'data/align_episodes'),
                teacher=getattr(args, 'align_judge_mode', 'heuristic'),
            )
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
        self._qa_fixed_gen = 0
        self._qa_wrist_gen = 0
        self._qa_global_viz_gen = 0
        self._qa_fine_viz_gen = 0
        self._depth_qa_stop_done = False
        self._last_mono_chord_meta: Optional[Dict] = None
        self._range_depth_started = False
        self._await_fresh_optical_until = 0.0
        self._probe_tri_chord_applied = False
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
        self._lock_request_written = False
        self._lock_wait_log_t = 0.0
        self._servo_step_idx = 0
        self._servo_deadline = 0.0
        self._servo_orient_frac: Optional[float] = None
        self._servo_step_scale = 1.0
        self._servo_settle_until = 0.0
        self._refine_wait_log_t = 0.0
        # REFINING: no ALIGN joint-anchor soft bands — Cartesian IK owns the path.
        # Anchors are recorded at entry for QA only.
        self._refine_j1_anchor = 0.0
        self._refine_j5_anchor = 0.0
        self._refine_j2_anchor = 0.0
        self._refine_j3_anchor = 0.0
        self._last_berry_base: Optional[Tuple[float, float, float]] = None
        self._last_z_cam: Optional[float] = None
        self._last_berry_t = 0.0
        self._near_mode = False
        # REFINING: one fruit locked at entry — BoT-SORT track_id for whole phase.
        self._refine_fruit_anchor: Optional[Tuple[float, float, float]] = None
        self._refine_locked_track_id: Optional[int] = None
        # After clear_lock: fine must reset BoT-SORT before we pin a new tid.
        self._refine_await_tracker_reset = False
        self._refine_tracker_reset_t = 0.0
        self._refine_tracker_reset_live_streak = 0
        self._refine_fine_lost_streak = 0
        self._refine_lock_snap_pending = False
        self._near_handoff_written = False
        self._near_frozen_berry: Optional[Tuple[float, float, float]] = None
        self._near_oneshot_sent = False
        self._near_oneshot_attempts = 0
        # Mono probe + oneshots: range→center_oneshot→range_depth→contact_oneshot.
        self._mono_probe_done = False
        self._mono_probe_obs: List[Dict] = []
        self._mono_probe_purpose = 'center'  # 'center' | 'contact'
        # Online EE/camera Cartesian ↔ UV from probe/center small steps (not joint Jac).
        self._ee_uv_jac: Optional[Dict[str, float]] = None
        self._center_probe_tri_ok = False
        # Soft depth scale for center aim when short-baseline tri says mono is deep.
        self._center_z_tri_m: Optional[float] = None
        self._center_z_mono_m: Optional[float] = None
        # REFINING: range_center → center_oneshot → [live replan] → range_depth → near
        self._refine_phase = 'range_center'
        self._center_ok_streak = 0
        self._center_oneshot_sent = False
        self._center_oneshot_attempts = 0
        self._center_last_cmd: Optional[Dict] = None
        self._center_live_replan_done = False
        # Pending REFINING step record → written as servo_XX.json after settle.
        self._servo_pending_record: Optional[Dict] = None
        self._servo_motion_frames: List[Dict] = []
        self._servo_motion_last_t = 0.0
        self._pbvs_qa_recorder = None
        self._pbvs_qa_finalized = False
        self._servo_ik_fail_n = 0
        self._servo_ik_fail_t = 0.0
        self._tcp_pose: Optional[PoseStamped] = None

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
            PoseStamped, '/feedback/tcp_pose', self._on_tcp_pose, 10,
            callback_group=self._cb_group)
        self.create_subscription(
            Image, '/camera_fixed/color/image_raw', self._on_fixed_img, qos_profile_sensor_data,
            callback_group=self._cb_group)
        self.create_subscription(
            Image, '/camera_fixed/depth/image_raw', self._on_fixed_depth, qos_profile_sensor_data,
            callback_group=self._cb_group)
        self.create_subscription(
            CameraInfo, '/camera_fixed/color/camera_info', self._on_fixed_cam_info,
            qos_profile_sensor_data, callback_group=self._cb_group)
        self.create_subscription(
            Image, '/camera_wrist/color/image_raw', self._on_wrist_img, qos_profile_sensor_data,
            callback_group=self._cb_group)
        self.create_subscription(
            Image, '/camera_wrist/depth/image_raw', self._on_wrist_depth, qos_profile_sensor_data,
            callback_group=self._cb_group)
        self.create_subscription(
            CameraInfo, '/camera_wrist/color/camera_info', self._on_wrist_info,
            qos_profile_sensor_data, callback_group=self._cb_group)
        self._wrist_K: Optional[Tuple[float, float, float, float]] = None  # fx,fy,cx,cy
        self._wrist_camera_K: Optional[np.ndarray] = None   # 3×3 for PBVS approach selector
        self._last_wrist_depth: Optional[np.ndarray] = None
        self._last_fixed_depth: Optional[np.ndarray] = None
        self._fixed_camera_K: Optional[np.ndarray] = None
        self._fixed_camera_frame: str = 'camera_fixed_color_optical_frame'
        self._precomputed_approach_dir: Optional[np.ndarray] = None
        self._fused_map: Optional[object] = None          # FusedApproachMap (created in _init_pbvs)
        self._pbvs_last_wrist_map_t: float = 0.0
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
        self._tool_traj_pub = self.create_publisher(
            ToolTrajectory4s, '/planning/tool_trajectory_4s', 10)
        if self._pbvs_via_executor or self._publish_tool_traj:
            self.create_subscription(
                ExecutorStatus, '/planning/executor_status', self._on_executor_status, 10,
                callback_group=self._cb_group)
        self._arm_enable = self.create_client(SetBool, '/enable_agx_arm', callback_group=self._cb_group)
        self._control_enable = self.create_client(SetBool, '/control_enable', callback_group=self._cb_group)
        self._arm_goal_handle = None
        os.makedirs(self._args.qa_dir, exist_ok=True)

        self.create_timer(0.1, self._tick, callback_group=self._cb_group)
        # Continuous PBVS execution at ~25 Hz (main tick stays 10 Hz for FSM/cmd).
        if self._use_pbvs:
            self.create_timer(
                0.04, self._tick_pbvs_rate, callback_group=self._cb_group)
        self._publish_status()
        self.get_logger().info(
            f'reach_fsm ready — LOCK_REGION(agent) → ALIGNING → wrist servo '
            f'(judge_mode={self._args.align_judge_mode}); '
            f'qa_dir={self._args.qa_dir}'
            + (f'; pbvs={getattr(self._args, "pbvs_mode", "stream")}@25Hz'
               if self._use_pbvs else '')
            + ('; via_executor' if self._pbvs_via_executor else ''))

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

    def _on_tcp_pose(self, msg: PoseStamped) -> None:
        self._tcp_pose = msg

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
            self._qa_fixed_gen += 1

    def _on_wrist_img(self, msg: Image) -> None:
        rgb = self._image_to_rgb(msg)
        if rgb is not None:
            self._qa_rgb_wrist = rgb
            self._qa_wrist_gen += 1

    def _on_wrist_depth(self, msg: Image) -> None:
        try:
            d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            self._last_wrist_depth = d.astype(np.float32) / 1000.0
        except Exception:
            pass

    def _on_fixed_depth(self, msg: Image) -> None:
        try:
            d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            self._last_fixed_depth = d.astype(np.float32) / 1000.0
        except Exception:
            pass

    def _on_fixed_cam_info(self, msg: CameraInfo) -> None:
        if msg.k[0] > 1.0 and msg.k[4] > 1.0:
            self._fixed_camera_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            if msg.header.frame_id:
                self._fixed_camera_frame = msg.header.frame_id

    def _on_wrist_info(self, msg: CameraInfo) -> None:
        if msg.k[0] > 1.0 and msg.k[4] > 1.0:
            self._wrist_K = (
                float(msg.k[0]), float(msg.k[4]),
                float(msg.k[2]), float(msg.k[5]),
            )
            self._wrist_camera_K = np.array([
                [float(msg.k[0]), 0.0,             float(msg.k[2])],
                [0.0,             float(msg.k[4]), float(msg.k[5])],
                [0.0,             0.0,             1.0            ],
            ], dtype=np.float64)

    def _wrist_intrinsics(self) -> Tuple[float, float, float, float]:
        """fx, fy, cx, cy — prefer live camera_info over CLI focal + image center."""
        w, h = self._wrist_image_wh()
        if self._wrist_K is not None:
            return self._wrist_K
        f = float(getattr(self._args, 'wrist_focal_px', 488.0))
        return f, f, 0.5 * w, 0.5 * h

    def _on_global_viz(self, msg: Image) -> None:
        rgb = self._image_to_rgb(msg)
        if rgb is not None:
            self._qa_rgb_global_viz = rgb
            self._qa_global_viz_gen += 1

    def _on_fine_viz(self, msg: Image) -> None:
        rgb = self._image_to_rgb(msg)
        if rgb is not None:
            self._qa_rgb_fine_viz = rgb
            self._qa_fine_viz_gen += 1

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

    def _snap_servo_motion_frame(self, *, force: bool = False) -> None:
        """Record motion frames with a 10 Hz cap, skipping writes when no camera updated."""
        import cv2  # type: ignore
        if not self._qa_session:
            return
        interval = float(getattr(self._args, 'servo_motion_snap_s', 0.10))
        now = time.time()
        if not force and (now - self._servo_motion_last_t) < interval:
            return
        if (self._qa_rgb_wrist is None and self._qa_rgb_fine_viz is None
                and self._qa_rgb_fixed is None and self._qa_rgb_global_viz is None):
            return
        gens = {
            'wrist_gen': int(self._qa_wrist_gen),
            'fine_viz_gen': int(self._qa_fine_viz_gen),
            'fixed_gen': int(self._qa_fixed_gen),
            'global_viz_gen': int(self._qa_global_viz_gen),
        }
        if not force and self._servo_motion_frames:
            prev = self._servo_motion_frames[-1]
            if all(int(prev.get(k, -1)) == v for k, v in gens.items()):
                return
        self._servo_motion_last_t = now
        idx = len(self._servo_motion_frames)
        tag = f'servo_{self._servo_step_idx:02d}_motion_{idx:02d}'
        d = os.path.join(self._args.qa_dir, self._qa_session)
        os.makedirs(d, exist_ok=True)
        wrist_name = None
        fine_name = None
        fixed_name = None
        global_name = None
        if self._qa_rgb_wrist is not None:
            wrist_name = f'{tag}_wrist.png'
            cv2.imwrite(
                os.path.join(d, wrist_name),
                cv2.cvtColor(self._qa_rgb_wrist, cv2.COLOR_RGB2BGR))
        if self._qa_rgb_fine_viz is not None:
            fine_name = f'{tag}_fine_viz.png'
            cv2.imwrite(
                os.path.join(d, fine_name),
                cv2.cvtColor(self._qa_rgb_fine_viz, cv2.COLOR_RGB2BGR))
        if self._qa_rgb_fixed is not None:
            fixed_name = f'{tag}_fixed.png'
            cv2.imwrite(
                os.path.join(d, fixed_name),
                cv2.cvtColor(self._qa_rgb_fixed, cv2.COLOR_RGB2BGR))
        if self._qa_rgb_global_viz is not None:
            global_name = f'{tag}_global_viz.png'
            cv2.imwrite(
                os.path.join(d, global_name),
                cv2.cvtColor(self._qa_rgb_global_viz, cv2.COLOR_RGB2BGR))
        tcp = self._tcp_or_ee_xyz()
        frame = {
            'i': idx,
            't': now,
            'tag': tag,
            'wrist': wrist_name,
            'fine_viz': fine_name,
            'fixed': fixed_name,
            'global_viz': global_name,
            **gens,
            'joints_deg': {
                'j1': math.degrees(self._joints[0]),
                'j2': math.degrees(self._joints[1]),
                'j3': math.degrees(self._joints[2]),
                'j5': math.degrees(self._joints[4]),
            },
            'tcp_base': list(tcp) if tcp is not None else None,
            'refine_phase': str(self._refine_phase),
        }
        self._servo_motion_frames.append(frame)
        if idx == 0 or force or idx % 5 == 0:
            self.get_logger().info(
                f'QA motion {self._qa_session}/{tag} '
                f'(n={len(self._servo_motion_frames)})')
        if self._use_pbvs:
            self._record_pbvs_tick(
                source='motion',
                motion_tag=tag,
                motion_i=idx,
                refine_phase=str(self._refine_phase),
            )

    def _qa_session_dir(self) -> Optional[str]:
        if not self._qa_session:
            return None
        d = os.path.join(self._args.qa_dir, self._qa_session)
        os.makedirs(d, exist_ok=True)
        return d

    def _write_qa_json(self, name: str, payload: Dict) -> Optional[str]:
        d = self._qa_session_dir()
        if d is None:
            return None
        path = os.path.join(d, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        return path

    def _ensure_pbvs_qa_recorder(self) -> None:
        if not self._use_pbvs:
            return
        if self._pbvs_qa_finalized:
            return
        if self._pbvs_qa_recorder is not None:
            return
        if not self._qa_session:
            self._qa_session = datetime.now().strftime('%Y%m%d_%H%M%S')
        from pbvs_qa_recorder import PbvsQaRecorder
        self._pbvs_qa_recorder = PbvsQaRecorder(
            self._args.qa_dir, self._qa_session)
        self.get_logger().info(
            f'PBVS QA recorder → {self._pbvs_qa_recorder.session_dir}')

    def _pbvs_qa_images(self) -> Dict[str, Optional[np.ndarray]]:
        return {
            'fixed': self._qa_rgb_fixed,
            'wrist': self._qa_rgb_wrist,
            'global_viz': self._qa_rgb_global_viz,
            'fine_viz': self._qa_rgb_fine_viz,
        }

    def _pbvs_build_tick_meta(self, **extra) -> Dict:
        """Snapshot PBVS + arm state for one QA jsonl row (25 Hz or motion)."""
        q = self._joint_positions_rad()
        tcp = self._tcp_or_ee_xyz()
        target = None
        cup_dist = None
        p_des = None
        err_m = None
        if hasattr(self, '_pbvs_kf'):
            target = self._pbvs_control_target()
            if target is not None:
                cup_dist = self._cup_berry_dist_from(target)
        cup_gap = None
        cup_open = None
        d_cam_m = None
        d_cam_surface_m = None
        berry = self._pick_probe_live_berry() if self._refine_fruit_anchor else None
        depth_mode = ''
        berry_xyz = None
        z_depth_m = None
        z_mono_m = None
        z_used_m = None
        cam_dist_m = None
        if berry is not None:
            try:
                base = self._berry_base_xyz(berry)
                if base is not None:
                    berry_xyz = list(base)
                depth_mode = str(getattr(berry, 'depth_mode', '') or '')
                z_d = float(getattr(berry, 'z_depth_m', -1.0))
                z_m = float(getattr(berry, 'z_mono_m', -1.0))
                if z_d > 1e-4:
                    z_depth_m = z_d
                if z_m > 1e-4:
                    z_mono_m = z_m
                cam = self._berry_cam_xyz(berry)
                if cam is not None and cam[2] > 1e-4:
                    z_used_m = float(cam[2])
                    cam_dist_m = float(
                        math.sqrt(cam[0] ** 2 + cam[1] ** 2 + cam[2] ** 2))
            except Exception:
                pass
        if berry is not None and berry_xyz is not None:
            cam_m = self._pbvs_cam_metrics(berry, np.asarray(berry_xyz, dtype=np.float64))
            if cam_m:
                d_cam_m = cam_m.get('d_cam_m')
                d_cam_surface_m = cam_m.get('d_cam_surface_m')
        # Always publish cup opening (= tip) in base_link for QA overlay.
        cup_open = self._cup_open_xyz(q)
        frozen_berry = getattr(self, '_pbvs_frozen_berry', None)
        if frozen_berry is None and self._refine_fruit_anchor is not None:
            frozen_berry = np.asarray(self._refine_fruit_anchor, dtype=np.float64)
        berry_base_locked = (
            np.asarray(frozen_berry, dtype=np.float64).flatten()[:3].tolist()
            if frozen_berry is not None else None)
        # Prefer live berry_xyz; else locked/frozen for oneshot after vision lost.
        if berry_xyz is None and berry_base_locked is not None:
            berry_xyz = list(berry_base_locked)
        cup_berry_dist_m = None
        cup_berry_dxyz_m = None
        if berry_base_locked is not None and cup_open is not None:
            dvec = np.asarray(berry_base_locked, dtype=np.float64).flatten()[:3] - np.asarray(
                cup_open, dtype=np.float64).flatten()[:3]
            cup_berry_dxyz_m = [float(v) for v in dvec]
            cup_berry_dist_m = float(np.linalg.norm(dvec))
        if target is not None:
            gap_axis = self._cup_gap_along_axis(
                np.asarray(target, dtype=np.float64),
                approach_dir=getattr(self, '_pbvs_frozen_approach_dir', None))
            if gap_axis is not None:
                cup_gap, _ = gap_axis
        tip = tip_xyz(q.tolist()) if q is not None else None
        if (target is not None and tip is not None
                and hasattr(self, '_pbvs_state')):
            approach_dir = getattr(self, '_pbvs_frozen_approach_dir', None)
            if approach_dir is None:
                approach_dir = getattr(self, '_pbvs_approach_dir_base', None)
            if approach_dir is not None:
                ad = np.asarray(approach_dir, dtype=np.float64)
                tip_standoff = self._tip_contact_standoff_m()
                standoff = 0.07
                if self._pbvs_state == 'APPROACH':
                    p_des = target - ad * tip_standoff
                elif self._pbvs_state == 'SERVO':
                    p_des = target - ad * standoff
                if p_des is not None:
                    err_m = float(np.linalg.norm(p_des - tip))
        meta: Dict = {
            'pbvs_state': getattr(self, '_pbvs_state', 'INIT'),
            'move_phase': self._move_phase,
            'track_id': (
                getattr(self, '_pbvs_lock_track_id', None)
                or self._refine_locked_track_id),
            'servo_step_idx': int(self._servo_step_idx),
            'depth_mode': depth_mode,
            'z_depth_m': z_depth_m,
            'z_mono_m': z_mono_m,
            'z_used_m': z_used_m,
            'cam_dist_m': cam_dist_m,
            'berry_xyz': berry_xyz,
            'berry_base_locked': berry_base_locked,
            'berry_z_m': (float(berry_xyz[2]) if berry_xyz else None),
            'target_xyz': (target.tolist() if target is not None else None),
            'frozen_target': (
                self._pbvs_frozen_target.tolist()
                if getattr(self, '_pbvs_frozen_target', None) is not None
                else berry_base_locked),
            'tip_xyz': (tip.tolist() if tip is not None else None),
            'p_des': (p_des.tolist() if p_des is not None else None),
            'cup_dist_m': cup_dist,
            'cup_gap_m': cup_gap,
            'd_cam_m': d_cam_m,
            'd_cam_surface_m': d_cam_surface_m,
            'cup_open_xyz': (cup_open.tolist() if cup_open is not None else None),
            'cup_berry_dist_m': cup_berry_dist_m,
            'cup_berry_dxyz_m': cup_berry_dxyz_m,
            'approach_blind': bool(getattr(self, '_pbvs_approach_blind', False)),
            'vision_lost_streak': int(getattr(self, '_pbvs_vision_lost_streak', 0)),
            'cam_snapshot_d_cam_m': (
                (self._pbvs_cam_snapshot or {}).get('d_cam_surface_m')),
            'err_m': err_m,
            'v_mps': float(getattr(self, '_pbvs_last_v_norm', 0.0)),
            'tcp_base': list(tcp) if tcp is not None else None,
            'joints_deg': {
                'j1': math.degrees(self._joints[0]),
                'j2': math.degrees(self._joints[1]),
                'j3': math.degrees(self._joints[2]),
                'j5': math.degrees(self._joints[4]),
            } if self._joints and len(self._joints) >= 6 else None,
            'live_uv': (
                list(self._berry_image_uv(berry))
                if berry is not None and self._berry_image_uv(berry) is not None
                else None),
            'live_depth_mode': depth_mode or None,
            'ray_scale_applied': (
                getattr(self, '_pbvs_scaled_berry', None) is not None),
            'kf_uncertainty': (
                float(self._pbvs_kf.uncertainty())
                if hasattr(self, '_pbvs_kf') and self._pbvs_kf.initialized
                else None),
            'kf_reject_streak': int(getattr(self, '_pbvs_kf_reject_streak', 0)),
            'camera_gens': {
                'fixed': int(self._qa_fixed_gen),
                'wrist': int(self._qa_wrist_gen),
                'global_viz': int(self._qa_global_viz_gen),
                'fine_viz': int(self._qa_fine_viz_gen),
            },
            'source': 'pbvs_tick',
        }
        meta.update(extra)
        return meta

    def _record_pbvs_tick(self, meta: Optional[Dict] = None, **extra) -> None:
        if not self._use_pbvs:
            return
        self._ensure_pbvs_qa_recorder()
        if self._pbvs_qa_recorder is None:
            return
        payload = self._pbvs_build_tick_meta(**(meta or {}), **extra)
        self._pbvs_qa_recorder.record_tick(payload, self._pbvs_qa_images())

    def _pbvs_qa_log_event(self, kind: str, detail: Optional[Dict] = None) -> None:
        self._ensure_pbvs_qa_recorder()
        if self._pbvs_qa_recorder is not None:
            self._pbvs_qa_recorder.log_event(kind, detail)

    def _finalize_pbvs_qa(
        self, outcome: str, extra: Optional[Dict] = None,
    ) -> Optional[str]:
        if self._pbvs_qa_finalized or self._pbvs_qa_recorder is None:
            return None
        self._pbvs_qa_finalized = True
        html = self._pbvs_qa_recorder.finalize(outcome, extra)
        self.get_logger().info(
            f'PBVS QA finalized ({outcome}): '
            f'{self._pbvs_qa_recorder.frame_count} frames → {html}')
        return html

    def _set_state(self, state: str, err: str = '') -> None:
        prev = self._state
        if state not in STATES:
            state = 'ERROR'
        if (prev == 'REFINING' and self._use_pbvs
                and state in ('ERROR', 'WAIT_CONFIRM', 'IDLE')
                and not self._pbvs_qa_finalized):
            outcome = (
                'success' if state == 'WAIT_CONFIRM'
                else ('error' if state == 'ERROR' else 'idle'))
            self._finalize_pbvs_qa(outcome, {'err': err, 'fsm_state': state})
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
        self._pbvs_executor_wait = False
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
            self._reset_refine_state()
            # Goal 4: close current episode on reset.
            if self._data_collector is not None:
                self._data_collector.end_episode()
            self._set_state('IDLE')
            return
        if cmd == 'clear_lock':
            self._locked = None
            if self._state in ('LOCKING', 'ALIGNING', 'REFINING', 'REFINING_WAIT_NEAR', 'PLANNING'):
                self._cancel_move()
                self._reset_align_loop()
                self._reset_refine_state()
                if self._data_collector is not None and self._state in ('LOCKING', 'ALIGNING'):
                    self._data_collector.end_episode()
                self._set_state('IDLE')
            return

        # Progress in-flight MoveIt / IK-align / wrist servo without blocking.
        # Keep confirm_near even while a premature/in-flight servo is running —
        # otherwise WAIT_NEAR confirmation is popped and discarded.
        if self._move_phase is not None:
            if cmd == 'confirm_near' and self._state == 'REFINING_WAIT_NEAR':
                self._near_mode = True
                self._freeze_near_berry()
                self._refine_deadline = time.time() + max(
                    45.0, float(self._args.refine_timeout_s) * 0.5)
                self.get_logger().info(
                    'REFINING near handoff confirmed (during motion) — '
                    f'will continue after settle '
                    f'(deadline +{self._refine_deadline - time.time():.0f}s)')
                self._set_state('REFINING')
            if self._move_phase in ('align', 'align_step', 'align_rollback', 'align_settle'):
                self._poll_align_motion()
            elif self._move_phase in ('servo', 'servo_ik', 'servo_settle'):
                self._poll_wrist_servo()
            else:
                self._poll_move()
            return

        if self._state == 'IDLE':
            if cmd == 'start_refine':
                self._plan = None
                self._cancel_move()
                self._reset_align_loop()
                self._servo_step_idx = 0
                self._servo_orient_frac = None
                self._servo_step_scale = 1.0
                if not self._qa_session:
                    self._qa_session = datetime.now().strftime('%Y%m%d_%H%M%S')
                self._reached_pub.publish(Bool(data=False))
                if self._locked is None and self._global is not None and self._global.berries:
                    self._apply_region_lock(
                        self._global.berries[0], source='start_refine region fallback')
                self._refine_j1_anchor = float(self._joints[0])
                self._refine_j5_anchor = float(self._joints[4])
                self._enter_fine_after_coarse('start_refine from current (align) pose')
                return
            if cmd == 'start':
                self._locked = None
                self._plan = None
                self._cancel_move()
                self._reset_align_loop()
                self._lock_request_written = False
                self._servo_step_idx = 0
                self._servo_orient_frac = None
                self._servo_step_scale = 1.0
                self._qa_session = datetime.now().strftime('%Y%m%d_%H%M%S')
                self._reached_pub.publish(Bool(data=False))
                self._set_state('LOCKING')
            return

        if self._state == 'LOCKING':
            if self._tick_lock_region():
                self._reset_align_loop()
                # Goal 4: start a new episode when a target is locked.
                if self._data_collector is not None:
                    plant_xyz = None
                    if self._locked is not None:
                        try:
                            plant_xyz = [
                                float(self._locked.pose.position.x),
                                float(self._locked.pose.position.y),
                                float(self._locked.pose.position.z),
                            ]
                        except Exception:
                            pass
                    self._data_collector.start_episode(plant_xyz=plant_xyz)
                self._set_state('ALIGNING')
            return

        if self._state == 'ALIGNING':
            if self._args.dry_run:
                self.get_logger().info('dry-run: skip ALIGNING loop')
                self._refine_j1_anchor = float(self._joints[0])
                self._refine_j5_anchor = float(self._joints[4])
                self._refine_deadline = time.time() + self._args.refine_timeout_s
                self._set_state('REFINING')
                return
            if not self._tick_aligning():
                self._set_state('ERROR', 'align loop failed')
            return

        if self._state == 'REFINING':
            # pbvs-vlm-reach-v2: continuous rate timer owns PBVS execution.
            if getattr(self, '_use_pbvs', False):
                return
            self._tick_wrist_servo_refine()
            return

        if self._state == 'REFINING_WAIT_NEAR':
            if cmd == 'confirm_near':
                self._near_mode = True
                self._freeze_near_berry()
                # Human confirm may take longer than mid-range servo budget.
                self._refine_deadline = time.time() + max(
                    45.0, float(self._args.refine_timeout_s) * 0.5)
                self.get_logger().info(
                    'REFINING near handoff confirmed — freeze berry, one-shot cup approach '
                    f'(deadline +{self._refine_deadline - time.time():.0f}s)')
                self._set_state('REFINING')
                return
            if cmd == 'abort':
                self._cancel_move()
                self._reset_refine_state()
                self._set_state('IDLE')
                return
            # Hold pose; user inspects near_handoff_pending QA snaps.
            return

        # Legacy open-loop states: should not be entered on approved path.
        if self._state in ('PLANNING', 'APPROACHING'):
            self._set_state(
                'ERROR',
                'open-loop PLANNING/APPROACHING disabled — use wrist visual servo in REFINING')
            return

        if self._state == 'WAIT_CONFIRM':
            if cmd == 'confirm_reset':
                self._set_state('RESETTING')
            elif cmd == 'next_fruit':
                self._prepare_for_next_fruit()
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
        if self._align_step_idx >= self._args.align_max_steps:
            self._set_state(
                'ERROR',
                'ALIGNING exhausted max steps without coarse_ok (no auto-refine)')
            return False

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
        elif self._args.align_judge_mode == 'vlm':
            from align_judge import decide_action_vlm
            global_img = self._annotate_fixed_for_vlm()
            if global_img is None or self._qa_rgb_wrist is None:
                self._set_state('ERROR', 'VLM align: fixed/wrist camera not ready')
                return False
            try:
                decision = decide_action_vlm(
                    obs,
                    global_img=global_img,
                    wrist_img=self._qa_rgb_wrist,
                    phase=phase,
                    failed_actions=self._align_failed_actions,
                    yaw_deadband_deg=self._args.align_yaw_deadband_deg,
                    pitch_target_rad=self._args.align_pitch_target_rad,
                    fine_visible_conf=self._args.align_fine_visible_conf,
                    ee_angle_trigger_deg=self._args.align_ee_angle_trigger_deg,
                )
            except Exception as exc:
                self._set_state('ERROR', f'VLM align failed: {exc}')
                return False
        elif self._args.align_judge_mode == 'bc':
            # BC policy decision (Goal 4).
            from align_judge import decide_action_bc
            decision = decide_action_bc(
                obs,
                global_img=self._qa_rgb_fixed,
                wrist_img=self._qa_rgb_wrist,
                model_path=getattr(self._args, 'bc_model_path', ''),
                phase=phase,
                fail_count=self._align_step_idx,
            )
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
        # Goal 4: log step for BC training (skip if cameras not yet warm).
        if self._data_collector is not None:
            if self._qa_rgb_fixed is not None and self._qa_rgb_wrist is not None:
                self._data_collector.log_step(
                    global_img=self._qa_rgb_fixed,
                    wrist_img=self._qa_rgb_wrist,
                    obs=obs,
                    action=decision,
                )
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
        # Goal 4: close ALIGNING episode, then open a REFINING episode.
        if self._data_collector is not None:
            self._data_collector.mark_success()
            self._data_collector.end_episode()
            # Immediately start a companion REFINING episode.
            plant_xyz = None
            if self._locked is not None:
                try:
                    plant_xyz = [
                        float(self._locked.pose.position.x),
                        float(self._locked.pose.position.y),
                        float(self._locked.pose.position.z),
                    ]
                except Exception:
                    pass
            self._data_collector.start_episode(plant_xyz=plant_xyz, phase='refining')
        self.get_logger().info(f'ALIGNING → fine control ({reason})')
        self._refine_deadline = time.time() + self._args.refine_timeout_s
        self._refine_j1_anchor = float(self._joints[0])
        self._refine_j5_anchor = float(self._joints[4])
        self._refine_j2_anchor = float(self._joints[1])
        self._refine_j3_anchor = float(self._joints[2])
        self._reset_refine_state(keep_anchors=True)
        # Do NOT lock in the same tick as clear — fine_detector must reset BoT-SORT first.
        self._refine_await_tracker_reset = True
        self._refine_tracker_reset_t = time.time()
        self._refine_tracker_reset_live_streak = 0
        self._fine = None  # drop pre-clear cache so we cannot pin a stale tid
        if 'start_refine' not in reason:
            self._save_refine_entry_pose(source=reason)
        self.get_logger().info(
            f'REFINING anchors j1={math.degrees(self._refine_j1_anchor):.1f}deg '
            f'j2={math.degrees(self._refine_j2_anchor):.1f}deg '
            f'j3={math.degrees(self._refine_j3_anchor):.1f}deg '
            f'j5={math.degrees(self._refine_j5_anchor):.1f}deg; '
            f'near_handoff_z={float(self._args.servo_near_handoff_z):.2f}m '
            f'fruit_assoc={float(self._args.refine_fruit_assoc_max_m):.2f}m '
            f'lock_min_conf={float(self._args.refine_lock_min_conf):.2f} '
            f'lock_max_base_y={float(self._args.refine_lock_max_base_y_m):.2f}m '
            f'near_confirm={int(self._args.servo_near_confirm)}; '
            f'await fine tracker reset before fruit lock')
        # Re-init PBVS first (creates _fused_map), then feed global RGB-D into it.
        # Must run in this order: _init_pbvs creates _fused_map; _build_approach_map_from_global feeds it.
        if getattr(self, '_use_pbvs', False) and not self._args.align_only:
            self._init_pbvs()
            # Wait for static fixed-cam TF (buffer fill race right after FSM start).
            self._build_approach_map_from_global(wait_tf_s=2.0)
        else:
            self._build_approach_map_from_global()
        if self._args.align_only:
            self._set_state('WAIT_CONFIRM')
        else:
            self._set_state('REFINING')

    def _clear_target_lock(self) -> None:
        """Tell fine_detector to drop BoT-SORT pin / base_coast and reset tracker."""
        self._refine_locked_track_id = None
        msg = DetectedBerry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._args.base_frame
        msg.track_id = -1
        msg.confidence = 0.0
        self._lock_pub.publish(msg)
        # Force at least one republish so fine_detector always sees clear even if
        # the previous lock was already cleared (same -1 payload).
        self._lock_pub.publish(msg)

    def _reset_refine_state(self, *, keep_anchors: bool = False) -> None:
        if (self._use_pbvs and self._pbvs_qa_recorder is not None
                and not self._pbvs_qa_finalized):
            self._finalize_pbvs_qa('reset')
        self._pbvs_qa_recorder = None
        self._pbvs_qa_finalized = False
        for attr in (
            '_pbvs_kf', '_pbvs_state', '_pbvs_target_cmd', '_pbvs_frozen_target',
            '_pbvs_contact_sent', '_fused_map',
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        self._servo_step_idx = 0
        self._near_mode = False
        self._near_handoff_written = False
        self._near_frozen_berry = None
        self._near_oneshot_sent = False
        self._near_oneshot_attempts = 0
        self._mono_probe_done = False
        self._mono_probe_obs = []
        self._mono_probe_purpose = 'center'
        self._ee_uv_jac = None
        self._center_probe_tri_ok = False
        self._center_z_tri_m = None
        self._center_z_mono_m = None
        self._refine_phase = 'range_center'
        self._depth_qa_stop_done = False
        self._last_mono_chord_meta = None
        self._range_depth_started = False
        self._await_fresh_optical_until = 0.0
        self._probe_tri_chord_applied = False
        self._center_ok_streak = 0
        self._center_oneshot_sent = False
        self._center_oneshot_attempts = 0
        self._center_last_cmd = None
        self._center_live_replan_done = False
        self._refine_fruit_anchor = None
        self._refine_locked_track_id = None
        self._refine_await_tracker_reset = False
        self._refine_tracker_reset_t = 0.0
        self._refine_tracker_reset_live_streak = 0
        self._refine_fine_lost_streak = 0
        self._refine_lock_snap_pending = False
        self._servo_pending_record = None
        self._servo_ik_fail_n = 0
        self._servo_ik_fail_t = 0.0
        self._last_berry_base = None
        self._last_z_cam = None
        self._last_berry_t = 0.0
        self._lock_wait_rec_t = 0.0
        self._lock_wait_snap_t = 0.0
        self._lock_wait_snap_idx = 0
        self._clear_target_lock()
        if not keep_anchors:
            self._refine_j1_anchor = float(self._joints[0])
            self._refine_j5_anchor = float(self._joints[4])
            self._refine_j2_anchor = float(self._joints[1])
            self._refine_j3_anchor = float(self._joints[2])

    def _prepare_for_next_fruit(self) -> None:
        """Pick cycle: leave WAIT_CONFIRM without homing; ready for next start_refine."""
        self._locked = None
        self._plan = None
        self._cancel_move()
        self._reset_refine_state()
        self._reached_pub.publish(Bool(data=False))
        self._set_state('IDLE', 'next_fruit')
        self.get_logger().info('WAIT_CONFIRM → IDLE (next_fruit, arm stays in place)')

    def _berry_depth_diag(self, berry: DetectedBerry) -> Dict:
        mode = str(getattr(berry, 'depth_mode', '') or '')
        z_d = float(getattr(berry, 'z_depth_m', -1.0))
        z_m = float(getattr(berry, 'z_mono_m', -1.0))
        return {
            'depth_mode': mode,
            'z_depth_m': z_d if z_d > 0.0 else None,
            'z_mono_m': z_m if z_m > 0.0 else None,
        }

    def _berry_has_rgbd(self, berry: DetectedBerry) -> bool:
        """True only when wrist RGB-D mask depth is valid (mono-only never qualifies)."""
        return float(getattr(berry, 'z_depth_m', -1.0)) > 1e-4

    def _berry_depth_trusted(self, berry: DetectedBerry) -> bool:
        """Wrist RGB-D ranging is authoritative — do not substitute mono depth."""
        mode = str(getattr(berry, 'depth_mode', '') or '')
        if mode in ('depth_raw', 'depth_coast'):
            return float(getattr(berry, 'z_depth_m', -1.0)) > 1e-4
        return False

    def _berry_range_z_m(
        self, berry: Optional[DetectedBerry],
    ) -> Tuple[Optional[float], str]:
        """Cam-frame Z for ranging: depth when available, else mono (legacy paths)."""
        if berry is None:
            return None, 'none'
        zd = float(getattr(berry, 'z_depth_m', -1.0))
        if self._berry_depth_trusted(berry) and 0.08 < zd < 1.5:
            return zd, 'depth'
        zm = float(getattr(berry, 'z_mono_m', -1.0))
        if 0.08 < zm < 1.5:
            return zm, 'mono'
        if zd > 1e-4 and 0.08 < zd < 1.5:
            return zd, 'depth'
        return None, 'none'

    def _berry_cam_range(self, berry: DetectedBerry) -> Optional[float]:
        """Distance from wrist camera optical origin to berry (m)."""
        cam = self._berry_cam_xyz(berry)
        if cam is None or cam[2] <= 1e-4:
            return None
        return math.sqrt(cam[0] ** 2 + cam[1] ** 2 + cam[2] ** 2)

    def _berry_cup_dist(self, berry: DetectedBerry) -> Optional[float]:
        base = self._berry_base_xyz(berry)
        tcp = self._tcp_or_ee_xyz()
        if base is None or tcp is None:
            return None
        return math.sqrt(
            (base[0] - tcp[0]) ** 2
            + (base[1] - tcp[1]) ** 2
            + (base[2] - tcp[2]) ** 2)

    def _berry_lock_base_y_ok(self, berry: DetectedBerry) -> bool:
        """Reject lock candidates too far in base_link +Y (workspace ceiling)."""
        max_y = float(getattr(self._args, 'refine_lock_max_base_y_m', 0.42))
        if max_y <= 0.0:
            return True
        base = self._berry_base_xyz(berry)
        if base is None:
            return False
        return float(base[1]) <= max_y

    def _pick_nearest_fine_berry_by_cup(self) -> Optional[DetectedBerry]:
        """REFINING entry lock: among conf≥min with valid wrist depth, nearest cam."""
        if self._fine is None or not self._fine.berries:
            return None
        if not self._fine_is_fresh():
            self._fine = None
            return None
        min_conf = float(self._args.refine_lock_min_conf)
        max_y = float(getattr(self._args, 'refine_lock_max_base_y_m', 0.42))
        ranked: List[Tuple[float, DetectedBerry]] = []
        skipped_low = 0
        skipped_high_y = 0
        for b in self._fine.berries:
            mode = str(getattr(b, 'depth_mode', '') or '')
            # Never pin a ghost / coast-only ID at entry — wait for live YOLO.
            if mode == 'base_coast':
                continue
            if not self._berry_depth_trusted(b):
                continue
            if float(b.confidence) < min_conf:
                skipped_low += 1
                continue
            if not self._berry_lock_base_y_ok(b):
                skipped_high_y += 1
                continue
            d_cam = self._berry_cam_range(b)
            if d_cam is not None:
                ranked.append((d_cam, b))
        if not ranked:
            if skipped_low > 0 or skipped_high_y > 0:
                self.get_logger().info(
                    f'REFINING: waiting for depth lock '
                    f'(low_conf={skipped_low} min_conf={min_conf:.2f} '
                    f'high_y={skipped_high_y} max_base_y={max_y:.2f}m)')
            return None
        ranked.sort(key=lambda x: x[0])
        return ranked[0][1]

    def _record_lock_wait_conf(self) -> None:
        """While waiting for lock: log all track confs + periodic fine_viz for threshold tuning."""
        if not self._qa_session:
            return
        now = time.time()
        last = float(getattr(self, '_lock_wait_rec_t', 0.0))
        if now - last < 0.5:
            return
        self._lock_wait_rec_t = now
        berries = []
        if self._fine is not None and self._fine.berries and self._fine_is_fresh():
            max_y = float(getattr(self._args, 'refine_lock_max_base_y_m', 0.42))
            for b in self._fine.berries:
                base = self._berry_base_xyz(b)
                base_y = float(base[1]) if base is not None else None
                berries.append({
                    'track_id': int(b.track_id),
                    'confidence': float(b.confidence),
                    'depth_mode': str(getattr(b, 'depth_mode', '') or ''),
                    'z_mono_m': float(getattr(b, 'z_mono_m', -1.0)),
                    'z_depth_m': float(getattr(b, 'z_depth_m', -1.0)),
                    'base_y_m': base_y,
                    'y_ok': (
                        base_y is None or max_y <= 0.0 or base_y <= max_y),
                })
        berries.sort(key=lambda x: -x['confidence'])
        best = berries[0]['confidence'] if berries else None
        gate = float(self._args.refine_lock_min_conf)
        idx = int(getattr(self, '_lock_wait_snap_idx', 0))
        rec = {
            't': now,
            'iso': datetime.now().isoformat(timespec='seconds'),
            'lock_min_conf': gate,
            'n': len(berries),
            'best_conf': best,
            'would_lock': bool(best is not None and best >= gate),
            'berries': berries,
            'snap_idx': idx,
        }
        d = self._qa_session_dir()
        if d:
            path = os.path.join(d, 'lock_wait_conf.jsonl')
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=True) + '\n')
        # Snap ~1 Hz for visual conf readout on overlays.
        snap_last = float(getattr(self, '_lock_wait_snap_t', 0.0))
        if now - snap_last >= 1.0:
            self._lock_wait_snap_t = now
            self._snap_qa(f'lock_wait_{idx:03d}')
            self._lock_wait_snap_idx = idx + 1
            self.get_logger().info(
                f'REFINING lock_wait[{idx}]: best_conf='
                f'{best if best is not None else -1:.3f} '
                f'gate={gate:.2f} n={len(berries)} '
                f'top={berries[:3]}')

    def _init_refine_fruit_lock(self) -> None:
        """Lock nearest-to-camera fine berry (mono pose) at REFINING entry."""
        self._record_lock_wait_conf()
        berry = self._pick_nearest_fine_berry_by_cup()
        if berry is None:
            self.get_logger().info(
                'REFINING: waiting for fine berries + TCP at entry '
                f'(min_conf={float(self._args.refine_lock_min_conf):.2f}, mono range)')
            return
        base = self._berry_base_xyz(berry)
        if base is None:
            return
        dist_cup = self._berry_cup_dist(berry)
        dist_cam = self._berry_cam_range(berry)
        tid = int(berry.track_id)
        self._refine_locked_track_id = tid if tid >= 0 else None
        self._refine_fruit_anchor = base
        self._last_berry_base = base
        self._last_berry_t = time.time()
        cam = self._berry_cam_xyz(berry)
        if cam is not None and cam[2] > 1e-4:
            self._last_z_cam = float(cam[2])
        berry.track_id = tid
        self._lock_pub.publish(berry)
        self.get_logger().info(
            f'REFINING fruit lock (nearest cam, mono pose, '
            f'conf>={float(self._args.refine_lock_min_conf):.2f}): '
            f'track_id={tid} conf={berry.confidence:.2f} '
            f'base=({base[0]:.3f},{base[1]:.3f},{base[2]:.3f}) '
            f'cam_dist={(dist_cam if dist_cam is not None else -1):.3f} '
            f'cup_dist={(dist_cup if dist_cup is not None else -1):.3f} '
            f'z_cam={(self._last_z_cam if self._last_z_cam is not None else -1):.3f} '
            f"mode={getattr(berry, 'depth_mode', '')} "
            f"z_depth={float(getattr(berry, 'z_depth_m', -1)):.3f} "
            f"z_mono={float(getattr(berry, 'z_mono_m', -1)):.3f}")
        # Ensure session dir exists before lock JSON (snap may come one tick later).
        if not self._qa_session:
            self._qa_session = datetime.now().strftime('%Y%m%d_%H%M%S')
        tcp = self._tcp_or_ee_xyz()
        lock_payload: Dict = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'session': self._qa_session,
            'track_id': tid,
            'confidence': float(berry.confidence),
            'lock_min_conf': float(self._args.refine_lock_min_conf),
            'track_min_conf': float(getattr(self._args, 'refine_track_min_conf', 0.12)),
            'berry_base': list(base),
            'cam_dist': float(dist_cam) if dist_cam is not None else None,
            'cup_dist': float(dist_cup) if dist_cup is not None else None,
            'z_cam': float(self._last_z_cam) if self._last_z_cam is not None else None,
            **self._berry_depth_diag(berry),
            'tcp_base': list(tcp) if tcp is not None else None,
            'joints_deg': [math.degrees(float(v)) for v in self._joints],
            'anchors_deg': {
                'j1': math.degrees(self._refine_j1_anchor),
                'j2': math.degrees(self._refine_j2_anchor),
                'j3': math.degrees(self._refine_j3_anchor),
                'j5': math.degrees(self._refine_j5_anchor),
            },
        }
        # Snapshot all visible tracks at lock (for choosing lock_min_conf).
        if self._fine is not None and self._fine.berries:
            lock_payload['candidates'] = [
                {
                    'track_id': int(b.track_id),
                    'confidence': float(b.confidence),
                    'depth_mode': str(getattr(b, 'depth_mode', '') or ''),
                }
                for b in sorted(
                    self._fine.berries, key=lambda x: -float(x.confidence))
            ]
        if tcp is not None:
            geo = self._cup_berry_geometry(base, tcp, berry_cam=cam)
            lock_payload.update({
                'berry_rel_cup': geo.get('berry_rel_cup'),
                'berry_cam': geo.get('berry_cam'),
                'cup_cam': geo.get('cup_cam'),
                'berry_uv': geo.get('berry_uv'),
                'cup_uv': geo.get('cup_uv'),
            })
            self._write_cup_rel_overlay('refine_fruit_lock', geo)
        self._write_qa_json('refine_fruit_lock.json', lock_payload)
        self._refine_lock_snap_pending = True

    def _await_fine_tracker_reset_before_lock(self) -> bool:
        """True when still waiting for fine BoT-SORT reset after clear_lock."""
        if not self._refine_await_tracker_reset:
            return False
        now = time.time()
        # Give fine_detector time to receive clear and reset_tracker().
        if now - self._refine_tracker_reset_t < 0.25:
            return True
        live = self._live_fine_berries()
        if live:
            self._refine_tracker_reset_live_streak += 1
        else:
            self._refine_tracker_reset_live_streak = 0
            self._record_lock_wait_conf()
            if now - self._refine_wait_log_t > 1.0:
                self._refine_wait_log_t = now
                self.get_logger().info(
                    'REFINING awaiting fine tracker reset '
                    '(need live YOLO after clear_lock)')
        if self._refine_tracker_reset_live_streak >= 3:
            self._refine_await_tracker_reset = False
            self.get_logger().info(
                f'REFINING fine tracker reset OK — live_n={len(live)}; fruit lock')
            self._init_refine_fruit_lock()
            return self._refine_fruit_anchor is None
        if now > self._refine_deadline:
            self._set_state(
                'ERROR',
                'refine: fine tracker reset / live berry timeout after clear_lock')
        return True

    def _annotate_refine_event_wrist(
        self,
        rgb: object,
        *,
        source: str,
        geo: Optional[Dict] = None,
        berry: Optional[DetectedBerry] = None,
    ) -> object:
        """Draw control-source markers: YOLO fresh lock vs probe-tri reproject."""
        import cv2  # type: ignore

        img = rgb.copy()
        h, w = img.shape[:2]
        ox, oy = int(w * 0.5), int(h * 0.5)
        cv2.drawMarker(
            img, (ox, oy), (160, 160, 160),
            markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
        cv2.putText(
            img, 'OPTICAL', (ox + 10, oy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        live_yolo: List[Dict] = []
        for b in self._live_fine_berries():
            u, v = float(b.image_u), float(b.image_v)
            if u < 0.0 or v < 0.0:
                continue
            live_yolo.append({
                'track_id': int(b.track_id),
                'u': u, 'v': v,
                'conf': float(b.confidence),
                'mode': str(getattr(b, 'depth_mode', '') or ''),
            })
            cv2.circle(img, (int(u), int(v)), 16, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(
                img, f'YOLO T{int(b.track_id)}', (int(u) + 14, int(v) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

        berry_uv_yolo = None
        if berry is not None and source.startswith('yolo'):
            u, v = float(berry.image_u), float(berry.image_v)
            if u >= 0.0 and v >= 0.0:
                berry_uv_yolo = [u, v]
                cv2.drawMarker(
                    img, (int(u), int(v)), (80, 255, 120),
                    markerType=cv2.MARKER_DIAMOND, markerSize=24, thickness=2)
                cv2.rectangle(
                    img, (int(u) - 28, int(v) - 28), (int(u) + 28, int(v) + 28),
                    (80, 255, 120), 2, cv2.LINE_AA)
                cv2.putText(
                    img, f'FSM YOLO LOCK T{int(berry.track_id)}',
                    (int(u) + 14, int(v) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 120), 2, cv2.LINE_AA)

        berry_uv_reproject = None
        pix_off = None
        if geo is not None:
            berry_uv = geo.get('berry_uv')
            optical = geo.get('optical_uv')
            if berry_uv is not None:
                berry_uv_reproject = [float(berry_uv[0]), float(berry_uv[1])]
                bu, bv = int(berry_uv[0]), int(berry_uv[1])
                cv2.drawMarker(
                    img, (bu, bv), (255, 80, 200),
                    markerType=cv2.MARKER_STAR, markerSize=26, thickness=2)
                cv2.putText(
                    img, 'PROBE TRI 3D', (bu + 12, bv + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 200), 2, cv2.LINE_AA)
            if berry_uv is not None and optical is not None:
                pix_off = math.hypot(
                    float(berry_uv[0]) - float(optical[0]),
                    float(berry_uv[1]) - float(optical[1]),
                )
                cv2.arrowedLine(
                    img,
                    (int(optical[0]), int(optical[1])),
                    (int(berry_uv[0]), int(berry_uv[1])),
                    (255, 80, 200), 2, tipLength=0.12)

        z_txt = f'{self._last_z_cam:.3f}' if self._last_z_cam is not None else '?'
        cv2.putText(
            img, f'SOURCE: {source}', (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(
            img,
            f'z_cam={z_txt}m  pix_off={(pix_off if pix_off is not None else -1):.1f}px  '
            f'live_yolo={len(live_yolo)}',
            (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 100), 1, cv2.LINE_AA)
        cv2.putText(
            img, 'grey=optical  green=YOLO lock  magenta=probe tri  yellow=live YOLO',
            (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        return img, {
            'source': source,
            'track_id': int(berry.track_id) if berry is not None else None,
            'berry_uv_yolo': berry_uv_yolo,
            'berry_uv_reproject': berry_uv_reproject,
            'optical_uv': [float(w * 0.5), float(h * 0.5)],
            'pix_off_reproject': pix_off,
            'live_yolo': live_yolo,
            'z_cam_m': self._last_z_cam,
            'fruit_anchor': list(self._refine_fruit_anchor or ()),
        }

    def _write_refine_event_meta(self, tag: str, meta: Dict) -> None:
        if not self._qa_session:
            return
        meta = dict(meta)
        meta['timestamp'] = datetime.now().isoformat(timespec='seconds')
        meta['session'] = self._qa_session
        meta['tag'] = tag
        path = os.path.join(self._args.qa_dir, self._qa_session, f'{tag}_meta.json')
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=True)
        except OSError as exc:
            self.get_logger().warn(f'REFINING event meta write failed: {exc}')

    def _snap_refine_lock_viz(
        self, tag: str, *, source: str = '', berry: Optional[DetectedBerry] = None,
    ) -> str:
        """Annotated wrist (source markers) + optional fine detection_viz."""
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        self._snap_qa(tag)
        if not self._qa_session:
            return ''
        sess_dir = os.path.join(self._args.qa_dir, self._qa_session)
        tune_dir = os.path.join(os.path.dirname(os.path.abspath(self._args.qa_dir)), 'refine_tune_viz')
        os.makedirs(tune_dir, exist_ok=True)

        geo = None
        if berry is not None and source.startswith('yolo'):
            cam = self._berry_cam_xyz(berry)
            tcp = self._tcp_or_ee_xyz()
            if cam is not None and tcp is not None:
                geo = self._cup_berry_geometry(
                    (float(berry.pose.pose.position.x),
                     float(berry.pose.pose.position.y),
                     float(berry.pose.pose.position.z)),
                    tcp,
                )
        elif 'probe_tri' in source or berry is None:
            geo = self._reproject_berry_geo()

        panels: List[Tuple[str, object]] = []
        event_meta: Dict = {'source': source or tag}
        if self._qa_rgb_wrist is not None:
            ann, event_meta = self._annotate_refine_event_wrist(
                self._qa_rgb_wrist, source=source or tag, geo=geo, berry=berry)
            panels.append(('wrist annotated', ann))
            cv2.imwrite(
                os.path.join(sess_dir, f'{tag}_annotated.png'),
                cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
        if self._qa_rgb_fine_viz is not None:
            panels.append(('BoT-SORT viz', self._qa_rgb_fine_viz))

        if not panels:
            return ''

        self._write_refine_event_meta(tag, event_meta)

        target_h = 480
        resized: List[object] = []
        for _, rgb in panels:
            img = rgb
            scale = target_h / float(img.shape[0])
            w = max(1, int(img.shape[1] * scale))
            resized.append(cv2.resize(img, (w, target_h)))

        gap = np.ones((target_h, 10, 3), dtype=np.uint8) * 48
        composite = resized[0]
        for img in resized[1:]:
            composite = np.hstack([composite, gap, img])

        for i, (label, _) in enumerate(panels):
            x0 = 12 if i == 0 else int(composite.shape[1] * i / len(panels)) + 12
            cv2.putText(
                composite, label, (x0, target_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        out_path = os.path.join(sess_dir, f'{tag}_sidebyside.png')
        latest = os.path.join(tune_dir, 'latest_lock_sidebyside.png')
        bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, bgr)
        cv2.imwrite(latest, bgr)
        self.get_logger().info(
            f'REFINING event viz ({source or tag}) → {out_path} (also {latest})')
        return out_path

    def _refine_entry_pose_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(self._args.qa_dir)),
            'refine_entry_pose.json')

    def _save_refine_entry_pose(self, *, source: str) -> None:
        path = self._refine_entry_pose_path()
        payload = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'session': self._qa_session,
            'source': source,
            'joints_rad': [float(v) for v in self._joints[:6]],
            'joints_deg': {
                ARM_JOINTS[i]: math.degrees(float(self._joints[i]))
                for i in range(min(6, len(self._joints)))
            },
            'fruit_anchor': list(self._refine_fruit_anchor or ()),
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=True)
            os.replace(tmp, path)
            self.get_logger().info(
                f'REFINING saved entry pose → {path} '
                f'j=({math.degrees(self._joints[0]):.1f},'
                f'{math.degrees(self._joints[1]):.1f},'
                f'{math.degrees(self._joints[2]):.1f},'
                f'{math.degrees(self._joints[4]):.1f})')
        except Exception as exc:
            self.get_logger().warn(f'REFINING could not save entry pose: {exc}')

    def _pick_fine_berry_unanchored(self) -> Optional[DetectedBerry]:
        """Legacy alias — entry lock uses nearest cup distance."""
        return self._pick_nearest_fine_berry_by_cup()

    def _filter_z_cam(self, z_cam: float) -> float:
        """Reject single-frame depth spikes (wrong YOLO association)."""
        last = self._last_z_cam
        jump = float(self._args.refine_z_cam_max_jump_m)
        if last is not None and jump > 0 and z_cam > last + jump:
            self.get_logger().warn(
                f'REFINING z_cam outlier {z_cam:.3f} vs last {last:.3f} — hold last')
            return last
        return z_cam

    def _base_xyz_to_cam(
        self, xyz: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float, float]]:
        cam = self._args.wrist_camera_frame
        try:
            tf = self._tf_buffer.lookup_transform(
                cam, self._args.base_frame, rclpy.time.Time())
        except Exception:
            return None
        T = _matrix_from_tf(tf)
        v = T @ np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
        return float(v[0]), float(v[1]), float(v[2])

    def _write_near_handoff_request(
        self,
        *,
        dist_cup: Optional[float],
        z_cam: Optional[float],
        berry_age: float,
    ) -> None:
        d = os.path.join(self._args.qa_dir, self._qa_session or 'latest')
        os.makedirs(d, exist_ok=True)
        payload = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'state': 'REFINING_WAIT_NEAR',
            'hint': (
                'Inspect near_handoff_pending QA images. '
                'If cup/berry geometry looks right for terminal approach, '
                'run: ros2 topic pub --once /reach/cmd std_msgs/msg/String '
                '"{data: confirm_near}"'
            ),
            'dist_cup_m': dist_cup,
            'z_cam_m': z_cam,
            'berry_age_s': berry_age,
            'fruit_anchor': list(self._refine_fruit_anchor or ()),
            'joints_deg': {
                'joint1': math.degrees(self._joints[0]),
                'joint2': math.degrees(self._joints[1]),
                'joint3': math.degrees(self._joints[2]),
                'joint5': math.degrees(self._joints[4]),
            },
        }
        path = os.path.join(d, 'near_handoff_request.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

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
                'camera_fixed_color_optical_frame', self._args.base_frame, rclpy.time.Time())
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
            self.get_logger().error(f'ALIGNING decision file invalid: {exc}')
            self._consume_align_decision_file(path)
            self._set_state('ERROR', f'invalid align decision file: {exc}')
            return None

        if int(data.get('step_idx', -1)) != int(self._align_step_idx):
            return None
        dec_phase = str(data.get('phase', phase)).strip().lower()
        if dec_phase and dec_phase != phase:
            # Wrong phase for current wait — keep waiting (do not consume).
            return None

        action = str(data.get('action', '')).strip().lower()
        allowed = set(ALIGN_ACTIONS) | {'restore_joints'}
        if action not in allowed:
            self.get_logger().error(f'ALIGNING decision action unsupported: {action!r}')
            self._consume_align_decision_file(path)
            self._set_state('ERROR', f'unsupported align action: {action}')
            return None

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
                joint6_limit_rad=self._args.align_joint6_limit_rad,
            )
        return apply_action_to_joints(
            action, self._joints, obs,
            yaw_step_deg=self._args.align_yaw_step_deg,
            pitch_step_deg=self._args.align_pitch_step_deg,
            joint1_limit_deg=self._args.align_joint1_limit_deg,
            joint5_min_rad=self._args.align_joint5_min_rad,
            joint5_max_rad=self._args.align_joint5_max_rad,
            joint6_limit_rad=self._args.align_joint6_limit_rad,
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

    def _lock_region_decision_path(self) -> str:
        d = os.path.join(self._args.qa_dir, self._qa_session or 'latest')
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, 'lock_region_decision.json')

    def _consume_lock_region_decision(self, path: str) -> None:
        used = path + f'.used_{self._qa_session or "x"}'
        try:
            os.replace(path, used)
        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass

    def _global_region_candidates(self) -> List[DetectedBerry]:
        if self._global is None or not self._global.berries:
            return []
        berries = list(self._global.berries)
        # Display order only — agent picks index; do not auto-swallow max conf.
        berries.sort(key=lambda b: (-float(b.confidence), abs(float(b.pose.pose.position.z))))
        return berries

    def _annotate_fixed_for_vlm(self) -> Optional[np.ndarray]:
        """Copy fixed RGB with locked cluster marked (red circle) for VLM."""
        if self._qa_rgb_fixed is None:
            return None
        img = self._qa_rgb_fixed.copy()
        if self._locked is None:
            return img
        p = self._locked.pose.pose.position
        uv = self._project_base_point_to_fixed_image(
            (float(p.x), float(p.y), float(p.z)))
        if uv is None:
            return img
        import cv2  # type: ignore
        u, v = int(uv[0]), int(uv[1])
        cv2.circle(img, (u, v), 36, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, 'CLUSTER', (max(u - 40, 4), max(v - 44, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
        return img

    def _write_lock_region_request(self, berries: List[DetectedBerry]) -> None:
        d = os.path.join(self._args.qa_dir, self._qa_session or 'latest')
        os.makedirs(d, exist_ok=True)
        cands = []
        for i, b in enumerate(berries):
            p = b.pose.pose.position
            cands.append({
                'index': i,
                'confidence': float(b.confidence),
                'xyz': [float(p.x), float(p.y), float(p.z)],
                'frame': b.pose.header.frame_id or b.header.frame_id,
            })
        payload = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'state': 'LOCKING',
            'hint': (
                'Pick a COLLECTION REGION (plant/cluster), not a single berry. '
                'Write lock_region_decision.json with action=lock_region and index=N '
                'from candidates (fixed mono QA image).'
            ),
            'candidates': cands,
        }
        path = os.path.join(d, 'lock_region_request.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        self._snap_qa('lock_region')

    def _apply_region_lock(self, berry: DetectedBerry, *, source: str) -> None:
        self._locked = berry
        self._lock_pub.publish(berry)
        p = berry.pose.pose.position
        self.get_logger().info(
            f'LOCK_REGION via {source}: conf={berry.confidence:.2f} '
            f'pos=({p.x:.3f},{p.y:.3f},{p.z:.3f}) frame={berry.pose.header.frame_id}')

    def _tick_lock_region(self) -> bool:
        """Pick cluster via lock_region_decision.json before ALIGN."""
        berries = self._global_region_candidates()
        if not berries:
            now = time.time()
            if now - self._lock_wait_log_t > 2.0:
                self._lock_wait_log_t = now
                self.get_logger().info('LOCKING waiting for /perception/global/berries')
            return False

        if not self._lock_request_written:
            self._write_lock_region_request(berries)
            self._lock_request_written = True
            self.get_logger().info(
                f'LOCKING wrote lock_region_request.json n={len(berries)}; '
                f'waiting for lock_region_decision.json')
            return False

        path = self._lock_region_decision_path()
        if not os.path.exists(path):
            now = time.time()
            if now - self._lock_wait_log_t > 2.0:
                self._lock_wait_log_t = now
                self.get_logger().info(
                    f'LOCKING waiting for decision file: {path} '
                    f'(action=lock_region, index=N)')
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            self._consume_lock_region_decision(path)
            self._set_state('ERROR', f'invalid lock_region decision: {exc}')
            return False

        action = str(data.get('action', '')).strip().lower()
        if action != 'lock_region':
            self._consume_lock_region_decision(path)
            self._set_state('ERROR', f'unsupported lock action: {action!r}')
            return False

        try:
            idx = int(data.get('index', -1))
        except (TypeError, ValueError):
            idx = -1
        # Refresh candidates in case detector updated.
        berries = self._global_region_candidates()
        if idx < 0 or idx >= len(berries):
            self._consume_lock_region_decision(path)
            self._set_state(
                'ERROR',
                f'lock_region index={idx} out of range n={len(berries)}')
            return False

        reason = str(data.get('reason', ''))
        self._consume_lock_region_decision(path)
        self._apply_region_lock(
            berries[idx],
            source=f'agent index={idx} reason={reason[:80]!r}')
        self._build_approach_map_from_global()
        return True

    def _berry_cam_xyz(
        self, berry: DetectedBerry,
    ) -> Optional[Tuple[float, float, float]]:
        """Berry XYZ in wrist optical frame (for z_cam + pixel error)."""
        p = berry.pose.pose.position
        bf = (berry.pose.header.frame_id or berry.header.frame_id
              or self._args.base_frame)
        cam = self._args.wrist_camera_frame
        try:
            tf = self._tf_buffer.lookup_transform(cam, bf, rclpy.time.Time())
        except Exception:
            try:
                tf = self._tf_buffer.lookup_transform(
                    cam, self._args.base_frame, rclpy.time.Time())
            except Exception:
                return None
        T = _matrix_from_tf(tf)
        v = T @ np.array([float(p.x), float(p.y), float(p.z), 1.0], dtype=np.float64)
        return float(v[0]), float(v[1]), float(v[2])

    def _cup_cam_xyz(self) -> Optional[Tuple[float, float, float]]:
        """Suction cup / TCP in wrist optical frame (image target ≠ optical center)."""
        tcp = self._tcp_or_ee_xyz()
        if tcp is None:
            return None
        return self._base_xyz_to_cam(tcp)

    def _cup_axis_aim_uv(
        self,
        z_cam: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """Image UV where a berry on the *cup axis* appears (not optical center).

        Cup sits ~8 cm below the camera. Projecting the near-field cup point gives
        an off-image UV; instead project the cup's lateral offset at the berry
        depth: (u,v) = optical_center + f * cup_cam_xy / z_berry.
        """
        cup = self._cup_cam_xyz()
        if cup is None:
            return None
        z = float(z_cam) if z_cam is not None and float(z_cam) > 1e-4 else float(cup[2])
        if z <= 1e-4:
            z = float(self._last_z_cam or 0.0)
        if z <= 1e-4:
            return None
        fx, fy, cx, cy = self._wrist_intrinsics()
        return (
            cx + fx * float(cup[0]) / z,
            cy + fy * float(cup[1]) / z,
        )

    def _aim_uv_for_center(
        self,
        z_cam: Optional[float] = None,
    ) -> Tuple[float, float, str]:
        """Preferred mid-range aim UV: cup-axis; fallback principal point."""
        fx, fy, cx, cy = self._wrist_intrinsics()
        optical = (cx, cy)
        aim = self._cup_axis_aim_uv(z_cam)
        if aim is None:
            return optical[0], optical[1], 'optical_center'
        return float(aim[0]), float(aim[1]), 'cup_axis'

    def _cup_berry_geometry(
        self,
        berry_xyz: Tuple[float, float, float],
        tcp: Tuple[float, float, float],
        berry_cam: Optional[Tuple[float, float, float]] = None,
    ) -> Dict:
        """Berry relative to suction-cup center (base + cam + image pixels)."""
        rel = (
            float(berry_xyz[0] - tcp[0]),
            float(berry_xyz[1] - tcp[1]),
            float(berry_xyz[2] - tcp[2]),
        )
        dist = math.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2)
        cup_cam = self._cup_cam_xyz()
        if berry_cam is None:
            berry_cam = self._base_xyz_to_cam(berry_xyz)
        fx, fy, cx, cy = self._wrist_intrinsics()
        wh = None
        if self._qa_rgb_wrist is not None:
            wh = (int(self._qa_rgb_wrist.shape[1]), int(self._qa_rgb_wrist.shape[0]))
        w = wh[0] if wh else 640
        h = wh[1] if wh else 480

        def _uv(cam_xyz: Optional[Tuple[float, float, float]]):
            if cam_xyz is None or cam_xyz[2] <= 1e-4:
                return None
            p = _project_cam_point(
                cam_xyz[0], cam_xyz[1], cam_xyz[2], w, h, fx,
                cx=cx, cy=cy, fx=fx, fy=fy)
            return [float(p[0]), float(p[1])] if p is not None else None

        return {
            'berry_rel_cup': list(rel),
            'dist_cup': float(dist),
            'berry_cam': list(berry_cam) if berry_cam is not None else None,
            'cup_cam': list(cup_cam) if cup_cam is not None else None,
            'berry_uv': _uv(berry_cam),
            'cup_uv': _uv(cup_cam),
            'optical_uv': [cx, cy],
            'aim_uv': list(self._aim_uv_for_center(
                float(berry_cam[2]) if berry_cam is not None else None)[:2]),
            'image_wh': [w, h],
        }

    def _write_cup_rel_overlay(self, tag: str, geo: Dict) -> None:
        """Annotate wrist RGB: optical center / cup / berry + XY/XZ cup-relative sketch."""
        import cv2  # type: ignore

        if self._qa_rgb_wrist is None or not self._qa_session:
            return
        d = os.path.join(self._args.qa_dir, self._qa_session)
        os.makedirs(d, exist_ok=True)
        img = self._qa_rgb_wrist.copy()
        h, w = img.shape[:2]
        opt = geo.get('optical_uv') or [w * 0.5, h * 0.5]
        ox, oy = float(opt[0]), float(opt[1])
        # Principal point (grey)
        cv2.drawMarker(img, (int(ox), int(oy)), (160, 160, 160),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=1)
        aim_uv = geo.get('aim_uv')
        if aim_uv is not None:
            cv2.drawMarker(
                img, (int(aim_uv[0]), int(aim_uv[1])), (0, 255, 0),
                markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
        cup_uv = geo.get('cup_uv')
        berry_uv = geo.get('berry_uv')
        if cup_uv is not None:
            cu, cv_ = int(cup_uv[0]), int(cup_uv[1])
            cv2.drawMarker(img, (cu, cv_), (0, 220, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
            cv2.circle(img, (cu, cv_), 10, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(img, 'CUP', (cu + 12, cv_ - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        if berry_uv is not None:
            bu, bv = int(berry_uv[0]), int(berry_uv[1])
            cv2.drawMarker(img, (bu, bv), (255, 80, 200),
                           markerType=cv2.MARKER_DIAMOND, markerSize=22, thickness=2)
            cv2.putText(img, 'BERRY', (bu + 12, bv + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 200), 1, cv2.LINE_AA)
        if cup_uv is not None and berry_uv is not None:
            cv2.arrowedLine(
                img, (int(cup_uv[0]), int(cup_uv[1])),
                (int(berry_uv[0]), int(berry_uv[1])),
                (80, 255, 120), 2, tipLength=0.12)
        dist = geo.get('dist_cup')
        rel = geo.get('berry_rel_cup') or [0, 0, 0]
        z_cam = geo.get('z_cam_m')
        travel = geo.get('travel_m')
        depth_src = geo.get('depth_source')
        line1 = (
            f'cup->berry dist={dist:.3f}m  rel=({rel[0]:+.3f},{rel[1]:+.3f},{rel[2]:+.3f})'
            if dist is not None else 'cup->berry ?')
        cv2.putText(
            img, line1,
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 120), 2, cv2.LINE_AA)
        if z_cam is not None or travel is not None or depth_src:
            line2 = (
                f'z_cam={(z_cam if z_cam is not None else -1):.3f}m  '
                f'travel={(travel if travel is not None else -1):.3f}m  '
                f'src={depth_src or "?"}')
            cv2.putText(
                img, line2,
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2, cv2.LINE_AA)
        cv2.putText(
            img, 'grey=optical  cyan=CUP  magenta=BERRY',
            (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        path = os.path.join(d, f'{tag}_cup_overlay.png')
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Schematic: berry in cup frame (cup at origin), XY top + XZ side
        sketch = np.zeros((360, 520, 3), dtype=np.uint8)
        sketch[:] = (28, 32, 40)
        scale = 400.0  # px per meter
        cx0, cy_xy = 130, 180
        cx1, cy_xz = 390, 180

        def _draw_panel(cx: int, cy: int, title: str, px: float, py: float, xlab: str, ylab: str):
            cv2.putText(sketch, title, (cx - 50, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(sketch, (cx - 90, cy), (cx + 90, cy), (60, 70, 90), 1)
            cv2.line(sketch, (cx, cy - 90), (cx, cy + 90), (60, 70, 90), 1)
            cv2.drawMarker(sketch, (cx, cy), (0, 220, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
            cv2.putText(sketch, 'CUP', (cx + 8, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)
            bx = int(cx + px * scale)
            by = int(cy - py * scale)
            bx = max(cx - 95, min(cx + 95, bx))
            by = max(cy - 95, min(cy + 95, by))
            cv2.arrowedLine(sketch, (cx, cy), (bx, by), (80, 255, 120), 2, tipLength=0.15)
            cv2.drawMarker(sketch, (bx, by), (255, 80, 200),
                           markerType=cv2.MARKER_DIAMOND, markerSize=12, thickness=2)
            cv2.putText(sketch, xlab, (cx + 70, cy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
            cv2.putText(sketch, ylab, (cx + 4, cy - 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

        _draw_panel(cx0, cy_xy, 'base XY (top)', rel[0], rel[1], '+X', '+Y')
        _draw_panel(cx1, cy_xz, 'base XZ (side)', rel[0], rel[2], '+X', '+Z')
        cv2.putText(
            sketch,
            f'|r|={dist:.3f}m  r=({rel[0]:+.3f},{rel[1]:+.3f},{rel[2]:+.3f})'
            if dist is not None else '',
            (16, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1, cv2.LINE_AA)
        cv2.imwrite(
            os.path.join(d, f'{tag}_cup_rel.png'),
            cv2.cvtColor(sketch, cv2.COLOR_RGB2BGR))

    def _wrist_image_wh(self) -> Tuple[int, int]:
        if self._qa_rgb_wrist is not None:
            h, w = self._qa_rgb_wrist.shape[:2]
            return int(w), int(h)
        return 640, 480

    def _center_aim_cam_xy(
        self, berry_cam: Tuple[float, float, float],
    ) -> Tuple[float, float, str]:
        """In-FOV aim for mid-range centering (cam xy at berry depth).

        Cup often projects below the 480p frame (~v=700+). Using that aim makes
        pix_err stick ~500–600px and falsely looks 'never centered'. Mid-range
        therefore aims at the optical axis so the berry stays trackable; cup-axis
        alignment happens naturally as we approach after depth is trusted.
        """
        cx, cy, cz = berry_cam
        if cz <= 1e-4:
            return 0.0, 0.0, 'optical'
        cup = self._cup_cam_xyz()
        if cup is not None and cup[2] > 1e-4:
            ax = float(cup[0] * cz / cup[2])
            ay = float(cup[1] * cz / cup[2])
            f = float(self._args.wrist_focal_px)
            w, h = self._wrist_image_wh()
            u = f * ax / cz + 0.5 * w
            v = f * ay / cz + 0.5 * h
            margin = 20.0
            if margin <= u < w - margin and margin <= v < h - margin:
                return ax, ay, 'cup'
        return 0.0, 0.0, 'optical'

    def _servo_image_error_px(
        self, cam_xyz: Tuple[float, float, float],
    ) -> Optional[float]:
        """Pixel error of berry vs in-FOV aim (cup if visible, else optical)."""
        x, y, z = cam_xyz
        if z <= 1e-4:
            return None
        f = float(self._args.wrist_focal_px)
        w, h = self._wrist_image_wh()
        bu = f * x / z + 0.5 * w
        bv = f * y / z + 0.5 * h
        ax, ay, _ = self._center_aim_cam_xy(cam_xyz)
        tu = f * ax / z + 0.5 * w
        tv = f * ay / z + 0.5 * h
        return math.hypot(bu - tu, bv - tv)

    def _cam_angular_errors(
        self, cam_xyz: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        """Angular error of berry relative to in-FOV aim (cup or optical)."""
        cx, cy, cz = cam_xyz
        if cz <= 1e-4:
            return 0.0, 0.0
        ax, ay, _ = self._center_aim_cam_xy(cam_xyz)
        return (
            math.atan2(cx, cz) - math.atan2(ax, cz),
            math.atan2(cy, cz) - math.atan2(ay, cz),
        )

    def _ibvs_reach_scale(
        self,
        pix_err: Optional[float],
        *,
        lost_streak: int = 0,
    ) -> float:
        """Approach scale after centering: 0 while off-center, full when tight."""
        if self._refine_phase == 'center':
            return 0.0
        min_s = float(self._args.servo_ibvs_reach_min_scale)
        min_s = max(0.0, min(1.0, min_s))
        if lost_streak >= 5:
            return 0.5 * min_s
        pix_tol = float(self._args.servo_ibvs_reach_pix_px)
        pix_soft = float(self._args.servo_ibvs_reach_pix_soft_px)
        if pix_err is None:
            return max(min_s, 0.45)
        if pix_err <= pix_tol:
            return 1.0
        if pix_err >= pix_soft:
            return min_s
        t = (pix_err - pix_tol) / max(pix_soft - pix_tol, 1.0)
        return 1.0 - t * (1.0 - min_s)

    def _cam_delta_to_base(
        self, d_cam: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float, float]]:
        T = self._lookup_T_base_cam()
        if T is None:
            return None
        v = T[:3, :3] @ np.array(
            [float(d_cam[0]), float(d_cam[1]), float(d_cam[2])], dtype=np.float64)
        return float(v[0]), float(v[1]), float(v[2])

    def _image_ready_for_near_handoff(
        self,
        pix_err: Optional[float],
        err_yaw: float,
        err_pitch: float,
    ) -> bool:
        """cup + image centered before near handoff."""
        pix_gate = float(self._args.servo_near_pix_tol_px)
        ang_gate = math.radians(float(self._args.servo_near_ang_tol_deg))
        dead = math.radians(float(self._args.servo_ang_deadband_deg))
        if pix_err is not None and pix_err <= pix_gate:
            return True
        if abs(err_yaw) <= max(dead, ang_gate) and abs(err_pitch) <= max(dead, ang_gate):
            return True
        return False

    def _tcp_or_ee_xyz(self) -> Optional[Tuple[float, float, float]]:
        """Suction-cup / TCP position in base (prefer /feedback/tcp_pose)."""
        if self._tcp_pose is not None:
            p = self._tcp_pose.pose.position
            return float(p.x), float(p.y), float(p.z)
        ee = self._current_ee_pose()
        if ee is None:
            return None
        p = ee.pose.position
        return float(p.x), float(p.y), float(p.z)

    def _berry_base_xyz(self, berry: DetectedBerry) -> Optional[Tuple[float, float, float]]:
        """Berry XYZ in base_frame."""
        p = berry.pose.pose.position
        bf = (berry.pose.header.frame_id or berry.header.frame_id
              or self._args.base_frame)
        if bf == self._args.base_frame or bf.endswith('/' + self._args.base_frame):
            return float(p.x), float(p.y), float(p.z)
        try:
            tf = self._tf_buffer.lookup_transform(
                self._args.base_frame, bf, rclpy.time.Time())
        except Exception:
            return float(p.x), float(p.y), float(p.z)
        T = _matrix_from_tf(tf)
        v = T @ np.array([float(p.x), float(p.y), float(p.z), 1.0], dtype=np.float64)
        return float(v[0]), float(v[1]), float(v[2])

    def _remember_berry(self, berry: DetectedBerry, z_cam: Optional[float]) -> None:
        base = self._berry_base_xyz(berry)
        if base is None:
            return
        mode = str(getattr(berry, 'depth_mode', '') or '')
        # Ghost lock after BoT-SORT loss — never update mid-range pose from this.
        if mode == 'base_coast':
            return
        # Pixel-coast without 3D update is still skipped; mono / depth / iou_pin OK.
        if mode in ('coast',) and not self._berry_has_rgbd(berry):
            # Pixel-coast without fresh RGB-D — skip unless depth still valid.
            if not self._berry_depth_trusted(berry):
                self.get_logger().debug(
                    f'REFINING skip remember mode={mode} '
                    f'z_d={float(getattr(berry, "z_depth_m", -1)):.3f} '
                    f'z_m={float(getattr(berry, "z_mono_m", -1)):.3f}')
                return
        # Reject ID-swap jumps while a refine lock is active.
        if self._last_berry_base is not None and self._refine_locked_track_id is not None:
            jump = math.sqrt(
                (base[0] - self._last_berry_base[0]) ** 2
                + (base[1] - self._last_berry_base[1]) ** 2
                + (base[2] - self._last_berry_base[2]) ** 2)
            max_jump = float(getattr(self._args, 'refine_track_max_jump_m', 0.08))
            if jump > max_jump:
                self.get_logger().warn(
                    f'REFINING reject track jump {jump:.3f}m > {max_jump:.3f}m '
                    f'(keep last base; tid={int(berry.track_id)})')
                return
        self._last_berry_base = base
        self._last_berry_t = time.time()
        if z_cam is not None:
            self._last_z_cam = float(z_cam)

    def _cup_berry_dist(self) -> Optional[float]:
        b = self._near_frozen_berry if self._near_mode and self._near_frozen_berry else (
            self._last_berry_base)
        if b is None:
            return None
        tcp = self._tcp_or_ee_xyz()
        if tcp is None:
            return None
        return math.sqrt(
            (b[0] - tcp[0]) ** 2 + (b[1] - tcp[1]) ** 2 + (b[2] - tcp[2]) ** 2)

    def _freeze_near_berry(self) -> None:
        """Pin 3D target for near handoff — no further wrist re-association."""
        if self._near_frozen_berry is not None:
            return
        # Absorb mid-range center residual: rebuild berry on live UV ray first.
        self._refresh_berry_from_live_for_contact(reason='near_freeze')
        if self._last_berry_base is None:
            return
        self._near_frozen_berry = tuple(self._last_berry_base)
        self._near_oneshot_sent = False
        self._near_oneshot_attempts = 0
        b = self._near_frozen_berry
        self.get_logger().info(
            f'REFINING freeze near berry=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f}) '
            f'tid={self._refine_locked_track_id}')

    def _update_berry_from_live_uv(
        self,
        *,
        uv: Tuple[float, float],
        z_cam: Optional[float] = None,
        berry: Optional[DetectedBerry] = None,
        reason: str = '',
    ) -> bool:
        """Replace last_berry_base with cam ray through live UV at z (mono/depth).

        Used after center oneshot residual and before near freeze so contact
        oneshot aims the *seen* berry, not the wrong-depth probe ray.
        """
        fx, fy, cx, cy = self._wrist_intrinsics()
        z: Optional[float] = float(z_cam) if z_cam is not None else None
        if z is None or z <= 1e-4:
            if berry is not None:
                z, _zsrc = self._berry_range_z_m(berry)
                if z is None or z <= 1e-4:
                    cam_b = self._berry_cam_xyz(berry)
                    if cam_b is not None and cam_b[2] > 1e-4:
                        z = float(cam_b[2])
            if (z is None or z <= 1e-4) and self._last_z_cam:
                z = float(self._last_z_cam)
        if z is None or z <= 1e-4:
            return False
        cam = (
            (float(uv[0]) - cx) / fx * float(z),
            (float(uv[1]) - cy) / fy * float(z),
            float(z),
        )
        T = self._lookup_T_base_cam()
        if T is None:
            return False
        p = T[:3, :3] @ np.array(cam, dtype=np.float64) + T[:3, 3]
        base = (float(p[0]), float(p[1]), float(p[2]))
        self._last_berry_base = base
        self._refine_fruit_anchor = base
        self._last_z_cam = float(z)
        self._last_berry_t = time.time()
        # Allow a later freeze to pick up the new base.
        self._near_frozen_berry = None
        self.get_logger().info(
            f'REFINING berry←live_uv ({reason}) '
            f'uv=({float(uv[0]):.0f},{float(uv[1]):.0f}) z={float(z):.3f} '
            f'base=({base[0]:.3f},{base[1]:.3f},{base[2]:.3f})')
        return True

    def _correct_berry_3d_from_fresh_lock(
        self,
        neu: DetectedBerry,
        *,
        reason: str,
    ) -> Tuple[bool, str]:
        """Rebuild berry_base from fresh-lock UV — never trust YOLO pose XYZ alone."""
        uv = self._berry_image_uv(neu)
        if uv is None:
            return False, 'no_uv'
        cam = self._berry_cam_xyz(neu)
        z_est, z_src = self._reestimate_z_on_live_uv(
            uv=(float(uv[0]), float(uv[1])),
            berry=neu,
            cam=cam,
        )
        if z_est is None and cam is not None and cam[2] > 1e-4:
            z_est, z_src = float(cam[2]), 'fresh_cam'
        if z_est is None:
            return False, 'no_z'
        ok = self._update_berry_from_live_uv(
            uv=(float(uv[0]), float(uv[1])),
            z_cam=float(z_est),
            berry=neu,
            reason=f'{reason}_{z_src}',
        )
        return ok, z_src

    def _refresh_berry_from_live_for_contact(self, *, reason: str) -> bool:
        """If a live box is near cup-aim, rebuild berry_base on that UV ray."""
        max_pix = max(
            float(getattr(self._args, 'servo_center_pix_tol_px', 35.0)),
            float(getattr(self._args, 'refine_fresh_lock_max_pix_px', 80.0)),
        )
        fresh_min = float(getattr(self._args, 'refine_fresh_lock_min_conf', 0.10))
        live = self._pick_live_nearest_optical(max_pix=max_pix, min_conf=fresh_min)
        if live is None:
            return False
        uv = self._berry_image_uv(live)
        if uv is None:
            return False
        cam = self._berry_cam_xyz(live)
        z = float(cam[2]) if cam is not None and cam[2] > 1e-4 else None
        return self._update_berry_from_live_uv(
            uv=(float(uv[0]), float(uv[1])),
            z_cam=z,
            berry=live,
            reason=reason,
        )

    def _fixed_mono_yaw_err(self) -> Optional[float]:
        """Plant/berry yaw − j1 from fixed mono global or last estimate."""
        plant = None
        if self._global is not None and self._global.berries:
            plant = max(self._global.berries, key=lambda b: float(b.confidence))
        if plant is not None:
            base = self._berry_base_xyz(plant)
            if base is not None:
                return math.atan2(base[1], base[0]) - float(self._joints[0])
        if self._last_berry_base is not None:
            bx, by, _ = self._last_berry_base
            return math.atan2(by, bx) - float(self._joints[0])
        return None

    def _lookup_T_base_cam(self) -> Optional[np.ndarray]:
        cam = self._args.wrist_camera_frame
        try:
            tf = self._tf_buffer.lookup_transform(
                self._args.base_frame, cam, rclpy.time.Time())
        except Exception:
            return None
        return _matrix_from_tf(tf)

    def _berry_uv_from_cam(
        self, cam_xyz: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float]]:
        if cam_xyz[2] <= 1e-4:
            return None
        fx, fy, cx, cy = self._wrist_intrinsics()
        w, h = self._wrist_image_wh()
        p = _project_cam_point(
            cam_xyz[0], cam_xyz[1], cam_xyz[2], w, h, fx,
            cx=cx, cy=cy, fx=fx, fy=fy)
        if p is None:
            return None
        return float(p[0]), float(p[1])

    def _berry_image_uv(self, berry: DetectedBerry) -> Optional[Tuple[float, float]]:
        """Prefer detector bbox center (image_u/v); fallback to mono cam projection."""
        try:
            iu = float(getattr(berry, 'image_u', -1.0))
            iv = float(getattr(berry, 'image_v', -1.0))
        except (TypeError, ValueError):
            iu, iv = -1.0, -1.0
        if iu >= 0.0 and iv >= 0.0:
            return iu, iv
        cam = self._berry_cam_xyz(berry)
        if cam is None:
            return None
        return self._berry_uv_from_cam(cam)

    def _capture_mono_probe_obs(self) -> Optional[Dict]:
        """One view: cam pose + berry ray UV. Live bbox only — never base_coast."""
        berry = self._pick_probe_live_berry()
        cam = None
        z_mono = None
        mode = ''
        tid = -1
        conf = 0.0
        uv_img: Optional[Tuple[float, float]] = None
        if berry is not None:
            tid = int(berry.track_id)
            conf = float(berry.confidence)
            mode = str(getattr(berry, 'depth_mode', '') or '')
            # Probe views: reject ghost / ID swap. After lock, only need a live
            # box on the locked id — do NOT reuse refine_lock_min_conf (entry gate).
            if mode == 'base_coast':
                return None
            if (
                self._refine_locked_track_id is not None
                and tid != int(self._refine_locked_track_id)
            ):
                self.get_logger().warn(
                    f'REFINING mono probe: skip tid={tid} ≠ lock='
                    f'{self._refine_locked_track_id}')
                return None
            track_min = float(getattr(self._args, 'refine_track_min_conf', 0.12))
            if conf < track_min:
                now = time.time()
                if now - self._refine_wait_log_t > 1.0:
                    self._refine_wait_log_t = now
                    self.get_logger().warn(
                        f'REFINING mono probe: skip low conf={conf:.2f} '
                        f'< track_min={track_min:.2f} (lock gate separate)')
                return None
            cam = self._berry_cam_xyz(berry)
            z_mono = float(getattr(berry, 'z_mono_m', -1.0))
            uv_img = self._berry_image_uv(berry)
            if not self._near_mode and cam is not None and cam[2] > 1e-4:
                self._remember_berry(berry, float(cam[2]))
        if cam is None or cam[2] <= 1e-4:
            return None
        uv = uv_img or self._berry_uv_from_cam(cam)
        if uv is None:
            return None
        # Continuity vs previous probe obs (wrong scene / wrong berry → abort).
        if self._mono_probe_obs:
            prev = self._mono_probe_obs[-1]
            duv = math.hypot(uv[0] - prev['uv'][0], uv[1] - prev['uv'][1])
            max_duv = float(getattr(self._args, 'servo_mono_probe_max_duv_px', 90.0))
            if duv > max_duv:
                self.get_logger().error(
                    f'REFINING mono probe: UV jump {duv:.0f}px > {max_duv:.0f} '
                    f'(prev={prev["uv"]} now={list(uv)}) — likely wrong detection, abort')
                self._mono_probe_done = True
                self._set_state(
                    'ERROR',
                    f'mono probe UV jump {duv:.0f}px — keep camera on plant')
                return None
        T = self._lookup_T_base_cam()
        if T is None:
            return None
        tcp = self._tcp_or_ee_xyz()
        ee = self._current_ee_pose()
        ee_xyz = None
        if ee is not None:
            ee_xyz = [
                float(ee.pose.position.x),
                float(ee.pose.position.y),
                float(ee.pose.position.z),
            ]
        # Explicit mono-ray base point matching this UV+cam (not a stale freeze).
        cam_v = np.array(
            [float(cam[0]), float(cam[1]), float(cam[2])], dtype=np.float64)
        p_base = T[:3, :3] @ cam_v + T[:3, 3]
        berry_base_est = [float(p_base[0]), float(p_base[1]), float(p_base[2])]
        return {
            'uv': [float(uv[0]), float(uv[1])],
            'uv_from_bbox': bool(uv_img is not None),
            'cam_xyz': [float(cam[0]), float(cam[1]), float(cam[2])],
            'z_mono': float(z_mono) if z_mono is not None and z_mono > 0 else float(cam[2]),
            'mode': mode,
            'track_id': tid,
            'confidence': conf,
            'T_base_cam': T.tolist(),
            'tcp_base': list(tcp) if tcp is not None else None,
            'ee_xyz': ee_xyz,
            'berry_base_est': berry_base_est,
            'joints_rad': [float(v) for v in self._joints[:6]],
            't': time.time(),
        }

    def _update_ee_uv_jac_from_delta(
        self,
        uv0,
        T0,
        uv1,
        T1,
        *,
        tag: str = 'probe',
        merge: bool = True,
    ) -> Optional[Dict[str, float]]:
        """Online calib: small-step camera/EE Cartesian move → ΔUV.

        Records how cam-frame translation (dx,dy) moved the berry in the image.
        Center oneshot uses aim_uv_ik (5-DOF UV+depth); Jac remains for QA.
        """
        if uv0 is None or uv1 is None or T0 is None or T1 is None:
            return None
        T0a = np.asarray(T0, dtype=np.float64)
        T1a = np.asarray(T1, dtype=np.float64)
        if T0a.shape != (4, 4) or T1a.shape != (4, 4):
            return None
        R0 = T0a[:3, :3]
        dp_base = T1a[:3, 3] - T0a[:3, 3]
        dp_cam = R0.T @ dp_base
        dx, dy, dz = float(dp_cam[0]), float(dp_cam[1]), float(dp_cam[2])
        du = float(uv1[0]) - float(uv0[0])
        dv = float(uv1[1]) - float(uv0[1])
        min_d = 0.0025
        min_duv = 2.5
        jac: Dict[str, float] = {
            'du': du, 'dv': dv,
            'dx_cam': dx, 'dy_cam': dy, 'dz_cam': dz,
            'dp_base': [float(dp_base[0]), float(dp_base[1]), float(dp_base[2])],
            'uv0': [float(uv0[0]), float(uv0[1])],
            'uv1': [float(uv1[0]), float(uv1[1])],
            'tag': tag,
        }
        if abs(dx) >= min_d and abs(du) >= min_duv:
            jac['du_dx'] = du / dx
        elif abs(dx) >= 0.008:
            jac['du_dx'] = du / dx
        if abs(dy) >= min_d and abs(dv) >= min_duv:
            jac['dv_dy'] = dv / dy
        elif abs(dy) >= 0.008:
            jac['dv_dy'] = dv / dy
        if abs(dx) >= min_d and abs(dv) >= min_duv:
            jac['dv_dx'] = dv / dx
        if abs(dy) >= min_d and abs(du) >= min_duv:
            jac['du_dy'] = du / dy
        # Isotropic fallback from whichever axis responded.
        if 'du_dx' in jac and 'dv_dy' not in jac:
            jac['dv_dy_mag'] = abs(float(jac['du_dx']))
        if 'dv_dy' in jac and 'du_dx' not in jac:
            jac['du_dx_mag'] = abs(float(jac['dv_dy']))

        prev = self._ee_uv_jac if merge and self._ee_uv_jac else {}
        for k in ('du_dx', 'dv_dy', 'du_dx_mag', 'dv_dy_mag', 'dv_dx', 'du_dy'):
            if k not in jac and k in prev:
                jac[k] = prev[k]

        if (
            'du_dx' not in jac and 'dv_dy' not in jac
            and 'du_dx_mag' not in jac and 'dv_dy_mag' not in jac
        ):
            self.get_logger().warn(
                f'REFINING ee-uv jac ({tag}): weak cam response '
                f'dcam=({dx*1000:+.1f},{dy*1000:+.1f},{dz*1000:+.1f})mm '
                f'duv=({du:+.1f},{dv:+.1f})')
            if not prev:
                self._ee_uv_jac = None
                return None
            return self._ee_uv_jac

        self._ee_uv_jac = jac
        self.get_logger().info(
            f'REFINING ee-uv jac from {tag}: '
            f'du/dx={jac.get("du_dx", float("nan")):.0f}px/m '
            f'dv/dy={jac.get("dv_dy", float("nan")):.0f}px/m '
            f'(Δcam=({dx*1000:+.1f},{dy*1000:+.1f})mm Δuv=({du:+.1f},{dv:+.1f})px)')
        self._write_qa_json('ee_uv_jac.json', {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'session': self._qa_session,
            **{k: (list(v) if isinstance(v, list) else v) for k, v in jac.items()},
        })
        return jac

    def _update_ee_uv_jac_from_probe(self) -> Optional[Dict[str, float]]:
        """Calibrate from mono-probe obs0→obs1 camera pose + UV."""
        obs = self._mono_probe_obs
        if len(obs) < 2:
            return None
        a, b = obs[0], obs[-1]
        return self._update_ee_uv_jac_from_delta(
            a.get('uv'), a.get('T_base_cam'),
            b.get('uv'), b.get('T_base_cam'),
            tag='probe', merge=False,
        )

    # Back-compat aliases used by older call sites during transition.
    def _update_lookat_jac_from_probe(self) -> Optional[Dict[str, float]]:
        return self._update_ee_uv_jac_from_probe()

    def _update_lookat_jac_from_delta(self, *args, **kwargs):
        # Old joint-space signature ignored — prefer pose form via finish_center.
        return self._ee_uv_jac

    def _start_mono_probe_step(self) -> bool:
        """Small orbit around locked berry + j1 yaw so wrist keeps facing the plant."""
        if not self._arm.server_is_ready():
            return False
        berry = self._last_berry_base
        tcp = self._tcp_or_ee_xyz()
        ee = self._current_ee_pose()
        T = self._lookup_T_base_cam()
        if berry is None or tcp is None or ee is None or T is None:
            return False
        step = float(self._args.servo_mono_probe_step_m)
        if step < 0.008:
            self.get_logger().warn(
                f'REFINING mono probe: step too small ({step:.4f}m) — finish early')
            self._finish_mono_probe()
            return False

        cam_o = T[:3, 3].copy()
        bx, by, bz = float(berry[0]), float(berry[1]), float(berry[2])
        vx, vy, vz = float(cam_o[0] - bx), float(cam_o[1] - by), float(cam_o[2] - bz)
        r_xy = math.hypot(vx, vy)
        # Orbit about vertical through berry (keeps range ≈const, plant stays in FOV).
        if r_xy >= 0.05:
            alpha = step / r_xy  # rad ≈ arc length / radius
            # Alternate sign by purpose index for baseline diversity.
            if int(len(self._mono_probe_obs)) % 2 == 1:
                alpha = -alpha
            ca, sa = math.cos(alpha), math.sin(alpha)
            vx2 = ca * vx - sa * vy
            vy2 = sa * vx + ca * vy
            cam_new = np.array([bx + vx2, by + vy2, bz + vz], dtype=np.float64)
            d_base = (
                float(cam_new[0] - cam_o[0]),
                float(cam_new[1] - cam_o[1]),
                float(cam_new[2] - cam_o[2]),
            )
            # Only partially yaw-cancel: intentional ΔUV from the same small step
            # is the look-at Jacobian sample (du/dj1, dv/dj5). Full cancel → Δu≈0.
            yaw_frac = float(getattr(self._args, 'servo_mono_probe_yaw_frac', 0.25))
            yaw_frac = max(0.0, min(1.0, yaw_frac))
            dj1 = yaw_frac * alpha
        else:
            # Too close on XY: small approach along cup→berry (still facing fruit).
            dx, dy, dz = bx - tcp[0], by - tcp[1], bz - tcp[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < 1e-4:
                return False
            step = min(step, max(0.0, dist - 0.05))
            if step < 0.008:
                self._finish_mono_probe()
                return False
            d_base = (dx / dist * step, dy / dist * step, dz / dist * step)
            dj1 = 0.0
            alpha = 0.0

        ex = float(ee.pose.position.x)
        ey = float(ee.pose.position.y)
        ez = float(ee.pose.position.z)
        tgt = (ex + d_base[0], ey + d_base[1], ez + d_base[2])
        # Seed IK with yaw already aimed at berry after the orbit.
        seed = list(self._joints)
        seed[0] = float(self._joints[0] + dj1)
        joints = position_ik_keep_orient(tgt, seed)
        if joints is None:
            self.get_logger().warn('REFINING mono probe: keep_orient IK failed')
            return False
        target = list(joints)
        # Explicitly keep yaw facing plant (orbit dj1).
        target[0] = float(self._joints[0] + dj1)
        target = self._clamp_servo_target(target)
        # Limit per-step joint change (still allow meaningful yaw).
        max_dj = [
            math.radians(8.0),
            math.radians(float(self._args.servo_max_dj2_deg)),
            math.radians(float(self._args.servo_max_dj3_deg)),
            math.radians(5.0),
            math.radians(float(self._args.servo_max_dj5_deg)),
            0.0,
        ]
        for i in range(6):
            d = target[i] - self._joints[i]
            d = max(-max_dj[i], min(max_dj[i], d))
            target[i] = float(self._joints[i] + d)

        travel = math.sqrt(d_base[0] ** 2 + d_base[1] ** 2 + d_base[2] ** 2)
        dist = math.sqrt(
            (bx - tcp[0]) ** 2 + (by - tcp[1]) ** 2 + (bz - tcp[2]) ** 2)
        self._servo_pending_record = {
            'step': int(self._servo_step_idx),
            'control': 'mono_probe_orbit',
            'probe_idx': len(self._mono_probe_obs),
            'purpose': str(self._mono_probe_purpose),
            'travel_m': float(travel),
            'orbit_alpha_deg': math.degrees(alpha) if r_xy >= 0.05 else 0.0,
            'dj1_deg': math.degrees(dj1),
            'dist_cup': float(dist),
            'berry_base': list(berry),
            'tcp_base': list(tcp),
            'target_xyz': list(tgt),
            'cmd_dxyz': list(d_base),
            'qa_tag': f'servo_{self._servo_step_idx:02d}_mono_probe',
            'source': 'step_json',
        }
        ok = self._send_joint_servo_goal(
            target,
            tag='servo',
            traj_s=max(float(self._args.servo_traj_s), 2.2),
            extra=(
                f'mono_probe_orbit travel={travel:.3f} α={math.degrees(alpha):+.1f}deg '
                f'dj1={math.degrees(dj1):+.1f}deg lock={self._refine_locked_track_id}'
            ),
        )
        return bool(ok)

    @staticmethod
    def _triangulate_rays(
        o0: np.ndarray, d0: np.ndarray, o1: np.ndarray, d1: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Midpoint of closest approach between two skew rays."""
        d0 = d0 / (np.linalg.norm(d0) + 1e-12)
        d1 = d1 / (np.linalg.norm(d1) + 1e-12)
        w0 = o0 - o1
        a = float(np.dot(d0, d0))
        b = float(np.dot(d0, d1))
        c = float(np.dot(d1, d1))
        d = float(np.dot(d0, w0))
        e = float(np.dot(d1, w0))
        denom = a * c - b * b
        if abs(denom) < 1e-10:
            return None
        t = (b * e - c * d) / denom
        s = (a * e - b * d) / denom
        p0 = o0 + t * d0
        p1 = o1 + s * d1
        return 0.5 * (p0 + p1)

    def _obs_to_ray_base(self, obs: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Back-project probe UV to a base-frame ray using live wrist intrinsics.

        Do NOT assume principal point = image center — Orbbec cy≈213 on 480p,
        which biases short-baseline triangulation depth if ignored.
        """
        T = np.array(obs['T_base_cam'], dtype=np.float64)
        u, v = obs['uv']
        fx, fy, cx, cy = self._wrist_intrinsics()
        ray_cam = np.array(
            [(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0], dtype=np.float64)
        ray_cam = ray_cam / (np.linalg.norm(ray_cam) + 1e-12)
        R = T[:3, :3]
        o = T[:3, 3].copy()
        d = R @ ray_cam
        return o, d

    def _apply_triangulation(self, tag: str) -> bool:
        """Update berry pose from mono_probe_obs. Returns True if triangulated OK."""
        obs = self._mono_probe_obs
        if len(obs) < 2:
            self.get_logger().warn(f'REFINING {tag}: <2 views — keep mono estimate')
            return False
        r0 = self._obs_to_ray_base(obs[0])
        r1 = self._obs_to_ray_base(obs[-1])
        if r0 is None or r1 is None:
            self.get_logger().warn(f'REFINING {tag}: ray build failed — keep mono')
            return False
        o0, d0 = r0
        o1, d1 = r1
        baseline = float(np.linalg.norm(o1 - o0))
        xyz = self._triangulate_rays(o0, d0, o1, d1)
        payload: Dict = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'session': self._qa_session,
            'purpose': self._mono_probe_purpose,
            'n_obs': len(obs),
            'baseline_m': baseline,
            'obs': [
                {k: v for k, v in o.items() if k != 'T_base_cam'} | {
                    'T_base_cam_t': list(np.array(o['T_base_cam'])[:3, 3]),
                    'T_base_cam_R': np.array(o['T_base_cam'], dtype=np.float64)[:3, :3].reshape(-1).tolist(),
                }
                for o in obs
            ],
        }
        ok = xyz is not None and baseline >= 0.008
        reject_reason = ''
        if not ok:
            reject_reason = (
                'xyz_none' if xyz is None else f'baseline_short_{baseline:.4f}m')
        if ok:
            z_tri_chk = float(np.linalg.norm(xyz - np.array(obs[0]['T_base_cam'], dtype=np.float64)[:3, 3]))
            z_mono_chk = float(obs[0].get('z_mono') or 0.0)
            payload['z_tri_from_cam0'] = z_tri_chk
            payload['z_mono_cam0'] = z_mono_chk
            self._center_z_tri_m = float(z_tri_chk)
            self._center_z_mono_m = float(z_mono_chk) if z_mono_chk > 1e-4 else None
            if z_mono_chk > 1e-4:
                payload['scale_tri_over_mono'] = z_tri_chk / z_mono_chk
            if z_tri_chk < 0.08 or z_tri_chk > 1.5:
                self.get_logger().warn(
                    f'REFINING {tag}: z_tri={z_tri_chk:.3f}m out of range — reject')
                ok = False
                reject_reason = f'z_tri_oor_{z_tri_chk:.3f}'
            elif z_mono_chk > 0.05:
                ratio = z_tri_chk / z_mono_chk
                # Short-baseline tri often collapses depth; reject unless close to mono.
                lo, hi = (0.85, 1.15) if str(self._mono_probe_purpose or '') == 'center' else (0.4, 2.5)
                if not (lo <= ratio <= hi):
                    self.get_logger().warn(
                        f'REFINING {tag}: z_tri/z_mono={ratio:.2f} outside '
                        f'[{lo:.2f},{hi:.2f}] — reject')
                    ok = False
                    reject_reason = f'scale_{ratio:.2f}_outside_[{lo:.2f},{hi:.2f}]'
        if not ok:
            self.get_logger().warn(
                f'REFINING {tag}: triangulation weak (baseline={baseline:.4f}m '
                f'reason={reject_reason or "unknown"})')
            payload['rejected'] = True
            payload['reject_reason'] = reject_reason or 'unknown'
            self._write_qa_json(f'{tag}.json', payload)
            return False
        corrected = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        T0 = np.array(obs[0]['T_base_cam'], dtype=np.float64)
        cam_o = T0[:3, 3]
        z_tri = float(np.linalg.norm(xyz - cam_o))
        z_mono = float(obs[0].get('z_mono') or 0.0)
        scale = (z_tri / z_mono) if z_mono > 1e-4 else None
        payload['berry_base_tri'] = list(corrected)
        payload['z_tri_from_cam0'] = z_tri
        payload['scale_tri_over_mono'] = scale
        self._last_berry_base = corrected
        self._refine_fruit_anchor = corrected
        self._last_berry_t = time.time()
        self._near_frozen_berry = None
        cam = self._base_xyz_to_cam(corrected)
        if cam is not None and cam[2] > 1e-4:
            self._last_z_cam = float(cam[2])
        self.get_logger().info(
            f'REFINING {tag} OK: berry=({corrected[0]:.3f},{corrected[1]:.3f},'
            f'{corrected[2]:.3f}) baseline={baseline:.3f}m '
            f'z_tri={z_tri:.3f} scale_tri/mono='
            f'{(scale if scale is not None else -1):.3f}')
        if str(self._mono_probe_purpose or '') == 'center':
            self._center_probe_tri_ok = True
        self._write_qa_json(f'{tag}.json', payload)
        return True

    def _scale_berry_along_cam_ray(
        self,
        berry: Tuple[float, float, float],
        scale: float,
        T_base_cam: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """Move berry along its camera ray (same UV, new depth). scale multiplies cam XYZ."""
        if scale <= 1e-4:
            return None
        T = T_base_cam
        if T is None:
            T = self._lookup_T_base_cam()
        if T is None:
            return None
        Tm = np.array(T, dtype=np.float64)
        try:
            T_inv = np.linalg.inv(Tm)
        except Exception:
            return None
        pb = np.array([berry[0], berry[1], berry[2], 1.0], dtype=np.float64)
        cam = (T_inv @ pb)[:3]
        if float(cam[2]) <= 1e-4:
            return None
        cam_s = cam * float(scale)
        p = Tm[:3, :3] @ cam_s + Tm[:3, 3]
        return (float(p[0]), float(p[1]), float(p[2]))

    def _berry_base_for_center_aim(
        self,
    ) -> Tuple[Optional[Tuple[float, float, float]], str]:
        """3D point for aim_uv_ik: must match the UV we are centering.

        Prefer last probe mono-ray back-project (berry_base_est). Short-baseline
        triangulation is often rejected — keep the mono ray as-is (no soft-scale).
        """
        if self._mono_probe_obs:
            last = self._mono_probe_obs[-1]
            est = last.get('berry_base_est')
            if isinstance(est, (list, tuple)) and len(est) >= 3:
                return (float(est[0]), float(est[1]), float(est[2])), 'probe_mono_ray'
            cam = last.get('cam_xyz')
            T = last.get('T_base_cam')
            if cam is not None and T is not None and float(cam[2]) > 1e-4:
                Tm = np.array(T, dtype=np.float64)
                cam_v = np.array(
                    [float(cam[0]), float(cam[1]), float(cam[2])], dtype=np.float64)
                p = Tm[:3, :3] @ cam_v + Tm[:3, 3]
                return (float(p[0]), float(p[1]), float(p[2])), 'probe_cam_backproject'
        if self._last_berry_base is not None:
            return tuple(self._last_berry_base), 'last_berry_base'
        return None, 'none'

    def _center_plan_uv_from_calib(
        self,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[List[float]], str]:
        """Open-loop center plan uses last calib probe UV — not tracking/coast."""
        if self._mono_probe_obs:
            last = self._mono_probe_obs[-1]
            uv = last.get('uv')
            c = last.get('cam_xyz')
            if uv is not None and len(uv) >= 2:
                cam = None
                if c is not None and len(c) >= 3 and float(c[2]) > 1e-4:
                    cam = (float(c[0]), float(c[1]), float(c[2]))
                return cam, [float(uv[0]), float(uv[1])], 'probe_calib'
        return None, None, 'none'

    def _center_oneshot_measure(
        self,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[List[float]], str]:
        """Read-only berry UV for QA. Never re-pin lock (fresh lock comes later)."""
        berry = None
        if self._fine is not None and self._fine.berries and self._fine_is_fresh():
            live = self._pick_live_nearest_optical(
                max_pix=float(getattr(self._args, 'refine_fresh_lock_max_pix_px', 120.0)))
            if live is not None:
                berry = live
            else:
                lock_tid = self._refine_locked_track_id
                for b in self._fine.berries:
                    if lock_tid is not None and int(b.track_id) == int(lock_tid):
                        mode = str(getattr(b, 'depth_mode', '') or '')
                        if mode != 'base_coast':
                            berry = b
                        break
        if berry is not None:
            mode = str(getattr(berry, 'depth_mode', '') or '')
            uv = self._berry_image_uv(berry)
            cam = self._berry_cam_xyz(berry)
            if uv is not None:
                if cam is None or cam[2] <= 1e-4:
                    fx, fy, cx, cy = self._wrist_intrinsics()
                    z, _ = self._berry_range_z_m(berry)
                    if (z is None or z <= 1e-4) and self._last_z_cam:
                        z = float(self._last_z_cam)
                    if z > 1e-4:
                        cam = (
                            (float(uv[0]) - cx) / fx * z,
                            (float(uv[1]) - cy) / fy * z,
                            z,
                        )
                src = 'live_bbox' if mode != 'base_coast' else 'coast_bbox'
                if src == 'coast_bbox' and (uv[0] < 0 or uv[1] < 0):
                    pass
                else:
                    return cam, [float(uv[0]), float(uv[1])], src
            if mode != 'base_coast' and cam is not None and cam[2] > 1e-4:
                uv2 = self._berry_uv_from_cam(cam)
                return cam, ([float(uv2[0]), float(uv2[1])] if uv2 else None), 'live'
        if int(getattr(self, '_center_oneshot_attempts', 0)) >= 1:
            return None, None, 'none'
        if self._mono_probe_obs:
            last = self._mono_probe_obs[-1]
            c = last.get('cam_xyz')
            uv = last.get('uv')
            cam = None
            if c is not None and len(c) >= 3 and float(c[2]) > 1e-4:
                cam = (float(c[0]), float(c[1]), float(c[2]))
            uvl = [float(uv[0]), float(uv[1])] if uv is not None and len(uv) >= 2 else None
            if cam is not None or uvl is not None:
                return cam, uvl, 'probe_obs'
        if self._last_berry_base is not None:
            cam = self._base_xyz_to_cam(self._last_berry_base)
            if cam is not None and cam[2] > 1e-4:
                uv = self._berry_uv_from_cam(cam)
                return cam, ([float(uv[0]), float(uv[1])] if uv else None), 'reproject'
        return None, None, 'none'

    def _lookup_T_link6_cam(self) -> Optional[np.ndarray]:
        """Fixed mount: camera pose in link6 (for 5-DOF aim IK)."""
        cam = self._args.wrist_camera_frame
        ee = self._args.ee_link
        try:
            tf = self._tf_buffer.lookup_transform(
                ee, cam, rclpy.time.Time())
        except Exception:
            return None
        return _matrix_from_tf(tf)

    def _start_center_oneshot(self) -> bool:
        """After probe calib: one 5-DOF aim_uv_ik to cup-axis (or optical) UV.

        Endpoint is image constraint + soft depth — not a free SE(3) translate.
        """
        if not self._arm.server_is_ready():
            return False
        max_att = max(1, int(getattr(self._args, 'servo_center_oneshot_max', 1)))
        if int(getattr(self, '_center_oneshot_attempts', 0)) >= max_att:
            self.get_logger().info(
                f'REFINING center oneshot: already sent ({max_att})')
            return False

        cam, uv, cam_src = self._center_plan_uv_from_calib()
        if uv is None:
            self.get_logger().warn('REFINING center oneshot: no calib probe UV')
            return False
        u, v = float(uv[0]), float(uv[1])
        berry, berry_src = self._berry_base_for_center_aim()
        if berry is None:
            self.get_logger().warn('REFINING center oneshot: no berry_base')
            return False
        # Depth for cup-aim UV + IK from probe mono ray / cam.
        z_aim = float(cam[2]) if cam is not None and cam[2] > 1e-4 else float(
            self._last_z_cam or 0.35)
        if self._mono_probe_obs and self._mono_probe_obs[-1].get('T_base_cam') is not None:
            T_obs = np.array(self._mono_probe_obs[-1]['T_base_cam'], dtype=np.float64)
            try:
                pb = np.array([berry[0], berry[1], berry[2], 1.0], dtype=np.float64)
                cam_b = (np.linalg.inv(T_obs) @ pb)[:3]
                if float(cam_b[2]) > 1e-4:
                    z_aim = float(cam_b[2])
            except Exception:
                pass
        self._last_z_cam = float(z_aim)
        cx_img, cy_img, aim_src = self._aim_uv_for_center(z_aim)
        du_des = cx_img - u
        dv_des = cy_img - v
        pix_off = math.hypot(du_des, dv_des)
        pix_tol = float(getattr(self._args, 'servo_center_pix_tol_px', 35.0))
        if pix_off <= pix_tol:
            self.get_logger().info(
                f'REFINING center oneshot skip: probe already near {aim_src} '
                f'pix_off={pix_off:.1f}')
            return False

        # Keep freeze consistent with the point aim_uv_ik uses (for after reproject).
        self._last_berry_base = berry
        self._last_berry_t = time.time()
        T_l6c = self._lookup_T_link6_cam()
        if T_l6c is None:
            self.get_logger().warn('REFINING center oneshot: no T_link6_cam')
            return False

        w, h = self._wrist_image_wh()
        fx, fy, cx0, cy0 = self._wrist_intrinsics()
        seed = list(self._joints)
        tol_pix = max(8.0, pix_tol * 0.5)
        joints = aim_uv_ik(
            berry,
            (cx_img, cy_img),
            seed,
            T_l6c,
            focal_px=0.5 * (fx + fy),
            fx=fx,
            fy=fy,
            cx=cx0,
            cy=cy0,
            image_wh=(float(w), float(h)),
            z_ref=z_aim,
            tol_pix=tol_pix,
        )
        if joints is None:
            self.get_logger().error(
                f'REFINING center oneshot: aim_uv_ik failed '
                f'uv=({u:.1f},{v:.1f})→({cx_img:.0f},{cy_img:.0f}) '
                f'aim={aim_src} berry_src={berry_src}')
            self._set_state(
                'ERROR',
                f'center aim_uv_ik failed ({aim_src} '
                f'pix_off={pix_off:.0f})')
            return False

        target = self._clamp_servo_target(list(joints))
        # Approximate travel for traj timing / QA.
        ee = self._current_ee_pose()
        if ee is None:
            return False
        ee0 = np.array([
            float(ee.pose.position.x),
            float(ee.pose.position.y),
            float(ee.pose.position.z),
        ], dtype=np.float64)
        ee1 = fk_xyz(target)
        d_base = ee1 - ee0
        lat = float(np.linalg.norm(d_base))
        traj_s = max(
            float(getattr(self._args, 'servo_center_oneshot_traj_s', 2.0)),
            min(8.0, 1.5 + lat / 0.03),
        )
        tcp = self._tcp_or_ee_xyz()
        cup = self._cup_cam_xyz()
        T_bc = self._lookup_T_base_cam()
        d_cam = None
        if T_bc is not None:
            d_cam = (T_bc[:3, :3].T @ d_base).tolist()
        method = f'aim_uv_ik/{aim_src}'
        self._servo_pending_record = {
            'step': int(self._servo_step_idx),
            'track_id': self._refine_locked_track_id,
            'refine_phase': 'center_oneshot',
            'berry_base': list(berry),
            'berry_base_src': berry_src,
            'tcp_base': list(tcp) if tcp else None,
            'ee_xyz_before': ee0.tolist(),
            'ee_xyz_cmd': ee1.tolist(),
            'cmd_dcam_m': d_cam,
            'cmd_d_base_m': d_base.tolist(),
            'berry_cam': list(cam) if cam else None,
            'cup_cam': list(cup) if cup is not None else None,
            'cam_src': cam_src,
            'aim_src': aim_src,
            'method': method,
            'pix_off': float(pix_off),
            'du_des': float(du_des),
            'dv_des': float(dv_des),
            'control': 'center_oneshot_cart',
            'berry_uv': [u, v],
            'aim_uv': [cx_img, cy_img],
            'optical_uv': [cx0, cy0],
            'attempt': 1,
            'joints_cmd_rad': [float(v) for v in target],
            'z_aim_m': float(z_aim),
            'qa_tag': f'servo_{self._servo_step_idx:02d}_center_oneshot',
            'source': 'step_json',
        }
        self._refine_phase = 'center_oneshot'
        # Drop probe-era lock so post-center can re-pin on a fresh YOLO box.
        self._clear_target_lock()
        self._center_oneshot_sent = True
        self._center_last_cmd = {
            'joints_cmd_rad': [float(v) for v in target],
            'ee_xyz_cmd': ee1.tolist(),
            'aim_uv': [cx_img, cy_img],
            'berry_uv': [u, v],
            'berry_base': list(berry),
            'z_aim_m': float(z_aim),
            'ee_xyz_before': ee0.tolist(),
            'tcp_before': list(tcp) if tcp else None,
            'cmd_d_base_m': d_base.tolist(),
            'berry_src': berry_src,
            'cam_src': cam_src,
            'aim_src': aim_src,
            'pix_off': float(pix_off),
        }
        self._write_center_climb_diag(
            phase='before_center',
            berry=berry,
            berry_src=berry_src,
            tcp=tcp,
            ee=ee0.tolist(),
            ee_cmd=ee1.tolist(),
            z_aim=float(z_aim),
            aim_uv=(cx_img, cy_img),
            berry_uv=(u, v),
            cmd_d_base=d_base.tolist(),
            extra={
                'cam_src': cam_src,
                'aim_src': aim_src,
                'pix_off': float(pix_off),
                'method': method,
            },
        )
        ok = self._send_joint_servo_goal(
            target,
            tag='center',
            traj_s=traj_s,
            extra=(
                f'center_arrive method={method} pix_off={pix_off:.1f} '
                f'uv=({u:.1f},{v:.1f})→({cx_img:.0f},{cy_img:.0f}) '
                f'dbase=({d_base[0]*1000:+.1f},{d_base[1]*1000:+.1f},'
                f'{d_base[2]*1000:+.1f})mm tip_above_berry_z='
                f'{((tcp[2] - berry[2]) * 1000.0) if tcp is not None else float("nan"):+.1f}mm '
                f'from={cam_src}'
            ),
        )
        if ok:
            self._center_oneshot_attempts = int(
                getattr(self, '_center_oneshot_attempts', 0)) + 1
            self.get_logger().info(
                f'REFINING center arrive[{self._servo_step_idx}]: method={method} '
                f'pix_off={pix_off:.1f} uv=({u:.1f},{v:.1f})→({cx_img:.0f},{cy_img:.0f}) '
                f'dbase=({d_base[0]*1000:+.1f},{d_base[1]*1000:+.1f},'
                f'{d_base[2]*1000:+.1f})mm from={cam_src} berry={berry_src}')
        else:
            self._servo_pending_record = None
            self._center_oneshot_sent = False
        return ok

    def _write_center_climb_diag(
        self,
        *,
        phase: str,
        berry: Optional[Tuple[float, float, float]] = None,
        berry_src: str = '',
        tcp: Optional[object] = None,
        ee: Optional[object] = None,
        ee_cmd: Optional[object] = None,
        z_aim: Optional[float] = None,
        aim_uv: Optional[Tuple[float, float]] = None,
        berry_uv: Optional[Tuple[float, float]] = None,
        cmd_d_base: Optional[object] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """Height / climb QA: tip vs berry Z in base — explains fixed-cam 'too high'.

        Wrist reproject on fruit + tip high on fixed cam usually means mono depth
        too deep → aim_uv_ik plans a large +Δz to put that 3D point on the cup axis.
        """
        import cv2  # type: ignore

        # Fresh fixed/wrist frames before climb overlay.
        self._snap_qa(f'center_climb_{phase}')
        if berry is None:
            berry = self._last_berry_base
        if tcp is None:
            tcp = self._tcp_or_ee_xyz()
        if ee is None:
            pose = self._current_ee_pose()
            if pose is not None:
                ee = [
                    float(pose.pose.position.x),
                    float(pose.pose.position.y),
                    float(pose.pose.position.z),
                ]
        tip = tcp if tcp is not None else ee
        tip_above_z_mm = None
        tip_above_xy_mm = None
        if berry is not None and tip is not None:
            tip_above_z_mm = (float(tip[2]) - float(berry[2])) * 1000.0
            tip_above_xy_mm = math.hypot(
                float(tip[0]) - float(berry[0]),
                float(tip[1]) - float(berry[1]),
            ) * 1000.0
        geo = None
        if berry is not None and tip is not None:
            geo = self._cup_berry_geometry(
                (float(berry[0]), float(berry[1]), float(berry[2])),
                (float(tip[0]), float(tip[1]), float(tip[2])),
            )
        last_cmd = getattr(self, '_center_last_cmd', None) or {}
        if ee_cmd is None and last_cmd.get('ee_xyz_cmd') is not None:
            ee_cmd = list(last_cmd['ee_xyz_cmd'])
        if cmd_d_base is None and last_cmd.get('cmd_d_base_m') is not None:
            cmd_d_base = list(last_cmd['cmd_d_base_m'])
        if z_aim is None and last_cmd.get('z_aim_m') is not None:
            z_aim = float(last_cmd['z_aim_m'])
        ee_err_mm = None
        if ee_cmd is not None and ee is not None:
            ee_err_mm = float(np.linalg.norm(
                np.array(ee, dtype=np.float64)
                - np.array(ee_cmd, dtype=np.float64)) * 1000.0)
        planned_dz_mm = (
            float(cmd_d_base[2]) * 1000.0
            if cmd_d_base is not None and len(cmd_d_base) >= 3 else None)
        payload: Dict = {
            'phase': phase,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'session': self._qa_session,
            'berry_base': list(berry) if berry is not None else None,
            'berry_src': berry_src or last_cmd.get('berry_src'),
            'tcp_base': list(tcp) if tcp is not None else None,
            'ee_xyz': list(ee) if ee is not None else None,
            'ee_xyz_cmd': list(ee_cmd) if ee_cmd is not None else None,
            'ee_err_mm': ee_err_mm,
            'cmd_d_base_m': list(cmd_d_base) if cmd_d_base is not None else None,
            'planned_dz_mm': planned_dz_mm,
            'tip_above_berry_z_mm': tip_above_z_mm,
            'tip_above_berry_xy_mm': tip_above_xy_mm,
            'z_aim_m': z_aim,
            'aim_uv': list(aim_uv) if aim_uv is not None else last_cmd.get('aim_uv'),
            'berry_uv': list(berry_uv) if berry_uv is not None else last_cmd.get('berry_uv'),
            'z_cam_m': self._last_z_cam,
            'dist_cup_m': geo.get('dist_cup') if geo else None,
            'berry_rel_cup': geo.get('berry_rel_cup') if geo else None,
            'note': (
                'tip_above_berry_z>0 means tip higher than model berry in base Z; '
                'large planned_dz with wrist reproject-on-fruit → mono depth too deep '
                '(IK climbs), not joint tracking shortfall'
            ),
        }
        if extra:
            payload.update(extra)
        tag = f'center_climb_{phase}'
        self._write_qa_json(f'{tag}.json', payload)
        self.get_logger().info(
            f'REFINING climb[{phase}]: tip_above_berry_z='
            f'{(tip_above_z_mm if tip_above_z_mm is not None else float("nan")):+.1f}mm '
            f'xy={(tip_above_xy_mm if tip_above_xy_mm is not None else float("nan")):.1f}mm '
            f'planned_dz={(planned_dz_mm if planned_dz_mm is not None else float("nan")):+.1f}mm '
            f'ee_err={(ee_err_mm if ee_err_mm is not None else float("nan")):.2f}mm '
            f'z_aim={(z_aim if z_aim is not None else -1):.3f}m '
)

        # Annotate fixed cam with climb numbers (side view of tip height).
        if self._qa_rgb_fixed is not None and self._qa_session:
            img = self._qa_rgb_fixed.copy()
            h, _w = img.shape[:2]
            lines = [
                f'CLIMB {phase}',
                f'tip_above_berry_z='
                f'{(tip_above_z_mm if tip_above_z_mm is not None else float("nan")):+.1f}mm',
                f'planned_dz='
                f'{(planned_dz_mm if planned_dz_mm is not None else float("nan")):+.1f}mm',
                f'|EE_fb-cmd|='
                f'{(ee_err_mm if ee_err_mm is not None else float("nan")):.2f}mm',
                f'z_aim={(z_aim if z_aim is not None else -1):.3f}m '
            ]
            y = 28
            for line in lines:
                cv2.putText(
                    img, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2, cv2.LINE_AA)
                y += 26
            d = os.path.join(self._args.qa_dir, self._qa_session)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(
                os.path.join(d, f'{tag}_fixed.png'),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _refresh_berry_cam_from_tri(self) -> None:
        """Refresh z_cam / berry age from frozen base_link triangulation (post-center)."""
        if self._last_berry_base is None:
            return
        cam = self._base_xyz_to_cam(self._last_berry_base)
        if cam is not None and cam[2] > 1e-4:
            self._last_z_cam = float(cam[2])
        self._last_berry_t = time.time()

    def _reproject_berry_geo(self) -> Optional[Dict]:
        """Cup/berry geometry from probe tri 3D + current TCP (no YOLO)."""
        if self._last_berry_base is None:
            return None
        tcp = self._tcp_or_ee_xyz()
        if tcp is None:
            return None
        return self._cup_berry_geometry(self._last_berry_base, tcp)

    def _reproject_pix_off(self) -> Optional[float]:
        """Optical-center error of probe-tri 3D reprojected to current wrist view."""
        geo = self._reproject_berry_geo()
        if geo is None:
            return None
        berry_uv = geo.get('berry_uv')
        optical = geo.get('optical_uv')
        if berry_uv is None or optical is None:
            return None
        return math.hypot(
            float(berry_uv[0]) - float(optical[0]),
            float(berry_uv[1]) - float(optical[1]),
        )

    def _apply_probe_tri_mono_chord_scale(self) -> Optional[str]:
        """Legacy: chord-scale tri depth via min-mono (disabled when RGB-D trusted)."""
        if self._probe_tri_chord_applied:
            return 'probe_tri_mono_chord'
        if not bool(getattr(self._args, 'refine_probe_tri_mono_chord', False)):
            return None
        live = self._pick_probe_live_berry()
        if live is not None and self._berry_depth_trusted(live):
            self.get_logger().info(
                'REFINING probe tri mono chord: skip — wrist depth_raw trusted')
            return None
        if not self._center_probe_tri_ok or self._last_berry_base is None:
            return None
        self._refresh_berry_cam_from_tri()
        tcp = self._tcp_or_ee_xyz()
        z_cam = self._last_z_cam
        if tcp is None or z_cam is None or z_cam <= 1e-4:
            return None
        geo = self._reproject_berry_geo()
        if geo is None or geo.get('berry_uv') is None:
            return None
        berry_uv = geo['berry_uv']
        max_pix = float(getattr(self._args, 'refine_reproject_mono_max_pix_px', 120.0))
        berry_d = float(getattr(self._args, 'berry_diameter_m', 0.015))
        focal = float(self._args.wrist_focal_px)
        scale_min = float(getattr(self._args, 'refine_mono_chord_scale_min', 0.45))
        scale_max = float(getattr(self._args, 'refine_mono_chord_scale_max', 1.05))

        candidates: List[Tuple[float, Dict]] = []

        rgb_sources: List[Tuple[str, object]] = []
        if self._qa_rgb_wrist is not None:
            rgb_sources.append(('wrist', self._qa_rgb_wrist))
        if self._qa_rgb_fine_viz is not None:
            rgb_sources.append(('fine_viz', self._qa_rgb_fine_viz))
        if rgb_sources:
            zm, meta = z_mono_near_reproject_from_sources(
                rgb_sources,
                float(berry_uv[0]),
                float(berry_uv[1]),
                focal_px=focal,
                berry_diameter_m=berry_d,
                max_pick_pix=max_pix,
            )
            if zm is not None and zm > 1e-4:
                meta['source'] = meta.get('rgb_source', 'yolo_near_uv')
                candidates.append((float(zm), meta))

        live = self._pick_live_near_uv(berry_uv)
        if live is not None:
            uv = self._berry_image_uv(live)
            zm = float(getattr(live, 'z_mono_m', -1.0))
            if (
                uv is not None
                and zm > 1e-4
                and math.hypot(uv[0] - float(berry_uv[0]), uv[1] - float(berry_uv[1]))
                <= max_pix
            ):
                candidates.append((zm, {
                    'source': 'live_z_mono',
                    'pix_off_px': math.hypot(
                        uv[0] - float(berry_uv[0]), uv[1] - float(berry_uv[1])),
                    'yolo_conf': float(live.confidence),
                }))

        relax_pix = max(max_pix, 200.0)
        ranked_live: List[Tuple[float, float, float]] = []
        for b in self._live_fine_berries():
            uv = self._berry_image_uv(b)
            zm = float(getattr(b, 'z_mono_m', -1.0))
            if uv is None or zm <= 1e-4:
                continue
            d = math.hypot(uv[0] - float(berry_uv[0]), uv[1] - float(berry_uv[1]))
            if d <= relax_pix:
                ranked_live.append((d, zm, float(b.confidence)))
        if ranked_live:
            ranked_live.sort(key=lambda x: x[0])
            d, zm, conf = ranked_live[0]
            candidates.append((zm, {
                'source': 'live_z_mono_relaxed',
                'pix_off_px': d,
                'yolo_conf': conf,
            }))

        if self._mono_probe_obs:
            zm = float(self._mono_probe_obs[-1].get('z_mono') or 0.0)
            if zm > 1e-4:
                candidates.append((zm, {
                    'source': 'probe_obs_z_mono',
                    'z_mono_m': zm,
                    'n_obs': len(self._mono_probe_obs),
                }))

        if not candidates:
            self.get_logger().warn(
                'REFINING probe tri mono chord: no z_mono candidates — keep tri')
            return None

        reject_log: List[str] = []
        for z_mono, mono_meta in candidates:
            berry_new, scale_meta = apply_mono_chord_scale(
                self._last_berry_base,
                tcp,
                z_cam_m=float(z_cam),
                z_mono_m=float(z_mono),
                scale_min=scale_min,
                scale_max=scale_max,
            )
            if berry_new is None:
                reject_log.append(
                    f'{mono_meta.get("source", "?")}: {scale_meta.get("rejected", "?")} '
                    f'scale={scale_meta.get("scale", -1):.3f}')
                continue
            old = self._last_berry_base
            self._last_berry_base = berry_new
            self._refine_fruit_anchor = berry_new
            self._near_frozen_berry = None
            self._refresh_berry_cam_from_tri()
            self._last_mono_chord_meta = mono_meta | scale_meta
            self._probe_tri_chord_applied = True
            dist = self._cup_berry_dist()
            self.get_logger().info(
                f'REFINING probe tri mono chord: '
                f'berry=({old[0]:.3f},{old[1]:.3f},{old[2]:.3f})→'
                f'({berry_new[0]:.3f},{berry_new[1]:.3f},{berry_new[2]:.3f}) '
                f'scale={scale_meta.get("scale", -1):.3f} '
                f'z_mono={z_mono:.3f} z_cam={z_cam:.3f} '
                f'dist_cup={(dist if dist is not None else -1):.3f}m '
                f'src={mono_meta.get("source", "?")}')
            return 'probe_tri_mono_chord'

        self.get_logger().warn(
            f'REFINING probe tri mono chord rejected all candidates: {"; ".join(reject_log)}')
        return None

    def _contact_from_probe_tri(self, *, reason: str) -> bool:
        """YOLO miss after center: probe tri 3D (+ optional mono chord) → contact oneshot."""
        if self._near_handoff_written or self._state == 'REFINING_WAIT_NEAR':
            return True
        if not self._center_probe_tri_ok or self._last_berry_base is None:
            return False
        self._apply_probe_tri_mono_chord_scale()
        self._refresh_berry_cam_from_tri()
        b = self._last_berry_base
        pix = self._reproject_pix_off()
        self.get_logger().info(
            f'REFINING contact from probe tri ({reason}): '
            f'berry=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f}) '
            f'z_cam={(self._last_z_cam if self._last_z_cam is not None else -1):.3f} '
            f'reproject_pix_off={(pix if pix is not None else -1):.1f}')
        ds = (
            'probe_tri_mono_chord' if self._probe_tri_chord_applied
            else 'probe_tri_reproject')
        self._snap_refine_lock_viz('probe_tri_contact', source=ds)
        self._enter_contact_oneshot(reason=reason)
        return True

    def _snap_center_oneshot_arrive_analysis(self, *, reason: str = '') -> str:
        """Always dump after-center wrist analysis: AIM / REPROJECT / LIVE.

        Reproject of frozen berry_base landing on aim only proves IK consistency
        in the model — NOT that the real fruit is there. When YOLO misses,
        berry_uv_after is often that same reproject (circular). This frame must
        still be written on ERROR so failures stay diagnosable.
        """
        import cv2  # type: ignore

        tag = 'center_oneshot_arrive'
        self._snap_qa(tag)
        if not self._qa_session or self._qa_rgb_wrist is None:
            return ''
        sess_dir = os.path.join(self._args.qa_dir, self._qa_session)
        os.makedirs(sess_dir, exist_ok=True)

        z_guess = float(self._last_z_cam or 0.35)
        au, av, aim_src = self._aim_uv_for_center(z_guess)
        fx, fy, cx, cy = self._wrist_intrinsics()
        geo = self._reproject_berry_geo()
        reproject_uv = None
        if geo is not None and geo.get('berry_uv') is not None:
            reproject_uv = [
                float(geo['berry_uv'][0]), float(geo['berry_uv'][1])]
            if geo.get('berry_cam') is not None and float(geo['berry_cam'][2]) > 1e-4:
                z_guess = float(geo['berry_cam'][2])
                au, av, aim_src = self._aim_uv_for_center(z_guess)

        live_max = max(
            float(getattr(self._args, 'servo_center_pix_tol_px', 35.0)),
            float(getattr(self._args, 'refine_fresh_lock_max_pix_px', 80.0)),
        )
        _cam_live, live_uv, live_src = self._center_arrive_measure_near_aim(
            aim_uv=(au, av), max_pix=live_max)
        # QA-only: also record optical-nearest live (may differ from cup-aim gate).
        _cam_qa, uv_qa, src_qa = self._center_oneshot_measure()

        img = self._qa_rgb_wrist.copy()
        h, w = img.shape[:2]
        # optical / principal
        cv2.drawMarker(
            img, (int(cx), int(cy)), (160, 160, 160),
            markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=1)
        cv2.putText(
            img, 'PRINCIPAL', (int(cx) + 8, int(cy) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
        # AIM
        cv2.drawMarker(
            img, (int(au), int(av)), (0, 255, 0),
            markerType=cv2.MARKER_CROSS, markerSize=26, thickness=2)
        cv2.putText(
            img, f'AIM/{aim_src}', (int(au) + 10, int(av) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        # REPROJECT (model berry_base — circular if no live)
        if reproject_uv is not None:
            ru, rv = int(reproject_uv[0]), int(reproject_uv[1])
            cv2.drawMarker(
                img, (ru, rv), (255, 80, 200),
                markerType=cv2.MARKER_STAR, markerSize=24, thickness=2)
            cv2.putText(
                img, 'REPROJECT(model)', (ru + 10, rv + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 200), 2, cv2.LINE_AA)
            cv2.arrowedLine(
                img, (int(au), int(av)), (ru, rv),
                (255, 80, 200), 1, tipLength=0.15)
        # All live YOLO
        live_all: List[Dict] = []
        for b in self._live_fine_berries():
            uv_b = self._berry_image_uv(b)
            if uv_b is None:
                continue
            live_all.append({
                'track_id': int(b.track_id),
                'u': float(uv_b[0]), 'v': float(uv_b[1]),
                'conf': float(b.confidence),
                'mode': str(getattr(b, 'depth_mode', '') or ''),
            })
            cv2.circle(
                img, (int(uv_b[0]), int(uv_b[1])), 14, (255, 200, 0), 2, cv2.LINE_AA)
        # LIVE near cup-aim (control-relevant)
        if live_uv is not None:
            lu, lv = int(live_uv[0]), int(live_uv[1])
            cv2.drawMarker(
                img, (lu, lv), (80, 255, 255),
                markerType=cv2.MARKER_DIAMOND, markerSize=28, thickness=2)
            cv2.putText(
                img, f'LIVE/{live_src}', (lu + 10, lv - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 255), 2, cv2.LINE_AA)
            cv2.arrowedLine(
                img, (int(au), int(av)), (lu, lv),
                (80, 255, 255), 2, tipLength=0.15)

        pix_repro = (
            math.hypot(reproject_uv[0] - au, reproject_uv[1] - av)
            if reproject_uv is not None else None)
        pix_live = (
            math.hypot(float(live_uv[0]) - au, float(live_uv[1]) - av)
            if live_uv is not None else None)
        joints_fb = [float(v) for v in self._joints[:6]]
        ee_fb = None
        try:
            ee_fk = fk_xyz(joints_fb)
            ee_fb = [float(ee_fk[0]), float(ee_fk[1]), float(ee_fk[2])]
        except Exception:
            ee_now = self._current_ee_pose()
            if ee_now is not None:
                ee_fb = [
                    float(ee_now.pose.position.x),
                    float(ee_now.pose.position.y),
                    float(ee_now.pose.position.z),
                ]

        circular = live_uv is None
        last_cmd = getattr(self, '_center_last_cmd', None) or {}
        ee_cmd = (
            [float(v) for v in last_cmd['ee_xyz_cmd']]
            if last_cmd.get('ee_xyz_cmd') is not None else None)
        joints_cmd = (
            [float(v) for v in last_cmd['joints_cmd_rad']]
            if last_cmd.get('joints_cmd_rad') is not None else None)
        ee_err_mm = None
        if ee_cmd is not None and ee_fb is not None:
            ee_err_mm = float(np.linalg.norm(
                np.array(ee_fb, dtype=np.float64)
                - np.array(ee_cmd, dtype=np.float64)) * 1000.0)
        joint_err_deg = None
        if joints_cmd is not None and len(joints_fb) >= 6:
            joint_err_deg = [
                math.degrees(float(joints_fb[i]) - float(joints_cmd[i]))
                for i in range(6)
            ]

        line0 = (
            f'CENTER ARRIVE  reason={reason or "finish"}  '
            f'circular_reproject={"YES" if circular else "no"}  '
            f'|EE_fb-cmd|={(ee_err_mm if ee_err_mm is not None else -1):.2f}mm')
        cv2.putText(
            img, line0, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(
            img,
            f'aim=({au:.0f},{av:.0f}) repro_pix='
            f'{(pix_repro if pix_repro is not None else -1):.1f} '
            f'live_pix={(pix_live if pix_live is not None else -1):.1f} '
            f'n_yolo={len(live_all)}',
            (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 100), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            'green=AIM  magenta=REPROJECT(model)  cyan=LIVE@aim  yellow=all YOLO  '
            'grey=principal',
            (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        if circular:
            cv2.putText(
                img,
                'WARN: no LIVE near aim — reproject~aim only proves IK model, '
                'not real fruit',
                (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 80), 2, cv2.LINE_AA)

        out_path = os.path.join(sess_dir, f'{tag}_annotated.png')
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # Keep fixed-cam snap next to it for climb diagnosis.
        if self._qa_rgb_fixed is not None:
            cv2.imwrite(
                os.path.join(sess_dir, f'{tag}_fixed.png'),
                cv2.cvtColor(self._qa_rgb_fixed, cv2.COLOR_RGB2BGR))
        if self._qa_rgb_fine_viz is not None:
            cv2.imwrite(
                os.path.join(sess_dir, f'{tag}_fine_viz.png'),
                cv2.cvtColor(self._qa_rgb_fine_viz, cv2.COLOR_RGB2BGR))

        meta = {
            'reason': reason or 'finish',
            'aim_uv': [float(au), float(av)],
            'aim_src': aim_src,
            'reproject_uv': reproject_uv,
            'reproject_pix_off': pix_repro,
            'live_uv_near_aim': (
                [float(live_uv[0]), float(live_uv[1])] if live_uv else None),
            'live_src_near_aim': live_src,
            'live_pix_off': pix_live,
            'qa_optical_live_uv': (
                [float(uv_qa[0]), float(uv_qa[1])] if uv_qa else None),
            'qa_optical_live_src': src_qa,
            'live_yolo': live_all,
            'circular_reproject_only': circular,
            'intrinsics_fx_fy_cx_cy': [fx, fy, cx, cy],
            'berry_base': list(self._last_berry_base or ()),
            'ee_xyz_cmd': ee_cmd,
            'ee_xyz_fb': ee_fb,
            'ee_err_mm': ee_err_mm,
            'joints_cmd_rad': joints_cmd,
            'joints_fb_rad': joints_fb,
            'joint_err_deg': joint_err_deg,
            'z_cam_m': self._last_z_cam,
        }
        self._write_refine_event_meta(tag, meta)
        self.get_logger().info(
            f'REFINING center arrive viz → {out_path} '
            f'repro_pix={(pix_repro if pix_repro is not None else -1):.1f} '
            f'live_pix={(pix_live if pix_live is not None else -1):.1f} '
            f'circular={circular} ee_err_mm='
            f'{(ee_err_mm if ee_err_mm is not None else -1):.2f}')
        return out_path

    def _reestimate_z_on_live_uv(
        self,
        *,
        uv: Tuple[float, float],
        berry: Optional[DetectedBerry] = None,
        cam: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Optional[float], str]:
        """Fresh depth on the live UV ray after center oneshot (arrive pose).

        Prefer wrist RGB-D when valid; mono only when depth is unavailable.
        """
        del uv  # ray built by caller; z only here
        if berry is not None:
            z, src = self._berry_range_z_m(berry)
            if z is not None and 0.08 < z < 1.5:
                return z, src
        if cam is not None and 0.08 < float(cam[2]) < 1.2:
            return float(cam[2]), 'live_cam'
        if self._last_z_cam is not None and 0.08 < float(self._last_z_cam) < 1.2:
            return float(self._last_z_cam), 'last_z_cam'
        return None, 'none'

    def _pick_live_berry_near_aim(
        self,
        *,
        aim_uv: Tuple[float, float],
        max_pix: float,
        min_conf: Optional[float] = None,
    ) -> Optional[DetectedBerry]:
        track_min = float(
            min_conf
            if min_conf is not None
            else getattr(self._args, 'refine_track_min_conf', 0.12))
        au, av = float(aim_uv[0]), float(aim_uv[1])
        ranked: List[Tuple[float, DetectedBerry]] = []
        for b in self._live_fine_berries():
            if float(b.confidence) < track_min:
                continue
            uv = self._berry_image_uv(b)
            if uv is None:
                continue
            e = math.hypot(float(uv[0]) - au, float(uv[1]) - av)
            if e <= float(max_pix):
                ranked.append((e, b))
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[0])
        return ranked[0][1]

    def _start_center_live_replan(
        self,
        *,
        uv: Tuple[float, float],
        berry: Tuple[float, float, float],
        z_aim: float,
        z_src: str,
    ) -> bool:
        """Second center oneshot: put live-UV berry on cup-aim with reestimated z."""
        if self._center_live_replan_done:
            return False
        if not self._arm.server_is_ready():
            return False
        max_att = max(2, int(getattr(self._args, 'servo_center_oneshot_max', 2)))
        # Fresh-lock UV correction may need one extra oneshot after climb replan.
        if 'fresh_lock' in str(z_src):
            max_att = max_att + 1
        if int(getattr(self, '_center_oneshot_attempts', 0)) >= max_att:
            self.get_logger().warn(
                f'REFINING center live-replan skipped: attempts≥{max_att}')
            return False

        u, v = float(uv[0]), float(uv[1])
        cx_img, cy_img, aim_src = self._aim_uv_for_center(float(z_aim))
        pix_off = math.hypot(cx_img - u, cy_img - v)
        pix_tol = float(getattr(self._args, 'servo_center_pix_tol_px', 35.0))
        # Even if already near aim, allow a small depth-driven tip correction.
        min_pix = max(6.0, pix_tol * 0.25)

        self._last_berry_base = berry
        self._last_berry_t = time.time()
        self._last_z_cam = float(z_aim)
        T_l6c = self._lookup_T_link6_cam()
        if T_l6c is None:
            self.get_logger().warn('REFINING center live-replan: no T_link6_cam')
            return False

        w, h = self._wrist_image_wh()
        fx, fy, cx0, cy0 = self._wrist_intrinsics()
        seed = list(self._joints)
        tol_pix = max(8.0, pix_tol * 0.5)
        joints = aim_uv_ik(
            berry,
            (cx_img, cy_img),
            seed,
            T_l6c,
            focal_px=0.5 * (fx + fy),
            fx=fx,
            fy=fy,
            cx=cx0,
            cy=cy0,
            image_wh=(float(w), float(h)),
            z_ref=float(z_aim),
            tol_pix=tol_pix,
        )
        if joints is None:
            self._center_live_replan_done = True
            self.get_logger().error(
                f'REFINING center live-replan: aim_uv_ik failed '
                f'uv=({u:.1f},{v:.1f})→({cx_img:.0f},{cy_img:.0f}) z={z_aim:.3f}')
            return False

        target = self._clamp_servo_target(list(joints))
        ee = self._current_ee_pose()
        if ee is None:
            return False
        ee0 = np.array([
            float(ee.pose.position.x),
            float(ee.pose.position.y),
            float(ee.pose.position.z),
        ], dtype=np.float64)
        ee1 = fk_xyz(target)
        d_base = ee1 - ee0
        lat = float(np.linalg.norm(d_base))
        # Skip tiny no-op if already on target in image and Cartesian.
        if pix_off < min_pix and lat < 0.008:
            self.get_logger().info(
                f'REFINING center live-replan skip: already near '
                f'pix_off={pix_off:.1f} |d|={lat*1000:.1f}mm')
            self._center_live_replan_done = True
            return False

        traj_s = max(
            float(getattr(self._args, 'servo_center_oneshot_traj_s', 2.0)),
            min(6.0, 1.2 + lat / 0.03),
        )
        tcp = self._tcp_or_ee_xyz()
        cup = self._cup_cam_xyz()
        T_bc = self._lookup_T_base_cam()
        d_cam = None
        if T_bc is not None:
            d_cam = (T_bc[:3, :3].T @ d_base).tolist()
        cam_xyz = self._base_xyz_to_cam(berry)

        self._write_qa_json('center_live_replan.json', {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'session': self._qa_session,
            'uv': [u, v],
            'aim_uv': [cx_img, cy_img],
            'aim_src': aim_src,
            'z_aim_m': float(z_aim),
            'z_src': z_src,
            'berry_base': list(berry),
            'pix_off': float(pix_off),
            'cmd_d_base_m': d_base.tolist(),
            'planned_dz_mm': float(d_base[2]) * 1000.0,
            'ee_xyz_before': ee0.tolist(),
            'ee_xyz_cmd': ee1.tolist(),
            'tip_above_berry_z_mm': (
                (float(tcp[2]) - float(berry[2])) * 1000.0
                if tcp is not None else None),
        })

        self._servo_pending_record = {
            'step': int(self._servo_step_idx),
            'track_id': self._refine_locked_track_id,
            'refine_phase': 'center_oneshot',
            'control': 'center_live_replan',
            'berry_base': list(berry),
            'berry_base_src': f'live_uv_{z_src}',
            'tcp_base': list(tcp) if tcp else None,
            'ee_xyz_before': ee0.tolist(),
            'ee_xyz_cmd': ee1.tolist(),
            'cmd_dcam_m': d_cam,
            'cmd_d_base_m': d_base.tolist(),
            'berry_cam': list(cam_xyz) if cam_xyz else None,
            'cup_cam': list(cup) if cup is not None else None,
            'cam_src': z_src,
            'aim_src': aim_src,
            'method': f'aim_uv_ik/live_replan/{z_src}',
            'pix_off': float(pix_off),
            'du_des': float(cx_img - u),
            'dv_des': float(cy_img - v),
            'berry_uv': [u, v],
            'aim_uv': [cx_img, cy_img],
            'optical_uv': [cx0, cy0],
            'attempt': int(getattr(self, '_center_oneshot_attempts', 0)) + 1,
            'joints_cmd_rad': [float(v) for v in target],
            'z_aim_m': float(z_aim),
            'qa_tag': f'servo_{self._servo_step_idx:02d}_center_live_replan',
            'source': 'step_json',
        }
        self._refine_phase = 'center_oneshot'
        self._center_oneshot_sent = True
        self._center_live_replan_done = True
        self._center_last_cmd = {
            'joints_cmd_rad': [float(v) for v in target],
            'ee_xyz_cmd': ee1.tolist(),
            'aim_uv': [cx_img, cy_img],
            'berry_uv': [u, v],
            'berry_base': list(berry),
            'z_aim_m': float(z_aim),
            'ee_xyz_before': ee0.tolist(),
            'tcp_before': list(tcp) if tcp else None,
            'cmd_d_base_m': d_base.tolist(),
            'berry_src': f'live_uv_{z_src}',
            'cam_src': z_src,
            'aim_src': aim_src,
            'pix_off': float(pix_off),
            'live_replan': True,
            'fresh_lock_replan': 'fresh_lock' in str(z_src),
        }
        self._write_center_climb_diag(
            phase='before_live_replan',
            berry=berry,
            berry_src=f'live_uv_{z_src}',
            tcp=tcp,
            ee=ee0.tolist(),
            ee_cmd=ee1.tolist(),
            z_aim=float(z_aim),
            aim_uv=(cx_img, cy_img),
            berry_uv=(u, v),
            cmd_d_base=d_base.tolist(),
            extra={'z_src': z_src, 'pix_off': float(pix_off)},
        )
        ok = self._send_joint_servo_goal(
            target,
            tag='center_replan',
            traj_s=traj_s,
            extra=(
                f'live_replan z={z_aim:.3f}/{z_src} pix_off={pix_off:.1f} '
                f'uv=({u:.1f},{v:.1f})→({cx_img:.0f},{cy_img:.0f}) '
                f'dbase=({d_base[0]*1000:+.1f},{d_base[1]*1000:+.1f},'
                f'{d_base[2]*1000:+.1f})mm'
            ),
        )
        if ok:
            self._center_oneshot_attempts = int(
                getattr(self, '_center_oneshot_attempts', 0)) + 1
            self.get_logger().info(
                f'REFINING center live-replan[{self._servo_step_idx}]: '
                f'z={z_aim:.3f}({z_src}) pix_off={pix_off:.1f} '
                f'dz={d_base[2]*1000:+.1f}mm')
        else:
            self._servo_pending_record = None
            self._center_oneshot_sent = False
            # Keep replan_done True to avoid retry loops on send failure.
        return ok

    def _finish_center_oneshot(self) -> None:
        """After center arrive: live UV depth re-estimate → optional replan → near.

        First oneshot uses probe mono (often wrong depth → tip climb). On arrive,
        rebuild berry on the *live* UV ray with fresh depth and, if residual or
        depth change warrants it, fire one more aim_uv_ik before contact.
        """
        # Always dump analysis frame first (including paths that ERROR below).
        self._snap_center_oneshot_arrive_analysis(reason='finish_center_oneshot')
        self._write_center_climb_diag(phase='after_center')
        self._center_oneshot_sent = False
        pix_tol = float(getattr(self._args, 'servo_center_pix_tol_px', 35.0))
        # Only accept live detections inside this radius of cup-aim.
        live_max = max(
            pix_tol,
            float(getattr(self._args, 'refine_fresh_lock_max_pix_px', 80.0)),
        )
        z_guess = float(self._last_z_cam or 0.35)
        au, av, aim_src = self._aim_uv_for_center(z_guess)

        cam, uv, src = self._center_arrive_measure_near_aim(
            aim_uv=(au, av), max_pix=live_max)
        live_berry = self._pick_live_berry_near_aim(
            aim_uv=(au, av), max_pix=live_max) if src.startswith('live') else None
        if uv is None:
            # Prefer mono-ray / last berry reproject (geometry of the oneshot).
            geo = self._reproject_berry_geo()
            if geo is not None and geo.get('berry_uv') is not None:
                uv = [float(geo['berry_uv'][0]), float(geo['berry_uv'][1])]
                bc = geo.get('berry_cam')
                cam = (
                    (float(bc[0]), float(bc[1]), float(bc[2]))
                    if bc is not None else None)
                src = 'reproject_aim'
            else:
                uv = [float(au), float(av)]
                cam = None
                src = 'aim_assumed'
                self.get_logger().warn(
                    f'REFINING center arrive: no live within {live_max:.0f}px of '
                    f'{aim_src} and no reproject — continue using aim UV')

        z_old = float(self._last_z_cam) if self._last_z_cam else z_guess
        if src.startswith('live'):
            z_est, z_src = self._reestimate_z_on_live_uv(
                uv=(float(uv[0]), float(uv[1])),
                berry=live_berry,
                cam=cam,
            )
            if z_est is None:
                z_est = float(cam[2]) if cam is not None and cam[2] > 1e-4 else z_guess
                z_src = 'fallback'
            self._update_berry_from_live_uv(
                uv=(float(uv[0]), float(uv[1])),
                z_cam=float(z_est),
                berry=live_berry,
                reason=f'center_arrive_live_{z_src}',
            )
            au, av, aim_src = self._aim_uv_for_center(float(z_est))
            pix = math.hypot(float(uv[0]) - au, float(uv[1]) - av)
            dz = abs(float(z_est) - float(z_old))
            if pix > pix_tol:
                self.get_logger().warn(
                    f'REFINING center arrive residual: live pix_off={pix:.1f}>'
                    f'{pix_tol:.0f} to {aim_src} '
                    f'uv=({float(uv[0]):.0f},{float(uv[1]):.0f})→'
                    f'({au:.0f},{av:.0f}) z={float(z_est):.3f}({z_src})')
            else:
                self.get_logger().info(
                    f'REFINING center arrive ok: pix_off={pix:.1f} to {aim_src} '
                    f'src={src} z={float(z_est):.3f}({z_src}) '
                    f'uv=({float(uv[0]):.0f},{float(uv[1]):.0f})')

            # One live-UV depth replan before handing off to contact.
            enable_replan = bool(
                getattr(self._args, 'servo_center_live_replan', True))
            last_cmd = getattr(self, '_center_last_cmd', None) or {}
            last_was_replan = bool(last_cmd.get('live_replan'))
            planned_dz = 0.0
            cmd_d = last_cmd.get('cmd_d_base_m')
            if isinstance(cmd_d, (list, tuple)) and len(cmd_d) >= 3:
                planned_dz = abs(float(cmd_d[2]))
            need_replan = (
                enable_replan
                and not self._center_live_replan_done
                and not last_was_replan
                and (
                    pix > max(8.0, pix_tol * 0.5)
                    or dz > 0.025
                    or planned_dz > 0.04
                )
            )
            if last_was_replan:
                self.get_logger().info(
                    'REFINING center live-replan arrive done — hand off to depth/near')
            if need_replan and self._last_berry_base is not None:
                if self._start_center_live_replan(
                    uv=(float(uv[0]), float(uv[1])),
                    berry=tuple(self._last_berry_base),
                    z_aim=float(z_est),
                    z_src=z_src,
                ):
                    return
        elif src.startswith('reproject'):
            z = float(cam[2]) if cam is not None and cam[2] > 1e-4 else z_guess
            au, av, aim_src = self._aim_uv_for_center(z)
            pix = math.hypot(float(uv[0]) - au, float(uv[1]) - av)
            if pix > max(pix_tol, live_max):
                self.get_logger().error(
                    f'REFINING center arrive miss: reproject pix_off={pix:.1f} '
                    f'to {aim_src} — geometry off')
                self._set_state(
                    'ERROR',
                    f'center arrive reproject miss pix_off={pix:.1f}px')
                return
            self.get_logger().info(
                f'REFINING center arrive ok: pix_off={pix:.1f} to {aim_src} '
                f'src={src} uv=({float(uv[0]):.0f},{float(uv[1]):.0f}) '
                f'live_max={live_max:.0f}px')
        else:
            z = float(cam[2]) if cam is not None and cam[2] > 1e-4 else z_guess
            au, av, aim_src = self._aim_uv_for_center(z)
            pix = math.hypot(float(uv[0]) - au, float(uv[1]) - av)
            self.get_logger().info(
                f'REFINING center arrive ok: pix_off={pix:.1f} to {aim_src} '
                f'src={src} uv=({float(uv[0]):.0f},{float(uv[1]):.0f}) '
                f'live_max={live_max:.0f}px')

        # Clear old probe lock → wait for fresh YOLO near cup-aim → re-pin.
        # Empty 1–2 frames after clear are normal; await covers that.
        self._clear_target_lock()
        self._await_fresh_optical_until = time.time() + float(
            getattr(self._args, 'refine_post_center_live_wait_s', 1.0))
        self._range_depth_started = False
        self._refine_phase = 'post_center'
        self._refresh_berry_cam_from_tri()
        qa_max = float(getattr(self._args, 'refine_fresh_lock_max_pix_px', 80.0)) * 1.5
        pix_live = self._optical_live_pix_off(max_pix=qa_max)
        pix_tri = self._reproject_pix_off()
        wait_s = float(getattr(self._args, 'refine_post_center_live_wait_s', 1.0))
        self.get_logger().info(
            f'REFINING center arrive done: live_pix_off='
            f'{(pix_live if pix_live is not None else -1):.1f}px '
            f'reproject_pix_off={(pix_tri if pix_tri is not None else -1):.1f}px '
            f'(lock cleared; await live@cup-aim ≤{wait_s:.1f}s then re-pin)')
        self._begin_range_depth()

    def _center_arrive_measure_near_aim(
        self,
        *,
        aim_uv: Tuple[float, float],
        max_pix: float,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[List[float]], str]:
        """Live YOLO whose *bbox* UV is within max_pix of cup-aim — else none."""
        au, av = float(aim_uv[0]), float(aim_uv[1])
        track_min = float(getattr(self._args, 'refine_track_min_conf', 0.12))
        ranked: List[Tuple[float, DetectedBerry]] = []
        for b in self._live_fine_berries():
            if float(b.confidence) < track_min:
                continue
            uv = self._berry_image_uv(b)
            if uv is None:
                continue
            e = math.hypot(float(uv[0]) - au, float(uv[1]) - av)
            if e <= float(max_pix):
                ranked.append((e, b))
        if not ranked:
            return None, None, 'none'
        ranked.sort(key=lambda x: x[0])
        live = ranked[0][1]
        uv = self._berry_image_uv(live)
        cam = self._berry_cam_xyz(live)
        if uv is None:
            return None, None, 'none'
        if cam is None or cam[2] <= 1e-4:
            fx, fy, cx, cy = self._wrist_intrinsics()
            z, _ = self._berry_range_z_m(live)
            if (z is None or z <= 1e-4) and self._last_z_cam:
                z = float(self._last_z_cam)
            if z > 1e-4:
                cam = (
                    (float(uv[0]) - cx) / fx * z,
                    (float(uv[1]) - cy) / fy * z,
                    z,
                )
        return cam, [float(uv[0]), float(uv[1])], 'live_near_aim'

    def _center_arrive_measure(
        self,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[List[float]], str]:
        """Compat: live near cup-aim only (no unlimited fallback)."""
        z = float(self._last_z_cam or 0.35)
        au, av, _ = self._aim_uv_for_center(z)
        max_pix = max(
            float(getattr(self._args, 'servo_center_pix_tol_px', 35.0)),
            float(getattr(self._args, 'refine_fresh_lock_max_pix_px', 80.0)),
        )
        return self._center_arrive_measure_near_aim(
            aim_uv=(au, av), max_pix=max_pix)

    def _center_oneshot_measure_for_residual(
        self,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[List[float]], str]:
        """Deprecated alias — residual center path removed."""
        return self._center_arrive_measure()

    def _optical_live_pix_off(self, max_pix: Optional[float] = None) -> Optional[float]:
        """Live berry nearest optical center — for QA after center, not control."""
        b = self._pick_live_nearest_optical(max_pix=max_pix)
        if b is None:
            return None
        return self._berry_optical_pix_err(b)

    def _center_live_pix_off(self) -> Optional[float]:
        return self._optical_live_pix_off()

    def _fresh_lock_optical_center(self, *, reason: str) -> Optional[DetectedBerry]:
        """After center: pick live YOLO near cup-aim and rebuild 3D on its UV.

        Called after clear_lock: old probe ID is gone; re-pin the YOLO box near
        cup-aim (not optical center) and re-estimate depth on that UV.
        """
        old = self._refine_locked_track_id
        max_pix = float(getattr(self._args, 'refine_fresh_lock_max_pix_px', 80.0))
        fresh_min = float(getattr(self._args, 'refine_fresh_lock_min_conf', 0.10))
        z_g = float(self._last_z_cam or 0.35)
        au, av, _ = self._aim_uv_for_center(z_g)
        # Cup-aim first; optical-nearest only as last resort inside radius.
        neu = self._pick_live_berry_near_aim(
            aim_uv=(au, av), max_pix=max_pix, min_conf=fresh_min)
        if neu is None:
            neu = self._pick_live_nearest_optical(max_pix=max_pix, min_conf=fresh_min)
        if neu is None:
            return None
        tid = int(neu.track_id)
        self._refine_locked_track_id = tid
        self._lock_pub.publish(neu)

        ok3d, z_src = self._correct_berry_3d_from_fresh_lock(
            neu, reason=f'fresh_lock_{reason}')
        if not ok3d:
            # Last resort: published pose (may be wrong depth — logged).
            bx = float(neu.pose.pose.position.x)
            by = float(neu.pose.pose.position.y)
            bz = float(neu.pose.pose.position.z)
            self._last_berry_base = (bx, by, bz)
            self._refine_fruit_anchor = (bx, by, bz)
            cam = self._berry_cam_xyz(neu)
            if cam is not None and cam[2] > 1e-4:
                self._last_z_cam = float(cam[2])
            self._last_berry_t = time.time()
            self.get_logger().warn(
                f'REFINING fresh optical lock {old}→{tid}: 3D rebuild failed '
                f'({z_src}) — fell back to YOLO pose XYZ')

        pix = self._berry_optical_pix_err(neu)
        uv = self._berry_image_uv(neu)
        aim_pix = None
        if uv is not None:
            aim_pix = math.hypot(float(uv[0]) - au, float(uv[1]) - av)
        b = self._last_berry_base
        base_s = (
            f'base=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f})' if b is not None else 'base=None')
        self.get_logger().info(
            f'REFINING fresh lock {old}→{tid} ({reason}) '
            f'cup_aim_pix={(aim_pix if aim_pix is not None else -1):.1f} '
            f'opt_pix={(pix if pix is not None else -1):.1f} '
            f'conf={float(neu.confidence):.2f} min_conf={fresh_min:.2f} '
            f'mode={getattr(neu, "depth_mode", "")} '
            f'3d={z_src if ok3d else "yolo_pose_fallback"} {base_s}')
        self._snap_refine_lock_viz('optical_fresh_lock', source='yolo_fresh_lock', berry=neu)
        return neu

    def _finish_depth_qa_stop(self, *, depth_source: str) -> None:
        """Stop-after-center: record approach depth (YOLO depth probe or probe-tri reproject)."""
        if self._depth_qa_stop_done:
            return
        self._depth_qa_stop_done = True
        self._clear_target_lock()
        if depth_source == 'probe_tri_reproject':
            self._refresh_berry_cam_from_tri()
        elif depth_source == 'probe_tri_mono_chord':
            self._refresh_berry_cam_from_tri()
        berry = self._last_berry_base
        tcp = self._tcp_or_ee_xyz()
        contact = float(self._args.cup_contact_offset)
        dist_cup = self._cup_berry_dist()
        travel = (
            float(dist_cup - contact)
            if dist_cup is not None and dist_cup > contact
            else None)
        geo: Dict = {}
        if berry is not None and tcp is not None:
            geo = self._cup_berry_geometry(berry, tcp)
            geo['z_cam_m'] = self._last_z_cam
            geo['travel_m'] = travel
            geo['depth_source'] = depth_source
            self._write_cup_rel_overlay('center_depth_qa', geo)
        pix_tri = self._reproject_pix_off()
        pix_live = self._optical_live_pix_off()
        payload = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'session': self._qa_session,
            'depth_source': depth_source,
            'berry_base': list(berry) if berry else None,
            'tcp_base': list(tcp) if tcp else None,
            'z_cam_m': self._last_z_cam,
            'dist_cup_m': dist_cup,
            'cup_contact_offset_m': contact,
            'travel_to_contact_m': travel,
            'reproject_pix_off_px': pix_tri,
            'live_pix_off_px': pix_live,
            'berry_uv': geo.get('berry_uv'),
            'cup_uv': geo.get('cup_uv'),
            'berry_rel_cup': geo.get('berry_rel_cup'),
            'center_probe_tri_ok': bool(self._center_probe_tri_ok),
        }
        mono_meta = getattr(self, '_last_mono_chord_meta', None)
        if mono_meta:
            payload['mono_chord'] = mono_meta
        if depth_source == 'yolo_depth_probe':
            mp = self._qa_session_dir()
            if mp:
                dp = os.path.join(mp, 'mono_probe_depth.json')
                if os.path.isfile(dp):
                    try:
                        with open(dp, encoding='utf-8') as f:
                            payload['mono_probe_depth'] = json.load(f)
                    except Exception:
                        pass
        self._write_qa_json('center_depth_qa.json', payload)
        self._snap_qa('center_depth_qa')
        self._refine_phase = 'center_done'
        self._snap_refine_lock_viz('center_done')
        self.get_logger().info(
            f'REFINING depth QA stop ({depth_source}): '
            f'z_cam={(self._last_z_cam if self._last_z_cam is not None else -1):.3f}m '
            f'dist_cup={(dist_cup if dist_cup is not None else -1):.3f}m '
            f'travel={(travel if travel is not None else -1):.3f}m '
            f'reproject_pix_off={(pix_tri if pix_tri is not None else -1):.1f}px '
            f'— measure travel_to_contact in replay')
        self._set_state('WAIT_CONFIRM')

    def _begin_range_depth(self) -> None:
        """After open-loop center: depth QA (stop) or full contact pipeline."""
        if (
            self._depth_qa_stop_done
            or self._refine_phase == 'center_done'
            or self._range_depth_started
            or self._near_handoff_written
            or self._state == 'REFINING_WAIT_NEAR'
        ):
            return
        self._refresh_berry_cam_from_tri()
        stop_after = bool(getattr(self._args, 'refine_stop_after_center', False))
        fresh_min = float(getattr(self._args, 'refine_fresh_lock_min_conf', 0.10))
        max_pix = float(getattr(self._args, 'refine_fresh_lock_max_pix_px', 80.0))

        if stop_after:
            neu = self._fresh_lock_optical_center(reason='stop_after_center_yolo')
            if neu is not None:
                self._range_depth_started = True
                self._await_fresh_optical_until = 0.0
                self.get_logger().info(
                    'REFINING stop-after-center: fresh YOLO near cup-aim '
                    '→ depth QA (skip orbit probe)')
                self._finish_depth_qa_stop(depth_source='yolo_fresh_lock')
                return
            if time.time() < float(self._await_fresh_optical_until or 0.0):
                now = time.time()
                if now - self._refine_wait_log_t > 0.4:
                    self._refine_wait_log_t = now
                    self.get_logger().info(
                        f'REFINING awaiting live@cup-aim after center '
                        f'(min_conf={fresh_min:.2f} max_pix={max_pix:.0f})')
                return
            self._range_depth_started = True
            if self._center_probe_tri_ok and self._last_berry_base is not None:
                ds = self._apply_probe_tri_mono_chord_scale()
                self._finish_depth_qa_stop(
                    depth_source=ds or 'probe_tri_reproject')
                return
            self._set_state(
                'ERROR',
                'refine: no live YOLO at center and center probe tri unavailable')
            return

        neu = self._fresh_lock_optical_center(reason='after_center_openloop')
        if neu is not None:
            self._await_fresh_optical_until = 0.0
            # Fresh lock already rebuilt 3D (UV ray + probe gate). Optionally
            # one more aim_uv_ik if cup-aim residual is still large — even when
            # an earlier live-replan already ran (162251: replan used wrong UV,
            # then fresh ID was right but 3D was skipped because replan_done).
            uv_n = self._berry_image_uv(neu)
            if (
                uv_n is not None
                and self._last_berry_base is not None
                and bool(getattr(self._args, 'servo_center_live_replan', True))
            ):
                z_now = float(self._last_z_cam or 0.35)
                au, av, _ = self._aim_uv_for_center(z_now)
                pix_n = math.hypot(float(uv_n[0]) - au, float(uv_n[1]) - av)
                pix_tol = float(
                    getattr(self._args, 'servo_center_pix_tol_px', 35.0))
                need_fresh_replan = pix_n > max(8.0, pix_tol * 0.5)
                # Allow one fresh-lock-driven replan even if climb replan ran.
                already = bool(getattr(self, '_center_live_replan_done', False))
                last_cmd = getattr(self, '_center_last_cmd', None) or {}
                last_was_fresh = bool(last_cmd.get('fresh_lock_replan'))
                if need_fresh_replan and not last_was_fresh:
                    if already:
                        # Permit a second oneshot attempt for fresh-lock UV.
                        self._center_live_replan_done = False
                    if self._start_center_live_replan(
                        uv=(float(uv_n[0]), float(uv_n[1])),
                        berry=tuple(self._last_berry_base),
                        z_aim=float(z_now),
                        z_src='fresh_lock_3d',
                    ):
                        return

            self._range_depth_started = True
            # Fresh YOLO at center already carries a base pose (mono and/or prior
            # probe-tri). A second contact mono probe often coasts the lock and
            # re-pins to a far box (20260807_142705). Skip that probe whenever we
            # already have berry_base — tri_ok is nice-to-have, not required.
            if self._last_berry_base is not None:
                why = (
                    'after_center_yolo_fresh_lock'
                    if self._center_probe_tri_ok
                    else 'after_center_yolo_fresh_lock_mono'
                )
                self.get_logger().info(
                    f'REFINING → contact: fresh optical YOLO + berry_base '
                    f'(skip contact mono probe; tri_ok={self._center_probe_tri_ok})')
                self._enter_contact_oneshot(reason=why)
                return
            self._mono_probe_done = False
            self._mono_probe_obs = []
            self._mono_probe_purpose = 'contact'
            self._refine_phase = 'range_depth'
            self.get_logger().info(
                'REFINING → range_depth: fresh optical lock without berry_base '
                '→ live depth probe → contact oneshot')
            return

        if time.time() < float(self._await_fresh_optical_until or 0.0):
            now = time.time()
            if now - self._refine_wait_log_t > 0.4:
                self._refine_wait_log_t = now
                n_live = len(self._live_fine_berries())
                z_g = float(self._last_z_cam or 0.35)
                au, av, _ = self._aim_uv_for_center(z_g)
                near_aim = self._pick_live_berry_near_aim(
                    aim_uv=(au, av), max_pix=max_pix)
                self.get_logger().info(
                    f'REFINING awaiting live@cup-aim after center '
                    f'(live_n={n_live} near_aim={int(near_aim is not None)} '
                    f'min_conf={fresh_min:.2f} max_pix={max_pix:.0f})')
            return

        self.get_logger().warn(
            f'REFINING fresh lock timeout after center '
            f'(min_conf={fresh_min:.2f} max_pix={max_pix:.0f}) — try live@aim 3D')
        # Timeout: if a live box near cup-aim appeared, rebuild 3D on it (no fake
        # deepen). Optional oneshot only when residual still large.
        if bool(getattr(self._args, 'servo_center_live_replan', True)):
            z_g = float(self._last_z_cam or 0.35)
            au, av, _ = self._aim_uv_for_center(z_g)
            live_max = max(
                float(getattr(self._args, 'servo_center_pix_tol_px', 35.0)),
                float(max_pix),
            )
            live_b = self._pick_live_berry_near_aim(
                aim_uv=(au, av), max_pix=live_max)
            if live_b is not None and float(live_b.confidence) >= fresh_min:
                uv_a = self._berry_image_uv(live_b)
                cam_a = self._berry_cam_xyz(live_b)
                if uv_a is not None:
                    z_est, z_src = self._reestimate_z_on_live_uv(
                        uv=(float(uv_a[0]), float(uv_a[1])),
                        berry=live_b,
                        cam=cam_a,
                    )
                    if z_est is None and cam_a is not None and cam_a[2] > 1e-4:
                        z_est, z_src = float(cam_a[2]), 'live_cam'
                    if z_est is not None:
                        self._update_berry_from_live_uv(
                            uv=(float(uv_a[0]), float(uv_a[1])),
                            z_cam=float(z_est),
                            berry=live_b,
                            reason=f'post_center_live_aim_{z_src}',
                        )
                        tid = int(live_b.track_id)
                        self._refine_locked_track_id = tid
                        self._lock_pub.publish(live_b)
                        pix_n = math.hypot(float(uv_a[0]) - au, float(uv_a[1]) - av)
                        pix_tol = float(
                            getattr(self._args, 'servo_center_pix_tol_px', 35.0))
                        if (
                            pix_n > max(8.0, pix_tol * 0.5)
                            and self._last_berry_base is not None
                            and not self._center_live_replan_done
                        ):
                            if self._start_center_live_replan(
                                uv=(float(uv_a[0]), float(uv_a[1])),
                                berry=tuple(self._last_berry_base),
                                z_aim=float(z_est),
                                z_src=z_src,
                            ):
                                return
                        # Live 3D updated — contact without inventing depth.
                        self._range_depth_started = True
                        self._enter_contact_oneshot(
                            reason='after_center_live_aim_timeout')
                        return

        self._range_depth_started = True
        use_tri = bool(getattr(self._args, 'refine_contact_from_probe_tri', True))
        if use_tri and self._contact_from_probe_tri(reason='center_probe_tri_no_yolo'):
            return

        # No usable live near cup-aim: keep probe mono-ray berry (no deepen).
        if self._last_berry_base is not None:
            self.get_logger().warn(
                'REFINING → contact: no live@cup-aim after center wait — '
                'use existing probe_mono_ray berry_base')
            self._enter_contact_oneshot(reason='after_center_probe_mono_ray_no_yolo')
            return

        self._set_state(
            'ERROR',
            'refine: no live berry near cup-aim after center and probe tri unavailable')

    def _enter_contact_oneshot(self, *, reason: str) -> None:
        """After depth is known: freeze berry → one planned traj to cup contact."""
        self._refine_phase = 'near'
        self._near_frozen_berry = None
        self._freeze_near_berry()
        dist_cup = self._cup_berry_dist()
        if self._args.servo_near_confirm:
            # Enter WAIT_NEAR BEFORE slow QA snaps so a concurrent refine tick
            # cannot auto-start near oneshot (phase==near used to enable it).
            self._near_mode = False
            self.get_logger().info(
                f'REFINING → WAIT_NEAR ({reason}): oneshot to contact '
                f'dist_cup={(dist_cup if dist_cup is not None else -1):.3f} '
                f'— publish confirm_near')
            self._set_state('REFINING_WAIT_NEAR')
            self._refine_deadline = time.time() + 600.0
            if not self._near_handoff_written:
                nh_src = 'near_handoff_yolo_lock'
                if self._refine_locked_track_id is None:
                    nh_src = (
                        'near_handoff_probe_tri_mono_chord'
                        if self._probe_tri_chord_applied
                        else 'near_handoff_probe_tri')
                berry = None
                if self._refine_locked_track_id is not None:
                    for b in self._live_fine_berries():
                        if int(b.track_id) == int(self._refine_locked_track_id):
                            berry = b
                            break
                self._snap_refine_lock_viz(
                    'near_handoff_pending', source=nh_src, berry=berry)
                self._write_near_handoff_request(
                    dist_cup=dist_cup, z_cam=self._last_z_cam, berry_age=0.0)
                self._near_handoff_written = True
            return
        self._near_mode = True
        self.get_logger().info(
            f'REFINING → near oneshot ({reason}): freeze berry, single traj to contact '
            f'dist_cup={(dist_cup if dist_cup is not None else -1):.3f}')

    def _pin_berry_from_probe_for_direct_contact(self) -> str:
        """Freeze berry for probe→near skip-center.

        Prefer accepted probe triangulation; else last probe mono-ray back-project.
        """
        if bool(getattr(self, '_center_probe_tri_ok', False)) and self._last_berry_base is not None:
            b = self._last_berry_base
            self.get_logger().info(
                f'REFINING skip-center: use probe_tri '
                f'base=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f})')
            return 'probe_tri'
        berry, src = self._berry_base_for_center_aim()
        if berry is None:
            self.get_logger().warn(
                'REFINING skip-center: no probe berry to freeze')
            return 'none'
        self._last_berry_base = berry
        self._refine_fruit_anchor = berry
        self._last_berry_t = time.time()
        self._near_frozen_berry = None
        cam = self._base_xyz_to_cam(berry)
        if cam is not None and cam[2] > 1e-4:
            self._last_z_cam = float(cam[2])
        self.get_logger().info(
            f'REFINING skip-center: freeze berry from {src} '
            f'base=({berry[0]:.3f},{berry[1]:.3f},{berry[2]:.3f})')
        return src

    def _finish_mono_probe(self) -> None:
        """End of a small-step range: triangulate then center-oneshot or contact-oneshot."""
        self._mono_probe_done = True
        purpose = str(self._mono_probe_purpose or 'contact')
        tag = 'mono_probe_center' if purpose == 'center' else 'mono_probe_depth'
        self._apply_triangulation(tag)
        # Ranging motion itself calibrates look-at Jac (Δq → ΔUV).
        if purpose == 'center':
            self._update_ee_uv_jac_from_probe()
            # Experiment: probe lock+depth → one open-loop push to berry (no center).
            if bool(getattr(self._args, 'refine_skip_center', False)):
                src = self._pin_berry_from_probe_for_direct_contact()
                if self._last_berry_base is None:
                    self._set_state(
                        'ERROR',
                        f'refine skip-center: no berry after probe ({src})')
                    return
                self._enter_contact_oneshot(reason=f'probe_skip_center_{src}')
                return
            if not self._start_center_oneshot():
                # Already centered or IK fail → still proceed to depth range.
                if self._refine_phase != 'range_depth':
                    self._begin_range_depth()
            return
        if bool(getattr(self._args, 'refine_stop_after_center', False)):
            self._finish_depth_qa_stop(depth_source='yolo_depth_probe')
            return
        self._enter_contact_oneshot(reason='probe_depth_done')

    def _tick_mono_probe(self) -> None:
        """Known orbit move + track UV → triangulate mono range."""
        n_move = max(1, int(getattr(self._args, 'servo_mono_probe_steps', 2)))
        # Need one obs before each move, then one after last move.
        if len(self._mono_probe_obs) == 0:
            obs = self._capture_mono_probe_obs()
            if obs is None:
                now = time.time()
                if now - self._refine_wait_log_t > 1.0:
                    self._refine_wait_log_t = now
                    self.get_logger().info('REFINING mono probe: waiting for berry UV')
                return
            self._mono_probe_obs.append(obs)
            self.get_logger().info(
                f'REFINING mono probe obs0 uv=({obs["uv"][0]:.1f},{obs["uv"][1]:.1f}) '
                f'z_mono={obs["z_mono"]:.3f}')
            if not self._start_mono_probe_step():
                return
            return

        # After a settle, move_phase is None and we land here again.
        need_obs = n_move + 1
        if len(self._mono_probe_obs) < need_obs:
            obs = self._capture_mono_probe_obs()
            if obs is None:
                return
            # Skip near-duplicate UV if arm hasn't moved yet.
            prev = self._mono_probe_obs[-1]
            du = abs(obs['uv'][0] - prev['uv'][0]) + abs(obs['uv'][1] - prev['uv'][1])
            tcp0 = prev.get('tcp_base')
            tcp1 = obs.get('tcp_base')
            tcp_move = 0.0
            if tcp0 and tcp1:
                tcp_move = math.sqrt(
                    (tcp1[0] - tcp0[0]) ** 2
                    + (tcp1[1] - tcp0[1]) ** 2
                    + (tcp1[2] - tcp0[2]) ** 2)
            if tcp_move < 0.005 and du < 2.0 and len(self._mono_probe_obs) >= 1:
                # Still starting first move — wait.
                if not self._arm.server_is_ready():
                    return
                return
            self._mono_probe_obs.append(obs)
            self.get_logger().info(
                f'REFINING mono probe obs{len(self._mono_probe_obs)-1} '
                f'uv=({obs["uv"][0]:.1f},{obs["uv"][1]:.1f}) '
                f'tcp_move={tcp_move:.3f}m duv={du:.1f}px')
            if len(self._mono_probe_obs) >= need_obs:
                self._finish_mono_probe()
                return
            self._start_mono_probe_step()
            return

        self._finish_mono_probe()

    # -----------------------------------------------------------------------
    # PBVS refining loop (pbvs-vlm-reach-v2)
    # -----------------------------------------------------------------------

    def _init_pbvs(self) -> None:
        """Called once when REFINING state is entered with use_pbvs=True."""
        import sys, os
        _scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from berry_kf_tracker import BerryKFTracker
        from approach_dir_selector import select_approach_direction
        from berry_approach_mapper import FusedApproachMap
        self._pbvs_kf = BerryKFTracker()
        self._pbvs_state = 'INIT'   # INIT → SERVO → PROBE → APPROACH → DONE
        self._pbvs_last_tick = time.time()
        self._pbvs_approach_dir_base = None   # (3,) unit vec in base_link, set by _build_approach_map_from_global
        self._pbvs_probe_q_before = None
        self._pbvs_probe_obs_list = []
        self._pbvs_select_approach_direction = select_approach_direction
        # Fresh fused map — receives global frame from _build_approach_map_from_global,
        # then wrist frames every 0.5 s during SERVO.
        self._fused_map = FusedApproachMap(max_age_s=4.0)
        self._pbvs_last_wrist_map_t = 0.0
        # Global RGB-D → map must succeed; entry-time TF races are retried in tick.
        self._pbvs_global_map_ok = False
        self._pbvs_global_retry_t = 0.0
        self._pbvs_stream_log_t = 0.0
        self._pbvs_stream_snap_t = 0.0
        # Conditioned PBVS setpoint (decoupled from raw KF posterior).
        self._pbvs_target_cmd: Optional[np.ndarray] = None
        self._pbvs_frozen_target: Optional[np.ndarray] = None  # QA legacy only
        self._pbvs_cam_snapshot: Optional[Dict] = None
        self._pbvs_approach_blind = False
        self._pbvs_vision_lost_streak = 0
        self._pbvs_had_live_near = False
        self._pbvs_near_logged = False
        self._pbvs_kf_reject_streak = 0
        self._pbvs_last_v_norm = 0.0
        self._pbvs_gate_log_t = 0.0
        self._pbvs_frozen_approach_dir: Optional[np.ndarray] = None
        self._pbvs_contact_sent = False
        self._pbvs_contact_finished = False
        self._pbvs_contact_sent_t = 0.0
        self._pbvs_blind_warn_t = 0.0
        self._pbvs_probe_started_t = 0.0
        self._pbvs_probe_return_pending = False
        self._pbvs_probe_last_t = 0.0
        self._pbvs_probe_entry_obs = None
        self._pbvs_lock_depth_mode = ''
        self._pbvs_entry_lock_xyz: Optional[np.ndarray] = None
        self._pbvs_probe_success_n = 0
        self._pbvs_probe_fail_n = 0
        self._pbvs_frozen_berry: Optional[np.ndarray] = None
        self._pbvs_scaled_berry: Optional[np.ndarray] = None  # lock-ray 1-DOF
        self._pbvs_lock_ray: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._pbvs_ray_scale: Optional[Dict] = None
        self._pbvs_ray_scale_best_gap: float = float('inf')  # keep best gap across ticks
        self._pbvs_frozen_normal: Optional[np.ndarray] = None  # outward, toward cup
        self._pbvs_surface_fit: Optional[Dict] = None
        self._pbvs_press_cup_start: Optional[np.ndarray] = None
        self._pbvs_oneshot_pending: Optional[str] = None  # 'approach'|'final'|'direct'|'align_n'|'press'
        self._pbvs_direct_shot_n = 0
        self._pbvs_depth_final_running = False
        self._pbvs_motion_min_d_cam: Optional[float] = None
        self._ensure_pbvs_qa_recorder()
        mode = str(getattr(self._args, 'pbvs_mode', 'stream') or 'stream')
        self._pbvs_qa_log_event('init', {
            'reason': 'pbvs_loop_start',
            'pbvs_mode': mode,
        })
        self.get_logger().info(
            f'PBVS loop initialised (mode={mode}, FusedApproachMap ready)')

    def _pbvs_oneshot_mode(self) -> bool:
        """Open-loop PBVS (blocking traj): ``single``/``oneshot``/``twostage``."""
        mode = str(getattr(self._args, 'pbvs_mode', 'single') or 'single')
        return self._use_pbvs and mode in ('oneshot', 'single', 'twostage')

    def _pbvs_direct_mode(self) -> bool:
        """Canonical path: lock → tip→surface oneshot(s) → press → WAIT_CONFIRM.

        ``oneshot`` is an alias of ``single``. Motion is open-loop to the frozen
        lock point (no live tracking). After each segment settles, cup↔frozen is
        checked; if still > exec_tol (0.5 mm), up to ``pbvs_direct_max_shots``
        residual pushes run, then an optional short press along −n, then a
        suction placeholder. Outer ``/reach/status`` stays REFINING until
        WAIT_CONFIRM. Human ``confirm_reset`` only after that.
        """
        mode = str(getattr(self._args, 'pbvs_mode', 'single') or 'single')
        return self._use_pbvs and mode in ('oneshot', 'single')

    def _pbvs_twostage_mode(self) -> bool:
        """Legacy opt-in: pre-grasp then depth_final."""
        mode = str(getattr(self._args, 'pbvs_mode', 'single') or 'single')
        return self._use_pbvs and mode == 'twostage'

    def _on_executor_status(self, msg: ExecutorStatus) -> None:
        self._last_executor_status = msg

    def _publish_pbvs_tool_trajectory(
        self,
        tip_start: Sequence[float],
        tip_goal: Sequence[float],
        approach_axis: Sequence[float],
        *,
        move_duration_s: float,
        tag: str = '',
    ) -> int:
        """Publish PBVS setpoint as /planning/tool_trajectory_4s (teacher source)."""
        if not self._publish_tool_traj and not self._pbvs_via_executor:
            return int(self._pbvs_tool_traj_seq)
        from tool_trajectory_utils import build_tool_trajectory_4s
        self._pbvs_tool_traj_seq += 1
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = 'base_link'
        msg = build_tool_trajectory_4s(
            tip_start, tip_goal, approach_axis,
            header=hdr,
            seq=int(self._pbvs_tool_traj_seq),
            move_duration_s=float(move_duration_s),
        )
        self._tool_traj_pub.publish(msg)
        self.get_logger().info(
            f'PBVS tool_traj seq={msg.seq} tag={tag or "-"} '
            f'move={move_duration_s:.2f}s exec_i={msg.execute_until_index} '
            f'tip→{np.round(np.asarray(tip_goal, dtype=float), 3)}')
        return int(msg.seq)

    def _begin_pbvs_executor_wait(self, *, traj_s: float, tool_seq: int) -> None:
        """Wait for trajectory_executor instead of local FollowJointTrajectory."""
        settle = float(self._args.servo_settle_s)
        dt = float(max(0.05, traj_s))
        self._pbvs_executor_wait = True
        self._pbvs_executor_motion_until = time.time() + dt
        self._servo_deadline = time.time() + max(
            float(self._args.servo_timeout_s), dt + settle + 2.0)
        self._servo_settle_until = time.time() + dt + settle
        self._move_phase = 'servo'
        self._ik_fut = None
        self._traj_goal_fut = None
        self._traj_result_fut = None
        self._servo_motion_frames = []
        self._servo_motion_last_t = 0.0
        self._pbvs_pending_tool_seq = int(tool_seq)
        self._snap_servo_motion_frame(force=True)
        self.get_logger().info(
            f'PBVS via_executor wait seq={tool_seq} traj={dt:.2f}s')

    def _tick_pbvs_rate(self) -> None:
        """25 Hz PBVS execution; skip while a blocking traj (probe) is in flight."""
        if self._state != 'REFINING' or not self._use_pbvs:
            return
        if self._move_phase is not None:
            return
        if self._pbvs_oneshot_mode():
            self._tick_refining_pbvs_oneshot()
        else:
            self._tick_refining_pbvs()
        self._record_pbvs_tick()

    def _stream_joint_goal(self, q: Sequence[float], *, traj_s: float = 0.15) -> None:
        """Fire-and-forget short FollowJointTrajectory (teleop-style streaming).

        Does NOT set ``_move_phase``, so the 25 Hz PBVS timer keeps running.
        """
        if not self._arm.server_is_ready():
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in list(q)[:6]]
        dt = float(max(0.05, traj_s))
        pt.time_from_start.sec = int(dt)
        pt.time_from_start.nanosec = int((dt - int(dt)) * 1e9)
        goal.trajectory.points = [pt]
        fut = self._arm.send_goal_async(goal)

        def _on_sent(f) -> None:
            try:
                gh = f.result()
            except Exception:
                return
            if gh is None or not gh.accepted:
                return
            prev = self._arm_goal_handle
            self._arm_goal_handle = gh
            if prev is not None and prev != gh:
                try:
                    prev.cancel_goal_async()
                except Exception:
                    pass

        fut.add_done_callback(_on_sent)

    def _pbvs_mono_depth_mode(self, mode: str) -> bool:
        m = str(mode or '').lower()
        return m in ('mono', 'mono_fallback')

    def _pbvs_rgbd_depth_mode(self, mode: str) -> bool:
        m = str(mode or '').lower()
        return m in (
            'rgbd', 'fused_rgbd', 'da2', 'fused_da2', 'depth', 'depth_raw',
            'iou_pin')

    def _pbvs_depth_trusted(
        self, mode: str, berry: Optional[DetectedBerry] = None,
    ) -> bool:
        if self._pbvs_rgbd_depth_mode(mode):
            return True
        return berry is not None and self._berry_has_rgbd(berry)

    def _pbvs_slew_target_cmd(
        self,
        raw: np.ndarray,
        *,
        max_step_m: float,
    ) -> np.ndarray:
        """Rate-limit PBVS setpoint changes (safety even if KF glitches)."""
        raw = np.asarray(raw, dtype=np.float64).flatten()[:3]
        if self._pbvs_target_cmd is None:
            self._pbvs_target_cmd = raw.copy()
            return self._pbvs_target_cmd
        delta = raw - self._pbvs_target_cmd
        dist = float(np.linalg.norm(delta))
        step = float(max(1e-6, max_step_m))
        if dist <= step:
            self._pbvs_target_cmd = raw.copy()
        else:
            self._pbvs_target_cmd = self._pbvs_target_cmd + delta * (step / dist)
        return self._pbvs_target_cmd

    def _pbvs_control_target(self) -> Optional[np.ndarray]:
        """PBVS setpoint: live/KF while vision up; last snapshot only for blind oneshot."""
        if (self._pbvs_approach_blind
                and self._pbvs_cam_snapshot is not None
                and self._pbvs_cam_snapshot.get('berry_base') is not None):
            return np.asarray(
                self._pbvs_cam_snapshot['berry_base'], dtype=np.float64)
        if self._pbvs_target_cmd is not None:
            return self._pbvs_target_cmd.copy()
        if hasattr(self, '_pbvs_kf') and self._pbvs_kf.initialized:
            return self._pbvs_kf.position
        return None

    def _pbvs_finish_contact(
        self,
        metric_m: float,
        *,
        tag: str,
        d_cam_m: Optional[float] = None,
    ) -> None:
        if getattr(self, '_pbvs_contact_finished', False):
            return
        self._pbvs_contact_finished = True
        tcp_d = None
        snap = getattr(self, '_pbvs_cam_snapshot', None) or {}
        berry_ref = snap.get('berry_base')
        if berry_ref is not None:
            tcp_d = self._cup_berry_dist_from(np.asarray(berry_ref, dtype=np.float64))
        self.get_logger().info(
            f'PBVS DONE ({tag}): d_cam={(d_cam_m if d_cam_m is not None else metric_m)*1000:.1f} mm '
            f'metric={metric_m*1000:.1f} mm '
            f'tcp_dist={(tcp_d if tcp_d is not None else -1)*1000:.1f} mm')
        self._pbvs_qa_log_event('done', {
            'tag': tag,
            'd_cam_m': float(d_cam_m if d_cam_m is not None else metric_m),
            'cup_gap_m': float(metric_m),
            'cup_dist_m': float(tcp_d) if tcp_d is not None else None,
            'snapshot_berry': berry_ref,
            'approach_blind': bool(getattr(self, '_pbvs_approach_blind', False)),
        })
        self._record_pbvs_tick(
            source='contact_done',
            cup_dist_m=tcp_d,
            cup_gap_m=metric_m,
            d_cam_m=d_cam_m,
        )
        self._pbvs_state = 'DONE'
        from std_msgs.msg import Bool as _BoolMsg
        reached = _BoolMsg()
        reached.data = True
        self._reached_pub.publish(reached)
        if self._data_collector is not None:
            self._data_collector.mark_success()
            self._data_collector.end_episode()
        self._set_state('WAIT_CONFIRM', 'pbvs_contact')

    def _log_pbvs_handoff_diag(
        self,
        *,
        frozen: np.ndarray,
        frozen_src: str,
        depth_mode: str,
        berry_xyz_base: Optional[np.ndarray],
        berry: Optional[DetectedBerry],
        cup_gap: Optional[float],
        tcp_dist: Optional[float],
        approach_dir: np.ndarray,
    ) -> None:
        """Rich handoff snapshot for frozen-anchor XY/Z forensics."""
        entry = getattr(self, '_pbvs_entry_lock_xyz', None)
        if entry is None and self._refine_fruit_anchor is not None:
            entry = np.asarray(self._refine_fruit_anchor, dtype=np.float64)
        kf = (self._pbvs_kf.position.copy()
              if hasattr(self, '_pbvs_kf') and self._pbvs_kf.initialized else None)
        target_cmd = (self._pbvs_target_cmd.copy()
                      if self._pbvs_target_cmd is not None else None)
        live = (np.asarray(berry_xyz_base, dtype=np.float64).copy()
                if berry_xyz_base is not None else None)
        frozen_a = np.asarray(frozen, dtype=np.float64).flatten()[:3]
        tcp = self._tcp_or_ee_xyz()
        cup_open = self._cup_open_xyz()
        tip = None
        q = self._joint_positions_rad()
        if q is not None:
            tip = tip_xyz(q.tolist())

        def _delta_mm(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[Dict]:
            if a is None or b is None:
                return None
            d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
            return {
                'dxyz_mm': [float(x * 1000) for x in d],
                'xy_mm': float(np.linalg.norm(d[:2]) * 1000),
                'dz_mm': float(d[2] * 1000),
                'norm_mm': float(np.linalg.norm(d) * 1000),
            }

        geo = None
        if tcp is not None and live is not None:
            geo = self._cup_berry_geometry(
                tuple(float(v) for v in live),
                tcp,
            )

        live_mode_raw = ''
        live_conf = None
        if berry is not None:
            live_mode_raw = str(getattr(berry, 'depth_mode', '') or '')
            live_conf = float(berry.confidence)

        diag: Dict = {
            'track_id': self._refine_locked_track_id,
            'frozen_source': frozen_src,
            'depth_mode_handoff': depth_mode,
            'live_depth_mode_raw': live_mode_raw,
            'live_conf': live_conf,
            'entry_lock_xyz': entry.tolist() if entry is not None else None,
            'live_handoff_xyz': live.tolist() if live is not None else None,
            'kf_xyz': kf.tolist() if kf is not None else None,
            'target_cmd_xyz': target_cmd.tolist() if target_cmd is not None else None,
            'frozen_xyz': frozen_a.tolist(),
            'approach_dir': np.asarray(approach_dir, dtype=np.float64).tolist(),
            'cup_gap_m': cup_gap,
            'tcp_dist_m': tcp_dist,
            'cup_open_xyz': cup_open.tolist() if cup_open is not None else None,
            'tip_xyz': tip.tolist() if tip is not None else None,
            'tcp_base': list(tcp) if tcp is not None else None,
            'delta_frozen_vs_entry': _delta_mm(frozen_a, entry),
            'delta_live_vs_entry': _delta_mm(live, entry),
            'delta_frozen_vs_live': _delta_mm(frozen_a, live),
            'delta_frozen_vs_kf': _delta_mm(frozen_a, kf),
            'delta_frozen_vs_target_cmd': _delta_mm(frozen_a, target_cmd),
            'berry_rel_cup': geo.get('berry_rel_cup') if geo else None,
            'berry_uv': geo.get('berry_uv') if geo else None,
            'cup_uv': geo.get('cup_uv') if geo else None,
            'kf_uncertainty': (
                float(self._pbvs_kf.uncertainty())
                if hasattr(self, '_pbvs_kf') and self._pbvs_kf.initialized
                else None),
            'kf_reject_streak': int(getattr(self, '_pbvs_kf_reject_streak', 0)),
        }
        self._write_qa_json('pbvs_handoff.json', diag)
        self._pbvs_qa_log_event('approach_handoff', diag)
        dfe = diag.get('delta_frozen_vs_entry') or {}
        dfl = diag.get('delta_frozen_vs_live') or {}
        self.get_logger().info(
            'PBVS handoff diag: '
            f'src={frozen_src} mode={depth_mode} '
            f'cup_gap={(cup_gap if cup_gap is not None else -1)*1000:.1f}mm '
            f'tcp_dist={(tcp_dist if tcp_dist is not None else -1)*1000:.1f}mm '
            f'frozen_vs_entry_xy={dfe.get("xy_mm", -1):.1f}mm '
            f'frozen_vs_live_xy={dfl.get("xy_mm", -1):.1f}mm '
            f'frozen={np.round(frozen_a, 3)}')

    def _pbvs_gap_approach_dir(self) -> Optional[np.ndarray]:
        ad = getattr(self, '_pbvs_frozen_approach_dir', None)
        if ad is None:
            ad = getattr(self, '_pbvs_approach_dir_base', None)
        if ad is None:
            return None
        a = np.asarray(ad, dtype=np.float64).flatten()[:3]
        n = float(np.linalg.norm(a))
        return a / n if n > 1e-6 else None

    def _pbvs_axial_surface_gap(self, berry_base: np.ndarray) -> Optional[float]:
        """Cup-axis gap to berry surface (m). Primary metric for oneshot depth_final."""
        adir = self._pbvs_gap_approach_dir()
        gap_axis = self._cup_gap_along_axis(
            np.asarray(berry_base, dtype=np.float64), approach_dir=adir)
        if gap_axis is None:
            return None
        return max(0.0, float(gap_axis[0]))

    def _pbvs_measure_d_cam_surface(
        self,
        *,
        live_only: bool = False,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Cam-frame cup↔berry surface gap (m). Returns (d_surface, live_cam_or_none).

        Oneshot / ``live_only``: never use frozen axial_gap — cam depth or cam-frame
        reproject only (industry pre-grasp + depth-final pattern).
        """
        frozen = getattr(self, '_pbvs_frozen_berry', None)
        tcp = self._tcp_or_ee_xyz()

        berry = self._pick_probe_live_berry()
        live_d = None
        if berry is not None:
            base = self._berry_base_xyz(berry)
            if base is not None:
                live_d = self._pbvs_live_d_cam_surface(
                    berry, np.asarray(base, dtype=np.float64))

        if self._pbvs_oneshot_mode() or live_only:
            if live_d is not None:
                return float(live_d), float(live_d)
            if frozen is not None:
                d_geom = self._pbvs_d_cam_surface_from_base(
                    np.asarray(frozen, dtype=np.float64))
                if d_geom is not None:
                    return d_geom, None
            return None, None

        axial = None
        if frozen is not None:
            axial = self._pbvs_axial_surface_gap(np.asarray(frozen, dtype=np.float64))

        if live_d is not None and tcp is not None and frozen is not None:
            tcp_dist = float(np.linalg.norm(
                np.asarray(frozen, dtype=np.float64) - np.asarray(tcp, dtype=np.float64)))
            if live_d > 0.12 and tcp_dist < self._tip_contact_standoff_m() + 0.04:
                if axial is not None:
                    return axial, None
            return float(live_d), float(live_d)

        if axial is not None:
            return axial, None

        if frozen is not None:
            d_geom = self._pbvs_d_cam_surface_from_base(
                np.asarray(frozen, dtype=np.float64))
            if d_geom is not None:
                return d_geom, None
        return None, None

    def _pbvs_send_cup_axis_oneshot(
        self,
        berry: np.ndarray,
        travel_m: float,
        *,
        tag: str,
        pending: str,
        d_cam_surface: Optional[float] = None,
        lookat: bool = True,
        allow_tip_residual: bool = True,
        dir_base: Optional[np.ndarray] = None,
    ) -> bool:
        """Blocking cup_axis oneshot; ``pending`` identifies phase on settle."""
        if not self._pbvs_via_executor and not self._arm.server_is_ready():
            return False
        travel = float(travel_m)
        min_travel = self._pbvs_min_oneshot_travel_m()
        if travel < min_travel:
            return False
        tcp = self._tcp_or_ee_xyz()
        ee = self._current_ee_pose()
        if tcp is None or ee is None:
            return False
        berry_t = tuple(float(v) for v in np.asarray(berry, dtype=np.float64).flatten()[:3])
        if dir_base is not None:
            d = np.asarray(dir_base, dtype=np.float64).flatten()[:3]
            nd = float(np.linalg.norm(d))
            if nd < 1e-9:
                return False
            d_unit = d / nd
            d_base = d_unit * float(travel)
            T = self._lookup_T_base_cam()
            if T is not None:
                d_cam = T[:3, :3].T @ d_base
            else:
                d_cam = np.array([0.0, 0.0, float(travel)], dtype=np.float64)
            method = 'along_normal'
            _plan_meta = {'dir_base': d_unit.tolist()}
            planned = (d_cam, d_base, method, _plan_meta)
        elif self._pbvs_direct_mode():
            planned = self._plan_tip_to_surface_delta(berry_t, travel)
        elif self._pbvs_oneshot_mode():
            planned = self._plan_cup_axis_oneshot_delta(berry_t, travel)
        else:
            planned = self._plan_near_oneshot_delta(berry_t, tcp, travel)
        if planned is None:
            self.get_logger().warn(f'PBVS {tag}: cup-axis plan failed')
            return False
        _d_cam, d_base, method, _plan_meta = planned
        ee0 = np.array([
            float(ee.pose.position.x),
            float(ee.pose.position.y),
            float(ee.pose.position.z),
        ], dtype=np.float64)
        tcp0 = np.array([float(tcp[0]), float(tcp[1]), float(tcp[2])], dtype=np.float64)
        tcp_goal = tcp0 + np.asarray(d_base, dtype=np.float64)

        # Approach axis for planner I/O: prefer −n, else tip→berry.
        approach_axis = None
        n_aim_pub = getattr(self, '_pbvs_frozen_normal', None)
        if n_aim_pub is not None:
            n_u = np.asarray(n_aim_pub, dtype=np.float64).flatten()[:3]
            nn = float(np.linalg.norm(n_u))
            if nn > 1e-9:
                approach_axis = (-n_u / nn).tolist()
        if approach_axis is None:
            look = np.asarray(berry_t, dtype=np.float64) - tcp_goal
            ln = float(np.linalg.norm(look))
            if ln > 1e-9:
                approach_axis = (look / ln).tolist()
            else:
                axis = self._cup_axis_base()
                approach_axis = (
                    axis.tolist() if axis is not None else [0.0, 0.0, 1.0])

        if tag == 'pbvs_single':
            traj_s = float(self._args.align_traj_s)
        elif tag == 'pbvs_single_residual':
            traj_s = float(self._args.servo_traj_s)
        else:
            traj_s = max(0.8, min(4.0, float(travel) / 0.04))

        tool_seq = self._publish_pbvs_tool_trajectory(
            tcp0.tolist(), tcp_goal.tolist(), approach_axis,
            move_duration_s=float(traj_s), tag=tag)

        if self._pbvs_via_executor:
            self._pbvs_oneshot_pending = pending
            pkg = {
                'berry_base': list(berry_t),
                'tcp_start': list(tcp0),
                'tcp_goal': list(tcp_goal),
                'travel_m': float(travel),
                'd_cam_surface_m': float(d_cam_surface) if d_cam_surface is not None else None,
                'method': str(method),
                'traj_s': float(traj_s),
                'phase': pending,
                'via_executor': True,
                'tool_traj_seq': int(tool_seq),
                'approach_axis': list(approach_axis),
            }
            self._write_qa_json(f'pbvs_{pending}.json', pkg)
            self._pbvs_qa_log_event(f'oneshot_{pending}', pkg)
            self._begin_pbvs_executor_wait(traj_s=float(traj_s), tool_seq=int(tool_seq))
            self.get_logger().info(
                f'PBVS {tag}: via_executor travel={travel*1000:.1f}mm '
                f'traj={traj_s:.1f}s seq={tool_seq}')
            return True

        joints = None
        ik_method = 'keep_orient'
        aim_t = berry_t
        use_cup_lookat = lookat and (
            self._pbvs_oneshot_mode() or travel >= 0.03 or dir_base is not None)
        if use_cup_lookat:
            T_l6c = self._lookup_T_link6_cam()
            if T_l6c is None:
                T_l6c = np.eye(4, dtype=np.float64)
                T_l6c[:3, 3] = [0.0, -0.08, -0.04]
            n_aim = getattr(self, '_pbvs_frozen_normal', None)
            if n_aim is not None:
                # Goal pose: cup at P, link6+Z ∥ −n (colinear cup / patch / center).
                # One 4s traj interpolates both; do not rotate after arrive.
                n_u = np.asarray(n_aim, dtype=np.float64).flatten()[:3]
                nn = float(np.linalg.norm(n_u))
                if nn > 1e-9:
                    n_u = n_u / nn
                    aim_t = tuple(
                        (np.asarray(berry_t, dtype=np.float64) - n_u * 0.08).tolist())
            joints = self._approach_axis_ik_for_tcp_goal(
                tcp_goal=tcp_goal,
                berry=berry_t,
                T_l6c=T_l6c,
                seed=list(self._joints),
                d_base=d_base,
                contact_m=self._tip_contact_standoff_m(),
                allow_tip_residual=allow_tip_residual,
                aim_xyz=aim_t,
            )
            if joints is not None:
                ik_method = 'cup_axis_lookat_n' if n_aim is not None else 'cup_axis_lookat'
        if joints is None:
            tgt_xyz = tuple((ee0 + d_base).tolist())
            joints = position_ik_keep_orient(tgt_xyz, self._joints)
            if joints is None:
                joints = position_ik_keep_orient_chunked(tgt_xyz, self._joints)
                ik_method = 'keep_orient_chunked'
        if joints is None:
            self.get_logger().warn(f'PBVS {tag}: IK failed')
            return False
        target = self._clamp_servo_target(list(joints))
        ok = self._send_joint_servo_goal(
            target,
            tag=tag,
            traj_s=traj_s,
            extra=(
                f'{ik_method} travel={travel:.3f}m plan={method} '
                f'traj={traj_s:.1f}s '
                f'd_cam={(d_cam_surface if d_cam_surface is not None else -1):.3f}'))
        if ok:
            self._pbvs_oneshot_pending = pending
            pkg = {
                'berry_base': list(berry_t),
                'tcp_start': list(tcp0),
                'tcp_goal': list(tcp_goal),
                'travel_m': float(travel),
                'd_cam_surface_m': float(d_cam_surface) if d_cam_surface is not None else None,
                'method': str(method),
                'traj_s': float(traj_s),
                'phase': pending,
                'n_base': (
                    np.asarray(self._pbvs_frozen_normal, dtype=np.float64).tolist()
                    if getattr(self, '_pbvs_frozen_normal', None) is not None else None),
                'dir_base': (
                    np.asarray(dir_base, dtype=np.float64).tolist()
                    if dir_base is not None else None),
                'aim_base': list(aim_t) if lookat else None,
                'tool_traj_seq': int(tool_seq),
            }
            self._write_qa_json(f'pbvs_{pending}.json', pkg)
            self._pbvs_qa_log_event(f'oneshot_{pending}', pkg)
            self.get_logger().info(
                f'PBVS {tag}: travel={travel*1000:.1f}mm '
                f'd_cam={(d_cam_surface if d_cam_surface is not None else -1)*1000:.1f}mm '
                f'traj={traj_s:.1f}s')
        return ok

    def _pbvs_start_approach_oneshot(
        self,
        berry: np.ndarray,
        *,
        d_cam_surface: Optional[float] = None,
    ) -> bool:
        """Pre-grasp approach: cup_axis oneshot until cam d_cam_surface ≈ pre-grasp."""
        berry_a = np.asarray(berry, dtype=np.float64).flatten()[:3]
        pre = self._pbvs_pre_grasp_d_cam_m()
        d_cam = d_cam_surface
        if d_cam is None:
            live = self._pick_probe_live_berry()
            base = self._berry_base_xyz(live) if live is not None else None
            if live is not None and base is not None:
                d_cam = self._pbvs_live_d_cam_surface(live, np.asarray(base))
            if d_cam is None:
                snap = getattr(self, '_pbvs_cam_snapshot', None) or {}
                d_cam = snap.get('d_cam_surface_m')
        if d_cam is None:
            self.get_logger().warn('PBVS approach: no d_cam_surface at lock')
            return False
        travel = float(d_cam) - pre
        if travel < 0.002:
            self.get_logger().info(
                f'PBVS approach skip: d_cam={d_cam*1000:.1f}mm '
                f'≤ pre_grasp={pre*1000:.1f}mm')
            return False
        return self._pbvs_send_cup_axis_oneshot(
            berry_a, travel,
            tag='pbvs_approach',
            pending='approach',
            d_cam_surface=float(d_cam),
            lookat=True,
            allow_tip_residual=False,
        )

    def _pbvs_travel_to_surface_m(
        self,
        berry: np.ndarray,
    ) -> Optional[float]:
        """3D cup-opening → locked surface contact point (m).

        ``berry`` is the depth-backprojected surface point at bbox center, not a
        sphere center — travel is full cup↔point distance minus clearance only."""
        cup = self._cup_open_xyz()
        if cup is None:
            return None
        berry_a = np.asarray(berry, dtype=np.float64).flatten()[:3]
        cup_a = np.asarray(cup, dtype=np.float64).flatten()[:3]
        dist = float(np.linalg.norm(berry_a - cup_a))
        return dist - self._tip_contact_standoff_m()

    def _pbvs_direct_travel_m(self, berry: np.ndarray) -> Optional[float]:
        """Main-push travel: cup opening → locked point P (position first)."""
        return self._pbvs_travel_to_surface_m(berry)

    def _pbvs_press_m(self) -> float:
        v = float(getattr(self._args, 'pbvs_press_m', 0.0) or 0.0)
        return max(0.0, min(0.005, v))

    def _pbvs_try_ray_scale(self, live: Optional[DetectedBerry]) -> None:
        """During 4s oneshot: if live UV + baseline, 1-DOF scale lock ray → P1.

        Does not cancel the 4s traj. P1 is consumed by residual shots.
        Keeps the result with smallest gap_m across all ticks (best triangulation).
        """
        ray = getattr(self, '_pbvs_lock_ray', None)
        P0 = getattr(self, '_pbvs_frozen_berry', None)
        if ray is None or P0 is None or live is None:
            return
        mode = str(getattr(live, 'depth_mode', '') or '')
        if mode == 'base_coast':
            return
        uv = self._berry_image_uv(live)
        T1 = self._lookup_T_base_cam()
        if uv is None or T1 is None:
            return
        from pbvs_ray_scale import scale_lock_ray
        o0, d0 = ray
        rec = scale_lock_ray(
            o0, d0, T1, uv, self._wrist_intrinsics(),
            np.asarray(P0, dtype=np.float64),
            min_lat_m=0.018, max_gap_m=0.010, max_dp_m=0.012)
        rec['live_uv'] = [float(uv[0]), float(uv[1])]
        rec['live_mode'] = mode
        rec['live_conf'] = float(getattr(live, 'confidence', 0.0) or 0.0)
        self._pbvs_ray_scale = rec
        if rec.get('ok') and rec.get('apply') and rec.get('P1') is not None:
            cur_gap = float(rec.get('gap_m', float('inf')))
            prev_gap = getattr(self, '_pbvs_ray_scale_best_gap', float('inf'))
            if cur_gap < prev_gap:
                self._pbvs_ray_scale_best_gap = cur_gap
                self._pbvs_scaled_berry = np.asarray(rec['P1'], dtype=np.float64)
                self._write_qa_json('pbvs_ray_scale.json', rec)
                self._pbvs_qa_log_event('ray_scale', rec)
                self.get_logger().info(
                    f'PBVS ray-scale APPLY |ΔP|={float(rec["dp_m"])*1000:.1f}mm '
                    f'lat={float(rec["lat_m"])*1000:.1f}mm '
                    f'skew={float(rec["gap_m"])*1000:.1f}mm '
                    f's {float(rec["s0_m"])*1000:.0f}→{float(rec["s1_m"])*1000:.0f}mm '
                    f'(prev_gap={prev_gap*1000:.1f}mm)')
        elif rec.get('lat_m', 0.0) >= 0.018:
            self._write_qa_json('pbvs_ray_scale.json', rec)

    def _pbvs_freeze_surface_normal(
        self,
        berry: Optional[DetectedBerry],
        berry_xyz_base: np.ndarray,
    ) -> None:
        """Lock-time PCA plane; freeze outward n (toward cup) or skip."""
        self._pbvs_frozen_normal = None
        self._pbvs_surface_fit = None
        qa: Dict = {'ok': False, 'skip': True, 'reason': ''}
        depth = getattr(self, '_last_wrist_depth', None)
        if depth is None:
            qa['reason'] = 'no_wrist_depth'
            self._write_qa_json('pbvs_surface_fit.json', qa)
            self.get_logger().warn(
                'PBVS surface_fit skip: no wrist depth — point contact')
            return
        uv = self._berry_image_uv(berry) if berry is not None else None
        if uv is None:
            cam = self._base_xyz_to_cam(
                tuple(float(v) for v in berry_xyz_base.flatten()[:3]))
            if cam is not None:
                uv = self._berry_uv_from_cam(cam)
        if uv is None:
            qa['reason'] = 'no_uv'
            self._write_qa_json('pbvs_surface_fit.json', qa)
            self.get_logger().warn('PBVS surface_fit skip: no UV — point contact')
            return
        fx, fy, cx, cy = self._wrist_intrinsics()
        T = self._lookup_T_base_cam()
        if T is None:
            qa['reason'] = 'no_tf'
            self._write_qa_json('pbvs_surface_fit.json', qa)
            self.get_logger().warn('PBVS surface_fit skip: no TF — point contact')
            return
        p = np.asarray(berry_xyz_base, dtype=np.float64).flatten()[:3]
        cup = self._cup_open_xyz()
        if cup is not None:
            toward = np.asarray(cup, dtype=np.float64).flatten()[:3] - p
        else:
            toward = -(T[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
        z_hint = None
        if berry is not None:
            bcam = self._berry_cam_xyz(berry)
            if bcam is not None and float(bcam[2]) > 0.05:
                z_hint = float(bcam[2])
        qa_dir = self._qa_session_dir()
        if qa_dir is not None:
            try:
                np.save(
                    os.path.join(qa_dir, 'pbvs_surface_fit_depth.npy'),
                    np.asarray(depth, dtype=np.float32))
                qa['depth_npy'] = 'pbvs_surface_fit_depth.npy'
                qa['depth_shape'] = [int(depth.shape[0]), int(depth.shape[1])]
            except Exception:
                pass
        qa['K'] = [float(fx), float(fy), float(cx), float(cy)]
        qa['T_base_cam'] = np.asarray(T, dtype=np.float64).tolist()
        qa['toward_base'] = np.asarray(toward, dtype=np.float64).tolist()
        fit = fit_contact_surface(
            depth, float(uv[0]), float(uv[1]),
            fx=fx, fy=fy, cx=cx, cy=cy,
            T_base_cam=T,
            toward_base=toward,
            z_hint_m=z_hint,
        )
        qa.update(fit)
        qa['skip'] = not bool(fit.get('ok'))
        qa['lock_uv'] = [float(uv[0]), float(uv[1])]
        qa['lock_berry_base'] = p.tolist()
        if fit.get('ok') and fit.get('n_base') is not None:
            n = np.asarray(fit['n_base'], dtype=np.float64).flatten()[:3]
            nn = float(np.linalg.norm(n))
            if nn > 1e-9:
                n = n / nn
            # Sign only: n outward toward cup. Do not clamp/skip vs cup−P or
            # lock-time cup_axis — those are not fit quality. Orientation
            # coincidence is a post-arrive traj.
            self._pbvs_frozen_normal = n
            qa['n_base'] = n.tolist()
            qa['n_raw_base'] = n.tolist()
            qa['ok'] = True
            qa['skip'] = False
            qa['reason'] = 'ok'
            axis = self._cup_axis_base()
            if axis is not None:
                c = float(np.clip(np.dot(n, -axis), -1.0, 1.0))
                qa['angle_to_cup_axis_at_lock_deg'] = math.degrees(math.acos(c))
            # Refine P0 with surface-fit centroid: avoids depth-edge bleeding in
            # depth_in_mask. contact_base uses only center+ring pixels filtered
            # to |z - z0| < 8mm, giving a cleaner depth estimate.
            cb = fit.get('contact_base')
            if cb is not None and self._pbvs_frozen_berry is not None:
                cb_arr = np.asarray(cb, dtype=np.float64).flatten()[:3]
                delta = float(np.linalg.norm(cb_arr - self._pbvs_frozen_berry))
                qa['p0_delta_m'] = float(delta)
                if delta < 0.020:
                    old_p0 = self._pbvs_frozen_berry.copy()
                    self._pbvs_frozen_berry = cb_arr.copy()
                    qa['p0_refined'] = True
                    self.get_logger().info(
                        f'P0 refined by surface_fit: Δ={delta*1000:.1f}mm '
                        f'{np.round(old_p0, 3)} → {np.round(cb_arr, 3)}')
                else:
                    qa['p0_refined'] = False
                    self.get_logger().warn(
                        f'P0 surface_fit delta={delta*1000:.1f}mm > 20mm — skip')
        self._pbvs_surface_fit = qa
        self._write_qa_json('pbvs_surface_fit.json', qa)
        if self._pbvs_frozen_normal is not None:
            self.get_logger().info(
                f'PBVS surface_fit n={np.round(self._pbvs_frozen_normal, 3)} '
                f'inliers={qa.get("n_inliers")} '
                f'ang_cup_lock={qa.get("angle_to_cup_axis_at_lock_deg")}deg')
        else:
            self.get_logger().warn(
                f'PBVS surface_fit skip ({qa.get("reason")}) — point contact')

    def _pbvs_angle_cup_to_neg_n_deg(self) -> Optional[float]:
        axis = self._cup_axis_base()
        n = getattr(self, '_pbvs_frozen_normal', None)
        if axis is None or n is None:
            return None
        n_u = np.asarray(n, dtype=np.float64).flatten()[:3]
        nn = float(np.linalg.norm(n_u))
        if nn < 1e-9:
            return None
        c = float(np.clip(np.dot(axis, -n_u / nn), -1.0, 1.0))
        return math.degrees(math.acos(c))

    def _pbvs_start_align_n(self) -> bool:
        """Hold cup position, rotate link6 +Z onto −n (colinear cup / patch / center)."""
        n = getattr(self, '_pbvs_frozen_normal', None)
        berry = getattr(self, '_pbvs_frozen_berry', None)
        if n is None or berry is None:
            return False
        if not self._arm.server_is_ready():
            return False
        ang0 = self._pbvs_angle_cup_to_neg_n_deg()
        if ang0 is not None and ang0 < 5.0:
            self.get_logger().info(
                f'PBVS align_n skip: already {ang0:.1f}deg (link6+Z ∥ −n)')
            return False
        ee = self._current_ee_pose()
        tcp = self._tcp_or_ee_xyz()
        if ee is None or tcp is None:
            return False
        n_u = np.asarray(n, dtype=np.float64).flatten()[:3]
        n_u = n_u / max(float(np.linalg.norm(n_u)), 1e-12)
        p = np.asarray(berry, dtype=np.float64).flatten()[:3]
        # Aim through P along −n so cup / contact / sphere-center are colinear.
        aim = tuple((p - n_u * 0.08).tolist())
        ee_tgt = (
            float(ee.pose.position.x),
            float(ee.pose.position.y),
            float(ee.pose.position.z),
        )
        seed = list(self._joints)
        T0 = fk_link6_T(seed)
        tcp0 = np.asarray(tcp, dtype=np.float64).flatten()[:3]
        off_l6 = T0[:3, :3].T @ (tcp0 - T0[:3, 3])
        ik_kw = {
            'tol_m': self._pbvs_ik_tol_m(),
            'w_pos': 0.5,
            'w_dir': 1.0,
            'tol_dir_rad': math.radians(4.0),
        }
        joints = cup_axis_ik(
            ee_tgt, aim, seed, tip_offset_link6=off_l6, **ik_kw)
        ik_method = 'cup_axis_hold_pos'
        if joints is None:
            joints = cup_axis_ik(ee_tgt, aim, seed, **ik_kw)
            ik_method = 'cup_axis'
        if joints is None:
            self.get_logger().warn('PBVS align_n: IK failed')
            return False
        target = self._clamp_servo_target(list(joints))
        traj_s = float(self._args.servo_traj_s)
        ok = self._send_joint_servo_goal(
            target, tag='pbvs_align_n', traj_s=traj_s,
            extra=(
                f'{ik_method} hold-pos axis→−n '
                f'from={(ang0 if ang0 is not None else -1):.1f}deg '
                f'traj={traj_s:.1f}s'))
        if ok:
            self._pbvs_oneshot_pending = 'align_n'
            self._pbvs_state = 'ALIGN_N'
            pkg = {
                'n_base': n_u.tolist(),
                'aim_base': list(aim),
                'berry_base': p.tolist(),
                'tcp_hold': tcp0.tolist(),
                'ee_hold': list(ee_tgt),
                'angle_before_deg': ang0,
                'traj_s': traj_s,
                'ik_method': ik_method,
            }
            self._write_qa_json('pbvs_align_n.json', pkg)
            self._pbvs_qa_log_event('align_n_start', pkg)
            self.get_logger().info(
                f'PBVS align_n: rotate link6+Z onto −n '
                f'({(ang0 if ang0 is not None else -1):.1f}deg) '
                f'traj={traj_s:.1f}s hold position')
        return ok

    def _pbvs_on_align_n_settle(self) -> None:
        ang = self._pbvs_angle_cup_to_neg_n_deg()
        cup = self._cup_open_xyz()
        tcp = self._tcp_or_ee_xyz()
        frozen = getattr(self, '_pbvs_frozen_berry', None)
        cup_lock = None
        if cup is not None and frozen is not None:
            cup_lock = float(np.linalg.norm(
                np.asarray(frozen, dtype=np.float64).flatten()[:3]
                - np.asarray(cup, dtype=np.float64).flatten()[:3]))
        arrive = {
            'angle_cup_axis_to_neg_n_deg': ang,
            'n_base': (
                np.asarray(self._pbvs_frozen_normal, dtype=np.float64).tolist()
                if getattr(self, '_pbvs_frozen_normal', None) is not None else None),
            'cup_axis_base': (
                self._cup_axis_base().tolist()
                if self._cup_axis_base() is not None else None),
            'cup_berry_dist_m': cup_lock,
            'tcp_base': list(tcp) if tcp is not None else None,
            'cup_open_xyz': list(cup) if cup is not None else None,
        }
        self._snap_qa(f'servo_{self._servo_step_idx:02d}_align_n')
        self._servo_step_idx += 1
        self._write_qa_json('pbvs_align_n_arrive.json', arrive)
        self._pbvs_qa_log_event('align_n_arrive', arrive)
        self.get_logger().info(
            'PBVS align_n ARRIVE  '
            f'axis∠−n={(ang if ang is not None else -1):.1f}deg '
            f'cup↔lock={(cup_lock if cup_lock is not None else -1)*1000:.1f}mm')
        self._pbvs_after_align_n()

    def _pbvs_start_press(self) -> bool:
        """Short keep-orient traj: +press_m into fruit along −n."""
        press = self._pbvs_press_m()
        n = getattr(self, '_pbvs_frozen_normal', None)
        if press < 1e-5 or n is None:
            return False
        if not self._arm.server_is_ready():
            return False
        ee = self._current_ee_pose()
        tcp = self._tcp_or_ee_xyz()
        cup = self._cup_open_xyz()
        if ee is None or tcp is None:
            return False
        n_a = np.asarray(n, dtype=np.float64).flatten()[:3]
        nn = float(np.linalg.norm(n_a))
        if nn < 1e-9:
            return False
        n_a = n_a / nn
        d_base = -n_a * press
        ee0 = np.array([
            float(ee.pose.position.x),
            float(ee.pose.position.y),
            float(ee.pose.position.z),
        ], dtype=np.float64)
        tgt = tuple((ee0 + d_base).tolist())
        joints = position_ik_keep_orient(tgt, self._joints)
        ik_method = 'keep_orient'
        if joints is None:
            joints = position_ik_keep_orient_chunked(tgt, self._joints)
            ik_method = 'keep_orient_chunked'
        if joints is None:
            self.get_logger().warn('PBVS press: IK failed')
            return False
        target = self._clamp_servo_target(list(joints))
        traj_s = 0.8
        tcp0 = np.asarray(tcp, dtype=np.float64).flatten()[:3]
        self._pbvs_press_cup_start = (
            np.asarray(cup, dtype=np.float64).flatten()[:3].copy()
            if cup is not None else None)
        ok = self._send_joint_servo_goal(
            target, tag='pbvs_press', traj_s=traj_s,
            extra=(
                f'{ik_method} press={press*1000:.1f}mm along -n '
                f'traj={traj_s:.1f}s'))
        if ok:
            self._pbvs_oneshot_pending = 'press'
            self._pbvs_state = 'PRESS'
            pkg = {
                'press_m': press,
                'n_base': n_a.tolist(),
                'd_base': d_base.tolist(),
                'tcp_start': tcp0.tolist(),
                'tcp_goal': (tcp0 + d_base).tolist(),
                'cup_start': (
                    self._pbvs_press_cup_start.tolist()
                    if self._pbvs_press_cup_start is not None else None),
                'traj_s': traj_s,
                'ik_method': ik_method,
            }
            self._write_qa_json('pbvs_press.json', pkg)
            self._pbvs_qa_log_event('press_start', pkg)
            self.get_logger().info(
                f'PBVS press: {press*1000:.1f}mm along -n traj={traj_s:.1f}s')
        return ok

    def _pbvs_on_press_settle(self) -> None:
        n = getattr(self, '_pbvs_frozen_normal', None)
        cup = self._cup_open_xyz()
        cup0 = getattr(self, '_pbvs_press_cup_start', None)
        along_n = None
        into_m = None
        if cup is not None and cup0 is not None and n is not None:
            dcup = (np.asarray(cup, dtype=np.float64).flatten()[:3]
                    - np.asarray(cup0, dtype=np.float64).flatten()[:3])
            n_a = np.asarray(n, dtype=np.float64).flatten()[:3]
            along_n = float(np.dot(dcup, n_a))
            into_m = -along_n
        arrive = {
            'press_m_cmd': self._pbvs_press_m(),
            'delta_cup_along_n_m': along_n,
            'press_into_m': into_m,
            'n_base': (
                np.asarray(n, dtype=np.float64).tolist() if n is not None else None),
            'cup_start': (
                np.asarray(cup0, dtype=np.float64).tolist()
                if cup0 is not None else None),
            'cup_end': list(cup) if cup is not None else None,
        }
        self._snap_qa(f'servo_{self._servo_step_idx:02d}_press_arrive')
        self._servo_step_idx += 1
        self._write_qa_json('pbvs_press_arrive.json', arrive)
        self._pbvs_qa_log_event('press_arrive', arrive)
        self.get_logger().info(
            'PBVS press ARRIVE  '
            f'along_n={(along_n if along_n is not None else 0)*1000:.2f}mm '
            f'into={(into_m if into_m is not None else 0)*1000:.2f}mm '
            f'(cmd={self._pbvs_press_m()*1000:.1f}mm) — ignore cup↔lock')
        self._pbvs_enter_suction_hold()

    def _pbvs_enter_suction_hold(self) -> None:
        self._pbvs_state = 'SUCTION_HOLD'
        pkg = {
            'gpio': False,
            'placeholder': True,
            'note': 'suction placeholder — no GPIO this round',
        }
        self._write_qa_json('pbvs_suction_hold.json', pkg)
        self._pbvs_qa_log_event('suction_placeholder', pkg)
        self.get_logger().info(
            'PBVS SUCTION_HOLD placeholder (no GPIO) → WAIT_CONFIRM')
        self._set_state('WAIT_CONFIRM', 'pbvs_suction_hold')

    def _pbvs_after_direct_arrive(self) -> None:
        """Main 4s traj already aimed −n; optional press then suction."""
        self._pbvs_after_align_n()

    def _pbvs_after_align_n(self) -> None:
        press = self._pbvs_press_m()
        n = getattr(self, '_pbvs_frozen_normal', None)
        if press >= 1e-5 and n is not None:
            if self._pbvs_start_press():
                return
            self.get_logger().warn(
                'PBVS press send failed — suction placeholder at current pose')
        elif press >= 1e-5 and n is None:
            self.get_logger().info(
                'PBVS press skip: no frozen normal — point contact')
        self._pbvs_enter_suction_hold()

    def _pbvs_start_direct_oneshot(
        self,
        berry: np.ndarray,
        *,
        d_cam_surface: Optional[float] = None,
    ) -> bool:
        """Single segment: lock → oneshot with tip on berry surface."""
        berry_a = np.asarray(berry, dtype=np.float64).flatten()[:3]
        travel = self._pbvs_direct_travel_m(berry_a)
        if travel is None:
            self.get_logger().warn('PBVS direct: no cup opening')
            return False
        standoff = self._tip_contact_standoff_m()
        exec_tol = self._pbvs_contact_exec_tol_m()
        if travel < exec_tol:
            self.get_logger().info(
                f'PBVS direct skip: already at surface '
                f'(travel<{exec_tol*1000:.1f}mm standoff={standoff*1000:.1f}mm)')
            return False
        axis = self._cup_axis_base()
        tcp = self._tcp_or_ee_xyz()
        cup = self._cup_open_xyz()
        n = getattr(self, '_pbvs_frozen_normal', None)
        plan_method = 'tip_to_surface'
        plan_diag = {
            'berry_base': list(berry_a),
            'd_cam_surface_lock_m': (
                float(d_cam_surface) if d_cam_surface is not None else None),
            'tip_standoff_m': float(standoff),
            'surface_clearance_m': self._cup_surface_clearance_m(),
            'berry_radius_m': self._berry_radius_m(),
            'travel_m': float(travel),
            'plan_method': plan_method,
            'n_base': (
                np.asarray(n, dtype=np.float64).tolist() if n is not None else None),
            'cup_axis_base': axis.tolist() if axis is not None else None,
            'cup_open_start': list(cup) if cup is not None else None,
            'tcp_start': list(tcp) if tcp is not None else None,
        }
        self._write_qa_json('pbvs_direct_plan.json', plan_diag)
        self._pbvs_qa_log_event('direct_plan', plan_diag)
        self.get_logger().info(
            f'PBVS single PLAN  {plan_method} travel={travel*1000:.1f}mm '
            f'(standoff={standoff*1000:.1f}mm = r+clearance)')
        if d_cam_surface is not None:
            self._pbvs_motion_min_d_cam = float(d_cam_surface)
        self._pbvs_direct_shot_n = 1
        return self._pbvs_send_cup_axis_oneshot(
            berry_a, travel,
            tag='pbvs_single',
            pending='direct',
            d_cam_surface=(
                float(d_cam_surface) if d_cam_surface is not None else None),
            lookat=True,
            allow_tip_residual=True,
        )

    def _pbvs_run_depth_final(self) -> None:
        """After approach oneshot: one depth sample → short final oneshot or DONE."""
        if self._move_phase is not None or getattr(self, '_pbvs_depth_final_running', False):
            return
        self._pbvs_depth_final_running = True
        try:
            surface = self._cup_surface_clearance_m()
            d_surf, live_d = self._pbvs_measure_d_cam_surface(live_only=True)
            if d_surf is None:
                self._set_state('ERROR', 'pbvs: depth_final — no d_cam (live or TF)')
                return
            self._pbvs_qa_log_event('depth_final_sample', {
                'd_cam_surface_m': float(d_surf),
                'live': live_d is not None,
                'pre_grasp_d_cam_m': self._pbvs_pre_grasp_d_cam_m(),
            })
            self.get_logger().info(
                f'PBVS depth_final sample: d_cam_surface={d_surf*1000:.1f}mm '
                f'live_cam={(live_d if live_d is not None else -1)*1000:.1f}mm '
                f'({"live" if live_d is not None else "cam_reproj"})')
            if d_surf <= surface + 0.002:
                self._snap_qa(f'servo_{self._servo_step_idx:02d}_contact')
                self._pbvs_finish_contact(
                    float(d_surf), tag='depth_at_standoff', d_cam_m=live_d or d_surf)
                return
            frozen = getattr(self, '_pbvs_frozen_berry', None)
            if frozen is None:
                self._set_state('ERROR', 'pbvs: depth_final — no frozen berry')
                return
            travel = float(d_surf) - surface
            # No hard 50mm cap: open-loop final must close the measured gap.
            # (Former 50mm safety cap left ~3cm short when pre-grasp residual was large.)
            if travel > 0.20:
                self.get_logger().warn(
                    f'PBVS depth_final travel large: {travel*1000:.0f}mm '
                    f'(still executing; check depth)')
            self._pbvs_state = 'FINAL_ONESHOT'
            if not self._pbvs_send_cup_axis_oneshot(
                    frozen, travel,
                    tag='pbvs_depth_final',
                    pending='final',
                    d_cam_surface=d_surf,
                    lookat=True,
                    allow_tip_residual=True):
                self._set_state('ERROR', 'pbvs: depth_final oneshot failed')
        finally:
            if self._pbvs_state not in ('FINAL_ONESHOT', 'DONE'):
                self._pbvs_depth_final_running = False

    def _pbvs_on_oneshot_settle(self, pending: str) -> None:
        """Called from _poll_wrist_servo when a PBVS oneshot traj finishes."""
        if pending == 'direct':
            surface = self._cup_surface_clearance_m()
            d_surf, live_d = self._pbvs_measure_d_cam_surface(live_only=True)
            min_d = getattr(self, '_pbvs_motion_min_d_cam', None)
            frozen = getattr(self, '_pbvs_frozen_berry', None)
            tcp = self._tcp_or_ee_xyz()
            cup = self._cup_open_xyz()
            tcp_dist = None
            cup_berry_dist = None
            cup_berry_dxyz = None
            axial_gap = None
            if frozen is not None and tcp is not None:
                tcp_dist = float(np.linalg.norm(
                    np.asarray(frozen, dtype=np.float64) - np.asarray(tcp, dtype=np.float64)))
                axial_gap = self._pbvs_axial_surface_gap(
                    np.asarray(frozen, dtype=np.float64))
            if frozen is not None and cup is not None:
                dvec = np.asarray(frozen, dtype=np.float64).flatten()[:3] - np.asarray(
                    cup, dtype=np.float64).flatten()[:3]
                cup_berry_dxyz = [float(v) for v in dvec]
                cup_berry_dist = float(np.linalg.norm(dvec))
            live_berry_base = None
            cup_live_dist = None
            cup_live_dxyz = None
            live = self._pick_probe_live_berry()
            if live is not None and cup is not None:
                try:
                    live_base = self._berry_base_xyz(live)
                    if live_base is not None:
                        live_berry_base = list(live_base)
                        dvec_l = np.asarray(live_base, dtype=np.float64).flatten()[:3] - np.asarray(
                            cup, dtype=np.float64).flatten()[:3]
                        cup_live_dxyz = [float(v) for v in dvec_l]
                        cup_live_dist = float(np.linalg.norm(dvec_l))
                except Exception:
                    pass
            self._snap_qa(f'servo_{self._servo_step_idx:02d}_direct_arrive')
            self._servo_step_idx += 1
            axis_now = self._cup_axis_base()
            n_now = getattr(self, '_pbvs_frozen_normal', None)
            ang_axis_n = None
            if axis_now is not None and n_now is not None:
                n_u = np.asarray(n_now, dtype=np.float64).flatten()[:3]
                nn = float(np.linalg.norm(n_u))
                if nn > 1e-9:
                    c = float(np.clip(np.dot(axis_now, -n_u / nn), -1.0, 1.0))
                    ang_axis_n = math.degrees(math.acos(c))
            arrive = {
                'd_cam_surface_settle_m': float(d_surf) if d_surf is not None else None,
                'd_cam_surface_live_m': live_d,
                'd_cam_surface_min_motion_m': min_d,
                'surface_clearance_m': float(surface),
                'tcp_dist_frozen_m': tcp_dist,
                'cup_berry_dist_m': cup_berry_dist,
                'cup_berry_dxyz_m': cup_berry_dxyz,
                'contact_exec_tol_m': self._pbvs_contact_exec_tol_m(),
                'live_berry_base': live_berry_base,
                'cup_live_berry_dist_m': cup_live_dist,
                'cup_live_berry_dxyz_m': cup_live_dxyz,
                'axial_gap_frozen_m': axial_gap,
                'shot_n': int(getattr(self, '_pbvs_direct_shot_n', 1) or 1),
                'berry_base_frozen': (
                    np.asarray(frozen, dtype=np.float64).tolist()
                    if frozen is not None else None),
                'n_base': (
                    np.asarray(n_now, dtype=np.float64).tolist()
                    if n_now is not None else None),
                'tcp_base': list(tcp) if tcp is not None else None,
                'cup_open_xyz': list(cup) if cup is not None else None,
                'cup_axis_base': axis_now.tolist() if axis_now is not None else None,
                'angle_cup_axis_to_neg_n_deg': ang_axis_n,
                'berry_base_scaled': (
                    np.asarray(getattr(self, '_pbvs_scaled_berry'), dtype=np.float64).tolist()
                    if getattr(self, '_pbvs_scaled_berry', None) is not None else None),
                'ray_scale': getattr(self, '_pbvs_ray_scale', None),
            }
            self._write_qa_json('pbvs_direct_arrive.json', arrive)
            self._pbvs_qa_log_event('direct_arrive', arrive)
            shot_n = int(getattr(self, '_pbvs_direct_shot_n', 1) or 1)
            max_shots = self._pbvs_direct_max_shots()
            exec_tol = self._pbvs_contact_exec_tol_m()
            contact_slack = max(exec_tol, float(surface) + exec_tol)
            scaled = getattr(self, '_pbvs_scaled_berry', None)
            residual_target = (
                np.asarray(scaled, dtype=np.float64) if scaled is not None
                else (np.asarray(frozen, dtype=np.float64) if frozen is not None else None))
            cup_res_dist = cup_berry_dist
            if scaled is not None and cup is not None:
                dvec_s = residual_target.flatten()[:3] - np.asarray(
                    cup, dtype=np.float64).flatten()[:3]
                cup_res_dist = float(np.linalg.norm(dvec_s))
            residual_ok = (
                residual_target is not None
                and shot_n < max_shots
                and (
                    (cup_res_dist is not None and cup_res_dist > contact_slack)
                    or (
                        scaled is None
                        and axial_gap is not None
                        and float(axial_gap) > contact_slack)
                )
            )
            if residual_ok:
                travel = self._pbvs_travel_to_surface_m(residual_target)
                min_travel = self._pbvs_min_oneshot_travel_m()
                if travel is not None and float(travel) >= min_travel:
                    travel = min(float(travel), 0.04)
                    self.get_logger().info(
                        'PBVS direct residual  '
                        f'{"P1" if scaled is not None else "P0"} '
                        f'cup↔tgt={(cup_res_dist if cup_res_dist is not None else -1)*1000:.1f}mm '
                        f'cup↔lock={(cup_berry_dist if cup_berry_dist is not None else -1)*1000:.1f}mm '
                        f'axial_gap={(axial_gap if axial_gap is not None else -1)*1000:.1f}mm '
                        f'travel={travel*1000:.1f}mm '
                        f'shot={shot_n + 1}/{max_shots} '
                        f'(tol={exec_tol*1000:.1f}mm)')
                    self._pbvs_direct_shot_n = shot_n + 1
                    if scaled is not None:
                        self._pbvs_frozen_berry = residual_target.copy()
                        self._last_berry_base = tuple(
                            float(v) for v in residual_target.tolist())
                    if self._pbvs_send_cup_axis_oneshot(
                            residual_target,
                            travel,
                            tag='pbvs_single_residual',
                            pending='direct',
                            d_cam_surface=(
                                float(d_surf) if d_surf is not None else None),
                            lookat=True,
                            allow_tip_residual=True):
                        return
                    self.get_logger().warn(
                        'PBVS direct residual oneshot failed — '
                        'arrive check at current pose')
            self.get_logger().info(
                'PBVS direct ARRIVE  '
                f'd_cam_settle={(d_surf if d_surf is not None else -1)*1000:.1f}mm '
                f'd_cam_min={(min_d if min_d is not None else -1)*1000:.1f}mm '
                f'cup↔lock={(cup_berry_dist if cup_berry_dist is not None else -1)*1000:.1f}mm '
                f'tcp↔lock={(tcp_dist if tcp_dist is not None else -1)*1000:.1f}mm '
                f'axial_gap={(axial_gap if axial_gap is not None else -1)*1000:.1f}mm '
                f'shot={shot_n}/{max_shots} tol={exec_tol*1000:.1f}mm '
                f'(target clearance={surface*1000:.1f}mm) '
                f'axis∠−n={(ang_axis_n if ang_axis_n is not None else -1):.1f}deg')
            self._pbvs_after_direct_arrive()
            return
        if pending == 'align_n':
            self._pbvs_on_align_n_settle()
            return
        if pending == 'press':
            self._pbvs_on_press_settle()
            return
        if pending == 'approach':
            self._pbvs_state = 'DEPTH_FINAL'
            self._pbvs_depth_final_running = False
            self._snap_qa(f'servo_{self._servo_step_idx:02d}_after_approach')
            self._servo_step_idx += 1
            self._pbvs_run_depth_final()
        elif pending == 'final':
            surface = self._cup_surface_clearance_m()
            d_surf, live_d = self._pbvs_measure_d_cam_surface(live_only=True)
            self._snap_qa(f'servo_{self._servo_step_idx:02d}_contact')
            self._servo_step_idx += 1
            self._pbvs_depth_final_running = False
            if d_surf is None:
                self._set_state('ERROR', 'pbvs: no depth after final oneshot')
                return
            self._pbvs_qa_log_event('depth_final_done', {
                'd_cam_surface_m': float(d_surf),
                'live_cam_m': live_d,
            })
            if d_surf > surface + 0.010:
                self._set_state(
                    'ERROR',
                    f'pbvs: final short of surface '
                    f'd_cam_surface={d_surf*1000:.1f}mm '
                    f'(need ≤{(surface+0.002)*1000:.1f}mm)')
                return
            self._pbvs_finish_contact(
                float(d_surf), tag='depth_final', d_cam_m=live_d or d_surf)

    def _pbvs_start_contact_oneshot(
        self,
        berry: np.ndarray,
        d_cam_surface: float,
        *,
        approach_dir: Optional[np.ndarray] = None,
    ) -> bool:
        """Stream-mode terminal oneshot (legacy PBVS path)."""
        if self._pbvs_contact_sent or not self._arm.server_is_ready():
            return False
        surface = self._cup_surface_clearance_m()
        if d_cam_surface <= surface + 1e-4:
            return False
        travel = float(d_cam_surface) - surface
        if travel < 0.002:
            return False
        ok = self._pbvs_send_cup_axis_oneshot(
            np.asarray(berry, dtype=np.float64),
            travel,
            tag='pbvs_contact',
            pending='contact',
            d_cam_surface=float(d_cam_surface),
        )
        if ok:
            self._pbvs_contact_sent = True
            self._pbvs_contact_sent_t = time.time()
            self._pbvs_state = 'CONTACT'
        return ok

    def _tick_refining_pbvs_oneshot(self) -> None:
        """Open-loop PBVS tick.

        ``single`` (and alias ``oneshot``): fruit lock + surface n freeze →
        one or more cup-axis pushes (auto residual until cup↔lock ≤ 0.5 mm
        or max shots) → optional press along −n → SUCTION_HOLD placeholder →
        outer WAIT_CONFIRM → human confirm_reset. No pre-grasp / depth_final.
        ``twostage``: legacy approach + depth_final.
        """
        if not hasattr(self, '_pbvs_kf'):
            self._init_pbvs()

        now = time.time()
        if not getattr(self, '_pbvs_global_map_ok', False):
            if now - getattr(self, '_pbvs_global_retry_t', 0.0) >= 0.5:
                self._pbvs_global_retry_t = now
                self._build_approach_map_from_global(wait_tf_s=0.75)

        if self._pbvs_state == 'DONE':
            return

        if self._refine_fruit_anchor is None:
            if self._await_fine_tracker_reset_before_lock():
                return
            self._init_refine_fruit_lock()
            if self._refine_fruit_anchor is None:
                if now - getattr(self, '_pbvs_init_log_t', 0.0) > 1.0:
                    self._pbvs_init_log_t = now
                    self.get_logger().info(
                        f'PBVS waiting fruit lock: {self._fine_reject_reason()}')
                return

        berry = self._pick_probe_live_berry()
        berry_xyz_base = None
        depth_mode = ''
        if self._pbvs_direct_mode() and self._refine_fruit_anchor is not None:
            berry_xyz_base = np.asarray(
                self._refine_fruit_anchor, dtype=np.float64).flatten()[:3]
            depth_mode = str(getattr(self, '_pbvs_lock_depth_mode', '') or '')
        if berry is not None and berry_xyz_base is None:
            try:
                base = self._berry_base_xyz(berry)
                if base is not None:
                    berry_xyz_base = np.array(base, dtype=np.float64)
                    depth_mode = str(getattr(berry, 'depth_mode', 'mono') or 'mono')
            except Exception:
                pass

        if self._pbvs_state == 'INIT':
            if berry_xyz_base is None:
                if now - getattr(self, '_pbvs_init_log_t', 0.0) > 1.0:
                    self._pbvs_init_log_t = now
                    self.get_logger().info(
                        f'PBVS INIT waiting berry: {self._fine_reject_reason()}')
                return
            self._pbvs_frozen_berry = berry_xyz_base.copy()
            self._pbvs_entry_lock_xyz = berry_xyz_base.copy()
            self._pbvs_scaled_berry = None
            self._pbvs_ray_scale = None
            self._pbvs_lock_ray = None
            T_lock = self._lookup_T_base_cam()
            uv_lock = self._berry_image_uv(berry) if berry is not None else None
            if uv_lock is None:
                cam0 = self._base_xyz_to_cam(
                    tuple(float(v) for v in berry_xyz_base.tolist()))
                if cam0 is not None:
                    uv_lock = self._berry_uv_from_cam(cam0)
            if T_lock is not None and uv_lock is not None:
                from pbvs_ray_scale import ray_from_uv
                o0, d0 = ray_from_uv(T_lock, uv_lock, self._wrist_intrinsics())
                self._pbvs_lock_ray = (o0, d0)
            if self._pbvs_direct_mode():
                self._pbvs_freeze_surface_normal(berry, berry_xyz_base)
            if berry is not None:
                depth_mode = str(getattr(berry, 'depth_mode', depth_mode) or depth_mode)
            self._pbvs_lock_depth_mode = depth_mode
            # Sticky lock id for QA (avoid re-assoc overwriting to small ints).
            self._pbvs_lock_track_id = self._refine_locked_track_id
            cam_m = (
                self._pbvs_cam_metrics(berry, berry_xyz_base)
                if berry is not None else None)
            tcp = self._tcp_or_ee_xyz()
            adir = self._pbvs_approach_dir_base
            if adir is None and tcp is not None:
                d = berry_xyz_base - np.asarray(tcp, dtype=np.float64)
                n = float(np.linalg.norm(d))
                adir = d / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
            if adir is not None:
                self._pbvs_frozen_approach_dir = np.asarray(
                    adir, dtype=np.float64).copy()
            if cam_m:
                self._pbvs_save_cam_snapshot(
                    berry, berry_xyz_base,
                    approach_dir=self._pbvs_frozen_approach_dir,
                    depth_mode=depth_mode,
                )
            d_cam = cam_m.get('d_cam_surface_m') if cam_m else None
            tcp_d = float(np.linalg.norm(
                berry_xyz_base - np.asarray(tcp, dtype=np.float64))) if tcp else None
            travel_surf = self._pbvs_direct_travel_m(berry_xyz_base)
            standoff = self._tip_contact_standoff_m()
            n_lock = getattr(self, '_pbvs_frozen_normal', None)

            # Canonical single path (oneshot is the same).
            if self._pbvs_direct_mode():
                self.get_logger().info(
                    f'PBVS single LOCK  berry={np.round(berry_xyz_base, 3)} '
                    f'tid={self._refine_locked_track_id} mode={depth_mode} '
                    f'd_cam={(d_cam if d_cam is not None else -1)*1000:.1f}mm '
                    f'travel→surface='
                    f'{(travel_surf if travel_surf is not None else -1)*1000:.1f}mm '
                    f'standoff={standoff*1000:.1f}mm '
                    f'n={"ok" if n_lock is not None else "skip"} '
                    f'tcp_dist={None if tcp_d is None else round(tcp_d, 3)}')
                if travel_surf is None:
                    self._set_state('ERROR', 'pbvs: lock — no cup→surface travel')
                    return
                if travel_surf < self._pbvs_contact_exec_tol_m():
                    self._snap_qa(f'servo_{self._servo_step_idx:02d}_direct_arrive')
                    self._pbvs_after_direct_arrive()
                    return
                self._pbvs_state = 'DIRECT_ONESHOT'
                if not self._pbvs_start_direct_oneshot(
                        berry_xyz_base,
                        d_cam_surface=(float(d_cam) if d_cam is not None else None)):
                    self._set_state('ERROR', 'pbvs: single oneshot failed')
                return

            # Legacy twostage only.
            if not self._pbvs_twostage_mode():
                self._set_state('ERROR', f'pbvs: unknown mode={getattr(self._args, "pbvs_mode", "")}')
                return
            pre = self._pbvs_pre_grasp_d_cam_m()
            self.get_logger().info(
                f'PBVS twostage LOCK  berry={np.round(berry_xyz_base, 3)} '
                f'tid={self._refine_locked_track_id} mode={depth_mode} '
                f'd_cam={(d_cam if d_cam is not None else -1)*1000:.1f}mm '
                f'pre_grasp={pre*1000:.1f}mm '
                f'tcp_dist={None if tcp_d is None else round(tcp_d, 3)}')
            self._pbvs_state = 'APPROACH_ONESHOT'
            if d_cam is None:
                self._set_state('ERROR', 'pbvs: lock — no d_cam_surface')
                return
            travel = float(d_cam) - pre
            if travel >= 0.002:
                if not self._pbvs_start_approach_oneshot(
                        berry_xyz_base, d_cam_surface=float(d_cam)):
                    self._set_state('ERROR', 'pbvs: approach oneshot failed')
                return
            self._pbvs_state = 'DEPTH_FINAL'
            self._pbvs_run_depth_final()
            return

        if self._pbvs_state == 'SUCTION_HOLD':
            self._pbvs_enter_suction_hold()
            return

        if self._pbvs_state in (
                'DIRECT_ONESHOT', 'ALIGN_N', 'PRESS',
                'APPROACH_ONESHOT', 'FINAL_ONESHOT', 'DEPTH_FINAL'):
            if self._move_phase is not None:
                live = self._pick_probe_live_berry()
                if self._pbvs_state == 'DIRECT_ONESHOT':
                    self._pbvs_try_ray_scale(live)
                base = self._berry_base_xyz(live) if live is not None else None
                if live is not None and base is not None:
                    ld = self._pbvs_live_d_cam_surface(
                        live, np.asarray(base, dtype=np.float64))
                    if ld is not None:
                        prev = getattr(self, '_pbvs_motion_min_d_cam', None)
                        self._pbvs_motion_min_d_cam = (
                            float(ld) if prev is None else min(float(prev), float(ld)))
            return

    def _tick_refining_pbvs(self) -> None:
        """Continuous PBVS loop: track berry in 3-D, servo to standoff, approach.

        Replaces the two-step center_oneshot → range_depth path when
        use_pbvs ROS param is True.
        """
        # Lazy init on first tick.
        if not hasattr(self, '_pbvs_kf'):
            self._init_pbvs()

        now = time.time()
        dt = min(now - self._pbvs_last_tick, 0.2)
        self._pbvs_last_tick = now

        # Global fixed-cam map: do not permanently fall back to EE→berry after a
        # one-shot TF discovery race at REFINING entry.
        if not getattr(self, '_pbvs_global_map_ok', False):
            if now - getattr(self, '_pbvs_global_retry_t', 0.0) >= 0.5:
                self._pbvs_global_retry_t = now
                self._build_approach_map_from_global(wait_tf_s=0.75)

        # ── Constants ──────────────────────────────────────────────────
        STANDOFF_M = 0.07
        D_NEAR_M = float(getattr(self._args, 'servo_near_handoff_dist_m', 0.08))
        READY_THRESH = 8e-4
        SURFACE_M = self._cup_surface_clearance_m()
        TIP_STANDOFF_M = self._tip_contact_standoff_m()
        PROBE_STEP_DEG = 3.0
        # Continuous PBVS (velocity-style): e → v = λ e → diff-IK → move_j
        PBVS_DT = 0.04
        PBVS_LAMBDA = 1.2
        PBVS_V_MAX = 0.08          # m/s mid-range
        PBVS_V_MAX_NEAR = 0.045    # m/s near contact
        PBVS_MAX_DQ = 0.05         # rad per tick
        PBVS_TARGET_SLEW_M = 0.010   # max setpoint step per 25 Hz tick
        PBVS_GATE_ABORT_STREAK = 15
        PBVS_SAFETY_TIP_TARGET_M = 0.35
        PBVS_MONO_V_REJECT_MPS = 0.02
        PBVS_PROBE_MIN_CUP_M = D_NEAR_M + 0.06   # no active probe near handoff
        PBVS_CUP_ONESHOT_M = 0.04                # switch to cup_axis below 4 cm
        PBVS_PROBE_TIMEOUT_S = 3.0
        PBVS_PROBE_COOLDOWN_S = 4.0
        PBVS_RGBD_PROBE_MIN_CUP_M = 0.20
        VISION_LOST_N = int(getattr(self._args, 'pbvs_vision_lost_frames', 3))
        Z_CAM_NEAR_M = float(getattr(self._args, 'pbvs_vision_z_cam_min', 0.08))
        NEAR_D_CAM_M = float(getattr(self._args, 'pbvs_near_d_cam_m', 0.12))
        PBVS_BLIND_CONFIRM_TIMEOUT_S = 20.0

        # Depth-source → measurement sigma mapping.
        _DEPTH_SIGMA = {
            'rgbd': 0.005, 'fused_rgbd': 0.005,
            'depth': 0.012, 'depth_raw': 0.012,
            'da2': 0.012,  'fused_da2': 0.012,
            'iou_pin': 0.012,
            'mono': 0.025, 'mono_fallback': 0.025,
        }

        # ── 0. Fruit lock (same association as legacy refine path) ─────
        if self._refine_fruit_anchor is None:
            if self._await_fine_tracker_reset_before_lock():
                return
            self._init_refine_fruit_lock()
            if self._refine_fruit_anchor is None:
                if now - getattr(self, '_pbvs_init_log_t', 0.0) > 1.0:
                    self._pbvs_init_log_t = now
                    self.get_logger().info(
                        f'PBVS waiting fruit lock: {self._fine_reject_reason()}')
                return

        # ── 1. Gather locked-track detection (never berries[0]) ───────
        berry = self._pick_probe_live_berry()
        berry_xyz_base = None
        berry_sigma = 0.025
        depth_mode = ''
        if berry is not None:
            try:
                base = self._berry_base_xyz(berry)
                if base is not None:
                    berry_xyz_base = np.array(base, dtype=np.float64)
                    depth_mode = str(getattr(berry, 'depth_mode', 'mono') or 'mono')
                    berry_sigma = _DEPTH_SIGMA.get(depth_mode, 0.020)
            except Exception:
                pass

        # ── 2. INIT state: wait for first valid detection ──────────────
        if self._pbvs_state == 'INIT':
            if berry_xyz_base is None:
                if now - getattr(self, '_pbvs_init_log_t', 0.0) > 1.0:
                    self._pbvs_init_log_t = now
                    self.get_logger().info(
                        f'PBVS INIT waiting berry: {self._fine_reject_reason()}')
                return
            lock_sigma = (berry_sigma if self._pbvs_depth_trusted(depth_mode, berry)
                          else berry_sigma * 3)
            self._pbvs_kf.initialize(berry_xyz_base, sigma_init=lock_sigma)
            self._pbvs_lock_depth_mode = depth_mode
            self._pbvs_entry_lock_xyz = berry_xyz_base.copy()
            self._pbvs_target_cmd = berry_xyz_base.copy()
            self._pbvs_kf_reject_streak = 0
            self._pbvs_state = 'SERVO'
            self.get_logger().info(
                f'PBVS INIT → SERVO  berry_base={np.round(berry_xyz_base, 3)} '
                f'tid={self._refine_locked_track_id} mode={depth_mode}')
            return

        # ── CONTACT / DONE ─────────────────────────────────────────────
        if self._pbvs_state == 'DONE':
            return

        if self._pbvs_state == 'CONTACT':
            live_d = self._pbvs_live_d_cam_surface(berry, berry_xyz_base)
            if live_d is not None and live_d <= SURFACE_M + 0.002:
                self._snap_qa(f'servo_{self._servo_step_idx:02d}_contact')
                self._pbvs_finish_contact(
                    float(live_d), tag='live_cam', d_cam_m=live_d)
                return
            if self._pbvs_approach_blind and self._pbvs_contact_sent:
                sent_t = getattr(self, '_pbvs_contact_sent_t', 0.0)
                oneshot_done = self._move_phase is None
                if oneshot_done and sent_t > 0.0:
                    elapsed = now - sent_t
                    if live_d is not None and live_d > SURFACE_M + 0.015:
                        if now - getattr(self, '_pbvs_blind_warn_t', 0.0) >= 2.0:
                            self._pbvs_blind_warn_t = now
                            self.get_logger().warn(
                                f'PBVS blind: oneshot done but live d_cam={live_d*1000:.1f}mm '
                                f'(need ≤{(SURFACE_M+0.002)*1000:.1f}mm)')
                    if elapsed > PBVS_BLIND_CONFIRM_TIMEOUT_S:
                        self._set_state(
                            'ERROR',
                            f'pbvs: blind contact unconfirmed '
                            f'(live d_cam={(live_d if live_d is not None else -1)*1000:.0f}mm '
                            f'after {elapsed:.0f}s)')
            return

        # ── 3. KF predict + gated update (SERVO + live APPROACH) ───────
        if self._pbvs_state in ('SERVO', 'APPROACH') and not self._pbvs_approach_blind:
            self._pbvs_kf.predict(dt)
            kf_update = None
            if berry_xyz_base is not None:
                skip_meas = False
                skip_reason = ''
                if (self._pbvs_mono_depth_mode(depth_mode)
                        and self._pbvs_last_v_norm > PBVS_MONO_V_REJECT_MPS):
                    skip_meas = True
                    skip_reason = 'mono_while_moving'
                elif str(depth_mode) == 'base_coast':
                    skip_meas = True
                    skip_reason = 'base_coast'

                if skip_meas:
                    if now - self._pbvs_gate_log_t >= 1.0:
                        self._pbvs_gate_log_t = now
                        self.get_logger().debug(
                            f'PBVS KF skip update: {skip_reason} '
                            f'mode={depth_mode} v={self._pbvs_last_v_norm:.3f}')
                else:
                    tip_ref = self._tcp_or_ee_xyz()
                    cup_dist_ref = None
                    if tip_ref is not None:
                        cup_dist_ref = float(np.linalg.norm(
                            berry_xyz_base - np.asarray(tip_ref, dtype=np.float64)))
                    physical_max = max(0.08, 0.5 * cup_dist_ref) if (
                        cup_dist_ref is not None) else 0.12
                    kf_update = self._pbvs_kf.gated_update(
                        berry_xyz_base,
                        berry_sigma,
                        physical_max_m=physical_max,
                    )
                    if kf_update.accepted:
                        self._pbvs_kf_reject_streak = 0
                    else:
                        self._pbvs_kf_reject_streak += 1
                        if now - self._pbvs_gate_log_t >= 1.0:
                            self._pbvs_gate_log_t = now
                            self.get_logger().warn(
                                f'PBVS KF gated: {kf_update.reason} '
                                f'|innov|={kf_update.innov_norm:.3f}m '
                                f'mahal={kf_update.mahal_sq:.1f} '
                                f'mode={depth_mode} streak={self._pbvs_kf_reject_streak}')

            if self._pbvs_kf_reject_streak >= PBVS_GATE_ABORT_STREAK:
                self._set_state(
                    'ERROR',
                    f'pbvs: {self._pbvs_kf_reject_streak} consecutive KF rejects '
                    f'(target drift protection)')
                return

            kf_pos = self._pbvs_kf.position
            if kf_pos is not None:
                self._pbvs_slew_target_cmd(kf_pos, max_step_m=PBVS_TARGET_SLEW_M)

        depth_img = getattr(self, '_last_wrist_depth', None)
        K_wrist = getattr(self, '_wrist_camera_K', None)

        # ── 4. PROBE state: active lateral triangulation ──────────────
        if self._pbvs_state == 'PROBE':
            self._tick_pbvs_probe(PROBE_STEP_DEG, berry_xyz_base)
            return

        # Decide whether to trigger an active probe (mid-range only).
        cup_for_probe = None
        if self._pbvs_target_cmd is not None:
            cup_for_probe = self._cup_berry_dist_from(self._pbvs_target_cmd)
        if (self._pbvs_state == 'SERVO'
                and self._pbvs_kf.needs_active_probe(READY_THRESH * 4)
                and self._pbvs_probe_q_before is None
                and (now - getattr(self, '_pbvs_probe_last_t', 0.0)
                     >= PBVS_PROBE_COOLDOWN_S)
                and (cup_for_probe is None or cup_for_probe > PBVS_PROBE_MIN_CUP_M)):
            allow_probe = True
            lock_mode = str(getattr(self, '_pbvs_lock_depth_mode', '') or '')
            depth_trusted = (
                (berry is not None and self._pbvs_depth_trusted(depth_mode, berry))
                or self._pbvs_depth_trusted(lock_mode))
            if (depth_trusted
                    and cup_for_probe is not None
                    and cup_for_probe > PBVS_RGBD_PROBE_MIN_CUP_M):
                allow_probe = False
                if now - self._pbvs_gate_log_t >= 1.0:
                    self._pbvs_gate_log_t = now
                    self.get_logger().info(
                        'PBVS probe suppressed: trusted RGB-D lock still far '
                        f'(cup={cup_for_probe:.3f}m mode={depth_mode or lock_mode})')
            if berry_xyz_base is None:
                allow_probe = False
            if allow_probe:
                self._start_pbvs_probe(PROBE_STEP_DEG, berry_xyz_base)
                return

        target_xyz = self._pbvs_control_target()
        if target_xyz is None:
            return

        tip_now_pre = tip_xyz(self._joint_positions_rad().tolist()) if (
            self._joint_positions_rad() is not None) else None
        if (tip_now_pre is not None
                and float(np.linalg.norm(target_xyz - tip_now_pre))
                > PBVS_SAFETY_TIP_TARGET_M):
            self._set_state(
                'ERROR',
                f'pbvs: target-tip {float(np.linalg.norm(target_xyz - tip_now_pre)):.3f}m '
                f'> {PBVS_SAFETY_TIP_TARGET_M:.2f}m safety limit')
            return

        # ── 5. Update fused map with wrist frame (every 0.5 s during SERVO) ─
        # The global frame was already added by _build_approach_map_from_global at
        # REFINING entry.  Here we continuously refine the direction as the arm
        # closes in and the wrist camera sees the berry from closer range.
        if (self._pbvs_state == 'SERVO'
                and depth_img is not None
                and K_wrist is not None
                and target_xyz is not None
                and getattr(self, '_fused_map', None) is not None):
            now_map = time.time()
            if now_map - self._pbvs_last_wrist_map_t >= 0.5:
                T_base_cam = self._tf_base_cam()
                if T_base_cam is not None:
                    n_pts = self._fused_map.add_frame(
                        depth_img, K_wrist, T_base_cam,
                        target_xyz, radius_m=0.15,
                        depth_min_m=0.10, depth_max_m=1.50)
                    self._pbvs_last_wrist_map_t = now_map
                    if n_pts > 0:
                        new_dir = self._fused_map.get_approach_dir(
                            target_xyz, self._tcp_or_ee_xyz())
                        self._pbvs_approach_dir_base = new_dir
                        self.get_logger().debug(
                            f'FusedApproachMap: +{n_pts} wrist pts '
                            f'(total {self._fused_map.n_points}), '
                            f'dir={np.round(new_dir, 3)}')

        approach_dir = self._pbvs_approach_dir_base
        if approach_dir is None:
            # Fallback: EE → berry (target_xyz already available from KF).
            ee = self._tcp_or_ee_xyz()
            if ee is not None:
                d = target_xyz - np.array(ee, dtype=np.float64)
                n = float(np.linalg.norm(d))
                approach_dir = d / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
            else:
                approach_dir = np.array([0.0, 0.0, 1.0])  # last resort: +Z
        if (self._pbvs_state in ('APPROACH', 'CONTACT')
                and self._pbvs_approach_blind
                and self._pbvs_frozen_approach_dir is not None):
            approach_dir = self._pbvs_frozen_approach_dir

        # ── 6. Cam metrics, live PBVS setpoint, vision-loss blind ────
        cup_dist = self._cup_berry_dist_from(target_xyz)
        cup_gap = None
        gap_axis = self._cup_gap_along_axis(
            target_xyz, approach_dir=approach_dir)
        if gap_axis is not None:
            cup_gap, _ = gap_axis

        cam_m = None
        d_cam_surface = None
        if (berry is not None and berry_xyz_base is not None
                and not self._pbvs_approach_blind):
            cam_m = self._pbvs_cam_metrics(berry, berry_xyz_base)
            if cam_m is not None:
                d_cam_surface = cam_m.get('d_cam_surface_m')
                self._pbvs_save_cam_snapshot(
                    berry, berry_xyz_base,
                    approach_dir=np.asarray(approach_dir, dtype=np.float64),
                    depth_mode=depth_mode,
                )
                self._pbvs_vision_lost_streak = 0
                z_d = cam_m.get('z_depth_m')
                near = (
                    (d_cam_surface is not None and d_cam_surface <= NEAR_D_CAM_M)
                    or (z_d is not None and z_d <= Z_CAM_NEAR_M))
                if near:
                    self._pbvs_had_live_near = True
                slew = PBVS_TARGET_SLEW_M * (
                    2.0 if self._pbvs_state == 'APPROACH' else 1.0)
                self._pbvs_slew_target_cmd(berry_xyz_base, max_step_m=slew)
                if (self._pbvs_state == 'SERVO' and near
                        and not getattr(self, '_pbvs_near_logged', False)):
                    self._pbvs_state = 'APPROACH'
                    self._pbvs_near_logged = True
                    self.get_logger().info(
                        f'PBVS SERVO → APPROACH (live)  d_cam={d_cam_surface*1000:.1f}mm '
                        f'tcp={cup_dist*1000:.1f}mm z_depth={(z_d or -1)*1000:.0f}mm '
                        f'pix_err={cam_m.get("pix_err_px")}')
                    self._pbvs_log_vision_event('near_enter', {
                        'd_cam_surface_m': d_cam_surface,
                        'tcp_dist_m': cup_dist,
                        'cup_gap_m': cup_gap,
                        'depth_mode': depth_mode,
                    })
                if (self._pbvs_state == 'APPROACH'
                        and d_cam_surface is not None
                        and not self._pbvs_contact_sent):
                    surface = self._cup_surface_clearance_m()
                    if d_cam_surface <= surface + 0.002:
                        self._pbvs_finish_contact(
                            float(d_cam_surface), tag='live_cam', d_cam_m=d_cam_surface)
                        return
                    if d_cam_surface <= PBVS_CUP_ONESHOT_M:
                        if self._pbvs_start_contact_oneshot(
                                berry_xyz_base, float(d_cam_surface),
                                approach_dir=np.asarray(
                                    approach_dir, dtype=np.float64)):
                            return
        elif not self._pbvs_approach_blind:
            if self._pbvs_blind_eligible(
                    getattr(self, '_pbvs_cam_snapshot', None),
                    near_d_cam_m=NEAR_D_CAM_M,
                    z_cam_near_m=Z_CAM_NEAR_M):
                self._pbvs_vision_lost_streak += 1
                if self._pbvs_vision_lost_streak >= VISION_LOST_N:
                    if self._pbvs_enter_blind_approach(
                            np.asarray(approach_dir, dtype=np.float64),
                            near_d_cam_m=NEAR_D_CAM_M,
                            z_cam_near_m=Z_CAM_NEAR_M):
                        return
            elif self._pbvs_vision_lost_streak > 0:
                self._pbvs_vision_lost_streak = 0

        if self._pbvs_approach_blind or self._pbvs_contact_sent:
            if self._pbvs_state == 'CONTACT':
                return
            # Blind segment handled by oneshot; skip streaming PBVS.
            return

        target_xyz = self._pbvs_control_target()
        if target_xyz is None:
            return

        # ── 7. Continuous velocity PBVS: v=λe → diff-IK → stream ──────
        q_current = self._joint_positions_rad()
        if q_current is None:
            return
        # Always trust joint feedback (never open-loop integrate commanded q).
        tip_now = tip_xyz(q_current.tolist())

        if self._pbvs_state == 'APPROACH':
            p_des = target_xyz - approach_dir * float(TIP_STANDOFF_M)
            v_max = PBVS_V_MAX_NEAR
        else:
            # Mid: tip standoff along approach direction.
            p_des = target_xyz - approach_dir * STANDOFF_M
            v_max = PBVS_V_MAX

        err = np.asarray(p_des, dtype=np.float64) - tip_now
        err_n = float(np.linalg.norm(err))
        if err_n < 1e-4:
            return
        # Classical PBVS: v = λ (p_des − p); never invert this sign.
        v = PBVS_LAMBDA * err
        speed = float(np.linalg.norm(v))
        if speed > v_max:
            v *= v_max / speed
        self._pbvs_last_v_norm = float(min(speed, v_max))

        q_new = cartesian_velocity_step(
            q_current.tolist(), v, dt=PBVS_DT, max_dq=PBVS_MAX_DQ)
        # Publish teacher tool traj ~1 Hz; optional executor drive.
        if now - getattr(self, '_pbvs_tool_traj_pub_t', 0.0) >= 1.0:
            self._pbvs_tool_traj_pub_t = now
            self._publish_pbvs_tool_trajectory(
                tip_now.tolist(), np.asarray(p_des, dtype=float).tolist(),
                np.asarray(approach_dir, dtype=float).tolist(),
                move_duration_s=1.0, tag=f'stream_{self._pbvs_state}')
        if self._pbvs_via_executor:
            # Executor owns joint streaming; do not dual-drive.
            return
        # Short preemptible traj (~teleop), not blocking settle.
        self._stream_joint_goal(q_new, traj_s=max(0.12, PBVS_DT * 3.0))

        if now - getattr(self, '_pbvs_stream_log_t', 0.0) >= 1.0:
            self._pbvs_stream_log_t = now
            self.get_logger().info(
                f'PBVS stream {self._pbvs_state}: '
                f'd_cam={None if d_cam_surface is None else round(d_cam_surface, 3)} '
                f'gap={None if cup_gap is None else round(cup_gap, 3)} '
                f'tcp={None if cup_dist is None else round(cup_dist, 3)} '
                f'|e|={err_n:.3f} |v|={min(speed, v_max):.3f} '
                f'tip={np.round(tip_now, 3)} des={np.round(p_des, 3)}')

        if now - getattr(self, '_pbvs_stream_snap_t', 0.0) >= 2.0:
            self._pbvs_stream_snap_t = now
            tag = f'servo_{self._servo_step_idx:02d}_stream'
            self._snap_qa(tag)
            self._servo_step_idx += 1

        # Goal 4: sparse BC log (not every 25 Hz tick).
        if (self._data_collector is not None and self._data_collector.active
                and int(now * 5) != int((now - PBVS_DT) * 5)):
            iw, ih = self._wrist_image_wh()
            refine_obs = {
                'joint1_deg': math.degrees(q_new[0]),
                'joint2_deg': math.degrees(q_new[1]),
                'joint3_deg': math.degrees(q_new[2]),
                'joint5_deg': math.degrees(q_new[4]),
                'cup_dist_m': float(cup_dist) if cup_dist is not None else 0.0,
                'kf_uncertainty': float(self._pbvs_kf.uncertainty()),
                'kf_reject_streak': int(self._pbvs_kf_reject_streak),
                'err_m': err_n,
                'v_mps': float(min(speed, v_max)),
            }
            wrist_img = (self._qa_rgb_fine_viz if self._qa_rgb_fine_viz is not None
                         else self._qa_rgb_wrist)
            if wrist_img is not None and self._qa_rgb_fixed is not None:
                self._data_collector.log_step(
                    global_img=self._qa_rgb_fixed,
                    wrist_img=wrist_img,
                    obs=refine_obs,
                    action={
                        'action': 'pbvs_stream',
                        'phase': self._pbvs_state,
                        'source': 'diff_ik',
                    },
                )

    def _joint_positions_rad(self) -> Optional[np.ndarray]:
        """Current arm joints as float64 rad vector (len 6), or None if unset."""
        if not self._joints or len(self._joints) < 6:
            return None
        return np.array([float(v) for v in self._joints[:6]], dtype=np.float64)

    def _start_pbvs_probe(self, step_deg: float,
                          berry_xyz_base: Optional[np.ndarray] = None) -> None:
        """Initiate a lateral probe step to triangulate berry 3-D position."""
        q = self._joint_positions_rad()
        if q is None:
            return
        self._pbvs_probe_q_before = q.copy()
        self._pbvs_probe_obs_list = []
        if berry_xyz_base is not None:
            self._pbvs_probe_entry_obs = np.asarray(
                berry_xyz_base, dtype=np.float64).copy()
            self._pbvs_probe_obs_list.append(self._pbvs_probe_entry_obs.copy())
        else:
            self._pbvs_probe_entry_obs = None
        self._pbvs_probe_return_pending = False
        self._pbvs_probe_started_t = time.time()
        self._pbvs_probe_last_t = self._pbvs_probe_started_t
        self._pbvs_state = 'PROBE'
        # Step joint1 laterally.
        q_probe = q.copy()
        q_probe[0] += float(np.radians(step_deg))
        self._send_joint_servo_goal(
            [float(v) for v in q_probe.tolist()],
            tag='pbvs_probe', traj_s=0.4)
        self._pbvs_qa_log_event('probe_start', {
            'step_deg': float(step_deg),
            'entry_obs': (self._pbvs_probe_entry_obs.tolist()
                          if self._pbvs_probe_entry_obs is not None else None),
        })
        self.get_logger().info(
            f'PBVS active probe: step_deg={step_deg} '
            f'entry_obs={len(self._pbvs_probe_obs_list)}')

    def _tick_pbvs_probe(self, step_deg: float,
                         berry_xyz_base) -> None:
        """Collect probe observation and return to original pose."""
        del step_deg  # fixed step configured at start
        now = time.time()
        if self._pbvs_probe_q_before is None:
            self._pbvs_state = 'SERVO'
            return

        if berry_xyz_base is not None:
            obs = np.asarray(berry_xyz_base, dtype=np.float64).copy()
            if not self._pbvs_probe_obs_list:
                self._pbvs_probe_obs_list.append(obs)
            else:
                prev = np.asarray(self._pbvs_probe_obs_list[-1], dtype=np.float64)
                if float(np.linalg.norm(obs - prev)) > 1e-4:
                    self._pbvs_probe_obs_list.append(obs)

        elapsed = now - float(getattr(self, '_pbvs_probe_started_t', now))

        if not self._pbvs_probe_return_pending:
            if self._move_phase is not None:
                return
            if elapsed < 0.45:
                return
            if len(self._pbvs_probe_obs_list) >= 2:
                triangulated = np.mean(self._pbvs_probe_obs_list, axis=0)
                gate = self._pbvs_kf.gated_update_triangulated(
                    triangulated, sigma_meas=0.004, physical_max_m=0.10)
                if gate.accepted:
                    kf_pos = self._pbvs_kf.position
                    if kf_pos is not None:
                        self._pbvs_slew_target_cmd(kf_pos, max_step_m=0.015)
                    self._pbvs_probe_success_n += 1
                else:
                    self._pbvs_probe_fail_n += 1
                self._pbvs_qa_log_event('probe_triangulated', {
                    'accepted': bool(gate.accepted),
                    'reason': gate.reason,
                    'n_obs': int(len(self._pbvs_probe_obs_list)),
                    'triangulated': np.asarray(triangulated).tolist(),
                    'mahal_sq': float(gate.mahal_sq),
                    'innov_norm': float(gate.innov_norm),
                    'uncertainty': float(self._pbvs_kf.uncertainty()),
                })
                self.get_logger().info(
                    f'PBVS probe triangulated: {triangulated}  '
                    f'gate={gate.reason} uncertainty→'
                    f'{self._pbvs_kf.uncertainty()*1e6:.1f} µm²')
            elif elapsed > 3.0:
                self._pbvs_probe_fail_n += 1
                self._pbvs_qa_log_event('probe_timeout', {
                    'elapsed_s': float(elapsed),
                    'n_obs': int(len(self._pbvs_probe_obs_list)),
                })
                self.get_logger().warn(
                    f'PBVS probe timeout ({elapsed:.1f}s, '
                    f'n_obs={len(self._pbvs_probe_obs_list)}) — skip triangulation')
            else:
                return
            q_back = [float(v) for v in list(self._pbvs_probe_q_before)[:6]]
            self._send_joint_servo_goal(
                q_back, tag='pbvs_probe_return', traj_s=0.4)
            self._pbvs_probe_return_pending = True
            return

        if self._move_phase is not None:
            return
        self._pbvs_probe_q_before = None
        self._pbvs_probe_obs_list = []
        self._pbvs_probe_entry_obs = None
        self._pbvs_probe_return_pending = False
        self._pbvs_probe_started_t = 0.0
        self._pbvs_state = 'SERVO'
        self.get_logger().info('PBVS probe → SERVO')

    def _cup_tip_offset_link6(self) -> np.ndarray:
        """Soft-cup tip (= opening) in link6. GT-fit 20260812 (table Z − 12mm)."""
        return np.array([0.0, 0.01883, 0.06152], dtype=np.float64)

    def _cup_rim_z_m(self) -> float:
        """Deprecated: tip IS the opening; kept only for CLI compat."""
        return float(getattr(self._args, 'cup_rim_z', 0.0))

    def _berry_radius_m(self) -> float:
        return float(getattr(self._args, 'berry_radius', 0.008))

    def _cup_surface_clearance_m(self) -> float:
        return float(getattr(self._args, 'cup_surface_clearance_m', 0.003))

    def _tip_contact_standoff_m(self) -> float:
        """Cup↔locked contact point clearance (m).

        Wrist depth at bbox/mask center is range to *surface* along the ray;
        UV+depth back-project yields that surface point in base_link — not a
        sphere center. Do not add/subtract berry_radius here."""
        return self._cup_surface_clearance_m()

    def _pbvs_contact_exec_tol_m(self) -> float:
        """Open-loop execution: cup↔frozen lock residual trigger (m).

        PBVS single runs blind after lock — no live tracking during motion."""
        if self._pbvs_direct_mode():
            return 0.0005
        return 0.002

    def _pbvs_direct_max_shots(self) -> int:
        return max(2, int(getattr(self._args, 'pbvs_direct_max_shots', 4)))

    def _pbvs_min_oneshot_travel_m(self) -> float:
        if self._pbvs_direct_mode():
            return 0.0005
        return 0.002

    def _pbvs_ik_tol_m(self) -> float:
        if self._pbvs_direct_mode():
            return 5e-4
        return 2.5e-3

    def _pbvs_pre_grasp_d_cam_m(self) -> float:
        """Cam-frame cup↔berry surface distance for oneshot pre-grasp (m)."""
        return float(getattr(self._args, 'pbvs_pre_grasp_d_cam_m', 0.08))

    def _pbvs_d_cam_surface_from_base(self, berry_base: np.ndarray) -> Optional[float]:
        """Cup↔locked surface contact gap in cam frame (m)."""
        berry_cam = self._base_xyz_to_cam(
            tuple(float(v) for v in np.asarray(berry_base, dtype=np.float64).flatten()[:3]))
        cup_open = self._cup_open_cam_xyz()
        if berry_cam is None or cup_open is None:
            return None
        d_cam = float(np.linalg.norm(
            np.asarray(berry_cam, dtype=np.float64) - np.asarray(cup_open, dtype=np.float64)))
        return d_cam - self._cup_surface_clearance_m()

    def _link6_pose(self, q=None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if q is None:
            q = self._joint_positions_rad()
        if q is None:
            return None
        R, p = fk_link6(list(q))
        return R, np.asarray(p, dtype=np.float64)

    def _cup_axis_base(self, q=None) -> Optional[np.ndarray]:
        lp = self._link6_pose(q)
        if lp is None:
            return None
        R, _ = lp
        axis = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        n = float(np.linalg.norm(axis))
        return axis / n if n > 1e-6 else None

    def _cup_open_cam_xyz(self) -> Optional[np.ndarray]:
        """Cup opening (not TCP) in wrist optical frame."""
        cup = self._cup_open_xyz()
        if cup is None:
            return None
        cam = self._base_xyz_to_cam(tuple(float(v) for v in cup))
        if cam is None:
            return None
        return np.asarray(cam, dtype=np.float64)

    def _pbvs_cam_metrics(
        self,
        berry: Optional[DetectedBerry],
        berry_xyz_base: Optional[np.ndarray],
    ) -> Optional[Dict]:
        """Camera-frame cup_open↔berry distance (primary PBVS near metric)."""
        if berry is None or berry_xyz_base is None:
            return None
        berry_cam = self._berry_cam_xyz(berry)
        cup_open_cam = self._cup_open_cam_xyz()
        tcp = self._tcp_or_ee_xyz()
        geo = None
        if tcp is not None:
            geo = self._cup_berry_geometry(
                tuple(float(v) for v in np.asarray(berry_xyz_base).flatten()[:3]),
                tcp,
                berry_cam,
            )
        d_cam = None
        d_cam_surface = None
        if berry_cam is not None and cup_open_cam is not None:
            bc = np.asarray(berry_cam, dtype=np.float64)
            cc = np.asarray(cup_open_cam, dtype=np.float64)
            d_cam = float(np.linalg.norm(bc - cc))
            d_cam_surface = d_cam - self._cup_surface_clearance_m()
        z_d = float(getattr(berry, 'z_depth_m', -1.0))
        z_m = float(getattr(berry, 'z_mono_m', -1.0))
        pix_err = None
        if geo and geo.get('berry_uv') and geo.get('cup_uv'):
            bu, bv = geo['berry_uv']
            cu, cv_ = geo['cup_uv']
            pix_err = float(math.hypot(bu - cu, bv - cv_))
        return {
            'd_cam_m': d_cam,
            'd_cam_surface_m': d_cam_surface,
            'berry_cam': geo.get('berry_cam') if geo else (
                list(berry_cam) if berry_cam else None),
            'cup_open_cam': list(cup_open_cam) if cup_open_cam is not None else None,
            'cup_cam': geo.get('cup_cam') if geo else None,
            'berry_uv': geo.get('berry_uv') if geo else None,
            'cup_uv': geo.get('cup_uv') if geo else None,
            'pix_err_px': pix_err,
            'z_depth_m': z_d if z_d > 1e-4 else None,
            'z_mono_m': z_m if z_m > 1e-4 else None,
            'berry_base': np.asarray(
                berry_xyz_base, dtype=np.float64).flatten()[:3].tolist(),
        }

    def _pbvs_save_cam_snapshot(
        self,
        berry: DetectedBerry,
        berry_xyz_base: np.ndarray,
        *,
        approach_dir: np.ndarray,
        depth_mode: str,
    ) -> None:
        cam_m = self._pbvs_cam_metrics(berry, berry_xyz_base)
        if cam_m is None:
            return
        self._pbvs_cam_snapshot = {
            **cam_m,
            'approach_dir': np.asarray(approach_dir, dtype=np.float64).tolist(),
            'depth_mode': depth_mode,
            'track_id': self._refine_locked_track_id,
            't': time.time(),
        }

    def _pbvs_live_d_cam_surface(
        self,
        berry: Optional[DetectedBerry],
        berry_xyz_base: Optional[np.ndarray],
    ) -> Optional[float]:
        if berry is None or berry_xyz_base is None:
            return None
        cam_m = self._pbvs_cam_metrics(berry, berry_xyz_base)
        if not cam_m:
            return None
        d = cam_m.get('d_cam_surface_m')
        return float(d) if d is not None else None

    def _pbvs_blind_eligible(
        self,
        snap: Optional[Dict],
        *,
        near_d_cam_m: float,
        z_cam_near_m: float,
    ) -> bool:
        """Blind oneshot only after genuine near-phase live vision."""
        if not getattr(self, '_pbvs_had_live_near', False):
            return False
        if self._pbvs_state != 'APPROACH':
            return False
        if not snap or snap.get('berry_base') is None:
            return False
        d_cam = snap.get('d_cam_surface_m')
        if d_cam is None:
            d_cam = snap.get('d_cam_m')
        if d_cam is None or float(d_cam) > float(near_d_cam_m):
            return False
        z_d = snap.get('z_depth_m')
        if z_d is not None and float(z_d) > float(z_cam_near_m):
            return False
        return True

    def _pbvs_log_vision_event(self, kind: str, extra: Optional[Dict] = None) -> None:
        snap = dict(self._pbvs_cam_snapshot or {})
        if extra:
            snap.update(extra)
        snap['event'] = kind
        fname = 'pbvs_vision_lost.json' if kind == 'vision_lost' else 'pbvs_vision_snapshot.json'
        self._write_qa_json(fname, snap)
        self._pbvs_qa_log_event(kind, snap)

    def _pbvs_enter_blind_approach(
        self,
        approach_dir: np.ndarray,
        *,
        near_d_cam_m: float,
        z_cam_near_m: float,
    ) -> bool:
        """Vision lost in near range: blind cup_axis oneshot from last cam snapshot."""
        if self._pbvs_approach_blind or self._pbvs_contact_sent:
            return False
        snap = self._pbvs_cam_snapshot
        if not self._pbvs_blind_eligible(
                snap, near_d_cam_m=near_d_cam_m, z_cam_near_m=z_cam_near_m):
            d_cam = (snap or {}).get('d_cam_surface_m')
            self.get_logger().warn(
                'PBVS blind rejected: need APPROACH+near live snapshot '
                f'(had_near={self._pbvs_had_live_near} state={self._pbvs_state} '
                f'd_cam={(d_cam if d_cam is not None else -1)*1000:.0f}mm '
                f'z={(snap or {}).get("z_depth_m")})')
            return False
        self._pbvs_approach_blind = True
        self._pbvs_state = 'APPROACH'
        self._pbvs_frozen_approach_dir = np.asarray(
            approach_dir, dtype=np.float64).copy()
        d_cam = snap.get('d_cam_surface_m')
        if d_cam is None:
            d_cam = snap.get('d_cam_m')
        berry = np.asarray(snap['berry_base'], dtype=np.float64)
        self._pbvs_log_vision_event('vision_lost', {
            'vision_lost_streak': int(self._pbvs_vision_lost_streak),
            'd_cam_at_loss_m': d_cam,
        })
        self.get_logger().info(
            f'PBVS vision lost → blind oneshot  d_cam={(d_cam if d_cam is not None else -1)*1000:.1f}mm '
            f'z_depth={(snap.get("z_depth_m") or -1)*1000:.0f}mm '
            f'pix_err={snap.get("pix_err_px")}')
        surface = self._cup_surface_clearance_m()
        if d_cam is not None and d_cam <= surface + 0.002:
            self._pbvs_finish_contact(float(d_cam), tag='blind_at_loss', d_cam_m=d_cam)
            return True
        if d_cam is not None and d_cam <= near_d_cam_m:
            return self._pbvs_start_contact_oneshot(
                berry, float(d_cam), approach_dir=self._pbvs_frozen_approach_dir)
        return False

    def _cup_open_xyz(self, q=None) -> Optional[np.ndarray]:
        """Cup opening in base_link — same point as tip (TCP), link6+[0,0,0.075]."""
        lp = self._link6_pose(q)
        if lp is None:
            return None
        R, p = lp
        return p + R @ self._cup_tip_offset_link6()

    def _cup_gap_along_axis(
        self,
        berry_xyz_base: np.ndarray,
        *,
        approach_dir: Optional[np.ndarray] = None,
        q=None,
    ) -> Optional[Tuple[float, np.ndarray]]:
        """Cup opening → locked surface contact gap along approach axis (m).

        ``berry_xyz_base`` is the depth-backprojected surface point (bbox center),
        not a modeled sphere center."""
        cup = self._cup_open_xyz(q)
        if cup is None:
            return None
        berry = np.asarray(berry_xyz_base, dtype=np.float64).flatten()[:3]
        axis = None
        if approach_dir is not None:
            axis = np.asarray(approach_dir, dtype=np.float64).flatten()[:3]
        if axis is None or float(np.linalg.norm(axis)) < 1e-6:
            axis = self._cup_axis_base(q)
        if axis is None or float(np.linalg.norm(axis)) < 1e-6:
            axis = berry - cup
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            return None
        axis = axis / n
        axial = float(np.dot(berry - cup, axis))
        gap = axial - self._cup_surface_clearance_m()
        return gap, axis

    def _cup_berry_dist_from(self, berry_xyz_base: np.ndarray) -> float | None:
        """Legacy TCP↔berry-center Euclidean distance (m)."""
        tcp = self._tcp_or_ee_xyz()
        if tcp is None:
            return None
        diff = np.array(berry_xyz_base) - np.array(tcp)
        return float(np.linalg.norm(diff))

    def _tf_base_cam(self, *, wait_s: float = 0.25):
        """Look up T_base←wrist_optical (4×4)."""
        parent = self._args.base_frame
        child = getattr(
            self._args, 'wrist_camera_frame', 'camera_wrist_color_optical_frame')
        try:
            if not self._tf_buffer.can_transform(
                    parent, child, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=max(0.0, float(wait_s)))):
                return None
            tf_msg = self._tf_buffer.lookup_transform(
                parent, child, rclpy.time.Time())
            return _matrix_from_tf(tf_msg)
        except Exception:
            return None

    def _tf_fixed_cam(self, *, wait_s: float = 0.25):
        """Look up T_base←fixed_optical (4×4). wait_s blocks until TF is ready."""
        parent = self._args.base_frame
        child = self._fixed_camera_frame
        try:
            # can_transform blocks up to wait_s — covers listener discovery race.
            if not self._tf_buffer.can_transform(
                    parent, child, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=max(0.0, float(wait_s)))):
                self.get_logger().warn(
                    f'FusedApproachMap: TF not ready {parent}←{child} '
                    f'after {wait_s:.2f}s')
                return None
            tf_msg = self._tf_buffer.lookup_transform(
                parent, child, rclpy.time.Time())
            # TransformStamped — local helper expects .transform.translation
            return _matrix_from_tf(tf_msg)
        except Exception as exc:
            self.get_logger().warn(
                f'FusedApproachMap: TF {parent}←{child} failed: {exc}')
            return None

    def _build_approach_map_from_global(self, *, wait_tf_s: float = 0.25) -> None:
        """Feed global RGB-D frame into fused map and update approach direction.

        Skipped when no global depth is available or _fused_map hasn't been
        created yet (non-PBVS mode / align_only). On TF race, leaves
        ``_pbvs_global_map_ok`` False so PBVS ticks can retry.
        """
        fused_map = getattr(self, '_fused_map', None)
        if fused_map is None:
            return
        if self._last_fixed_depth is None or self._fixed_camera_K is None:
            if time.time() - getattr(self, '_pbvs_global_retry_t', 0.0) < 0.05:
                # Avoid log spam when called from rapid entry+tick.
                pass
            else:
                self.get_logger().info(
                    'FusedApproachMap: waiting for global depth/K')
            return
        if self._locked is None:
            self.get_logger().warn(
                'FusedApproachMap: skipped global frame (no locked berry)')
            return
        try:
            T = self._tf_fixed_cam(wait_s=wait_tf_s)
            if T is None:
                return
            p = self._locked.pose.pose.position
            berry = np.array([float(p.x), float(p.y), float(p.z)])
            ee = self._tcp_or_ee_xyz()
            n = fused_map.add_frame(
                self._last_fixed_depth, self._fixed_camera_K, T, berry)
            self._pbvs_approach_dir_base = fused_map.get_approach_dir(berry, ee)
            self._pbvs_global_map_ok = True
            self.get_logger().info(
                f'FusedApproachMap: +{n} pts from global cam '
                f'(total {fused_map.n_points}), '
                f'approach_dir={np.round(self._pbvs_approach_dir_base, 3)}')
        except Exception as exc:
            self.get_logger().warn(f'FusedApproachMap (global frame) failed: {exc}')

    # -----------------------------------------------------------------------
    # End PBVS (pbvs-vlm-reach-v2)
    # -----------------------------------------------------------------------

    def _tick_wrist_servo_refine(self) -> None:
        """range_center → center_oneshot → range_depth → contact oneshot."""
        if self._refine_lock_snap_pending:
            self._refine_lock_snap_pending = False
            self._snap_refine_lock_viz('refine_fruit_lock')
        if self._await_fine_tracker_reset_before_lock():
            return
        if self._refine_fruit_anchor is None:
            self._init_refine_fruit_lock()
            if self._refine_fruit_anchor is None:
                if time.time() > self._refine_deadline:
                    self._set_state(
                        'ERROR', 'refine: no fruit lock at REFINING entry')
                return

        if self._args.dry_run:
            berry = self._pick_fine_berry()
            if berry is not None or time.time() > self._refine_deadline:
                self.get_logger().info('dry-run: skip wrist servo, pretend reached')
                self._reached_pub.publish(Bool(data=True))
                self._set_state('REACHED')
                self._set_state('WAIT_CONFIRM')
            return

        if time.time() > self._refine_deadline:
            why = self._fine_reject_reason()
            self._set_state(
                'ERROR',
                f'refine/servo timeout — need wrist fine berry + contact; {why}')
            return

        # No small-step range: plan center oneshot from current mono lock immediately.
        if (
            not bool(getattr(self._args, 'servo_mono_probe', True))
            and self._refine_phase == 'range_center'
            and not self._near_mode
        ):
            if not self._start_center_oneshot():
                self._enter_contact_oneshot(reason='no_probe_skip_center')
            return

        # Small-step tracking ranges (measure only — not incremental centering).
        if (
            bool(getattr(self._args, 'servo_mono_probe', True))
            and self._refine_phase in ('range_center', 'range_depth', 'probe')
            and not self._mono_probe_done
            and not self._near_mode
        ):
            self._tick_mono_probe()
            return

        # Waiting for center oneshot traj / settle.
        if self._refine_phase == 'center_oneshot':
            if self._move_phase is not None:
                return
            if not self._center_oneshot_sent:
                if not self._start_center_oneshot():
                    self._refine_phase = 'post_center'
                    self._begin_range_depth()
            return

        # After center arrive: await live YOLO / live-replan / contact — never
        # re-fire open-loop center oneshot (153728 bug: phase stuck on
        # center_oneshot with sent=False → spurious second start).
        if self._refine_phase == 'post_center':
            if self._move_phase is not None:
                return
            self._begin_range_depth()
            return

        berry = self._pick_fine_berry()
        z_cam: Optional[float] = None
        pix_err: Optional[float] = None
        using_estimate = False

        # Legacy mid PBVS removed.
        if self._refine_phase in ('approach', 'center') and not self._near_mode:
            if self._last_berry_base is not None:
                self._start_center_oneshot()
            else:
                self._refine_phase = 'range_center'
                self._mono_probe_purpose = 'center'
                self._mono_probe_done = False
            return

        if berry is not None:
            self._refine_fine_lost_streak = 0
            self._lock_pub.publish(berry)
            cam = self._berry_cam_xyz(berry)
            if cam is not None and cam[2] > 1e-4:
                z_cam = self._filter_z_cam(float(cam[2]))
                pix_err = self._servo_image_error_px((cam[0], cam[1], z_cam))
                if not self._near_mode:
                    self._remember_berry(berry, z_cam)
        else:
            self._refine_fine_lost_streak += 1
            berry_age = (
                time.time() - self._last_berry_t if self._last_berry_t > 0 else 1e9)
            if self._refine_phase in ('range_center', 'range_depth', 'center_oneshot'):
                now = time.time()
                if now - self._refine_wait_log_t > 1.0:
                    self._refine_wait_log_t = now
                    self.get_logger().info(
                        f'REFINING {self._refine_phase}: lost live track '
                        f'(streak={self._refine_fine_lost_streak}) — hold')
                return
            if (
                not self._near_mode
                and self._refine_phase == 'near'
                and self._last_berry_base is not None
                and berry_age <= float(self._args.servo_estimate_max_age_s)
            ):
                self._last_berry_t = time.time()
                cam = self._base_xyz_to_cam(self._last_berry_base)
                if cam is not None and cam[2] > 1e-4:
                    z_cam = self._filter_z_cam(float(cam[2]))
                    pix_err = self._servo_image_error_px((cam[0], cam[1], z_cam))
                    using_estimate = True
                elif self._last_z_cam is not None and self._last_z_cam > 1e-4:
                    z_cam = float(self._last_z_cam)
                    using_estimate = True

        contact = float(self._args.cup_contact_offset)
        dist_cup = self._cup_berry_dist()
        contact_tol = contact + 0.005
        if dist_cup is not None and dist_cup <= contact_tol:
            self.get_logger().info(
                f'REFINING contact (cup↔berry): dist={dist_cup:.3f}m <= {contact_tol:.3f}m '
                f'(offset={contact:.3f}m) '
                f'z_cam={(z_cam if z_cam is not None else -1.0):.3f} near={int(self._near_mode)}')
            self._snap_qa(f'servo_{self._servo_step_idx:02d}_contact')
            self._reached_pub.publish(Bool(data=True))
            self._set_state('REACHED')
            self._set_state('WAIT_CONFIRM')
            return

        # Contact oneshot only after human confirm sets _near_mode (or auto path).
        # refine_phase=='near' alone is NOT enough — WAIT_NEAR sets phase first.
        if not self._near_mode:
            now = time.time()
            if now - self._refine_wait_log_t > 2.0:
                self._refine_wait_log_t = now
                self.get_logger().info(
                    f'REFINING idle in phase={self._refine_phase} '
                    f'(waiting range/oneshot pipeline or confirm_near)')
            return

        berry_age = time.time() - self._last_berry_t if self._last_berry_t > 0 else 1e9
        estimate_fresh = (
            (self._near_frozen_berry is not None)
            or (
                self._last_berry_base is not None
                and berry_age <= float(self._args.servo_estimate_max_age_s)))
        if not estimate_fresh:
            now = time.time()
            if now - self._refine_wait_log_t > 2.0:
                self._refine_wait_log_t = now
                self.get_logger().warn('REFINING near: no frozen berry pose')
            return

        self._freeze_near_berry()
        if self._near_oneshot_sent or self._move_phase is not None:
            return
        if not self._start_near_oneshot_step(dist_cup):
            now = time.time()
            if now - self._refine_wait_log_t > 2.0:
                self._refine_wait_log_t = now
                dist_now = self._cup_berry_dist()
                contact = float(self._args.cup_contact_offset)
                soft = contact + 0.008
                if dist_now is not None and dist_now <= soft:
                    self.get_logger().info(
                        f'REFINING near oneshot: close enough dist={dist_now:.3f}m '
                        f'≤ {soft:.3f}m — contact')
                    self._snap_qa(f'servo_{self._servo_step_idx:02d}_contact')
                    self._reached_pub.publish(Bool(data=True))
                    self._set_state('REACHED')
                    self._set_state('WAIT_CONFIRM')
                    return
                self.get_logger().warn(
                    f'REFINING near oneshot: waiting '
                    f'(attempts={self._near_oneshot_attempts} '
                    f'dist={(dist_now if dist_now is not None else -1):.3f})')

    def _send_joint_servo_goal(
        self,
        target: List[float],
        *,
        tag: str,
        extra: str = '',
        traj_s: Optional[float] = None,
    ) -> bool:
        max_dj = max(abs(target[i] - self._joints[i]) for i in range(6))
        min_dj = math.radians(0.03 if tag == 'pbvs_press' else 0.12)
        if max_dj < min_dj:
            self.get_logger().info(f'REFINING {tag}: joint delta tiny — waiting')
            if tag == 'servo':
                self._servo_pending_record = None
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
        dt = float(self._args.servo_traj_s if traj_s is None else traj_s)
        pt.time_from_start.sec = int(dt)
        pt.time_from_start.nanosec = int((dt - int(dt)) * 1e9)
        goal.trajectory.points = [pt]

        settle = float(self._args.servo_settle_s)
        # Deadline BEFORE move_phase — MultiThreadedExecutor can poll between
        # these assignments; deadline==0 would instant-timeout.
        self._servo_deadline = time.time() + max(
            float(self._args.servo_timeout_s), dt + settle + 2.0)
        self._servo_settle_until = time.time() + dt + settle
        self._move_phase = 'servo'
        self._ik_fut = None
        self._traj_goal_fut = self._arm.send_goal_async(goal)
        self._traj_result_fut = None
        self._servo_motion_frames = []
        self._servo_motion_last_t = 0.0
        # First frame at command start (pre-motion / early motion).
        self._snap_servo_motion_frame(force=True)
        self.get_logger().info(
            f'REFINING {tag}[{self._servo_step_idx}]: {extra} '
            f'Δj1={math.degrees(target[0]-self._joints[0]):+.1f}deg '
            f'Δj2={math.degrees(target[1]-self._joints[1]):+.1f}deg '
            f'Δj3={math.degrees(target[2]-self._joints[2]):+.1f}deg '
            f'Δj5={math.degrees(target[4]-self._joints[4]):+.1f}deg '
            f'→ j=({math.degrees(target[0]):.1f},{math.degrees(target[1]):.1f},'
            f'{math.degrees(target[2]):.1f},{math.degrees(target[4]):.1f})')
        return True

    def _clamp_servo_target(self, target: List[float]) -> List[float]:
        """URDF soft limits — j6 unlocked (±align_joint6_limit_rad, default ±165°)."""
        j1_lim = math.radians(float(self._args.align_joint1_limit_deg))
        target[0] = max(-j1_lim, min(j1_lim, target[0]))
        target[1] = max(0.0, min(math.pi, target[1]))
        target[2] = max(-math.pi, min(0.0, target[2]))
        # joint4 URDF ±1.553343 rad (~±89°)
        j4_lim = 1.553343
        target[3] = max(-j4_lim, min(j4_lim, float(target[3])))
        target[4] = max(
            float(self._args.align_joint5_min_rad),
            min(float(self._args.align_joint5_max_rad), target[4]))
        j6_lim = abs(float(self._args.align_joint6_limit_rad))
        if j6_lim <= 1e-9:
            target[5] = 0.0
        else:
            target[5] = max(-j6_lim, min(j6_lim, float(target[5])))
        return target

    def _clamp_oneshot_joints_from_ik(
        self,
        joints: List[float],
        *,
        profile: str,
        step_m: float = 0.02,
    ) -> List[float]:
        """Soft per-step joint cap after IK — legacy IBVS path only."""
        if profile == 'center':
            max_dj = [
                math.radians(10.0),
                math.radians(float(self._args.servo_max_dj2_deg) + 2.0),
                math.radians(float(self._args.servo_max_dj3_deg) + 2.0),
                math.radians(5.0),
                math.radians(float(self._args.servo_max_dj5_deg) + 4.0),
                0.0,
            ]
        else:
            # Near full oneshot uses URDF-only clamp in _start_near_oneshot_step.
            # This branch is unused for contact; keep a generous residual path.
            j1_cap = min(45.0, 12.0 + step_m * 120.0)
            max_dj = [
                math.radians(j1_cap),
                math.radians(min(60.0, float(self._args.servo_max_dj2_deg) + 40.0)),
                math.radians(min(60.0, float(self._args.servo_max_dj3_deg) + 40.0)),
                math.radians(45.0),
                math.radians(min(60.0, float(self._args.servo_max_dj5_deg) + 40.0)),
                0.0,
            ]
        target = list(self._joints)
        for i in range(6):
            d = float(joints[i]) - float(self._joints[i])
            d = max(-max_dj[i], min(max_dj[i], d))
            target[i] = float(self._joints[i] + d)
        return self._clamp_servo_target(target)

    def _plan_tip_to_surface_delta(
        self,
        berry: Tuple[float, float, float],
        step_m: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, Dict]]:
        """Single/direct: translate tip along cup→berry by ``step_m``.

        Landing is the surface point on the current tip→berry ray
        (berry − standoff × dir). Same geometry as the old tip-residual nudge,
        computed once instead of after a cup-axis slide.
        """
        tcp = self._tcp_or_ee_xyz()
        cup = self._cup_open_xyz()
        origin = cup if cup is not None else tcp
        if origin is None or step_m < 1e-4:
            return None
        berry_a = np.asarray(berry, dtype=np.float64).flatten()[:3]
        p0 = np.asarray(origin, dtype=np.float64).flatten()[:3]
        rel = berry_a - p0
        dist = float(np.linalg.norm(rel))
        if dist < 1e-4:
            return None
        d_base = rel * (float(step_m) / dist)
        T = self._lookup_T_base_cam()
        if T is not None:
            d_cam = T[:3, :3].T @ d_base
        else:
            d_cam = np.array([0.0, 0.0, float(step_m)], dtype=np.float64)
        meta: Dict = {
            'dir_base': (rel / dist).tolist(),
            'dist_center_m': dist,
            'cup_open': p0.tolist(),
        }
        return d_cam, d_base, 'tip_to_surface', meta

    def _plan_cup_axis_oneshot_delta(
        self,
        berry: Tuple[float, float, float],
        step_m: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, Dict]]:
        """Legacy twostage: translate along link6 +Z (cup axis) toward berry."""
        axis = self._cup_axis_base()
        cup = self._cup_open_xyz()
        if axis is None or cup is None or step_m < 1e-4:
            return None
        berry_a = np.asarray(berry, dtype=np.float64)
        cup_a = np.asarray(cup, dtype=np.float64)
        rel = berry_a - cup_a
        sign = 1.0 if float(np.dot(rel, axis)) >= 0.0 else -1.0
        d_base = axis * (sign * float(step_m))
        T = self._lookup_T_base_cam()
        if T is not None:
            R_c_b = T[:3, :3].T
            d_cam = R_c_b @ d_base
        else:
            d_cam = np.array([0.0, 0.0, float(step_m)], dtype=np.float64)
        berry_cam = self._base_xyz_to_cam(berry)
        cup_cam = self._cup_cam_xyz()
        meta: Dict = {
            'cup_axis_base': axis.tolist(),
            'sign': sign,
            'berry_cam': list(berry_cam) if berry_cam is not None else None,
            'cup_cam': list(cup_cam) if cup_cam is not None else None,
        }
        return d_cam, d_base, 'cup_axis', meta

    def _plan_near_oneshot_delta(
        self,
        berry: Tuple[float, float, float],
        tcp: Tuple[float, float, float],
        step_m: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, Dict]]:
        """Plan near oneshot: translate EE along cup→berry (keep_orient).

        Mid-range aims the berry onto the *cup axis* (image UV below optical
        center). Terminal contact then follows (berry − tcp). If mid-range still
        left the berry on the optical axis, cup→berry includes the ~8 cm mount
        climb — that is expected parallax, not a separate height bug.
        """
        T = self._lookup_T_base_cam()
        berry_cam = self._base_xyz_to_cam(berry)
        cup_cam = self._cup_cam_xyz()
        if berry_cam is None or cup_cam is None:
            return None
        bc = np.array(berry_cam, dtype=np.float64)
        cc = np.array(cup_cam, dtype=np.float64)
        rel = np.array(
            [berry[0] - tcp[0], berry[1] - tcp[1], berry[2] - tcp[2]],
            dtype=np.float64,
        )
        dist = float(np.linalg.norm(rel))
        if dist < 1e-4 or step_m < 1e-4:
            return None
        d_base = rel * (float(step_m) / dist)
        if T is not None:
            R_c_b = T[:3, :3].T  # base → cam
            d_cam = R_c_b @ d_base
        else:
            d_cam = np.array([0.0, 0.0, float(step_m)], dtype=np.float64)
        chord_cam = bc - cc
        chord_n = float(np.linalg.norm(chord_cam))
        z_frac = float(abs(chord_cam[2]) / chord_n) if chord_n > 1e-6 else 0.0
        ang_optical_deg = 0.0
        if chord_n > 1e-6:
            ang_optical_deg = float(math.degrees(math.acos(
                max(-1.0, min(1.0, float(chord_cam[2]) / chord_n)))))
        meta = {
            'berry_cam': bc.tolist(),
            'cup_cam': cc.tolist(),
            'cup_to_berry_base': rel.tolist(),
            'cam_axis_frac': z_frac,
            'angle_optical_deg': ang_optical_deg,
        }
        return d_cam, d_base, 'cup_to_berry', meta

    def _approach_axis_ik_for_tcp_goal(
        self,
        *,
        tcp_goal: np.ndarray,
        berry: Tuple[float, float, float],
        T_l6c: np.ndarray,
        seed: List[float],
        d_base: np.ndarray,
        contact_m: float,
        allow_tip_residual: bool = True,
        aim_xyz: Optional[Tuple[float, float, float]] = None,
    ) -> Optional[List[float]]:
        """Near contact: cup axis aimed at ``aim_xyz`` (default berry), tip to P.

        Primary: chunked ``cup_axis_ik`` (link6 +Z from tip → aim). Camera
        ``approach_axis`` is only a fallback — cam is ~8 cm off tip so cam-look
        ≠ cup-look near contact. keep_orient_chunked drifts R_des (161209).
        """
        tcp_now = self._tcp_or_ee_xyz()
        ee_now = self._current_ee_pose()
        if tcp_now is None or ee_now is None:
            return None
        T0 = fk_link6_T(seed)
        tcp0 = np.array(
            [float(tcp_now[0]), float(tcp_now[1]), float(tcp_now[2])],
            dtype=np.float64)
        off_l6 = T0[:3, :3].T @ (tcp0 - T0[:3, 3])
        ee0 = np.array([
            float(ee_now.pose.position.x),
            float(ee_now.pose.position.y),
            float(ee_now.pose.position.z),
        ], dtype=np.float64)
        # EE target that would put tip at contact if attitude held.
        ee_tgt = ee0 + np.asarray(d_base, dtype=np.float64)
        berry_a = np.asarray(berry, dtype=np.float64)
        aim = aim_xyz if aim_xyz is not None else berry
        ik_kw = {'tol_m': self._pbvs_ik_tol_m()}
        if aim_xyz is not None:
            ik_kw['w_dir'] = 0.8

        joints = cup_axis_ik_chunked(
            tuple(ee_tgt.tolist()), aim, seed,
            tip_offset_link6=off_l6, **ik_kw)
        method = 'cup_axis_chunked'
        if joints is None:
            joints = cup_axis_ik(
                tuple(ee_tgt.tolist()), aim, seed,
                tip_offset_link6=off_l6, **ik_kw)
            method = 'cup_axis'
        if joints is None:
            joints = approach_axis_ik_chunked(
                tuple(ee_tgt.tolist()), aim, seed, T_l6c, **ik_kw)
            method = 'approach_axis_chunked'
        if joints is None:
            joints = approach_axis_ik(
                tuple(ee_tgt.tolist()), aim, seed, T_l6c, **ik_kw)
            method = 'approach_axis'
        if joints is None:
            return None

        T_m = fk_link6_T(joints)
        tcp_m = T_m[:3, 3] + T_m[:3, :3] @ off_l6
        dist_m = float(np.linalg.norm(berry_a - tcp_m))
        exec_tol = self._pbvs_contact_exec_tol_m()
        self.get_logger().info(
            f'REFINING near lookat[{method}]: tcp-berry={dist_m*1000:.1f}mm '
            f'(contact={contact_m*1000:.0f}mm tol={exec_tol*1000:.1f}mm)')

        slack_m = exec_tol if self._pbvs_direct_mode() else 0.002
        if dist_m <= contact_m + slack_m:
            return list(joints)

        if not allow_tip_residual:
            return list(joints)

        # Tip still short: one more cup_axis (or keep_orient) residual.
        travel2 = dist_m - float(contact_m)
        min_travel = self._pbvs_min_oneshot_travel_m()
        if travel2 < min_travel:
            return list(joints)
        direction = (berry_a - tcp_m) / dist_m
        ee_final = tuple((fk_xyz(joints) + direction * travel2).tolist())
        joints_f = cup_axis_ik(
            ee_final, aim, joints, tip_offset_link6=off_l6, **ik_kw)
        if joints_f is None:
            joints_f = position_ik_keep_orient(ee_final, joints)
        if joints_f is None:
            joints_f = position_ik_keep_orient_chunked(ee_final, joints)
        if joints_f is None:
            self.get_logger().warn(
                'REFINING near lookat mid OK but tip residual failed — '
                'using mid (may finish short; arrive residual may retry)')
            return list(joints)

        T_f = fk_link6_T(joints_f)
        tcp_f = T_f[:3, 3] + T_f[:3, :3] @ off_l6
        dist_f = float(np.linalg.norm(berry_a - tcp_f))
        self.get_logger().info(
            f'REFINING near tip residual: '
            f'tcp-berry {dist_m*1000:.1f}→{dist_f*1000:.1f}mm')
        if dist_f <= dist_m + 1e-4:
            return list(joints_f)
        self.get_logger().warn(
            f'REFINING near tip residual worse '
            f'({dist_m*1000:.1f}→{dist_f*1000:.1f}mm) — keep mid')
        return list(joints)

    def _start_near_oneshot_step(self, dist_cup: Optional[float]) -> bool:
        """Near terminal: one open-loop traj along cup→berry to contact.

        Frozen 3D sets travel = dist_cup − cup_contact_offset (0 → tip on berry).
        IK may use chunked cup_axis / keep_orient for long Cartesian moves.
        """
        if not self._arm.server_is_ready():
            return False
        berry = self._near_frozen_berry or self._last_berry_base
        if berry is None:
            return False
        if self._near_frozen_berry is None:
            self._freeze_near_berry()
            berry = self._near_frozen_berry
            if berry is None:
                return False

        # True oneshot: at most one trajectory (default max=1).
        max_attempts = max(1, int(getattr(self._args, 'servo_near_oneshot_max', 1)))
        if self._near_oneshot_attempts >= max_attempts:
            self.get_logger().warn(
                f'REFINING near oneshot: already sent {self._near_oneshot_attempts}/'
                f'{max_attempts} dist={(dist_cup if dist_cup is not None else -1):.3f}')
            return False

        tcp = self._tcp_or_ee_xyz()
        ee = self._current_ee_pose()
        if tcp is None or ee is None:
            return False

        bx, by, bz = berry
        if dist_cup is None:
            dist_cup = math.sqrt(
                (bx - tcp[0]) ** 2 + (by - tcp[1]) ** 2 + (bz - tcp[2]) ** 2)
        contact = float(self._args.cup_contact_offset)
        if dist_cup <= contact + 1e-4:
            return False
        if dist_cup < 1e-4:
            return False

        travel_total = dist_cup - contact
        # Optional legacy cap (≤0 or unset → full remaining travel).
        step_cap = float(getattr(self._args, 'servo_near_oneshot_step_m', 0.0))
        if step_cap > 1e-6:
            step_m = min(travel_total, step_cap)
        else:
            step_m = travel_total
        if step_m < 0.002:
            return False

        planned = self._plan_near_oneshot_delta(berry, tcp, step_m)
        if planned is None:
            self.get_logger().warn('REFINING near oneshot: cup→berry plan failed')
            return False
        d_cam, d_base, method, plan_meta = planned

        ex = float(ee.pose.position.x)
        ey = float(ee.pose.position.y)
        ez = float(ee.pose.position.z)
        ee0 = np.array([ex, ey, ez], dtype=np.float64)
        # keep_orient: same translation on EE moves TCP by the same vector.
        tgt_xyz = tuple((ee0 + d_base).tolist())
        # Desired TCP after contact move (plan assumes pure translate).
        tcp0 = np.array(
            [float(tcp[0]), float(tcp[1]), float(tcp[2])], dtype=np.float64)
        tcp_goal = tcp0 + d_base

        # Near contact: cup axis (link6+Z) aimed tip→berry. Pure keep_orient
        # chunking drifts attitude (161209). Camera approach_axis is fallback.
        ik_method = 'keep_orient'
        joints = None
        T_l6c = self._lookup_T_link6_cam()
        # Always prefer lookat for contact moves ≥3cm; T_l6c only needed for
        # camera fallback inside the helper.
        use_lookat = float(step_m) >= 0.03
        if use_lookat:
            if T_l6c is None:
                # Fallback matches CAMERA_MOUNT_* (Gemini: R=I until pitch GT fit).
                T_l6c = np.eye(4, dtype=np.float64)
                T_l6c[:3, 3] = [0.0, -0.08, -0.04]
            joints = self._approach_axis_ik_for_tcp_goal(
                tcp_goal=tcp_goal,
                berry=berry,
                T_l6c=T_l6c,
                seed=list(self._joints),
                d_base=d_base,
                contact_m=contact,
            )
            if joints is not None:
                ik_method = 'cup_axis_lookat'
                tgt_xyz = tuple(fk_xyz(joints).tolist())
                self.get_logger().info(
                    'REFINING near oneshot: cup_axis look-at berry → tip contact')
        if joints is None:
            joints = position_ik_keep_orient(tgt_xyz, self._joints)
            if joints is None:
                joints = position_ik_keep_orient_chunked(tgt_xyz, self._joints)
                if joints is not None:
                    ik_method = 'keep_orient_chunked'
                    self.get_logger().warn(
                        'REFINING near oneshot: cup_axis lookat failed — '
                        'fallback keep_orient_chunked (attitude may drift)')
            if joints is None:
                self._servo_ik_fail_n = int(getattr(self, '_servo_ik_fail_n', 0)) + 1
                self.get_logger().warn(
                    f'REFINING near oneshot: IK failed '
                    f'tcp_goal=({tcp_goal[0]:.3f},{tcp_goal[1]:.3f},{tcp_goal[2]:.3f})')
                if self._servo_ik_fail_n >= 8:
                    self._set_state(
                        'ERROR',
                        f'near oneshot IK failed tcp_goal='
                        f'({tcp_goal[0]:.3f},{tcp_goal[1]:.3f},{tcp_goal[2]:.3f})')
                return False
        self._servo_ik_fail_n = 0

        # Full contact move needs the IK Δq; soft per-step Δj caps would truncate it.
        target = self._clamp_servo_target(list(joints))

        # ~3 cm/s class: 0.25 m → ~8–10 s (still within servo timeout budget).
        traj_s = max(
            float(self._args.servo_traj_s),
            min(12.0, 2.0 + step_m / 0.03),
        )

        geo = self._cup_berry_geometry(berry, tcp)
        ang_opt = plan_meta.get('angle_optical_deg')
        method_full = f'{method}+{ik_method}'
        self._servo_pending_record = {
            'step': int(self._servo_step_idx),
            'track_id': self._refine_locked_track_id,
            'refine_phase': 'near',
            'berry_base': list(berry),
            'tcp_base': list(tcp),
            'ee_base': [ex, ey, ez],
            'ee_xyz_before': [ex, ey, ez],
            'ee_xyz_cmd': list(tgt_xyz),
            'target_xyz': list(tgt_xyz),
            'cmd_dxyz': list(d_base),
            'cmd_d_base_m': list(d_base),
            'cmd_dcam_m': list(d_cam),
            'berry_cam': plan_meta.get('berry_cam'),
            'cup_cam': plan_meta.get('cup_cam'),
            'cup_to_berry_base': plan_meta.get('cup_to_berry_base'),
            'angle_optical_deg': ang_opt,
            'berry_rel_cup': geo.get('berry_rel_cup'),
            'dist_cup': float(dist_cup),
            'travel_m': float(step_m),
            'travel_total_m': float(travel_total),
            'tcp_goal': tcp_goal.tolist(),
            'ik_method': ik_method,
            'control': 'near_oneshot_cart',
            'method': method_full,
            'near_attempt': int(self._near_oneshot_attempts) + 1,
            'delta_j_deg': {
                'j1': math.degrees(target[0] - self._joints[0]),
                'j2': math.degrees(target[1] - self._joints[1]),
                'j3': math.degrees(target[2] - self._joints[2]),
                'j4': math.degrees(target[3] - self._joints[3]),
                'j5': math.degrees(target[4] - self._joints[4]),
            },
            'target_j_deg': {
                'j1': math.degrees(target[0]),
                'j2': math.degrees(target[1]),
                'j3': math.degrees(target[2]),
                'j4': math.degrees(target[3]),
                'j5': math.degrees(target[4]),
            },
            'qa_tag': f'servo_{self._servo_step_idx:02d}_near_oneshot',
            'source': 'step_json',
        }

        ok = self._send_joint_servo_goal(
            target,
            tag='near',
            traj_s=traj_s,
            extra=(
                f'oneshot {method_full} travel={step_m:.3f}m '
                f'dbase=({float(d_base[0])*1000:+.1f},{float(d_base[1])*1000:+.1f},'
                f'{float(d_base[2])*1000:+.1f})mm dist_cup={dist_cup:.3f}'
                + (f' ∠opt={float(ang_opt):.1f}deg' if ang_opt is not None else '')
            ),
        )
        if ok:
            self._near_oneshot_attempts += 1
            self._near_oneshot_sent = True
            self.get_logger().info(
                f'REFINING near oneshot[{self._servo_step_idx}]: '
                f'{method_full} travel={step_m:.3f}m traj={traj_s:.1f}s '
                f'tgt_ee=({tgt_xyz[0]:.3f},{tgt_xyz[1]:.3f},{tgt_xyz[2]:.3f}) '
                f'Δj1={math.degrees(target[0]-self._joints[0]):+.1f}deg '
                f'Δj2={math.degrees(target[1]-self._joints[1]):+.1f}deg '
                f'Δj4={math.degrees(target[3]-self._joints[3]):+.1f}deg'
                + (f' ∠opt={float(ang_opt):.1f}deg' if ang_opt is not None else ''))
        else:
            self._servo_pending_record = None
        return ok

    def _start_wrist_servo_step(
        self,
        berry: Optional[DetectedBerry],
        z_cam: float,
        pix_err: Optional[float],
        *,
        using_estimate: bool = False,
        dist_cup: Optional[float] = None,
        lost_streak: int = 0,
    ) -> bool:
        """Closed-loop mid-range: EE Cartesian waypoint → position IK → joints.

        j6 is locked (5-DOF). MoveIt `/compute_ik` asks for full SE(3) and
        returns NO_IK_SOLUTION for almost any absolute pose off the manifold.
        We therefore solve *position-only* Jacobian IK (joints 1–6) so cm-scale
        cup→berry waypoints map to FollowJointTrajectory without joint heuristics.
        """
        if not self._arm.server_is_ready():
            return False

        if berry is not None:
            cam = self._berry_cam_xyz(berry)
        elif self._last_berry_base is not None:
            cam = self._base_xyz_to_cam(self._last_berry_base)
        else:
            return False

        berry_xyz = self._last_berry_base
        if berry is not None:
            berry_xyz = self._berry_base_xyz(berry) or berry_xyz
        tcp = self._tcp_or_ee_xyz()
        ee = self._current_ee_pose()
        if berry_xyz is None or tcp is None or ee is None:
            return False

        if cam is not None and cam[2] > 1e-4:
            cx, cy, cz = cam[0], cam[1], z_cam if z_cam > 1e-4 else float(cam[2])
        else:
            # No wrist projection (FOV / lost) — pure cup→berry PBVS.
            cx, cy, cz = 0.0, 0.0, float(z_cam) if z_cam > 1e-4 else 0.20

        if dist_cup is None:
            dist_cup = math.sqrt(
                (berry_xyz[0] - tcp[0]) ** 2
                + (berry_xyz[1] - tcp[1]) ** 2
                + (berry_xyz[2] - tcp[2]) ** 2)

        if cam is not None and cam[2] > 1e-4:
            err_yaw, err_pitch = self._cam_angular_errors((cx, cy, cz))
        else:
            err_yaw, err_pitch = 0.0, 0.0
        dead = math.radians(float(self._args.servo_ang_deadband_deg))
        if abs(err_yaw) < dead:
            err_yaw = 0.0
        if abs(err_pitch) < dead:
            err_pitch = 0.0

        contact = float(self._args.cup_contact_offset)
        step_cap = float(self._args.servo_step_m)
        if self._refine_phase == 'center':
            # Smaller lateral steps so BoT-SORT can keep the lock.
            step_cap = min(step_cap, float(getattr(
                self._args, 'servo_center_step_m', 0.012)))
        if dist_cup <= contact + 1e-4:
            return False

        ex = float(ee.pose.position.x)
        ey = float(ee.pose.position.y)
        ez = float(ee.pose.position.z)
        control = 'position_diff_ik'
        aim_kind = 'cup'
        reach_scale = self._ibvs_reach_scale(pix_err, lost_streak=lost_streak)

        if self._refine_phase == 'center' and cam is not None and cam[2] > 1e-4:
            # Pure lateral IBVS in cam: no range close until image is centered.
            ax, ay, aim_kind = self._center_aim_cam_xy((cx, cy, cz))
            off_x = cx - ax
            off_y = cy - ay
            lat_mag = math.hypot(off_x, off_y)
            if lat_mag < 1e-4:
                return False
            gain = float(getattr(self._args, 'servo_center_gain', 0.45))
            step_m = min(step_cap, max(0.004, gain * lat_mag))
            d_cam = (-step_m * off_x / lat_mag, -step_m * off_y / lat_mag, 0.0)
            d_base = self._cam_delta_to_base(d_cam)
            if d_base is None:
                return False
            tgt_xyz = (ex + d_base[0], ey + d_base[1], ez + d_base[2])
            ux, uy, uz = d_base[0] / step_m, d_base[1] / step_m, d_base[2] / step_m
            reach_scale = 0.0
            control = 'center_ibvs'
        else:
            # Approach PBVS: step EE along TCP→berry after center (+probe).
            ux = (berry_xyz[0] - tcp[0]) / dist_cup
            uy = (berry_xyz[1] - tcp[1]) / dist_cup
            uz = (berry_xyz[2] - tcp[2]) / dist_cup
            step_m = min(step_cap, dist_cup - contact) * reach_scale
            if step_m < 1e-4:
                lat = 0.004 * max(abs(err_yaw), abs(err_pitch)) / max(dead, 1e-3)
                step_m = min(step_cap * 0.35, max(0.002, lat))
            tgt_xyz = (ex + ux * step_m, ey + uy * step_m, ez + uz * step_m)
            control = 'approach_pbvs'

        # Predicted EE move vs cup→berry vector (for replay diagnosis).
        geo = self._cup_berry_geometry(berry_xyz, tcp, berry_cam=(cx, cy, cz))

        joints = position_ik_keep_orient(tgt_xyz, self._joints)
        if joints is None:
            self._servo_ik_fail_t = time.time()
            self._servo_ik_fail_n = int(getattr(self, '_servo_ik_fail_n', 0)) + 1
            self.get_logger().warn(
                f'REFINING servo[{self._servo_step_idx}]: keep_orient IK failed '
                f'n={self._servo_ik_fail_n} '
                f'tgt=({tgt_xyz[0]:.3f},{tgt_xyz[1]:.3f},{tgt_xyz[2]:.3f})')
            return False
        self._servo_ik_fail_n = 0

        max_dj = [
            math.radians(float(self._args.servo_max_dj1_deg)),
            math.radians(float(self._args.servo_max_dj2_deg)),
            math.radians(float(self._args.servo_max_dj3_deg)),
            math.radians(5.0),
            math.radians(float(self._args.servo_max_dj5_deg)),
            math.radians(8.0),  # j6 unlocked
        ]
        if self._refine_phase == 'center':
            # Tighter joint caps during centering to keep tracker alive.
            max_dj = [0.55 * v for v in max_dj]
        target = list(self._joints)
        for i in range(6):
            d = joints[i] - self._joints[i]
            d = max(-max_dj[i], min(max_dj[i], d))
            target[i] = float(self._joints[i] + d)
        target = self._clamp_servo_target(target)

        est_tag = ' est' if using_estimate else ''
        self._servo_pending_record = {
            'step': int(self._servo_step_idx),
            'track_id': self._refine_locked_track_id,
            'refine_phase': str(self._refine_phase),
            'aim_kind': aim_kind,
            'berry_base': list(berry_xyz),
            'tcp_base': list(tcp),
            'ee_base': [ex, ey, ez],
            'target_xyz': list(tgt_xyz),
            'cmd_dxyz': [ux * step_m, uy * step_m, uz * step_m],
            'berry_rel_cup': geo.get('berry_rel_cup'),
            'berry_cam': geo.get('berry_cam'),
            'cup_cam': geo.get('cup_cam'),
            'berry_uv': geo.get('berry_uv'),
            'cup_uv': geo.get('cup_uv'),
            'optical_uv': geo.get('optical_uv'),
            'z_cam': float(z_cam),
            'dist_cup': float(dist_cup),
            'pix_err': float(pix_err) if pix_err is not None else None,
            'err_yaw_deg': math.degrees(err_yaw),
            'err_pitch_deg': math.degrees(err_pitch),
            'reach_scale': float(reach_scale),
            'step_m': float(step_m),
            'using_estimate': bool(using_estimate),
            'lost_streak': int(lost_streak),
            'control': control,
            'delta_j_deg': {
                'j1': math.degrees(target[0] - self._joints[0]),
                'j2': math.degrees(target[1] - self._joints[1]),
                'j3': math.degrees(target[2] - self._joints[2]),
                'j5': math.degrees(target[4] - self._joints[4]),
            },
            'target_j_deg': {
                'j1': math.degrees(target[0]),
                'j2': math.degrees(target[1]),
                'j3': math.degrees(target[2]),
                'j5': math.degrees(target[4]),
            },
            'qa_tag': f'servo_{self._servo_step_idx:02d}_after',
            'source': 'step_json',
        }
        if berry is not None:
            self._servo_pending_record.update(self._berry_depth_diag(berry))

        self.get_logger().info(
            f'REFINING servo[{self._servo_step_idx}]: {control}{est_tag} '
            f'phase={self._refine_phase} aim={aim_kind} '
            f'z_cam={z_cam:.3f} dist_cup={dist_cup:.3f} step_m={step_m:.3f} '
            f'pix_err={(pix_err if pix_err is not None else -1.0):.1f} '
            f'err_yaw_deg={math.degrees(err_yaw):.1f} '
            f'err_pitch_deg={math.degrees(err_pitch):.1f} '
            f'reach_scale={reach_scale:.2f} '
            f'tgt=({tgt_xyz[0]:.3f},{tgt_xyz[1]:.3f},{tgt_xyz[2]:.3f})')
        traj_s = float(self._args.servo_traj_s)
        if self._refine_phase == 'center':
            traj_s = max(traj_s, float(getattr(self._args, 'servo_center_traj_s', 2.5)))
        return self._send_joint_servo_goal(
            target,
            tag='servo',
            traj_s=traj_s,
            extra=(
                f'{control} step_m={step_m:.3f} reach_scale={reach_scale:.2f} '
                f'dist_cup={dist_cup:.3f} phase={self._refine_phase}'
            ),
        )

    def _poll_wrist_servo(self) -> None:
        if self._servo_deadline > 0.0 and time.time() > self._servo_deadline:
            self._cancel_move()
            self._servo_pending_record = None
            self._pbvs_executor_wait = False
            self._set_state('ERROR', 'wrist servo step timeout')
            return

        # PBVS → executor path: no local FollowJointTrajectory handle.
        if self._pbvs_executor_wait and self._move_phase == 'servo':
            st = self._last_executor_status
            done = False
            if (st is not None
                    and int(getattr(st, 'tool_traj_seq', -1))
                    == int(getattr(self, '_pbvs_pending_tool_seq', -2))):
                if str(st.status) == 'ok':
                    done = True
                elif str(st.status) in ('ik_fail', 'jump_reject'):
                    self._pbvs_executor_wait = False
                    self._cancel_move()
                    self._set_state('ERROR', f'executor {st.status}: {st.detail}')
                    return
            if time.time() >= float(self._pbvs_executor_motion_until):
                done = True
            if done:
                self._pbvs_executor_wait = False
                self._snap_servo_motion_frame(force=True)
                self._servo_settle_until = time.time() + float(self._args.servo_settle_s)
                self._move_phase = 'servo_settle'
            else:
                self._snap_servo_motion_frame()
            return

        if self._move_phase == 'servo_ik':
            # Legacy absolute-/compute_ik path removed (j6 lock → always -31).
            self.get_logger().warn('REFINING: unexpected servo_ik phase — clearing')
            self._ik_fut = None
            self._move_phase = None
            self._servo_pending_record = None
            return

        if self._traj_goal_fut is not None:
            if not self._traj_goal_fut.done():
                self._snap_servo_motion_frame()
                return
            gh = self._traj_goal_fut.result()
            self._traj_goal_fut = None
            if gh is None or not gh.accepted:
                self._cancel_move()
                self._servo_pending_record = None
                self._servo_motion_frames = []
                self._set_state('ERROR', 'servo traj rejected')
                return
            self._arm_goal_handle = gh
            self._traj_result_fut = gh.get_result_async()
            self._snap_servo_motion_frame(force=True)
            return

        if self._traj_result_fut is not None:
            if not self._traj_result_fut.done():
                self._snap_servo_motion_frame()
                return
            self._traj_result_fut = None
            self._arm_goal_handle = None
            self._snap_servo_motion_frame(force=True)
            # Brief settle so next image is not from mid-motion.
            self._servo_settle_until = time.time() + float(self._args.servo_settle_s)
            self._move_phase = 'servo_settle'
            return

        if self._move_phase == 'servo_settle':
            if time.time() < self._servo_settle_until:
                self._snap_servo_motion_frame()
                return
            self._move_phase = None
            qa_tag = f'servo_{self._servo_step_idx:02d}_after'
            self._snap_qa(qa_tag)
            fb = {
                'j1': math.degrees(self._joints[0]),
                'j2': math.degrees(self._joints[1]),
                'j3': math.degrees(self._joints[2]),
                'j5': math.degrees(self._joints[4]),
            }
            self.get_logger().info(
                f'REFINING servo step OK fb_j=('
                f'{fb["j1"]:.1f},{fb["j2"]:.1f},{fb["j3"]:.1f},{fb["j5"]:.1f})')
            rec = self._servo_pending_record
            near_oneshot_done = False
            if rec is not None and int(rec.get('step', -1)) == int(self._servo_step_idx):
                near_oneshot_done = rec.get('control') in (
                    'near_oneshot_cart', 'near_oneshot_position_ik')
                rec = dict(rec)
                rec['fb_j_deg'] = fb
                rec['timestamp'] = datetime.now().isoformat(timespec='seconds')
                rec['session'] = self._qa_session
                rec['qa_tag'] = qa_tag
                rec['motion_frames'] = list(self._servo_motion_frames)
                rec['n_motion_frames'] = len(self._servo_motion_frames)
                tcp_after = self._tcp_or_ee_xyz()
                berry_after = self._last_berry_base
                if tcp_after is not None and berry_after is not None:
                    geo_after = self._cup_berry_geometry(berry_after, tcp_after)
                    rec['tcp_base_after'] = list(tcp_after)
                    rec['berry_rel_cup_after'] = geo_after.get('berry_rel_cup')
                    rec['dist_cup_after'] = geo_after.get('dist_cup')
                    rec['berry_uv_after'] = geo_after.get('berry_uv')
                    # Always keep model reproject separate from live (avoid circular QA).
                    rec['berry_uv_after_reproject'] = geo_after.get('berry_uv')
                    # Read-only: live bbox near center only (no coast / no re-pin).
                    if rec.get('refine_phase') == 'center_oneshot':
                        _cam_a, uv_a, src_a = self._center_oneshot_measure()
                        if uv_a is not None and src_a in ('live_bbox', 'live'):
                            rec['berry_uv_after'] = [float(uv_a[0]), float(uv_a[1])]
                            rec['berry_uv_after_src'] = src_a
                            rec['berry_uv_after_live'] = [
                                float(uv_a[0]), float(uv_a[1])]
                        else:
                            rec['berry_uv_after_src'] = 'yolo_miss_at_center'
                            rec['berry_uv_after_live'] = None
                        # Executed vs planned endpoint (control QA).
                        last_cmd = getattr(self, '_center_last_cmd', None) or {}
                        if last_cmd.get('ee_xyz_cmd') is not None:
                            rec['ee_xyz_cmd'] = list(last_cmd['ee_xyz_cmd'])
                        if last_cmd.get('joints_cmd_rad') is not None:
                            rec['joints_cmd_rad'] = list(last_cmd['joints_cmd_rad'])
                        rec['joints_fb_rad'] = [
                            float(v) for v in self._joints[:6]]
                        try:
                            ee_fb = fk_xyz(rec['joints_fb_rad'])
                            rec['ee_xyz_fb'] = [
                                float(ee_fb[0]), float(ee_fb[1]), float(ee_fb[2])]
                            if rec.get('ee_xyz_cmd') is not None:
                                rec['ee_err_mm'] = float(np.linalg.norm(
                                    np.array(rec['ee_xyz_fb'])
                                    - np.array(rec['ee_xyz_cmd'])) * 1000.0)
                        except Exception:
                            pass
                        # Independent control-gate live near cup-aim.
                        z_g = float(self._last_z_cam or 0.35)
                        au_g, av_g, _ = self._aim_uv_for_center(z_g)
                        live_max_g = max(
                            float(getattr(self._args, 'servo_center_pix_tol_px', 35.0)),
                            float(getattr(
                                self._args, 'refine_fresh_lock_max_pix_px', 80.0)),
                        )
                        _c2, uv2, src2 = self._center_arrive_measure_near_aim(
                            aim_uv=(au_g, av_g), max_pix=live_max_g)
                        rec['live_uv_near_aim'] = (
                            [float(uv2[0]), float(uv2[1])] if uv2 else None)
                        rec['live_src_near_aim'] = src2
                    rec['cup_uv_after'] = geo_after.get('cup_uv')
                    # Achieved Cartesian step at TCP (diagnose joint-clamp distortion).
                    tcp0 = rec.get('tcp_base')
                    if tcp0 is not None and len(tcp0) >= 3:
                        rec['achieved_dxyz'] = [
                            float(tcp_after[0] - tcp0[0]),
                            float(tcp_after[1] - tcp0[1]),
                            float(tcp_after[2] - tcp0[2]),
                        ]
                    self._write_cup_rel_overlay(qa_tag, geo_after)
                elif rec.get('berry_rel_cup') is not None:
                    self._write_cup_rel_overlay(qa_tag, {
                        'berry_rel_cup': rec.get('berry_rel_cup'),
                        'dist_cup': rec.get('dist_cup'),
                        'berry_uv': rec.get('berry_uv'),
                        'cup_uv': rec.get('cup_uv'),
                    })
                self._write_qa_json(f'servo_{self._servo_step_idx:02d}.json', rec)
            self._servo_pending_record = None
            self._servo_motion_frames = []
            self._servo_motion_last_t = 0.0
            self._servo_step_idx += 1
            # PBVS oneshot mode: approach / depth_final blocking trajs.
            pbvs_pending = getattr(self, '_pbvs_oneshot_pending', None)
            if pbvs_pending and self._pbvs_oneshot_mode():
                self._pbvs_oneshot_pending = None
                self._pbvs_on_oneshot_settle(pbvs_pending)
                return
            # Center lookat finished → residual or depth/stop.
            if self._refine_phase == 'center_oneshot':
                self._finish_center_oneshot()
            elif near_oneshot_done and (
                self._near_mode or self._refine_phase == 'near'):
                self._near_oneshot_sent = False
                dist_now = self._cup_berry_dist()
                contact = float(self._args.cup_contact_offset)
                # Tip↔berry-center ≤ contact (+8mm) = soft cup on fruit surface.
                # Former +25mm gate declared SUCCESS at ~40mm standoff (0.039m).
                tol = contact + 0.008
                if dist_now is not None and dist_now <= tol:
                    self.get_logger().info(
                        f'REFINING near oneshot done: contact dist={dist_now:.3f}m '
                        f'≤ {tol:.3f}m')
                    self._snap_qa(f'servo_{self._servo_step_idx:02d}_contact')
                    self._reached_pub.publish(Bool(data=True))
                    self._set_state('REACHED')
                    self._set_state('WAIT_CONFIRM')
                else:
                    max_att = max(1, int(getattr(self._args, 'servo_near_oneshot_max', 2)))
                    if int(self._near_oneshot_attempts) < max_att:
                        self.get_logger().warn(
                            f'REFINING near oneshot short dist='
                            f'{(dist_now if dist_now is not None else -1):.3f}m '
                            f'(need ≤{tol:.3f}m) — residual attempt '
                            f'{int(self._near_oneshot_attempts)+1}/{max_att}')
                    else:
                        self._set_state(
                            'ERROR',
                            f'near oneshot finished short of contact '
                            f'dist={(dist_now if dist_now is not None else -1):.3f}m '
                            f'(need ≤{tol:.3f}m)')
            return

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

    def _fine_reject_reason(self) -> str:
        """Why REFINING cannot yet promote a wrist fine berry (for logs / ERROR)."""
        if self._fine is None:
            return 'no /perception/fine/berries received yet'
        if not self._fine.berries:
            n = 0
            return (
                f'fine msg empty (n={n}) frame={self._fine.header.frame_id!r} '
                f'age={self._fine_msg_age_s():.2f}s')
        if not self._fine_is_fresh():
            return (
                f'fine msg stale age={self._fine_msg_age_s():.2f}s '
                f'> max={self._args.fine_max_age_s:.2f}s (n={len(self._fine.berries)})')
        berries = list(self._fine.berries)
        max_d = self._args.fine_assoc_max_m
        lock_tid = self._refine_locked_track_id
        parts = []
        for i, b in enumerate(berries[:5]):
            p = b.pose.pose.position
            bf = b.pose.header.frame_id or self._fine.header.frame_id
            parts.append(
                f'[{i}] id={int(b.track_id)} conf={b.confidence:.2f} '
                f'xyz=({p.x:.3f},{p.y:.3f},{p.z:.3f}) frame={bf!r}')
        if lock_tid is not None and lock_tid >= 0:
            if any(int(b.track_id) == int(lock_tid) for b in berries):
                modes = [
                    str(getattr(b, 'depth_mode', '') or '')
                    for b in berries if int(b.track_id) == int(lock_tid)]
                return (
                    f'locked track_id={lock_tid} present but not live-synced '
                    f'(modes={modes}; n={len(berries)}); ' + '; '.join(parts))
            return (
                f'locked track_id={lock_tid} not in frame (n={len(berries)}); '
                + '; '.join(parts))
        if max_d > 0 and self._locked is not None:
            lx = self._locked.pose.pose.position.x
            ly = self._locked.pose.pose.position.y
            lz = self._locked.pose.pose.position.z
            lock_frame = self._locked.pose.header.frame_id or self._locked.header.frame_id
            nearest = min(
                math.sqrt(
                    (b.pose.pose.position.x - lx) ** 2
                    + (b.pose.pose.position.y - ly) ** 2
                    + (b.pose.pose.position.z - lz) ** 2)
                for b in berries)
            if nearest > max_d:
                return (
                    f'assoc fail nearest={nearest:.3f}m > fine_assoc_max_m={max_d:.3f}m '
                    f'lock=({lx:.3f},{ly:.3f},{lz:.3f}) frame={lock_frame!r}; '
                    + '; '.join(parts))
        max_y = float(getattr(self._args, 'refine_lock_max_base_y_m', 0.42))
        if max_y > 0:
            high_y = []
            for b in berries:
                base = self._berry_base_xyz(b)
                if base is not None and float(base[1]) > max_y:
                    high_y.append(
                        f'id={int(b.track_id)} y={float(base[1]):.3f}m')
            if high_y:
                return (
                    f'have n={len(berries)} but pick rejected '
                    f'(base_y>{max_y:.2f}m: {", ".join(high_y[:5])}); '
                    + '; '.join(parts))
        return f'have n={len(berries)} but pick rejected; ' + '; '.join(parts)

    def _berry_uv_pix(
        self, berry: DetectedBerry,
    ) -> Optional[Tuple[float, float]]:
        cam = self._berry_cam_xyz(berry)
        if cam is None or cam[2] <= 1e-4:
            return None
        return self._berry_uv_from_cam(cam)

    def _berry_optical_pix_err(self, berry: DetectedBerry) -> Optional[float]:
        """Pixel distance of berry UV from cup-axis aim (fallback optical center)."""
        uv = self._berry_uv_pix(berry)
        if uv is None:
            return None
        cam = self._berry_cam_xyz(berry)
        z = float(cam[2]) if cam is not None and cam[2] > 1e-4 else None
        au, av, _ = self._aim_uv_for_center(z)
        return math.hypot(uv[0] - au, uv[1] - av)

    def _live_fine_berries(self) -> List[DetectedBerry]:
        if self._fine is None or not self._fine.berries or not self._fine_is_fresh():
            return []
        out: List[DetectedBerry] = []
        for b in self._fine.berries:
            mode = str(getattr(b, 'depth_mode', '') or '')
            if mode == 'base_coast':
                continue
            out.append(b)
        return out

    def _pick_probe_live_berry(self) -> Optional[DetectedBerry]:
        """Live locked berry for probe calib (center/depth). Never base_coast."""
        lock_tid = self._refine_locked_track_id
        track_min = float(getattr(self._args, 'refine_track_min_conf', 0.12))
        candidates: List[DetectedBerry] = []
        for b in self._live_fine_berries():
            if lock_tid is not None and int(b.track_id) != int(lock_tid):
                continue
            if float(b.confidence) < track_min:
                continue
            candidates.append(b)
        if not candidates:
            return None
        if lock_tid is not None:
            return candidates[0]
        # Entry calib before lock settled: nearest cam range.
        ranked: List[Tuple[float, DetectedBerry]] = []
        for b in candidates:
            d = self._berry_cam_range(b)
            if d is not None:
                ranked.append((d, b))
        if ranked:
            ranked.sort(key=lambda x: x[0])
            return ranked[0][1]
        return candidates[0]

    def _pick_live_nearest_optical(
        self,
        max_pix: Optional[float] = None,
        *,
        min_conf: Optional[float] = None,
    ) -> Optional[DetectedBerry]:
        """Live YOLO closest to cup-axis aim UV (gate / fresh lock after center)."""
        if min_conf is None:
            track_min = float(getattr(self._args, 'refine_track_min_conf', 0.12))
        else:
            track_min = float(min_conf)
        ranked: List[Tuple[float, DetectedBerry]] = []
        for b in self._live_fine_berries():
            if float(b.confidence) < track_min:
                continue
            e = self._berry_optical_pix_err(b)
            if e is not None:
                ranked.append((e, b))
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[0])
        if max_pix is not None and ranked[0][0] > float(max_pix):
            return None
        return ranked[0][1]

    def _pick_live_near_uv(self, uv_ref) -> Optional[DetectedBerry]:
        """Live detection closest to a reference UV (continue same berry)."""
        if uv_ref is None or len(uv_ref) < 2:
            return None
        ranked: List[Tuple[float, DetectedBerry]] = []
        for b in self._live_fine_berries():
            uv = self._berry_image_uv(b)
            if uv is None:
                continue
            d = math.hypot(float(uv[0]) - float(uv_ref[0]), float(uv[1]) - float(uv_ref[1]))
            ranked.append((d, b))
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[0])
        # Reject if nearest live jumped too far (different berry / ghost).
        if ranked[0][0] > 120.0:
            return None
        return ranked[0][1]

    def _relock_sight_center(self, *, reason: str) -> Optional[DetectedBerry]:
        """After centering: lock the berry nearest cup-axis aim for depth/contact."""
        neu = self._pick_live_nearest_optical()
        if neu is None:
            return None
        old = self._refine_locked_track_id
        tid = int(neu.track_id)
        self._refine_locked_track_id = tid
        self._lock_pub.publish(neu)
        cam = self._berry_cam_xyz(neu)
        if cam is not None and cam[2] > 1e-4:
            self._remember_berry(neu, float(cam[2]))
        pix = self._berry_optical_pix_err(neu)
        self.get_logger().info(
            f'REFINING sight-center lock {old}→{tid} ({reason}) '
            f'cup_aim_pix={(pix if pix is not None else -1):.1f} '
            f'mode={getattr(neu, "depth_mode", "")}')
        return neu

    def _pick_fine_berry(self) -> Optional[DetectedBerry]:
        if self._fine is None or not self._fine.berries:
            return None
        if not self._fine_is_fresh():
            self._fine = None
            return None
        berries = list(self._fine.berries)

        lock_tid = self._refine_locked_track_id
        if lock_tid is not None and lock_tid >= 0:
            live = self._live_fine_berries()
            locked: Optional[DetectedBerry] = None
            for b in berries:
                if int(b.track_id) == lock_tid:
                    locked = b
                    break

            if locked is not None:
                mode = str(getattr(locked, 'depth_mode', '') or '')
                if mode != 'base_coast':
                    return locked
                # Ghost lock: prefer live; near may keep coast only if nothing live.
                if self._near_mode or self._refine_phase == 'near':
                    if not live:
                        return locked

            # Lost/coast: re-pin only during free centering — NEVER during probe
            # ranging (optical-nearest will grab wrong clutter after a bad move).
            if live and self._refine_phase in ('center',):
                neu = self._pick_live_nearest_optical()
                if neu is None:
                    # Fallback: nearest cam range.
                    ranked: List[Tuple[float, DetectedBerry]] = []
                    for b in live:
                        d = self._berry_cam_range(b)
                        if d is not None:
                            ranked.append((d, b))
                    if ranked:
                        ranked.sort(key=lambda x: x[0])
                        neu = ranked[0][1]
                if neu is not None:
                    old = lock_tid
                    self._refine_locked_track_id = int(neu.track_id)
                    self._lock_pub.publish(neu)
                    self.get_logger().warn(
                        f'REFINING re-pin lock {old}→{int(neu.track_id)} '
                        f'phase={self._refine_phase} '
                        f'(old lost/coast; mode={getattr(neu, "depth_mode", "")})')
                    return neu
            return None

        max_d = self._args.fine_assoc_max_m
        if max_d > 0 and self._locked is not None:
            lx = self._locked.pose.pose.position.x
            ly = self._locked.pose.pose.position.y
            lz = self._locked.pose.pose.position.z

            def d2(b: DetectedBerry) -> float:
                dx = b.pose.pose.position.x - lx
                dy = b.pose.pose.position.y - ly
                dz = b.pose.pose.position.z - lz
                return dx * dx + dy * dy + dz * dz

            berries.sort(key=d2)
            if math.sqrt(d2(berries[0])) > max_d:
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
                        default=float(os.environ.get('PICK_CUP_CONTACT_OFFSET', '0.015')),
                        help='Legacy tip↔berry-center standoff for non-PBVS refine path')
    parser.add_argument('--cup-surface-clearance-m', type=float, default=0.003,
                        help='PBVS: cup rim ↔ berry surface gap at contact (m)')
    parser.add_argument('--cup-rim-z', type=float, default=0.0,
                        help='Deprecated (tip=opening); ignored for cup geometry')
    parser.add_argument('--berry-radius', type=float, default=0.008,
                        help='PBVS: nominal berry radius for cup_gap (m)')
    parser.add_argument('--pbvs-vision-lost-frames', type=int, default=3,
                        help='PBVS: consecutive lost live frames before blind oneshot')
    parser.add_argument('--pbvs-vision-z-cam-min', type=float, default=0.08,
                        help='PBVS: z_depth (m) below which near phase begins')
    parser.add_argument('--pbvs-near-d-cam-m', type=float, default=0.12,
                        help='PBVS stream: cam-frame cup↔berry surface distance for near phase (m)')
    parser.add_argument('--pbvs-pre-grasp-d-cam-m', type=float, default=0.08,
                        help='PBVS oneshot: segment-1 stop when d_cam_surface reaches this (m)')
    parser.add_argument('--refine-skip-center', dest='refine_skip_center',
                        action='store_true', default=False,
                        help='After mono_probe orbit+depth: skip center oneshot; '
                             'freeze probe berry and go straight to near contact oneshot')
    parser.add_argument('--no-refine-skip-center', dest='refine_skip_center',
                        action='store_false',
                        help='Keep center oneshot after probe (default)')
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
    parser.add_argument('--align-judge-mode', choices=('heuristic', 'file', 'vlm'), default='heuristic',
                        help='Action judge source during ALIGNING; file waits for align_decision.json; vlm uses local Ollama')
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
    # pbvs-vlm-reach-v2
    parser.add_argument('--use-pbvs', action='store_true', default=False,
                        help='Use PBVS refining loop instead of center_oneshot→range_depth')
    parser.add_argument('--pbvs-mode', choices=('single', 'oneshot', 'twostage', 'stream'),
                        default='single',
                        help='single(=oneshot alias): lock→1×cup_axis to berry surface; '
                             'twostage: legacy pre-grasp+depth_final; stream: 25Hz PBVS')
    parser.add_argument('--publish-tool-traj', dest='publish_tool_traj',
                        action='store_true', default=True,
                        help='Publish PBVS setpoints as /planning/tool_trajectory_4s')
    parser.add_argument('--no-publish-tool-traj', dest='publish_tool_traj',
                        action='store_false',
                        help='Disable PBVS→tool_trajectory_4s publishing')
    parser.add_argument('--pbvs-via-executor', action='store_true', default=False,
                        help='P0b: PBVS only publishes tool_trajectory_4s; '
                             'trajectory_executor drives the arm')
    parser.add_argument('--pbvs-press-m', type=float, default=0.0,
                        help='After direct settle, extra press into fruit along -n (m). '
                             '0=skip press (still SUCTION_HOLD placeholder). Cap 5mm.')
    parser.add_argument('--pbvs-normal-max-deg', type=float, default=20.0,
                        help='Clamp fitted n toward cup−P (not current cup_axis) if angle exceeds this')
    parser.add_argument('--pbvs-normal-skip-deg', type=float, default=45.0,
                        help='If raw n vs cup−P exceeds this, skip fit (point contact). '
                             'Do not gate on lock-time cup_axis — that coincidence is the contact goal')
    # Goal 4: BC training data collection
    parser.add_argument('--collect-data', action='store_true', default=False,
                        help='Enable AlignDataCollector: log every ALIGNING episode')
    parser.add_argument('--collect-data-dir', default='data/align_episodes',
                        help='Directory to save collected episodes')
    parser.add_argument('--bc-model-path', default='',
                        help='Path to trained BCAlignPolicy .pt file (used with --align-judge-mode bc)')
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
    parser.add_argument('--align-joint6-limit-rad', type=float, default=2.8797933,
                        help='ALIGN/REFINE |joint6| soft clamp rad; 0 locks j6 at 0')
    parser.add_argument('--qa-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'log', 'real_robot', 'qa'),
        help='Directory for before/after align screenshots')
    parser.add_argument('--velocity-scale', type=float, default=0.12)
    parser.add_argument('--position-tolerance', type=float, default=0.025)
    parser.add_argument('--orient-tolerance', type=float, default=0.35)
    parser.add_argument('--fine-assoc-max-m', type=float, default=0.0,
                        help='Optional max 3D distance global plant lock→fine berry; '
                             '0=disable (default: wrist tracker berry[0] after ALIGN)')
    parser.add_argument('--fine-max-age-s', type=float, default=0.5,
                        help='Ignore /perception/fine/berries older than this (wall-clock recv age); '
                             '0 disables freshness gate')
    parser.add_argument('--refine-timeout-s', type=float, default=45.0,
                        help='REFINING total budget for fine berry + wrist servo to contact; '
                             'timeout → ERROR (never open-loop cup_axis / coarse lock)')
    parser.add_argument('--servo-step-m', type=float, default=0.025,
                        help='Wrist servo approach step per closed-loop move (m)')
    parser.add_argument('--servo-blend', type=float, default=0.55,
                        help='look_at position blend toward berry each servo step')
    parser.add_argument('--servo-orient-frac', type=float, default=0.0,
                        help='Slerp EE quat toward look-at (0=keep orientation for reliable IK)')
    parser.add_argument('--servo-lateral-gain', type=float, default=0.35,
                        help='Unused (CLI compat); lateral now via Cartesian PBVS')
    parser.add_argument('--servo-yaw-gain', type=float, default=0.22,
                        help='Legacy joint IBVS (unused by cartesian_ik path)')
    parser.add_argument('--servo-pitch-gain', type=float, default=0.18,
                        help='Legacy joint IBVS (unused by cartesian_ik path)')
    parser.add_argument('--servo-reach-gain', type=float, default=5.5,
                        help='Legacy joint reach map (unused by cartesian_ik path)')
    parser.add_argument('--servo-ang-deadband-deg', type=float, default=3.0,
                        help='Ignore image angular error smaller than this')
    parser.add_argument('--servo-approach-ang-deg', type=float, default=18.0,
                        help='Legacy alias; prefer --servo-approach-yaw-deg')
    parser.add_argument('--servo-approach-yaw-deg', type=float, default=18.0,
                        help='Legacy joint gate (unused by cartesian_ik path)')
    parser.add_argument('--servo-near-handoff-z', type=float, default=0.05,
                        help='Switch to pose+fixed-mono only below this wrist z_cam (m). '
                             'With cam retracted ~4cm, wrist usable to ~3cm before fruit.')
    parser.add_argument('--servo-near-handoff-dist-m', type=float, default=0.08,
                        help='Max cup-gap (m) for PBVS near handoff / freeze')
    parser.add_argument('--servo-near-pix-tol-px', type=float, default=55.0,
                        help='Max pixel error for cup+image near handoff gate')
    parser.add_argument('--servo-near-ang-tol-deg', type=float, default=8.0,
                        help='Max |err_yaw|/|err_pitch| deg for near handoff gate')
    parser.add_argument('--servo-ibvs-reach-pix-px', type=float, default=80.0,
                        help='Full Cartesian step when pix_err (vs cup) below this')
    parser.add_argument('--servo-ibvs-reach-pix-soft-px', type=float, default=280.0,
                        help='Above this pix_err, step scale = min_scale (not zero)')
    parser.add_argument('--servo-ibvs-reach-min-scale', type=float, default=0.30,
                        help='Minimum Cartesian approach scale at large image error')
    parser.add_argument('--servo-center-pix-tol-px', type=float, default=35.0,
                        help='pix_err vs optical center to leave center phase')
    parser.add_argument(
        '--servo-center-cup-aim-frac', type=float, default=1.0,
        help='Deprecated/debug: fraction of cup-axis UV from optical for mid-range '
             'center (1=full cup-aim). Do not use <1 as a climb workaround')
    parser.add_argument('--servo-center-ok-frames', type=int, default=3,
                        help='Consecutive centered frames before center→probe/approach')
    parser.add_argument('--servo-center-step-m', type=float, default=0.012,
                        help='Max EE step during center lateral IBVS (m)')
    parser.add_argument('--servo-center-gain', type=float, default=0.45,
                        help='Fraction of cam-lateral error taken per center step')
    parser.add_argument('--servo-center-traj-s', type=float, default=2.5,
                        help='Slower traj during center so BoT-SORT can track')
    parser.add_argument('--servo-center-oneshot-tol-m', type=float, default=0.008,
                        help='Skip center oneshot if berry already within this cam-lateral (m)')
    parser.add_argument('--servo-center-oneshot-traj-s', type=float, default=2.0,
                        help='Traj time for one-shot optical centering move')
    parser.add_argument('--servo-center-oneshot-gain', type=float, default=1.0,
                        help='Fraction of UV/geo error closed by center arrive (default 1=full)')
    parser.add_argument('--servo-center-oneshot-max-m', type=float, default=0.0,
                        help='Max lateral EE step for center arrive (m); 0=no cap (full oneshot)')
    parser.add_argument('--servo-center-oneshot-max', type=int, default=2,
                        help='Max center aim_uv_ik shots (1 open-loop + 1 live-replan)')
    parser.add_argument('--servo-center-live-replan', dest='servo_center_live_replan',
                        action='store_true', default=True,
                        help='After center arrive: live UV depth re-estimate + second oneshot')
    parser.add_argument('--servo-center-oneshot-max-dj1-deg', type=float, default=10.0,
                        help='Max |Δj1| for one center look-at')
    parser.add_argument('--servo-center-oneshot-max-dj5-deg', type=float, default=10.0,
                        help='Max |Δj5| for one center look-at')
    parser.add_argument('--refine-stop-after-center', dest='refine_stop_after_center',
                        action='store_true', default=False,
                        help='Step1 test: after center oneshot pause at WAIT_CONFIRM (no depth/contact)')
    parser.add_argument('--servo-near-confirm', dest='servo_near_confirm',
                        action='store_true', default=True,
                        help='Pause at REFINING_WAIT_NEAR for confirm_near (default on)')
    parser.add_argument('--no-servo-near-confirm', dest='servo_near_confirm',
                        action='store_false',
                        help='Auto-enter near mode without user confirm_near')
    parser.add_argument('--refine-fruit-assoc-max-m', type=float, default=0.12,
                        help='Max 3D distance from REFINING entry fruit lock to accept detections')
    parser.add_argument('--refine-lock-min-conf', type=float, default=0.20,
                        help='Min YOLO/BoT-SORT confidence to lock a fruit at REFINING entry')
    parser.add_argument('--refine-lock-max-base-y-m', type=float, default=0.42,
                        help='Skip REFINING entry lock when berry base_link Y exceeds this (m); 0=off')
    parser.add_argument('--refine-track-min-conf', type=float, default=0.12,
                        help='After lock: min conf for ranging/track obs (box on locked id; << lock gate)')
    parser.add_argument('--refine-fresh-lock-max-pix-px', type=float, default=80.0,
                        help='Fresh lock after center: max live YOLO offset from cup-aim (px)')
    parser.add_argument('--refine-fresh-lock-min-conf', type=float, default=0.10,
                        help='After center re-lock: min conf (lower than entry; FOV is centered)')
    parser.add_argument('--refine-post-center-live-wait-s', type=float, default=1.0,
                        help='After center: wait for live YOLO near cup-aim before probe fallback (~10 frames @10Hz)')
    parser.add_argument('--refine-contact-from-probe-tri', dest='refine_contact_from_probe_tri',
                        action='store_true', default=True,
                        help='YOLO miss after center: contact from center probe tri 3D (default on)')
    parser.add_argument('--no-refine-contact-from-probe-tri', dest='refine_contact_from_probe_tri',
                        action='store_false',
                        help='Require live YOLO fresh lock after center (legacy ERROR on miss)')
    parser.add_argument('--refine-probe-tri-mono-chord', dest='refine_probe_tri_mono_chord',
                        action='store_true', default=False,
                        help='Legacy: chord-scale probe tri via min-mono (off when wrist depth valid)')
    parser.add_argument('--no-refine-probe-tri-mono-chord', dest='refine_probe_tri_mono_chord',
                        action='store_false',
                        help='Keep raw probe tri / depth (default for Gemini wrist)')
    parser.add_argument('--refine-mono-chord-scale-min', type=float, default=0.45,
                        help='Min z_mono/z_cam scale for probe tri mono chord')
    parser.add_argument('--refine-mono-chord-scale-max', type=float, default=1.05,
                        help='Max z_mono/z_cam scale for probe tri mono chord')
    parser.add_argument('--refine-reproject-mono-max-pix-px', type=float, default=120.0,
                        help='Max YOLO offset from reproject UV for mono chord')
    parser.add_argument('--berry-diameter-m', type=float, default=0.015,
                        help='Berry diameter for mono depth sizing (m)')
    parser.add_argument('--refine-z-cam-max-jump-m', type=float, default=0.22,
                        help='Reject single-frame z_cam increases larger than this (m)')
    parser.add_argument('--refine-track-max-jump-m', type=float, default=0.08,
                        help='Reject locked-berry base jumps larger than this (m)')
    parser.add_argument('--refine-lost-frames-for-near', type=int, default=8,
                        help='Consecutive lost frames before near handoff (with dist gate)')
    parser.add_argument('--servo-near-oneshot-max', type=int, default=2,
                        help='Max near contact trajs (2=allow residual after approach_axis)')
    parser.add_argument('--servo-near-oneshot-step-m', type=float, default=0.0,
                        help='Optional Cartesian cap per near traj (m); '
                        '0 = full remaining dist_cup-contact in one shot')
    parser.add_argument('--servo-near-cam-lat-gain', type=float, default=0.12,
                        help='Near cam-depth: lateral trim fraction from berry_cam XY')
    parser.add_argument('--servo-mono-probe', dest='servo_mono_probe',
                        action='store_true', default=True,
                        help='After center: small-step triangulate depth, then oneshot contact')
    parser.add_argument('--no-servo-mono-probe', dest='servo_mono_probe',
                        action='store_false',
                        help='Skip depth probe; after center go straight to oneshot')
    parser.add_argument('--servo-mono-probe-steps', type=int, default=1,
                        help='Known moves for mono depth confirm (1=one step then approach)')
    parser.add_argument('--servo-mono-probe-step-m', type=float, default=0.02,
                        help='Cartesian length of each mono probe move (m)')
    parser.add_argument('--servo-mono-probe-yaw-frac', type=float, default=0.25,
                        help='Fraction of orbit yaw to cancel (0=max ΔUV for Jac, 1=keep facing)')
    parser.add_argument('--servo-mono-probe-max-duv-px', type=float, default=90.0,
                        help='Abort probe if berry UV jumps more than this between views')
    parser.add_argument('--servo-estimate-max-age-s', type=float, default=8.0,
                        help='Max age of last berry base pose for near handoff after wrist loss')
    parser.add_argument('--servo-j1-anchor-band-deg', type=float, default=0.0,
                        help='Deprecated unused (ALIGN anchor bands removed)')
    parser.add_argument('--servo-j5-anchor-band-deg', type=float, default=0.0,
                        help='Deprecated unused (ALIGN anchor bands removed)')
    parser.add_argument('--servo-j2-anchor-band-deg', type=float, default=0.0,
                        help='Deprecated unused (ALIGN anchor bands removed)')
    parser.add_argument('--servo-j3-anchor-band-deg', type=float, default=0.0,
                        help='Deprecated unused (ALIGN anchor bands removed)')
    parser.add_argument('--servo-max-dj1-deg', type=float, default=3.0)
    parser.add_argument('--servo-max-dj5-deg', type=float, default=3.5)
    parser.add_argument('--servo-max-dj2-deg', type=float, default=5.0)
    parser.add_argument('--servo-max-dj3-deg', type=float, default=5.0)
    parser.add_argument('--servo-settle-s', type=float, default=0.6,
                        help='Settle time after each wrist servo traj before next image')
    parser.add_argument('--servo-motion-snap-s', type=float, default=0.10,
                        help='During servo traj, save replay frames at most every N seconds')
    parser.add_argument('--servo-pixel-tol-px', type=float, default=40.0,
                        help='Berry center pixel error threshold for contact')
    parser.add_argument('--servo-traj-s', type=float, default=2.2,
                        help='FollowJointTrajectory duration per wrist servo step')
    parser.add_argument('--servo-timeout-s', type=float, default=12.0,
                        help='Timeout for one servo step')
    parser.add_argument('--wrist-camera-frame',
                        default='camera_wrist_color_optical_frame')
    parser.add_argument('--wrist-focal-px', type=float, default=488.0,
                        help='Fallback wrist fx=fy if /camera_wrist/color/camera_info missing')
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
