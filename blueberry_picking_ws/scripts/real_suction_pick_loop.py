#!/usr/bin/env python3
"""Real-robot suction pick loop — full pipeline for video / unattended runs.

Each cycle (--move):
  1. Home          — joint zero (initial pose)
  2. Detect        — YOLO + FP at home; optional camera +Z fine search
  3. Patrol        — only if step 2 finds no berry: joint grid sweep + detect
  4. Lock + plan   — locked berry -> plan_suction (pre_grasp / grasp / post_grasp)
  5. Approach      — MoveIt to pre_grasp, then grasp
  6. Suction       — press G to confirm suction on, then post_grasp retreat

Keyboard (terminal must be focused):
  R / H     STOP: cancel motion + MoveIt home (same joints as teleop R)
  G         After grasp pose: continue to post_grasp (suction should be on)
  Q / Esc   Quit loop (go home first)
  ?         Help

Requires: arm bringup + perception + grasp_planner. Run viz separately or use
run_real_suction_pick_loop.sh which starts detection overlay.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import threading
import tty
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
)
from picking_msgs.srv import PlanSuction, TriggerFineDetection
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener

_WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# tcp_link +0.05m from link6; cup mouth on tcp +Z (same axis as camera / CS7 +Z)
DEFAULT_CUP_CONTACT_OFFSET_M = 0.04
for rel in (
    'install/picking_msgs/lib/python3.12/site-packages',
    'install/picking_perception/lib/python3.12/site-packages',
):
    p = os.path.join(_WS, rel)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

_DEFAULT_SCAN_ANCHOR = [0.0, -0.40, 0.55, 0.0, 0.30, 0.0]
_DEFAULT_SCAN_J1_VALUES = [-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0]
_DEFAULT_SCAN_J2_DELTAS = [0.0]
_DEFAULT_SCAN_MODE = 'z_rotate'

_MOVEIT_ERROR_HINTS = {
    -4: 'CONTROL_FAILED: MoveIt 无法执行轨迹，常见原因是无 /joint_states（臂驱动未运行或未使能）',
    -1: 'PLANNING_FAILED: 规划失败',
    -2: 'INVALID_MOTION_PLAN',
    -3: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
    99999: 'FAILURE: OMPL 无法到达目标（常见：6D 姿态约束过严或目标不可达）',
}


def _moveit_error_hint(code: int) -> str:
    hint = _MOVEIT_ERROR_HINTS.get(code, '')
    return f'code={code}' + (f' ({hint})' if hint else '')


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.replace(',', ' ').split()]


def _load_scan_env(path: str) -> dict:
    anchor = list(_DEFAULT_SCAN_ANCHOR)
    j1_values = list(_DEFAULT_SCAN_J1_VALUES)
    j1_deltas = list(_DEFAULT_SCAN_J1_VALUES)  # legacy grid
    j2_deltas = list(_DEFAULT_SCAN_J2_DELTAS)
    mode = _DEFAULT_SCAN_MODE
    settle_s = 3.0
    detect_attempts = 3
    detect_interval_s = 1.5
    recorded_path = os.path.join(_WS, 'config', 'pick_scan_recorded.txt')
    if not os.path.isfile(path):
        return {
            'mode': mode,
            'anchor': anchor,
            'j1_values': j1_values,
            'j1_deltas': j1_deltas,
            'j2_deltas': j2_deltas,
            'settle_s': settle_s,
            'detect_attempts': detect_attempts,
            'detect_interval_s': detect_interval_s,
            'recorded_path': recorded_path,
        }
    with open(path, encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            val = val.strip().strip('"').strip("'")
            if key == 'PICK_SCAN_MODE':
                mode = val
            elif key == 'PICK_SCAN_ANCHOR':
                anchor = _parse_float_list(val)
            elif key == 'PICK_SCAN_J1_VALUES':
                j1_values = _parse_float_list(val)
            elif key == 'PICK_SCAN_J1_DELTAS':
                j1_deltas = _parse_float_list(val)
            elif key == 'PICK_SCAN_J2_DELTAS':
                j2_deltas = _parse_float_list(val)
            elif key == 'PICK_SCAN_SETTLE_S':
                settle_s = float(val)
            elif key == 'PICK_SCAN_DETECT_ATTEMPTS':
                detect_attempts = int(val)
            elif key == 'PICK_SCAN_DETECT_INTERVAL_S':
                detect_interval_s = float(val)
            elif key == 'PICK_SCAN_RECORDED':
                recorded_path = val
                if not os.path.isabs(recorded_path):
                    recorded_path = os.path.join(_WS, recorded_path)
    if len(anchor) != 6:
        raise ValueError(f'{path}: PICK_SCAN_ANCHOR needs 6 joint values')
    return {
        'mode': mode,
        'anchor': anchor,
        'j1_values': j1_values,
        'j1_deltas': j1_deltas,
        'j2_deltas': j2_deltas,
        'settle_s': settle_s,
        'detect_attempts': detect_attempts,
        'detect_interval_s': detect_interval_s,
        'recorded_path': recorded_path,
    }


def _load_recorded_poses(path: str) -> List[Tuple[str, List[float]]]:
    if not os.path.isfile(path):
        return []
    poses: List[Tuple[str, List[float]]] = []
    pending_label = ''
    idx = 0
    with open(path, encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                pending_label = line.lstrip('#').strip()
                continue
            joints = _parse_float_list(line)
            if len(joints) != 6:
                continue
            idx += 1
            name = pending_label or f'recorded{idx}'
            pending_label = ''
            poses.append((name.replace(' ', '_'), joints))
    return poses


def _build_z_rotate_pose_list(anchor: List[float],
                              j1_values: List[float],
                              j2_deltas: List[float]) -> List[Tuple[str, List[float]]]:
    """Patrol: rotate about base_link Z via joint1, optional joint2 height tiers."""
    poses: List[Tuple[str, List[float]]] = []
    for dj2 in j2_deltas:
        for j1 in j1_values:
            joints = list(anchor)
            joints[0] = j1
            joints[1] += dj2
            j2tag = f'_h{dj2:+.2f}'.replace('+', 'p').replace('-', 'm') if dj2 != 0.0 else ''
            name = f'zrot_j1{j1:+.2f}'.replace('+', 'p').replace('-', 'm') + j2tag
            poses.append((name, joints))
    return poses


def _build_grid_pose_list(anchor: List[float],
                          j1_deltas: List[float],
                          j2_deltas: List[float]) -> List[Tuple[str, List[float]]]:
    poses: List[Tuple[str, List[float]]] = []
    for dj2 in j2_deltas:
        for dj1 in j1_deltas:
            joints = list(anchor)
            joints[0] += dj1
            joints[1] += dj2
            name = f'j1{dj1:+.2f}_j2{dj2:+.2f}'.replace('+', 'p').replace('-', 'm')
            poses.append((name, joints))
    return poses


def _dedupe_scan_poses(poses: List[Tuple[str, List[float]]]) -> List[Tuple[str, List[float]]]:
    out: List[Tuple[str, List[float]]] = []
    last_key: Optional[Tuple[float, ...]] = None
    for name, joints in poses:
        key = tuple(round(v, 3) for v in joints)
        if key == last_key:
            continue
        last_key = key
        out.append((name, joints))
    return out


def _resolve_scan_poses(cfg: dict,
                        cli_poses: Optional[List[str]]) -> List[Tuple[str, List[float]]]:
    if cli_poses:
        out: List[Tuple[str, List[float]]] = []
        for i, pose in enumerate(cli_poses, start=1):
            joints = _parse_float_list(pose)
            if len(joints) != 6:
                raise ValueError(f'--scan-pose needs 6 joints, got {len(joints)}: {pose}')
            out.append((f'pose{i}', joints))
        return out

    recorded = _load_recorded_poses(cfg['recorded_path'])
    if recorded:
        return _dedupe_scan_poses(recorded)

    mode = cfg.get('mode', _DEFAULT_SCAN_MODE)
    if mode == 'recorded':
        return recorded

    if mode == 'grid':
        return _build_grid_pose_list(cfg['anchor'], cfg['j1_deltas'], cfg['j2_deltas'])

    return _build_z_rotate_pose_list(cfg['anchor'], cfg['j1_values'], cfg['j2_deltas'])


KEY_HELP = """
真机吸盘循环 — 按键介入（终端需聚焦）:
  R / H   急停并回零（与遥操 R 相同，MoveIt 关节回 home）
  G       到达 grasp 后：确认吸盘已开，继续 post_grasp 撤离
  Q / Esc 退出循环（先回零）
  ?       显示本帮助
  拍视频可加 --auto-retreat，到 grasp 后自动等待再撤离（无需按 G）
"""


def _quat_z_axis(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Unit Z axis of rotation quaternion (camera optical +Z in parent frame)."""
    return (
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )


def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-9:
        return 0.0, 0.0, 1.0
    return v[0] / n, v[1] / n, v[2] / n


def _normalize3(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return _normalize(v)


def _quat_align_z_to_dir(dx: float, dy: float, dz: float) -> Tuple[float, float, float, float]:
    """Quaternion so link +Z points along (dx,dy,dz)."""
    nx, ny, nz = _normalize3((dx, dy, dz))
    dot = nz
    if dot > 1.0 - 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    if dot < -1.0 + 1e-9:
        return 1.0, 0.0, 0.0, 0.0
    cx, cy = -ny, nx
    w = 1.0 + dot
    n = math.sqrt(cx * cx + cy * cy + w * w)
    return cx / n, cy / n, 0.0, w / n


def _quat_cup_toward_berry(dx: float, dy: float, dz: float) -> Tuple[float, float, float, float]:
    """Quaternion so tcp +Z (cup opening, camera-forward) points toward the berry."""
    return _quat_align_z_to_dir(dx, dy, dz)


def _print_pose(label: str, ps: PoseStamped) -> None:
    p = ps.pose.position
    print(f'  {label} [{ps.header.frame_id}] ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')


@dataclass
class ApproachGeometry:
    berry_cam: Tuple[float, float, float]
    cam_depth_m: float
    berry_l6: Tuple[float, float, float]
    tcp_l6: Tuple[float, float, float]
    berry_base: Optional[Tuple[float, float, float]]
    tcp_base: Optional[Tuple[float, float, float]]
    dist_tcp_berry_l6_m: float
    dist_tcp_berry_base_m: Optional[float]
    approach_l6: Tuple[float, float, float]
    approach_base: Optional[Tuple[float, float, float]]
    cup_offset_m: float
    pre_offset_m: float
    post_offset_m: float


class KeyboardMonitor:
    """Non-blocking single-key reader (like link6 teleop)."""

    def __init__(self) -> None:
        self._stop = False
        self._home = False
        self._continue_grasp = False
        self._quit = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._fd: Optional[int] = None
        self._old_term: Optional[list] = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            print('[keys] WARN: stdin not a TTY — keyboard estop disabled', file=sys.stderr)
            return
        self._fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._fd is not None and self._old_term is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
        self._quit = True

    def _loop(self) -> None:
        while not self._quit:
            if self._fd is None:
                break
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if not ch:
                continue
            key = ch.lower()
            with self._lock:
                if key in ('r', 'h'):
                    self._home = True
                    self._stop = True
                    print('\n[keys] HOME requested — cancel motion + go zero')
                elif key == 'g':
                    self._continue_grasp = True
                    print('\n[keys] continue -> post_grasp')
                elif key in ('q', '\x1b') or ch == '\x03':
                    self._quit = True
                    self._home = True
                    self._stop = True
                    print('\n[keys] quit')
                elif key == '?':
                    print(KEY_HELP)

    def consume_stop(self) -> bool:
        with self._lock:
            if self._stop:
                self._stop = False
                return True
            return False

    def consume_home(self) -> bool:
        with self._lock:
            if self._home:
                self._home = False
                return True
            return False

    def consume_continue_grasp(self) -> bool:
        with self._lock:
            if self._continue_grasp:
                self._continue_grasp = False
                return True
            return False

    def should_quit(self) -> bool:
        return self._quit


class PickLoopNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('real_suction_pick_loop')
        self._args = args
        self._keys = KeyboardMonitor()
        self._fp = self.create_client(TriggerFineDetection, 'trigger_fine_detection')
        self._plan = self.create_client(PlanSuction, 'plan_suction')
        self._enable_arm = self.create_client(SetBool, 'enable_agx_arm')
        self._move = ActionClient(self, MoveGroup, args.move_action)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._grasp_cup_pub = self.create_publisher(
            PointStamped, '/pick/suction_cup_contact', 1)
        self._grasp_eef_pub = self.create_publisher(
            PoseStamped, '/pick/suction_grasp_target', 1)
        self._locked_berry_pub = self.create_publisher(
            PointStamped, '/pick/locked_berry', 1)
        self._active_goal = None
        self._home_joints = list(args.home_joints)
        self._joint_positions: dict[str, float] = {}
        self._joint_state_time: Optional[rclpy.time.Time] = None
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)
        self._resolved_camera_frame: Optional[str] = None
        self._scan_poses: List[Tuple[str, List[float]]] = []
        self._scan_mode = 'z_rotate'
        self._scan_settle_s = 3.0
        self._scan_detect_attempts = 3
        self._scan_detect_interval_s = 1.5
        self._direct_teleop = False
        if not args.no_scan:
            cfg = _load_scan_env(args.scan_config)
            if getattr(args, 'scan_settle_s', None) is not None:
                cfg['settle_s'] = args.scan_settle_s
            if getattr(args, 'scan_detect_attempts', None) is not None:
                cfg['detect_attempts'] = args.scan_detect_attempts
            if getattr(args, 'scan_detect_interval_s', None) is not None:
                cfg['detect_interval_s'] = args.scan_detect_interval_s
            self._scan_settle_s = cfg['settle_s']
            self._scan_detect_attempts = cfg['detect_attempts']
            self._scan_detect_interval_s = cfg['detect_interval_s']
            self._scan_poses = _resolve_scan_poses(cfg, args.scan_pose or None)
            recorded = _load_recorded_poses(cfg['recorded_path'])
            if recorded:
                self._scan_mode = 'teleop (direct)'
                self._direct_teleop = not args.search_at_home
            elif cfg.get('mode') == 'grid':
                self._scan_mode = 'grid'
            else:
                self._scan_mode = 'z_rotate (joint1 / base Z)'
            if args.direct_scan:
                self._direct_teleop = True

    def _want_scan_move(self) -> bool:
        return bool(self._args.move or self._args.probe_only)

    def _want_grasp_move(self) -> bool:
        return bool(self._args.move and not self._args.probe_only)

    def _on_joint_states(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self._joint_positions[name] = float(pos)
        self._joint_state_time = Time.from_msg(msg.header.stamp)

    def _wait_joint_states(self, timeout_sec: float) -> bool:
        deadline = self.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            if self._joint_state_time is not None:
                age = (self.get_clock().now() - self._joint_state_time).nanoseconds / 1e9
                if age < 1.0 and all(j in self._joint_positions for j in ARM_JOINTS):
                    return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _try_enable_arm(self) -> bool:
        if not self._enable_arm.wait_for_service(timeout_sec=2.0):
            print('[pick] WARN: /enable_agx_arm service unavailable')
            return False
        req = SetBool.Request()
        req.data = True
        fut = self._enable_arm.call_async(req)
        deadline = self.get_clock().now().nanoseconds + int(8e9)
        while rclpy.ok() and not fut.done():
            if self.get_clock().now().nanoseconds > deadline:
                return False
            rclpy.spin_once(self, timeout_sec=0.05)
        res = fut.result()
        if res is None or not res.success:
            print('[pick] WARN: enable_agx_arm call failed')
            return False
        print('[pick] enable_agx_arm -> true')
        return True

    def _ensure_arm_ready(self) -> bool:
        print('[pick] Waiting for /joint_states ...')
        if self._wait_joint_states(5.0):
            return True
        print('[pick] No joint feedback — trying /enable_agx_arm ...')
        self._try_enable_arm()
        if self._wait_joint_states(10.0):
            return True
        print('[pick] ERROR: arm not ready (no fresh /joint_states).', file=sys.stderr)
        print('  agx_arm_ctrl may have crashed. Check:', file=sys.stderr)
        print('    tail -40 log/real_robot/arm.log', file=sys.stderr)
        print('  Fix:', file=sys.stderr)
        print('    bash scripts/real_robot_shutdown.sh', file=sys.stderr)
        print('    bash scripts/real_robot_bringup.sh --perception', file=sys.stderr)
        print('  Then verify:', file=sys.stderr)
        print('    ros2 topic hz /feedback/joint_states', file=sys.stderr)
        return False

    def _joints_near(self, target: List[float], tol: float) -> bool:
        for name, goal in zip(ARM_JOINTS, target):
            cur = self._joint_positions.get(name)
            if cur is None or abs(cur - goal) > tol:
                return False
        return True

    def _wait_service(self, client, name: str, timeout: float = 30.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f'{name} unavailable')
            return False
        return True

    def _abort_check(self) -> bool:
        return self._keys.consume_stop() or self._keys.should_quit()

    def _cancel_active_move(self) -> None:
        if self._active_goal is not None:
            try:
                self._active_goal.cancel_goal_async()
            except Exception:
                pass
            self._active_goal = None

    def _build_home_goal(self) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self._args.move_group
        req.num_planning_attempts = 10
        req.allowed_planning_time = 8.0
        req.max_velocity_scaling_factor = self._args.velocity_scale
        req.max_acceleration_scaling_factor = self._args.velocity_scale
        constraints = Constraints()
        for joint_name, position in zip(ARM_JOINTS, self._home_joints):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(position)
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

    def _build_joint_goal(self, joints: List[float],
                          velocity_scale: Optional[float] = None) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self._args.move_group
        req.num_planning_attempts = 10
        req.allowed_planning_time = 8.0
        scale = self._args.velocity_scale if velocity_scale is None else velocity_scale
        req.max_velocity_scaling_factor = scale
        req.max_acceleration_scaling_factor = scale
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

    def _go_home(self) -> bool:
        self._cancel_active_move()
        if not self._move.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('move_action unavailable for home')
            return False
        if self._joints_near(self._home_joints, self._args.home_tolerance):
            print(f'[pick] Already at home (within {self._args.home_tolerance:.3f} rad)')
            return True
        print(f'[pick] Moving home joints={[round(v, 3) for v in self._home_joints]}')
        return self._execute_move_goal(self._build_home_goal(), label='home')

    def _build_position_goal(self, target: PoseStamped,
                            tolerance_m: Optional[float] = None) -> MoveGroup.Goal:
        """Position-only goal — easier for OMPL from teleop scan poses."""
        tol = self._args.position_tolerance if tolerance_m is None else tolerance_m
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
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [tol]
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

    def _build_pose_goal(self, target: PoseStamped) -> MoveGroup.Goal:
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
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [0.02]
        pc.constraint_region.primitives = [region]
        pc.constraint_region.primitive_poses = [target.pose]
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.link_name = self._args.ee_link
        oc.header = target.header
        oc.orientation = target.pose.orientation
        oc.absolute_x_axis_tolerance = 0.35
        oc.absolute_y_axis_tolerance = 0.35
        oc.absolute_z_axis_tolerance = 0.35
        oc.weight = 1.0
        req.goal_constraints = [Constraints(
            position_constraints=[pc], orientation_constraints=[oc])]
        goal.request = req
        opts = PlanningOptions()
        opts.plan_only = False
        opts.replan = True
        opts.replan_attempts = 3
        goal.planning_options = opts
        return goal

    def _send_pose(self, target: PoseStamped, label: str,
                   position_only: bool = False) -> bool:
        mode = 'position' if position_only else '6D'
        print(f'[pick] moving -> {label} ({mode})')
        builder = self._build_position_goal if position_only else self._build_pose_goal
        return self._execute_move_goal(builder(target), label=label)

    def _send_pose_with_fallback(self, target: PoseStamped, label: str,
                                 position_only: bool) -> bool:
        if self._send_pose(target, label, position_only=position_only):
            return True
        if position_only:
            return False
        print(f'[pick] WARN: {label} 6D plan failed — retry position-only')
        return self._send_pose(target, label, position_only=True)

    def _execute_move_goal(self, goal: MoveGroup.Goal, label: str = 'move') -> bool:
        send = self._move.send_goal_async(goal)
        deadline = self.get_clock().now().nanoseconds + int(30e9)
        while rclpy.ok() and not send.done():
            if self._abort_check():
                return False
            if self.get_clock().now().nanoseconds > deadline:
                self.get_logger().error(f'{label}: send goal timeout')
                return False
            rclpy.spin_once(self, timeout_sec=0.05)

        handle = send.result()
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: goal rejected')
            return False
        self._active_goal = handle
        result_fut = handle.get_result_async()
        deadline = self.get_clock().now().nanoseconds + int(120e9)
        while rclpy.ok() and not result_fut.done():
            if self._abort_check():
                self._cancel_active_move()
                return False
            if self.get_clock().now().nanoseconds > deadline:
                self._cancel_active_move()
                self.get_logger().error(f'{label}: move timeout')
                return False
            rclpy.spin_once(self, timeout_sec=0.05)

        self._active_goal = None
        try:
            result = result_fut.result().result
        except Exception as exc:
            self.get_logger().error(f'{label}: move failed: {exc}')
            return False
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f'{label}: move failed {_moveit_error_hint(result.error_code.val)}')
            return False
        return True

    def _send_joints(self, joints: List[float], label: str,
                     velocity_scale: Optional[float] = None) -> bool:
        print(f'[pick] moving -> {label} joints={[round(v, 3) for v in joints]}')
        return self._execute_move_goal(
            self._build_joint_goal(joints, velocity_scale=velocity_scale), label=label)

    def _dwell(self, seconds: float, label: str) -> bool:
        if seconds <= 0.0:
            return True
        print(f'[pick] dwell {seconds:.1f}s — {label}')
        deadline = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            if self._abort_check():
                return False
            rclpy.spin_once(self, timeout_sec=0.05)
        return True

    def _settle_after_scan_move(self) -> bool:
        return self._dwell(self._scan_settle_s, 'arm/camera settling before detect')

    def _berry_position_in_frame(
            self, berry, frame: str) -> Optional[Tuple[float, float, float]]:
        ps = berry.pose
        src = ps.header.frame_id or self._args.base_frame
        if src == frame:
            p = ps.pose.position
            return float(p.x), float(p.y), float(p.z)
        try:
            tf = self._tf_buffer.lookup_transform(frame, src, Time())
        except Exception as exc:
            self.get_logger().warn(f'berry TF {src}->{frame}: {exc}')
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        px, py, pz = ps.pose.position.x, ps.pose.position.y, ps.pose.position.z
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        rx = (1 - 2 * (qy * qy + qz * qz)) * px + 2 * (qx * qy - qw * qz) * py + 2 * (qx * qz + qw * qy) * pz
        ry = 2 * (qx * qy + qw * qz) * px + (1 - 2 * (qx * qx + qz * qz)) * py + 2 * (qy * qz - qw * qx) * pz
        rz = 2 * (qx * qz - qw * qy) * px + 2 * (qy * qz + qw * qx) * py + (1 - 2 * (qx * qx + qy * qy)) * pz
        return t.x + rx, t.y + ry, t.z + rz

    def _berry_position_base(self, berry) -> Optional[Tuple[float, float, float]]:
        return self._berry_position_in_frame(berry, self._args.base_frame)

    @staticmethod
    def _rotate_vec3(q, vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        rx = (1 - 2 * (qy * qy + qz * qz)) * vx + 2 * (qx * qy - qw * qz) * vy + 2 * (qx * qz + qw * qy) * vz
        ry = 2 * (qx * qy + qw * qz) * vx + (1 - 2 * (qx * qx + qz * qz)) * vy + 2 * (qy * qz - qw * qx) * vz
        rz = 2 * (qx * qz - qw * qy) * vx + 2 * (qy * qz + qw * qx) * vy + (1 - 2 * (qx * qx + qy * qy)) * vz
        return rx, ry, rz

    def _point_link6_to_base(self, p_l6: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        flange = self._args.flange_frame
        try:
            tf = self._tf_buffer.lookup_transform(
                self._args.base_frame, flange, Time())
        except Exception as exc:
            self.get_logger().warn(f'TF {self._args.base_frame}->{flange}: {exc}')
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        rx, ry, rz = self._rotate_vec3(q, p_l6[0], p_l6[1], p_l6[2])
        return t.x + rx, t.y + ry, t.z + rz

    def _dir_link6_to_base(self, d_l6: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        flange = self._args.flange_frame
        try:
            tf = self._tf_buffer.lookup_transform(
                self._args.base_frame, flange, Time())
        except Exception as exc:
            self.get_logger().warn(f'TF dir {flange}->{self._args.base_frame}: {exc}')
            return None
        return self._rotate_vec3(tf.transform.rotation, d_l6[0], d_l6[1], d_l6[2])

    def _tcp_position_link6(self) -> Tuple[float, float, float]:
        flange = self._args.flange_frame
        try:
            tf = self._tf_buffer.lookup_transform(flange, self._args.ee_link, Time())
            t = tf.transform.translation
            return float(t.x), float(t.y), float(t.z)
        except Exception:
            return 0.0, 0.0, float(self._args.tcp_mount_z)

    def _berry_position_link6_from_camera(
            self, berry_cam: Tuple[float, float, float],
            cam_frame: str) -> Optional[Tuple[float, float, float]]:
        """Camera optical coords -> link6 (CS7), using TF or mount fallback."""
        flange = self._args.flange_frame
        try:
            tf = self._tf_buffer.lookup_transform(flange, cam_frame, Time())
            o = tf.transform.translation
            rx, ry, rz = self._rotate_vec3(
                tf.transform.rotation, berry_cam[0], berry_cam[1], berry_cam[2])
            return o.x + rx, o.y + ry, o.z + rz
        except Exception as exc:
            self.get_logger().warn(
                f'TF {flange}->{cam_frame} ({exc}) — using mount fallback '
                f'y={self._args.camera_mount_y:.3f}m')
        # link6 -> optical: same axes (Z=光轴=CS7+Z), camera origin on link6 -Y
        mount_y = float(self._args.camera_mount_y)
        cx, cy, cz = berry_cam[0], berry_cam[1], berry_cam[2]
        return cx, cy + mount_y, cz

    def _publish_locked_berry(self, berry_p: Tuple[float, float, float]) -> None:
        pt = PointStamped()
        pt.header.frame_id = self._args.base_frame
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x, pt.point.y, pt.point.z = berry_p
        self._locked_berry_pub.publish(pt)

    def _refine_suction_plan_toward_berry(self, plan, berry) -> bool:
        """Plan tcp poses in base_link: locked berry + current tcp, keep observation orientation."""
        berry_base = self._berry_position_base(berry)
        if berry_base is None:
            return self._refine_suction_plan_base_fallback(plan, berry)

        ee = self._current_ee_pose()
        bx, by, bz = berry_base
        if ee is not None:
            ex = ee.pose.position.x
            ey = ee.pose.position.y
            ez = ee.pose.position.z
            ax, ay, az = bx - ex, by - ey, bz - ez
            orient = ee.pose.orientation
        else:
            ax, ay, az = bx, by, bz
            orient = None

        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-6:
            return self._refine_suction_plan_base_fallback(plan, berry)
        ax, ay, az = ax / n, ay / n, az / n

        if orient is None:
            qx, qy, qz, qw = _quat_cup_toward_berry(ax, ay, az)
            orient_x, orient_y, orient_z, orient_w = qx, qy, qz, qw
        else:
            orient_x = orient.x
            orient_y = orient.y
            orient_z = orient.z
            orient_w = orient.w

        cup = self._args.cup_contact_offset
        pre_d = self._args.pre_grasp_offset
        post_d = self._args.post_grasp_offset
        stamp = self.get_clock().now().to_msg()

        for label, extra in (('pre_grasp', pre_d), ('grasp', 0.0), ('post_grasp', post_d)):
            reach = cup + extra
            ps = getattr(plan, label)
            ps.header.frame_id = self._args.base_frame
            ps.header.stamp = stamp
            ps.pose.position.x = bx - ax * reach
            ps.pose.position.y = by - ay * reach
            ps.pose.position.z = bz - az * reach
            ps.pose.orientation.x = orient_x
            ps.pose.orientation.y = orient_y
            ps.pose.orientation.z = orient_z
            ps.pose.orientation.w = orient_w

        self._publish_grasp_viz(plan, berry_base)
        self._publish_locked_berry(berry_base)

        berry_cam = self._berry_position_in_frame(berry, self._resolved_camera_frame or self._args.camera_frame)
        berry_l6 = self._berry_position_link6_from_camera(berry_cam, self._resolved_camera_frame or self._args.camera_frame) if berry_cam else None
        depth_m = math.sqrt(sum(c * c for c in berry_cam)) if berry_cam else 0.0
        if berry_l6 is not None:
            bl6 = f'berry_l6=({berry_l6[0]:.3f},{berry_l6[1]:.3f},{berry_l6[2]:.3f})'
        else:
            bl6 = 'berry_l6=(n/a)'
        ex_s = f'({ee.pose.position.x:.3f},{ee.pose.position.y:.3f},{ee.pose.position.z:.3f})' if ee else '(n/a)'
        print(f'[pick] eye-in-hand plan: berry_base=({bx:.3f},{by:.3f},{bz:.3f}) '
              f'tcp_now={ex_s} approach=({ax:.2f},{ay:.2f},{az:.2f}) '
              f'cam_depth={depth_m:.3f}m {bl6} ee={self._args.ee_link}')
        return True

    def _refine_suction_plan_base_fallback(self, plan, berry) -> bool:
        """Legacy: base_link EE -> berry vector."""
        berry_p = self._berry_position_base(berry)
        if berry_p is None:
            return False
        ee = self._current_ee_pose()
        bx, by, bz = berry_p
        if ee is None:
            ax, ay, az = bx, by, bz
        else:
            ex = ee.pose.position.x
            ey = ee.pose.position.y
            ez = ee.pose.position.z
            ax, ay, az = bx - ex, by - ey, bz - ez
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-6:
            ax, ay, az = bx, by, bz
            n = math.sqrt(ax * ax + ay * ay + az * az)
            if n < 1e-6:
                ax, ay, az = 0.0, 0.0, 1.0
                n = 1.0
        ax, ay, az = ax / n, ay / n, az / n
        cup = self._args.cup_contact_offset
        pre_d = self._args.pre_grasp_offset
        post_d = self._args.post_grasp_offset
        qx, qy, qz, qw = _quat_cup_toward_berry(ax, ay, az)
        stamp = self.get_clock().now().to_msg()
        for label, extra in (('pre_grasp', pre_d), ('grasp', 0.0), ('post_grasp', post_d)):
            ps = getattr(plan, label)
            ps.header.frame_id = self._args.base_frame
            ps.header.stamp = stamp
            reach = cup + extra
            ps.pose.position.x = bx - ax * reach
            ps.pose.position.y = by - ay * reach
            ps.pose.position.z = bz - az * reach
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
        self._publish_grasp_viz(plan, berry_p)
        print('[pick] WARN: using base_link fallback plan (check link6/camera TF)')
        return True

    def _compute_approach_geometry(self, berry) -> Optional[ApproachGeometry]:
        cam_frame = self._resolved_camera_frame or self._args.camera_frame
        berry_cam = self._berry_position_in_frame(berry, cam_frame)
        if berry_cam is None:
            berry_cam = self._berry_position_in_frame(
                berry, 'camera_wrist_color_optical_frame')
        if berry_cam is None:
            return None

        cam_depth = math.sqrt(
            berry_cam[0] ** 2 + berry_cam[1] ** 2 + berry_cam[2] ** 2)
        berry_l6 = self._berry_position_link6_from_camera(berry_cam, cam_frame)
        if berry_l6 is None:
            return None

        tcp_l6 = self._tcp_position_link6()
        bx, by, bz = berry_l6
        tx, ty, tz = tcp_l6
        dx, dy, dz = bx - tx, by - ty, bz - tz
        dist_l6 = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist_l6 > 1e-6:
            approach_l6 = (dx / dist_l6, dy / dist_l6, dz / dist_l6)
        else:
            approach_l6 = _normalize((bx, by, bz))

        berry_base = self._point_link6_to_base(berry_l6)
        ee = self._current_ee_pose()
        tcp_base: Optional[Tuple[float, float, float]] = None
        dist_base: Optional[float] = None
        if ee is not None:
            tcp_base = (
                float(ee.pose.position.x),
                float(ee.pose.position.y),
                float(ee.pose.position.z),
            )
        if berry_base is not None and tcp_base is not None:
            dist_base = math.sqrt(
                (berry_base[0] - tcp_base[0]) ** 2
                + (berry_base[1] - tcp_base[1]) ** 2
                + (berry_base[2] - tcp_base[2]) ** 2)

        return ApproachGeometry(
            berry_cam=berry_cam,
            cam_depth_m=cam_depth,
            berry_l6=berry_l6,
            tcp_l6=tcp_l6,
            berry_base=berry_base,
            tcp_base=tcp_base,
            dist_tcp_berry_l6_m=dist_l6,
            dist_tcp_berry_base_m=dist_base,
            approach_l6=approach_l6,
            approach_base=self._dir_link6_to_base(approach_l6),
            cup_offset_m=self._args.cup_contact_offset,
            pre_offset_m=self._args.pre_grasp_offset,
            post_offset_m=self._args.post_grasp_offset,
        )

    def _print_probe_report(self, plan, geom: ApproachGeometry) -> None:
        cx, cy, cz = geom.berry_cam
        bx, by, bz = geom.berry_l6
        tx, ty, tz = geom.tcp_l6
        ax, ay, az = geom.approach_l6
        pre_m = geom.cup_offset_m + geom.pre_offset_m
        grasp_back_m = geom.cup_offset_m
        post_m = geom.cup_offset_m + geom.post_offset_m

        print('')
        print('[probe] ========== 观测探针报告（未执行抓取）==========')
        print(f'[probe] 相机深度 cam_depth        = {geom.cam_depth_m:.3f} m')
        print(f'[probe] 果实@相机光学系 berry_cam  = ({cx:+.3f}, {cy:+.3f}, {cz:+.3f}) m')
        print(f'[probe] 果实@link6(坐标系7)        = ({bx:+.3f}, {by:+.3f}, {bz:+.3f}) m')
        print(f'[probe] 末端tcp@link6              = ({tx:+.3f}, {ty:+.3f}, {tz:+.3f}) m')
        print(f'[probe] 相机相对link6安装          = Y {self._args.camera_mount_y:+.3f} m')
        print(f'[probe] tcp相对link6安装           = Z {self._args.tcp_mount_z:+.3f} m')
        print(f'[probe] —— 我认为的距离 ——')
        print(f'[probe]   tcp → 果实 (link6系)     = {geom.dist_tcp_berry_l6_m:.3f} m')
        if geom.dist_tcp_berry_base_m is not None:
            print(f'[probe]   tcp → 果实 (base_link系) = {geom.dist_tcp_berry_base_m:.3f} m')
        else:
            print('[probe]   tcp → 果实 (base_link系) = (TF 不可用)')
        print(f'[probe]   相机 → 果实 (直线)       = {geom.cam_depth_m:.3f} m')
        print(f'[probe] 接近方向@link6             = ({ax:+.2f}, {ay:+.2f}, {az:+.2f})')
        if geom.approach_base is not None:
            abx, aby, abz = geom.approach_base
            print(f'[probe] 接近方向@base_link         = ({abx:+.2f}, {aby:+.2f}, {abz:+.2f})')
        print(f'[probe] —— 若继续执行，tcp 将这样走 ——')
        print(f'[probe]   1) pre_grasp : 沿接近方向后退 {pre_m:.2f}m '
              f'(杯口{geom.cup_offset_m:.2f}m + 预接近{geom.pre_offset_m:.2f}m)')
        _print_pose('pre_grasp', plan.pre_grasp)
        print(f'[probe]   2) grasp     : 再靠近 {pre_m - grasp_back_m:.2f}m，杯口对准果实')
        _print_pose('grasp', plan.grasp)
        print(f'[probe]   3) post_grasp: 吸住后沿接近方向后退 {post_m:.2f}m')
        _print_pose('post_grasp', plan.post_grasp)
        if geom.berry_base is not None:
            print(f'[probe] 果实中心@base_link           = '
                  f'({geom.berry_base[0]:+.3f}, {geom.berry_base[1]:+.3f}, '
                  f'{geom.berry_base[2]:+.3f}) m')
        if geom.tcp_base is not None:
            print(f'[probe] 当前tcp@base_link            = '
                  f'({geom.tcp_base[0]:+.3f}, {geom.tcp_base[1]:+.3f}, '
                  f'{geom.tcp_base[2]:+.3f}) m')
        print('[probe] ================================================')
        print('[probe] 本次停在观测位，未执行 pre_grasp/grasp/post_grasp')
        print('')

    def _publish_grasp_viz(self, plan, berry_p: Tuple[float, float, float]) -> None:
        bx, by, bz = berry_p
        stamp = plan.grasp.header.stamp
        cup_pt = PointStamped()
        cup_pt.header.frame_id = self._args.base_frame
        cup_pt.header.stamp = stamp
        cup_pt.point.x = bx
        cup_pt.point.y = by
        cup_pt.point.z = bz
        self._grasp_cup_pub.publish(cup_pt)
        self._grasp_eef_pub.publish(plan.grasp)

    def _detect_with_retries(self, pose_name: str, reset_lock: bool = False) -> Optional[object]:
        last = None
        for attempt in range(1, self._scan_detect_attempts + 1):
            if self._abort_check():
                return None
            print(f'[pick] detect at {pose_name} — attempt {attempt}/'
                  f'{self._scan_detect_attempts}')
            det = self._detect(reset_lock=reset_lock and attempt == 1)
            if det is None:
                return None
            last = det
            if det.success and det.detected_berries:
                print(f'[pick] {pose_name}: {det.message}')
                return det
            print(f'[pick] {pose_name} attempt {attempt}: {det.message}')
            if attempt < self._scan_detect_attempts:
                if not self._dwell(self._scan_detect_interval_s, 'wait before re-detect'):
                    return None
        return last

    def _current_link6_pose(self) -> Optional[PoseStamped]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._args.base_frame, self._args.flange_frame, Time())
        except Exception as exc:
            self.get_logger().warn(
                f'TF {self._args.base_frame}->{self._args.flange_frame}: {exc}')
            return None
        ps = PoseStamped()
        ps.header.frame_id = self._args.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = tf.transform.translation.x
        ps.pose.position.y = tf.transform.translation.y
        ps.pose.position.z = tf.transform.translation.z
        ps.pose.orientation = tf.transform.rotation
        return ps

    def _current_ee_pose(self) -> Optional[PoseStamped]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._args.base_frame, self._args.ee_link, Time())
        except Exception as exc:
            self.get_logger().warn(
                f'TF {self._args.base_frame}->{self._args.ee_link}: {exc}')
            return None
        ps = PoseStamped()
        ps.header.frame_id = self._args.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = tf.transform.translation.x
        ps.pose.position.y = tf.transform.translation.y
        ps.pose.position.z = tf.transform.translation.z
        ps.pose.orientation = tf.transform.rotation
        return ps

    def _camera_frame_candidates(self) -> List[str]:
        raw = [
            self._args.camera_frame,
            'camera_wrist_color_optical_frame',
            'camera_wrist_color_frame',
            'camera_wrist_link',
        ]
        out: List[str] = []
        for f in raw:
            if f and f not in out:
                out.append(f)
        return out

    def _resolve_camera_frame(self, timeout_sec: float) -> Optional[str]:
        if self._resolved_camera_frame:
            try:
                self._tf_buffer.lookup_transform(
                    self._args.base_frame, self._resolved_camera_frame, Time())
                return self._resolved_camera_frame
            except Exception:
                self._resolved_camera_frame = None

        deadline = self.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            for frame in self._camera_frame_candidates():
                try:
                    self._tf_buffer.lookup_transform(
                        self._args.base_frame, frame, Time())
                except Exception:
                    continue
                self._resolved_camera_frame = frame
                if frame != self._args.camera_frame:
                    print(f'[pick] camera TF ready: {self._args.base_frame} -> {frame}')
                return frame
            if self._abort_check():
                return None
            rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def _camera_optical_z_search_pose(self, step_m: float) -> Optional[PoseStamped]:
        """Move EE so wrist camera translates +step_m along optical +Z."""
        link6 = self._current_link6_pose()
        if link6 is None:
            return None
        cam_frame = self._resolved_camera_frame or self._args.camera_frame
        try:
            tf = self._tf_buffer.lookup_transform(
                self._args.base_frame, cam_frame, Time())
        except Exception as exc:
            self.get_logger().warn(f'TF camera frame ({cam_frame}): {exc}')
            return None
        q = tf.transform.rotation
        z_axis = _normalize(_quat_z_axis(q.x, q.y, q.z, q.w))
        out = PoseStamped()
        out.header = link6.header
        out.pose = link6.pose
        out.pose.position.x += z_axis[0] * step_m
        out.pose.position.y += z_axis[1] * step_m
        out.pose.position.z += z_axis[2] * step_m
        return out

    def _detect(self, reset_lock: bool = False):
        req = TriggerFineDetection.Request()
        req.reset_lock = reset_lock
        fut = self._fp.call_async(req)
        deadline = self.get_clock().now().nanoseconds + int(self._args.fp_timeout * 1e9)
        while rclpy.ok() and not fut.done():
            if self._abort_check():
                return None
            if self.get_clock().now().nanoseconds > deadline:
                return None
            rclpy.spin_once(self, timeout_sec=0.05)
        return fut.result()

    def _z_search(self) -> Tuple[Optional[object], bool]:
        """Returns (detection_or_none, aborted). Skipped TF -> (None, False)."""
        print('[pick] No berry — searching along camera +Z ...')
        for step in range(1, self._args.max_search_steps + 1):
            if self._abort_check():
                return None, True
            target = self._camera_optical_z_search_pose(self._args.search_step_m)
            if target is None:
                print('[pick] z-search: TF not ready — skip optical fine search')
                return None, False
            print(f'[pick] z-search step {step}/{self._args.max_search_steps} '
                  f'dz={self._args.search_step_m:.3f}m')
            if not self._send_pose(target, f'z_search_{step}'):
                if self._keys.consume_home() or self._keys.should_quit():
                    return None, True
                continue
            det = self._detect(reset_lock=False)
            if det is None:
                return None, True
            if det.success and det.detected_berries:
                print(f'[pick] z-search found berries at step {step}: {det.message}')
                return det, False
            print(f'[pick] z-search step {step}: still no berry ({det.message if det else "timeout"})')
        return None, False

    def _detect_with_z_search(self) -> Optional[object]:
        det = self._detect(reset_lock=False)
        if det is None or self._abort_check():
            return None
        if det.success and det.detected_berries:
            return det
        if not self._resolve_camera_frame(self._args.tf_wait_s):
            print('[pick] WARN: camera TF missing (need bringup camera + camera_tf). '
                  'Skip z-search; patrol may still find berries.')
            return det
        z_det, aborted = self._z_search()
        if aborted:
            return None
        if z_det is not None and z_det.success and z_det.detected_berries:
            return z_det
        return det

    def _begin_cycle_home(self) -> bool:
        """Move to joint zero at the start of each pick cycle."""
        if not self._want_scan_move():
            return True
        if not self._ensure_arm_ready():
            return False
        print('[pick] [HOME] Moving to zero pose (all joints 0) ...')
        if not self._go_home():
            return False
        if not self._dwell(self._scan_settle_s, 'settled at home'):
            return False
        return True

    def _find_berry_at_teleop_poses(self) -> Optional[object]:
        """Home -> teleop poses -> dwell -> detect (no z-search)."""
        if self._want_scan_move() and not self._args.skip_home:
            if not self._begin_cycle_home():
                return None

        if not self._scan_poses:
            print('[pick] ERROR: no teleop poses in config/pick_scan_recorded.txt')
            return None

        print(f'[pick] [1/4] TELEOP SCAN — {len(self._scan_poses)} poses '
              f'(settle={self._scan_settle_s:.1f}s, '
              f'{self._scan_detect_attempts}x detect, '
              f'interval={self._scan_detect_interval_s:.1f}s)')
        last_det = None
        for idx, (name, joints) in enumerate(self._scan_poses, start=1):
            if self._abort_check():
                return None
            print(f'[pick] teleop pose {idx}/{len(self._scan_poses)} -> {name}')
            if self._want_scan_move():
                if not self._send_joints(
                        joints, f'teleop_{name}',
                        velocity_scale=self._args.scan_velocity_scale):
                    print(f'[pick] skip {name} (move failed)')
                    continue
            if not self._settle_after_scan_move():
                return None
            det = self._detect_with_retries(name, reset_lock=True)
            if det is None:
                return None
            last_det = det
            if det.success and det.detected_berries:
                print(f'[pick] berry found at teleop pose {name}')
                return det
            print(f'[pick] no berry at {name} — next pose ...')
        print('[pick] all teleop poses tried — no blueberry')
        return last_det

    def _find_berry(self) -> Optional[object]:
        """Find berry: direct teleop poses OR home -> detect -> patrol."""
        if self._direct_teleop and self._scan_poses:
            return self._find_berry_at_teleop_poses()

        if self._want_scan_move() and not self._args.skip_home:
            if not self._begin_cycle_home():
                return None

        if self._want_scan_move() and self._args.skip_home:
            if not self._ensure_arm_ready():
                return None

        print('[pick] [2/6] DETECT — YOLO + FoundationPose at initial pose ...')
        det = self._detect_with_z_search()
        if det is None or self._abort_check():
            return None
        if det.success and det.detected_berries:
            print(f'[pick] berry found at home: {det.message}')
            return det

        if self._args.no_scan or not self._want_scan_move():
            print('[pick] no berry at initial pose (patrol disabled)')
            return det

        if not self._scan_poses:
            print('[pick] no berry at initial pose (no patrol poses configured)')
            return det

        print(f'[pick] [3/6] PATROL — no berry at home, {self._scan_mode} '
              f'({len(self._scan_poses)} poses) ...')
        self._resolve_camera_frame(self._args.tf_wait_s)
        for idx, (name, joints) in enumerate(self._scan_poses, start=1):
            if self._abort_check():
                return None
            print(f'[pick] patrol {idx}/{len(self._scan_poses)} -> {name}')
            if not self._send_joints(joints, f'scan_{name}'):
                print(f'[pick] patrol: skip {name} (move failed)')
                continue
            if not self._settle_after_scan_move():
                return None
            det = self._detect(reset_lock=False)
            if det is None:
                return None
            if det.success and det.detected_berries:
                print(f'[pick] patrol found berries at {name}: {det.message}')
                return det
            print(f'[pick] patrol {name}: no berry ({det.message}) — z-search ...')
            z_det, aborted = self._z_search()
            if aborted:
                return None
            if z_det is not None and z_det.success and z_det.detected_berries:
                print(f'[pick] patrol + z-search found berries at {name}')
                return z_det
        print('[pick] patrol exhausted — no blueberry in view')
        return None

    def _wait_continue_grasp(self) -> bool:
        if self._args.auto_retreat:
            pause = self._args.grasp_pause_s
            print(f'[pick] At grasp — turn ON suction; auto post_grasp in {pause:.0f}s '
                  f'(--auto-retreat)')
            deadline = self.get_clock().now().nanoseconds + int(pause * 1e9)
            while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
                if self._keys.should_quit():
                    return False
                if self._keys.consume_home():
                    print('[pick] WARN: R/H during grasp wait — still continuing auto-retreat '
                          '(use Q to abort)')
                rclpy.spin_once(self, timeout_sec=0.1)
            return True

        print('[pick] At grasp — turn ON suction, then press G to retreat (Q=quit)')
        print('[pick]   NOTE: R/H here will abort retreat and go home')
        while rclpy.ok() and not self._keys.should_quit():
            if self._keys.consume_continue_grasp():
                return True
            if self._keys.consume_home():
                print('[pick] grasp aborted by R/H — skipping post_grasp')
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _run_once(self) -> int:
        if self._abort_check():
            return -1

        det = self._find_berry()
        if det is None:
            return -1 if self._keys.should_quit() else 1
        if not det.success or not det.detected_berries:
            print('[pick] cycle aborted — no blueberry found')
            return 1

        lock_i = int(getattr(det, 'locked_berry_index', -1))
        step_lock = '2/4' if self._direct_teleop else '4/6'
        print(f'[pick] [{step_lock}] LOCK — {len(det.detected_berries)} berries, {det.message}')
        if lock_i < 0 or lock_i >= len(det.detected_berries):
            print('[pick] ERROR: no locked berry index — refusing to plan (stale TRACK?). '
                  'Retry detect or restart perception node.')
            return 1
        berries = [det.detected_berries[lock_i]]
        print(f'[pick] locked berry index={lock_i} (planning 1 of {len(det.detected_berries)})')

        print(f'[pick] [{"3/4" if self._direct_teleop else "5/6"}] PLAN — approach pre_grasp / grasp / post_grasp ...')
        pfut = self._plan.call_async(PlanSuction.Request(berries=berries))
        while rclpy.ok() and not pfut.done():
            if self._abort_check():
                return -1
            rclpy.spin_once(self, timeout_sec=0.05)
        plan_res = pfut.result()
        if plan_res is None or not plan_res.success:
            print(f'[pick] plan failed: {plan_res.message if plan_res else "timeout"}')
            return 1

        plan = plan_res.plan
        if not self._refine_suction_plan_toward_berry(plan, berries[0]):
            print('[pick] WARN: could not refine plan toward berry (TF?) — using planner output')
            berry_p = self._berry_position_base(berries[0])
            if berry_p is not None:
                self._publish_grasp_viz(plan, berry_p)
        if not self._args.probe_only:
            print('[pick] plan (tcp +Z / cup forward toward berry, mouth on berry center):')
            _print_pose('pre_grasp', plan.pre_grasp)
            _print_pose('grasp', plan.grasp)
            _print_pose('post_grasp', plan.post_grasp)

        geom = self._compute_approach_geometry(berries[0])
        if self._args.probe_only:
            if geom is not None:
                self._print_probe_report(plan, geom)
            else:
                print('[pick] WARN: could not compute approach geometry (camera/link6 TF?)')
                _print_pose('pre_grasp', plan.pre_grasp)
                _print_pose('grasp', plan.grasp)
                _print_pose('post_grasp', plan.post_grasp)

        if self._args.probe_only:
            return 0

        if not self._want_grasp_move():
            print('[pick] dry-run (add --move to execute grasp motion)')
            return 0

        if self._current_ee_pose() is None:
            print(f'[pick] ERROR: TF {self._args.base_frame}->{self._args.ee_link} missing — '
                  f'MoveIt cannot plan to this link.', file=sys.stderr)
            print('[pick] Real agx_arm uses tcp_link (not simulation eef_link). Restart with:',
                  file=sys.stderr)
            print('  bash scripts/run_real_suction_pick.sh   # auto --ee-link tcp_link',
                  file=sys.stderr)
            return 1

        print(f'[pick] [{"4/4" if self._direct_teleop else "6/6"}] MOVE — pre_grasp -> grasp (suction) -> post_grasp ...')
        if not self._send_pose_with_fallback(plan.pre_grasp, 'pre_grasp', position_only=False):
            return -1 if self._abort_check() else 1
        if not self._send_pose_with_fallback(plan.grasp, 'grasp', position_only=False):
            return -1 if self._abort_check() else 1

        if not self._wait_continue_grasp():
            return -1

        if not self._send_pose_with_fallback(plan.post_grasp, 'post_grasp', position_only=False):
            return -1 if self._abort_check() else 1

        print('[pick] cycle complete — turn OFF suction if needed')
        reset_req = TriggerFineDetection.Request()
        reset_req.reset_lock = True
        self._fp.call_async(reset_req)
        return 0

    def run(self) -> int:
        if not self._wait_service(self._fp, 'trigger_fine_detection'):
            return 1
        if not self._wait_service(self._plan, 'plan_suction'):
            print('Start: bash scripts/run_grasp_planner.sh', file=sys.stderr)
            return 1
        if self._want_scan_move() and not self._move.wait_for_server(timeout_sec=10.0):
            print('Start arm bringup for /move_action', file=sys.stderr)
            return 1
        if self._want_scan_move() and not self._ensure_arm_ready():
            return 1

        self._keys.start()
        print(KEY_HELP.strip())
        print('')
        if self._args.probe_only:
            print('[pick] PROBE: HOME -> teleop观测位 -> LOCK -> PLAN -> 打印距离/路径 (不抓取)')
        else:
            print('[pick] Full cycle: HOME -> DETECT -> (PATROL if needed) -> LOCK -> PLAN -> MOVE')
        if self._want_scan_move() and not self._args.no_scan:
            if self._direct_teleop:
                print(f'[pick] Mode: HOME -> teleop poses ({len(self._scan_poses)}), '
                      f'dwell {self._scan_settle_s:.1f}s')
            else:
                print(f'[pick] Patrol: {self._scan_mode}, {len(self._scan_poses)} poses')
            print(f'[pick]   config: {self._args.scan_config}')
            print('[pick]   record poses: bash scripts/record_scan_pose.sh <name>')
        print('[pick] Viz overlay: /camera_wrist/color/detection_viz')
        print('')

        try:
            while rclpy.ok() and not self._keys.should_quit():
                rc = self._run_once()
                if rc < 0:
                    self._go_home()
                    if self._keys.should_quit():
                        print('[pick] quit.')
                        break
                    if self._args.once:
                        print('[pick] cycle interrupted — exiting (--once).')
                        break
                    print('[pick] cycle interrupted — home done; ready for next cycle')
                    continue
                if self._keys.consume_home():
                    self._go_home()
                    if self._args.once:
                        print('[pick] home done — exiting (--once).')
                        break
                    print('[pick] home done — ready for next cycle')
                if self._args.once:
                    if rc == 0:
                        print('[pick] cycle complete — exiting (--once).')
                    else:
                        print(f'[pick] cycle finished with code={rc} — exiting (--once).')
                    break
                if rc == 0:
                    print(f'[pick] waiting {self._args.cycle_pause_s:.0f}s before next cycle ...')
                    deadline = self.get_clock().now().nanoseconds + int(
                        self._args.cycle_pause_s * 1e9)
                    while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
                        if self._keys.should_quit() or self._keys.consume_home():
                            self._go_home()
                            break
                        rclpy.spin_once(self, timeout_sec=0.1)
        finally:
            self._keys.stop()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--move', action='store_true',
                        help='Execute scan + grasp motions (use without --probe-only for full pick)')
    parser.add_argument('--probe-only', action='store_true',
                        help='Move to teleop观测位, lock, print distance/plan — no grasp')
    parser.add_argument('--once', action='store_true', help='Single pick cycle then exit')
    parser.add_argument('--auto-retreat', action='store_true',
                        help='At grasp: wait grasp-pause-s then post_grasp (no G key)')
    parser.add_argument('--grasp-pause-s', type=float, default=4.0,
                        help='Seconds at grasp before auto post_grasp (--auto-retreat)')
    parser.add_argument('--tf-wait-s', type=float, default=20.0,
                        help='Wait for camera TF before z-search (seconds)')
    parser.add_argument('--fp-timeout', type=float, default=120.0)
    parser.add_argument('--move-action', default='/move_action')
    parser.add_argument('--move-group', default='arm')
    parser.add_argument('--ee-link', default='eef_link',
                        help='MoveIt pose target link (SRDF tip: eef_link = cup mount)')
    parser.add_argument('--cup-contact-offset', type=float, default=DEFAULT_CUP_CONTACT_OFFSET_M,
                        help='tcp +Z distance to cup mouth (m); real mount: cup opens along +Z')
    parser.add_argument('--base-frame', default='base_link')
    parser.add_argument('--flange-frame', default='link6',
                        help='Piper X 坐标系7 / flange (camera mount parent)')
    parser.add_argument('--camera-mount-y', type=float, default=-0.08,
                        help='Camera on link6 -Y offset (m); matches CAMERA_MOUNT_TY')
    parser.add_argument('--tcp-mount-z', type=float, default=0.05,
                        help='tcp_link on link6 +Z (m); matches ARM_TCP_OFFSET')
    parser.add_argument('--camera-frame', default='camera_wrist_color_optical_frame')
    parser.add_argument('--velocity-scale', type=float, default=0.12)
    parser.add_argument('--position-tolerance', type=float, default=0.025,
                        help='Position goal sphere radius (m) for pre_grasp/post_grasp')
    parser.add_argument('--search-step-m', type=float, default=0.03,
                        help='Camera +Z step per search attempt (m)')
    parser.add_argument('--max-search-steps', type=int, default=6)
    parser.add_argument('--cycle-pause-s', type=float, default=3.0)
    parser.add_argument('--home-joints', type=float, nargs=6,
                        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        help='Home joint targets (same as teleop R)')
    parser.add_argument('--home-tolerance', type=float, default=0.03,
                        help='Skip home move if all joints within this (rad)')
    default_scan_cfg = os.path.join(_WS, 'config', 'pick_scan_poses.env')
    parser.add_argument('--scan-config', default=default_scan_cfg,
                        help='Patrol grid config (PICK_SCAN_ANCHOR + J1/J2 deltas)')
    parser.add_argument('--scan-pose', action='append', metavar='J1,J2,J3,J4,J5,J6',
                        help='Explicit patrol pose (rad); overrides grid from --scan-config')
    parser.add_argument('--no-scan', action='store_true',
                        help='Skip patrol; detect + camera Z search at current pose only')
    parser.add_argument('--direct-scan', action='store_true',
                        help='Go straight to teleop poses (default when recorded file has poses)')
    parser.add_argument('--search-at-home', action='store_true',
                        help='Detect at home + z-search before teleop poses (old behavior)')
    parser.add_argument('--scan-settle-s', type=float, default=None,
                        help='Seconds to wait at each teleop pose before detect (default 3)')
    parser.add_argument('--scan-detect-attempts', type=int, default=None,
                        help='Detection attempts per teleop pose (default 3)')
    parser.add_argument('--scan-detect-interval-s', type=float, default=None,
                        help='Pause between detect attempts at same pose (default 1.5s)')
    parser.add_argument('--scan-velocity-scale', type=float, default=0.08,
                        help='MoveIt velocity for moves between teleop scan poses')
    parser.add_argument('--pre-grasp-offset', type=float, default=0.15)
    parser.add_argument('--post-grasp-offset', type=float, default=0.12)
    parser.add_argument('--skip-home', action='store_true',
                        help='Do not move to zero at cycle start (debug only)')
    args = parser.parse_args()

    rclpy.init()
    node = PickLoopNode(args)
    try:
        return max(node.run(), 0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
