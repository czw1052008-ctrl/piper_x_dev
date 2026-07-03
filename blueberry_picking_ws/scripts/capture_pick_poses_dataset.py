#!/usr/bin/env python3
"""Capture wrist-camera frames at teleop scan poses (same poses as pick patrol).

Moves the arm through deduplicated poses from pick_scan_recorded.txt, dwells at
each pose, then saves RGB frames for YOLO annotation.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import List, Optional, Tuple

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    PlanningOptions,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    cv2 = None

_WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _WS not in sys.path:
    sys.path.insert(0, os.path.join(_WS, 'scripts'))

from capture_annotation_batch import _image_to_rgb, _resolve_start_index  # noqa: E402
from real_suction_pick_loop import (  # noqa: E402
    ARM_JOINTS,
    _load_scan_env,
    _moveit_error_hint,
    _resolve_scan_poses,
)


def _save_png_rgb(path: str, rgb) -> None:
    if cv2 is None:
        raise RuntimeError('opencv-python (cv2) required to save PNG')
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _split_capture_counts(total: int, n_poses: int) -> List[int]:
    base = total // n_poses
    rem = total % n_poses
    return [base + (1 if i < rem else 0) for i in range(n_poses)]


class PoseCaptureNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('capture_pick_poses_dataset')
        self._args = args
        self._move = ActionClient(self, MoveGroup, args.move_action)
        self._rgb_msg: Image | None = None
        self.create_subscription(Image, args.color_topic, self._on_rgb, 1)
        self._active_goal = None

        cfg = _load_scan_env(args.scan_config)
        self._settle_s = args.settle_s if args.settle_s is not None else cfg['settle_s']
        self._poses = _resolve_scan_poses(cfg, args.scan_pose or None)
        if not self._poses:
            raise RuntimeError(
                f'no scan poses — record some: bash scripts/record_scan_pose.sh <name>')

    def _on_rgb(self, msg: Image) -> None:
        self._rgb_msg = msg

    @staticmethod
    def _stamp_key(msg: Image) -> tuple[int, int]:
        return msg.header.stamp.sec, msg.header.stamp.nanosec

    def wait_rgb(self, timeout_sec: float) -> bool:
        t0 = time.time()
        while rclpy.ok() and self._rgb_msg is None and (time.time() - t0) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.2)
        return self._rgb_msg is not None

    def wait_new_rgb(self, after_stamp: tuple[int, int] | None, timeout_sec: float) -> bool:
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._rgb_msg is None:
                continue
            if after_stamp is None or self._stamp_key(self._rgb_msg) > after_stamp:
                return True
        return False

    def spin_for(self, duration_sec: float) -> None:
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < duration_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

    def grab_rgb(self):
        if self._rgb_msg is None:
            raise RuntimeError('no RGB frame')
        return _image_to_rgb(self._rgb_msg)

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

    def _build_home_goal(self) -> MoveGroup.Goal:
        return self._build_joint_goal(list(self._args.home_joints))

    def _execute_move_goal(self, goal: MoveGroup.Goal, label: str = 'move') -> bool:
        if not self._move.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'{label}: move_action unavailable')
            return False
        send = self._move.send_goal_async(goal)
        deadline = self.get_clock().now().nanoseconds + int(30e9)
        while rclpy.ok() and not send.done():
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
            if self.get_clock().now().nanoseconds > deadline:
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
        print(f'[capture] move -> {label} joints={[round(v, 3) for v in joints]}')
        return self._execute_move_goal(
            self._build_joint_goal(joints, velocity_scale=velocity_scale), label=label)

    def run_capture(self) -> int:
        out_dir = os.path.abspath(self._args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        start_idx = _resolve_start_index(out_dir, self._args.prefix, self._args.start_index)
        per_pose = _split_capture_counts(self._args.count, len(self._poses))

        print(f'[capture] Waiting for {self._args.color_topic} ...')
        if not self.wait_rgb(self._args.startup_timeout):
            print('ERROR: no RGB frames — is the camera running?', file=sys.stderr)
            return 1

        if not self._move.wait_for_server(timeout_sec=self._args.move_timeout):
            print(f'ERROR: {self._args.move_action} unavailable — is arm bringup running?',
                  file=sys.stderr)
            return 1

        if self._args.home_first:
            print(f'[capture] Homing joints={list(self._args.home_joints)}')
            if not self._execute_move_goal(self._build_home_goal(), label='home'):
                return 1
            self.spin_for(1.0)

        print(f'[capture] {len(self._poses)} poses, {self._args.count} frames total '
              f'-> {out_dir}')
        for i, ((name, joints), n_cap) in enumerate(zip(self._poses, per_pose), start=1):
            print(f'[capture] pose {i}/{len(self._poses)}: {name} ({n_cap} frames)')
            if not self._send_joints(
                    joints, label=name, velocity_scale=self._args.scan_velocity_scale):
                return 1
            print(f'[capture] settle {self._settle_s:.1f}s at {name}')
            self.spin_for(self._settle_s)

            last_stamp: tuple[int, int] | None = None
            for j in range(n_cap):
                if not self.wait_new_rgb(last_stamp, 5.0):
                    print(f'[capture] WARN: no new frame at {name} ({j + 1}/{n_cap})',
                          file=sys.stderr)
                    self.spin_for(self._args.interval)
                    continue
                rgb = self.grab_rgb()
                last_stamp = self._stamp_key(self._rgb_msg)
                frame_idx = start_idx
                start_idx += 1
                path = os.path.join(
                    out_dir, f'{self._args.prefix}frame_{frame_idx:04d}.png')
                _save_png_rgb(path, rgb)
                print(f'[capture] saved {path}')
                if j + 1 < n_cap:
                    self.spin_for(self._args.interval)

        print(f'[capture] Done: saved frames to {out_dir}')
        return 0


def _preflight(color_topic: str) -> bool:
    import subprocess
    try:
        out = subprocess.run(
            ['ros2', 'topic', 'list'],
            capture_output=True, text=True, timeout=10, check=False)
        if color_topic not in out.stdout:
            print(f'ERROR: topic {color_topic} not found.', file=sys.stderr)
            return False
    except Exception as exc:
        print(f'WARN: ros2 topic list failed: {exc}', file=sys.stderr)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_scan_cfg = os.path.join(_WS, 'config', 'pick_scan_poses.env')
    parser.add_argument('--color-topic', default='/camera_wrist/color/image_raw')
    parser.add_argument('--out-dir', default='datasets/blueberry/images_batch3')
    parser.add_argument('--prefix', default='b3_')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Seconds between frames at the same pose')
    parser.add_argument('--startup-timeout', type=float, default=60.0)
    parser.add_argument('--move-timeout', type=float, default=30.0)
    parser.add_argument('--move-action', default='/move_action')
    parser.add_argument('--move-group', default='arm')
    parser.add_argument('--velocity-scale', type=float, default=0.12)
    parser.add_argument('--scan-velocity-scale', type=float, default=0.08)
    parser.add_argument('--scan-config', default=default_scan_cfg)
    parser.add_argument('--scan-pose', action='append', metavar='J1,J2,J3,J4,J5,J6')
    parser.add_argument('--settle-s', type=float, default=None)
    parser.add_argument('--home-first', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--home-joints', type=float, nargs=6,
                        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    args = parser.parse_args()

    if not _preflight(args.color_topic):
        return 1

    rclpy.init()
    node = PoseCaptureNode(args)
    try:
        return node.run_capture()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
