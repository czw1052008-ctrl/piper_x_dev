#!/usr/bin/env python3
"""Live FK arm skeleton overlay on fixed camera RGB.

Move the physical DaBai until the green skeleton hugs the real arm
(assumes current FIXED_CAM_* TF stays unchanged).

  ros2 run / python3 scripts/fixed_fk_skeleton_live.py
  rqt_image_view /planning/viz/fixed_fk_skeleton
"""
from __future__ import annotations

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener

from end_effector_profile import load_profile
from obstacle_extractor import project_base_to_uv
from piper_position_ik import fk_link6, fk_link_origins, tip_xyz

ARM = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def _bgr_of(msg: Image) -> np.ndarray:
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == 'rgb8':
        return cv2.cvtColor(raw.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
    if msg.encoding == 'bgr8':
        return raw.reshape(msg.height, msg.width, 3).copy()
    raise RuntimeError(f'unsupported encoding {msg.encoding}')


def _T_from_tf(tf) -> np.ndarray:
    t = tf.transform.translation
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    # Hamilton xyzw (same as scipy / tf2); R[1,2] = 2(yz - xw)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class FixedFkSkeletonLive(Node):
    def __init__(self) -> None:
        super().__init__('fixed_fk_skeleton_live')
        self.declare_parameter('end_effector', 'agx_gripper_v1')
        self.declare_parameter('hz', 15.0)
        profile = load_profile(str(self.get_parameter('end_effector').value))
        self._tip_off = profile.tip_offset_link6
        self._buf = Buffer()
        TransformListener(self._buf, self)
        self._bgr = None
        self._K = None
        self._q = None
        self._pub = self.create_publisher(Image, '/planning/viz/fixed_fk_skeleton', 10)
        self.create_subscription(
            Image, '/camera_fixed/color/image_raw', self._on_color, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, '/camera_fixed/color/camera_info', self._on_info, qos_profile_sensor_data)
        self.create_subscription(JointState, '/feedback/joint_states', self._on_joints, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        hz = max(float(self.get_parameter('hz').value), 1.0)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(
            'LIVE FK skeleton → /planning/viz/fixed_fk_skeleton — '
            'move fixed cam until green matches real arm')

    def _on_color(self, msg: Image) -> None:
        try:
            self._bgr = _bgr_of(msg)
        except Exception as exc:
            self.get_logger().warn(f'color: {exc}')

    def _on_info(self, msg: CameraInfo) -> None:
        self._K = np.array(msg.k, dtype=float).reshape(3, 3)

    def _on_joints(self, msg: JointState) -> None:
        m = dict(zip(msg.name, msg.position))
        if all(j in m for j in ARM):
            self._q = [float(m[j]) for j in ARM]

    def _tick(self) -> None:
        if self._bgr is None or self._K is None or self._q is None:
            return
        try:
            tf = self._buf.lookup_transform(
                'base_link', 'camera_fixed_color_optical_frame', rclpy.time.Time())
        except Exception:
            return
        T = _T_from_tf(tf)
        out = self._bgr.copy()
        h, w = out.shape[:2]
        origins = fk_link_origins(self._q)
        tip = tip_xyz(self._q, tip_offset_link6=self._tip_off)
        _, p6 = fk_link6(self._q)
        chain = [np.zeros(3)] + [origins[i] for i in range(6)] + [p6, tip]
        names = ['base', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'link6', 'tip']
        uvs = []
        for p in chain:
            uv = project_base_to_uv(p, self._K, T)
            if uv is None:
                uvs.append(None)
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))
            uvs.append((u, v) if 0 <= u < w and 0 <= v < h else None)
        for a, b in zip(uvs, uvs[1:]):
            if a and b:
                cv2.line(out, a, b, (0, 255, 0), 3)
        for name, uv in zip(names, uvs):
            if not uv:
                continue
            color = (
                (0, 255, 255) if name == 'tip'
                else ((255, 255, 0) if name == 'base' else (0, 165, 255))
            )
            cv2.circle(out, uv, 8 if name == 'tip' else 6, color, -1)
            cv2.putText(
                out, name, (uv[0] + 6, uv[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.rectangle(out, (0, 0), (w, 52), (20, 20, 20), -1)
        cv2.putText(
            out,
            'MOVE fixed cam until GREEN skeleton hugs REAL arm  (TF unchanged)',
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            out, 'yellow=base  orange=links  cyan=tip',
            (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_fixed'
        msg.height = h
        msg.width = w
        msg.encoding = 'bgr8'
        msg.step = w * 3
        msg.data = out.tobytes()
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = FixedFkSkeletonLive()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
