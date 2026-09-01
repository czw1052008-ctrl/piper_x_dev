#!/usr/bin/env python3
"""DEPRECATED: top-K obstacle spheres. Use occupancy_map_node (Occupancy+ESDF).

Kept for bag compatibility only. Prefer:
  python3 scripts/occupancy_map_node.py
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from picking_msgs.msg import (
    PerceptionScene,
    SceneBerry,
    SceneObstacle,
    SceneObstacleArray,
)
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener

from end_effector_profile import load_profile
from obstacle_extractor import (
    BBox2D,
    DepthFrameSpec,
    ObstacleExtractConfig,
    decode_depth_m,
    extract_obstacle_spheres,
    project_base_to_uv,
)
from piper_position_ik import fk_link_origins, tip_xyz

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def _image_to_depth(msg: Image) -> np.ndarray:
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ('32FC1', '32FC'):
        raw = np.frombuffer(msg.data, dtype=np.float32)
    elif msg.encoding in ('16UC1', 'mono16'):
        raw = np.frombuffer(msg.data, dtype=np.uint16)
    return decode_depth_m(msg.encoding, raw, msg.height, msg.width)


def _K_from_info(msg: CameraInfo) -> np.ndarray:
    return np.array(msg.k, dtype=np.float64).reshape(3, 3)


def _T_from_tf(tf: TransformStamped) -> np.ndarray:
    t = tf.transform.translation
    q = tf.transform.rotation
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


class ObstacleExtractorNode(Node):
    def __init__(self) -> None:
        super().__init__('obstacle_extractor_node')
        self.declare_parameter('publish_hz', 5.0)
        self.declare_parameter('max_obstacles', 12)
        self.declare_parameter('end_effector', 'agx_gripper_v1')
        self.declare_parameter('fixed_depth_topic', '/camera_fixed/depth/image_raw')
        self.declare_parameter('fixed_info_topic', '/camera_fixed/color/camera_info')
        self.declare_parameter('fixed_frame', 'camera_fixed_color_optical_frame')
        self.declare_parameter('wrist_depth_topic', '/camera_wrist/depth/image_raw')
        self.declare_parameter('wrist_info_topic', '/camera_wrist/color/camera_info')
        self.declare_parameter('wrist_frame', 'camera_wrist_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')

        profile = load_profile(str(self.get_parameter('end_effector').value))
        self._tip_off = profile.tip_offset_link6
        self._cfg = ObstacleExtractConfig(
            max_obstacles=int(self.get_parameter('max_obstacles').value),
        )

        self._fixed_depth = None
        self._fixed_K = None
        self._wrist_depth = None
        self._wrist_K = None
        self._scene: Optional[PerceptionScene] = None
        self._joints: Optional[List[float]] = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(
            Image, self.get_parameter('fixed_depth_topic').value,
            self._on_fixed_depth, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.get_parameter('fixed_info_topic').value,
            self._on_fixed_info, qos_profile_sensor_data)
        self.create_subscription(
            Image, self.get_parameter('wrist_depth_topic').value,
            self._on_wrist_depth, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.get_parameter('wrist_info_topic').value,
            self._on_wrist_info, qos_profile_sensor_data)
        self.create_subscription(PerceptionScene, '/perception/scene_graph', self._on_scene, 10)
        self.create_subscription(JointState, '/feedback/joint_states', self._on_joints, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)

        self._pub = self.create_publisher(SceneObstacleArray, '/perception/obstacles', 10)
        hz = max(float(self.get_parameter('publish_hz').value), 0.5)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(f'obstacle_extractor @ {hz:.1f} Hz → /perception/obstacles')

    def _on_fixed_depth(self, msg: Image) -> None:
        try:
            self._fixed_depth = _image_to_depth(msg)
        except Exception as exc:
            self.get_logger().warn(f'fixed depth decode: {exc}')

    def _on_fixed_info(self, msg: CameraInfo) -> None:
        self._fixed_K = _K_from_info(msg)

    def _on_wrist_depth(self, msg: Image) -> None:
        try:
            self._wrist_depth = _image_to_depth(msg)
        except Exception as exc:
            self.get_logger().warn(f'wrist depth decode: {exc}')

    def _on_wrist_info(self, msg: CameraInfo) -> None:
        self._wrist_K = _K_from_info(msg)

    def _on_scene(self, msg: PerceptionScene) -> None:
        self._scene = msg

    def _on_joints(self, msg: JointState) -> None:
        m = dict(zip(msg.name, msg.position))
        if all(j in m for j in ARM_JOINTS):
            self._joints = [float(m[j]) for j in ARM_JOINTS]

    def _lookup_T(self, child: str) -> Optional[np.ndarray]:
        base = str(self.get_parameter('base_frame').value)
        try:
            tf = self._tf_buffer.lookup_transform(base, child, rclpy.time.Time())
            return _T_from_tf(tf)
        except Exception:
            return None

    def _berry_positions(self) -> List[Tuple[float, float, float]]:
        out: List[Tuple[float, float, float]] = []
        if self._scene is None:
            return out
        for b in self._scene.berries:
            p = b.position
            if all(math.isfinite(float(v)) for v in p):
                out.append((float(p[0]), float(p[1]), float(p[2])))
        return out

    def _wrist_berry_bboxes(self) -> List[BBox2D]:
        out: List[BBox2D] = []
        if self._scene is None:
            return out
        for b in self._scene.berries:
            if not b.visible_wrist:
                continue
            if float(b.bbox_u1) <= float(b.bbox_u0):
                continue
            out.append(BBox2D(
                float(b.bbox_u0), float(b.bbox_v0),
                float(b.bbox_u1), float(b.bbox_v1)))
        return out

    def _tick(self) -> None:
        frames: List[DepthFrameSpec] = []
        berries = self._berry_positions()
        wrist_bboxes = self._wrist_berry_bboxes()

        if self._fixed_depth is not None and self._fixed_K is not None:
            T = self._lookup_T(str(self.get_parameter('fixed_frame').value))
            if T is not None:
                frames.append(DepthFrameSpec(
                    self._fixed_depth, self._fixed_K, T, berry_bboxes=[]))

        if self._wrist_depth is not None and self._wrist_K is not None:
            T = self._lookup_T(str(self.get_parameter('wrist_frame').value))
            if T is not None:
                frames.append(DepthFrameSpec(
                    self._wrist_depth, self._wrist_K, T,
                    berry_bboxes=wrist_bboxes))

        tip = None
        arm_links = None
        if self._joints is not None:
            t = tip_xyz(self._joints, tip_offset_link6=self._tip_off)
            tip = (float(t[0]), float(t[1]), float(t[2]))
            origins = fk_link_origins(self._joints)
            arm_links = [
                (float(origins[i, 0]), float(origins[i, 1]), float(origins[i, 2]))
                for i in range(origins.shape[0])
            ]

        spheres = extract_obstacle_spheres(
            frames,
            berry_positions_base=berries,
            tip_xyz_base=tip,
            arm_link_xyz_base=arm_links,
            cfg=self._cfg)

        # Fill bbox_uv on fixed camera for viz.
        if self._fixed_K is not None:
            T_fix = self._lookup_T(str(self.get_parameter('fixed_frame').value))
            if T_fix is not None:
                for s in spheres:
                    uv = project_base_to_uv(s.position, self._fixed_K, T_fix)
                    if uv is not None:
                        u, v = uv
                        s.bbox_uv = (u - 6, v - 6, u + 6, v + 6)

        arr = SceneObstacleArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = 'base_link'
        obs: List[SceneObstacle] = []
        for i, s in enumerate(spheres):
            o = SceneObstacle()
            o.id = i
            o.position = [float(s.position[0]), float(s.position[1]), float(s.position[2])]
            o.bbox_uv = [float(v) for v in s.bbox_uv]
            o.radius_m = float(s.radius_m)
            o.source = str(s.source)
            obs.append(o)
        arr.obstacles = obs
        self._pub.publish(arr)
        if obs:
            self.get_logger().debug(
                f'obstacles n={len(obs)} top_score={spheres[0].score:.0f}')


def main() -> None:
    rclpy.init()
    node = ObstacleExtractorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
