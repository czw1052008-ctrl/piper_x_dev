#!/usr/bin/env python3
"""Hemisphere occupancy + ESDF: fixed+wrist late fusion (log-odds).

Publishes:
  /perception/occupancy_esdf
  /perception/occupancy_local
  /planning/viz/occupancy_slice
  /planning/viz/occupancy_overlay
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from picking_msgs.msg import OccupancyEsdf, OccupancyLocal, PerceptionScene, SceneBerry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from end_effector_profile import load_profile
from occupancy_map import (
    BERRY,
    EGO,
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyConfig,
    OccupancyVolume,
)
from obstacle_extractor import project_base_to_uv
from piper_position_ik import fk_link_origins, tip_xyz
import struct

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def _quat_align_z(direction: np.ndarray) -> Tuple[float, float, float, float]:
    """Quaternion rotating +Z onto ``direction`` (unit)."""
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    d = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    d = d / n
    c = float(np.dot(z, d))
    if c > 0.999999:
        return 0.0, 0.0, 0.0, 1.0
    if c < -0.999999:
        return 1.0, 0.0, 0.0, 0.0
    axis = np.cross(z, d)
    axis = axis / float(np.linalg.norm(axis))
    s = math.sqrt(max(0.0, (1.0 - c) * 0.5))
    w = math.sqrt(max(0.0, (1.0 + c) * 0.5))
    return float(axis[0] * s), float(axis[1] * s), float(axis[2] * s), float(w)


def _image_to_depth(msg: Image) -> np.ndarray:
    if msg.encoding in ('32FC1', '32FC'):
        raw = np.frombuffer(msg.data, dtype=np.float32)
        return raw.reshape(msg.height, msg.width)
    if msg.encoding in ('16UC1', 'mono16'):
        raw = np.frombuffer(msg.data, dtype=np.uint16)
        return raw.reshape(msg.height, msg.width).astype(np.float32) / 1000.0
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    return raw.reshape(msg.height, msg.width, -1)[:, :, 0].astype(np.float32)


def _K_from_info(msg: CameraInfo) -> np.ndarray:
    return np.array(msg.k, dtype=np.float64).reshape(3, 3)


def _T_from_tf(tf: TransformStamped) -> np.ndarray:
    t = tf.transform.translation
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    # Hamilton xyzw (scipy / tf2)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _bgr_to_imgmsg(bgr: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.step = msg.width * 3
    msg.data = bgr.tobytes()
    return msg


def _labels_to_bgr(lab2d: np.ndarray) -> np.ndarray:
    out = np.zeros((*lab2d.shape, 3), dtype=np.uint8)
    out[lab2d == FREE] = (60, 180, 60)
    out[lab2d == OCCUPIED] = (40, 40, 220)
    out[lab2d == EGO] = (220, 180, 40)
    out[lab2d == BERRY] = (220, 60, 220)
    out[lab2d == UNKNOWN] = (30, 30, 30)
    return out


class OccupancyMapNode(Node):
    def __init__(self) -> None:
        super().__init__('occupancy_map_node')
        self.declare_parameter('publish_hz', 1.5)
        self.declare_parameter('end_effector', 'agx_gripper_v1')
        self.declare_parameter('fixed_depth_topic', '/camera_fixed/depth/image_raw')
        self.declare_parameter('fixed_info_topic', '/camera_fixed/color/camera_info')
        self.declare_parameter('fixed_color_topic', '/camera_fixed/color/image_raw')
        self.declare_parameter('fixed_frame', 'camera_fixed_color_optical_frame')
        self.declare_parameter('wrist_depth_topic', '/camera_wrist/depth/image_raw')
        self.declare_parameter('wrist_info_topic', '/camera_wrist/color/camera_info')
        self.declare_parameter('wrist_frame', 'camera_wrist_color_optical_frame')
        self.declare_parameter('enable_wrist', True)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('voxel_m', 0.02)
        self.declare_parameter('radius_m', 0.85)
        self.declare_parameter('z_min', 0.02)
        self.declare_parameter('crop_size', 32)
        self.declare_parameter('ray_stride_px', 2)
        self.declare_parameter('hit_inflate_voxels', 1)
        self.declare_parameter('slice_z_m', 0.08)
        self.declare_parameter('reset_each_tick', False)
        self.declare_parameter('overlay_max_points', 30000)
        self.declare_parameter('overlay_alpha', 0.45)
        self.declare_parameter('cloud_max_points', 80000)
        self.declare_parameter('publish_3d_viz', True)

        profile = load_profile(str(self.get_parameter('end_effector').value))
        self._tip_off = profile.tip_offset_link6
        cfg = OccupancyConfig(
            radius_m=float(self.get_parameter('radius_m').value),
            z_min=float(self.get_parameter('z_min').value),
            voxel_m=float(self.get_parameter('voxel_m').value),
            crop_size=int(self.get_parameter('crop_size').value),
            ray_stride_px=int(self.get_parameter('ray_stride_px').value),
            hit_inflate_voxels=int(self.get_parameter('hit_inflate_voxels').value),
        )
        self._vol = OccupancyVolume(cfg)
        self._reset_each = bool(self.get_parameter('reset_each_tick').value)
        self._enable_wrist = bool(self.get_parameter('enable_wrist').value)

        self._fixed_depth = None
        self._fixed_K = None
        self._fixed_bgr = None
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
            Image, self.get_parameter('fixed_color_topic').value,
            self._on_fixed_color, qos_profile_sensor_data)
        if self._enable_wrist:
            self.create_subscription(
                Image, self.get_parameter('wrist_depth_topic').value,
                self._on_wrist_depth, qos_profile_sensor_data)
            self.create_subscription(
                CameraInfo, self.get_parameter('wrist_info_topic').value,
                self._on_wrist_info, qos_profile_sensor_data)
        self.create_subscription(PerceptionScene, '/perception/scene_graph', self._on_scene, 10)
        self.create_subscription(JointState, '/feedback/joint_states', self._on_joints, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)

        self._pub_full = self.create_publisher(OccupancyEsdf, '/perception/occupancy_esdf', 10)
        self._pub_local = self.create_publisher(OccupancyLocal, '/perception/occupancy_local', 10)
        self._pub_slice = self.create_publisher(Image, '/planning/viz/occupancy_slice', 10)
        self._pub_overlay = self.create_publisher(Image, '/planning/viz/occupancy_overlay', 10)
        # 3D structured viz (RViz): separate clouds so layers can be toggled
        self._pub_cloud_occ = self.create_publisher(
            PointCloud2, '/planning/viz/occupancy_cloud/occupied', 10)
        self._pub_cloud_free = self.create_publisher(
            PointCloud2, '/planning/viz/occupancy_cloud/free', 10)
        self._pub_cloud_ego = self.create_publisher(
            PointCloud2, '/planning/viz/occupancy_cloud/ego', 10)
        self._pub_cloud_berry = self.create_publisher(
            PointCloud2, '/planning/viz/occupancy_cloud/berry', 10)
        self._pub_markers = self.create_publisher(
            MarkerArray, '/planning/viz/occupancy_markers', 10)

        hz = max(float(self.get_parameter('publish_hz').value), 0.2)
        self.create_timer(1.0 / hz, self._tick)
        nx, ny, nz = [int(x) for x in self._vol.dims]
        self.get_logger().info(
            f'occupancy hemisphere R={cfg.radius_m} z≥{cfg.z_min} voxel={cfg.voxel_m} '
            f'grid={nx}x{ny}x{nz} wrist={self._enable_wrist} @ {hz:.1f} Hz; '
            f'3D → /planning/viz/occupancy_cloud/{{occupied,free,ego,berry}}')

    def _on_fixed_depth(self, msg: Image) -> None:
        try:
            self._fixed_depth = _image_to_depth(msg)
        except Exception as exc:
            self.get_logger().warn(f'fixed depth: {exc}')

    def _on_fixed_info(self, msg: CameraInfo) -> None:
        self._fixed_K = _K_from_info(msg)

    def _on_fixed_color(self, msg: Image) -> None:
        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'rgb8':
                rgb = raw.reshape(msg.height, msg.width, 3)
                self._fixed_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                self._fixed_bgr = raw.reshape(msg.height, msg.width, 3).copy()
        except Exception as exc:
            self.get_logger().warn(f'fixed color: {exc}')

    def _on_wrist_depth(self, msg: Image) -> None:
        try:
            self._wrist_depth = _image_to_depth(msg)
        except Exception as exc:
            self.get_logger().warn(f'wrist depth: {exc}')

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

    def _berry_split(self) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        active: List[Tuple[float, float, float]] = []
        others: List[Tuple[float, float, float]] = []
        if self._scene is None:
            return active, others
        for b in self._scene.berries:
            p = b.position
            if not all(math.isfinite(float(v)) for v in p):
                continue
            xyz = (float(p[0]), float(p[1]), float(p[2]))
            if not self._vol.in_workspace(xyz):
                continue
            if int(getattr(b, 'pick_role', 0)) == int(SceneBerry.PICK_ROLE_ACTIVE):
                active.append(xyz)
            else:
                others.append(xyz)
        return active, others

    def _active_or_tip_center(self) -> Tuple[float, float, float]:
        active, _ = self._berry_split()
        if active:
            return active[0]
        if self._joints is not None:
            t = tip_xyz(self._joints, tip_offset_link6=self._tip_off)
            return float(t[0]), float(t[1]), float(t[2])
        return 0.0, 0.35, 0.25

    def _tick(self) -> None:
        if self._fixed_depth is None or self._fixed_K is None:
            return
        T_fixed = self._lookup_T(str(self.get_parameter('fixed_frame').value))
        if T_fixed is None:
            return

        if self._reset_each:
            self._vol.reset()
        else:
            self._vol.decay()

        # 1) Ego capsules first (self-filter for depth)
        if self._joints is not None:
            tip = tip_xyz(self._joints, tip_offset_link6=self._tip_off)
            origins = fk_link_origins(self._joints)
            links = [
                (float(origins[i, 0]), float(origins[i, 1]), float(origins[i, 2]))
                for i in range(6)
            ]
            self._vol.apply_ego(
                (float(tip[0]), float(tip[1]), float(tip[2])), links)
        else:
            self._vol.clear_ego_mask()

        # 2) Dual-cam late fusion
        hits_f = self._vol.insert_depth_frame(
            self._fixed_depth, self._fixed_K, T_fixed,
            weight=float(self._vol.cfg.w_fixed))
        hits_w = 0
        if (self._enable_wrist and self._wrist_depth is not None
                and self._wrist_K is not None):
            T_wrist = self._lookup_T(str(self.get_parameter('wrist_frame').value))
            if T_wrist is not None:
                # near-field boost from median valid depth
                d = self._wrist_depth
                valid = d[np.isfinite(d) & (d > 0.1) & (d < 1.5)]
                near = bool(valid.size and float(np.median(valid)) < float(self._vol.cfg.wrist_near_z_m))
                w = float(self._vol.cfg.w_wrist_near if near else self._vol.cfg.w_wrist)
                hits_w = self._vol.insert_depth_frame(
                    self._wrist_depth, self._wrist_K, T_wrist, weight=w)

        # 3) Berry semantics
        active, others = self._berry_split()
        self._vol.apply_berries(active, others)
        self._vol.materialize_labels()
        self._vol.compute_esdf()

        stamp = self.get_clock().now().to_msg()
        hdr = Header(stamp=stamp, frame_id='base_link')

        full = OccupancyEsdf()
        full.header = hdr
        full.origin_xyz = [float(x) for x in self._vol.origin]
        full.voxel_m = float(self._vol.voxel_m)
        full.size_xyz = [int(x) for x in self._vol.dims]
        full.labels = self._vol.flatten_labels().tolist()
        full.esdf = self._vol.flatten_esdf().tolist()
        full.occupied_count = self._vol.occupied_count()
        full.free_count = self._vol.free_count()
        self._pub_full.publish(full)

        center = self._active_or_tip_center()
        crop = self._vol.crop_around(center)
        local = OccupancyLocal()
        local.header = hdr
        local.origin_xyz = [float(x) for x in crop.origin_xyz]
        local.voxel_m = float(crop.voxel_m)
        local.size_xyz = [int(x) for x in crop.size_xyz]
        local.labels = crop.labels.reshape(-1).tolist()
        local.esdf = crop.esdf.reshape(-1).tolist()
        local.crop_center_xyz = [float(x) for x in center]
        self._pub_local.publish(local)

        z_slice = float(self.get_parameter('slice_z_m').value)
        lab2d, _ = self._vol.z_slice_labels(z_slice)
        viz = _labels_to_bgr(np.flipud(lab2d.T))
        viz = cv2.resize(viz, (lab2d.shape[0] * 3, lab2d.shape[1] * 3),
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            viz,
            f'hemi z={z_slice:.2f} occ={full.occupied_count} free={full.free_count} '
            f'hitsF={hits_f} hitsW={hits_w}',
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        self._pub_slice.publish(_bgr_to_imgmsg(viz, stamp, 'occupancy_slice'))

        if self._fixed_bgr is not None:
            overlay = self._render_fixed_overlay(
                T_fixed, hits_f, hits_w, full.occupied_count, full.free_count)
            if overlay is not None:
                self._pub_overlay.publish(_bgr_to_imgmsg(overlay, stamp, 'camera_fixed'))

        if bool(self.get_parameter('publish_3d_viz').value):
            self._publish_3d_viz(hdr, active, others)

    def _xyzrgb_cloud(
        self,
        header: Header,
        idxs: np.ndarray,
        rgb: Tuple[int, int, int],
        max_pts: int,
    ) -> PointCloud2:
        if idxs.size == 0:
            msg = PointCloud2()
            msg.header = header
            msg.height = 1
            msg.width = 0
            msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            msg.is_bigendian = False
            msg.point_step = 16
            msg.row_step = 0
            msg.data = b''
            msg.is_dense = True
            return msg
        stride = max(1, len(idxs) // max(1, max_pts))
        idxs = idxs[::stride][:max_pts]
        r, g, b = rgb
        rgb_f = struct.unpack('f', struct.pack('I', (int(r) << 16) | (int(g) << 8) | int(b)))[0]
        buf = bytearray()
        for i, j, k in idxs:
            x, y, z = self._vol.idx_to_world(int(i), int(j), int(k))
            buf.extend(struct.pack('<ffff', float(x), float(y), float(z), float(rgb_f)))
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(idxs)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.data = bytes(buf)
        msg.is_dense = True
        return msg

    def _publish_3d_viz(
        self,
        header: Header,
        active: List[Tuple[float, float, float]],
        others: List[Tuple[float, float, float]],
    ) -> None:
        max_pts = int(self.get_parameter('cloud_max_points').value)
        labs = self._vol.labels
        # Space layers only (arm mesh comes from RobotModel / arm_model markers)
        self._pub_cloud_occ.publish(self._xyzrgb_cloud(
            header, np.argwhere(labs == OCCUPIED), (220, 40, 40), max_pts // 2))
        self._pub_cloud_free.publish(self._xyzrgb_cloud(
            header, np.argwhere(labs == FREE), (40, 200, 60), max_pts // 4))
        self._pub_cloud_ego.publish(self._xyzrgb_cloud(
            header, np.argwhere(labs == EGO), (240, 200, 40), max_pts // 6))
        self._pub_cloud_berry.publish(self._xyzrgb_cloud(
            header, np.argwhere(labs == BERRY), (240, 40, 240), max_pts // 8))

        ma = MarkerArray()
        mid = 0

        # --- Arm model (FK capsules) — readable geometry, not TF axes ---
        if self._joints is not None:
            tip = tip_xyz(self._joints, tip_offset_link6=self._tip_off)
            origins = fk_link_origins(self._joints)
            pts = [np.zeros(3)]
            for i in range(6):
                pts.append(origins[i].astype(float))
            pts.append(np.asarray(tip, dtype=float))
            # joint spheres
            for i, p in enumerate(pts):
                m = Marker()
                m.header = header
                m.ns = 'arm_model'
                m.id = mid
                mid += 1
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(p[0])
                m.pose.position.y = float(p[1])
                m.pose.position.z = float(p[2])
                m.pose.orientation.w = 1.0
                r = 0.028 if i < len(pts) - 1 else 0.022
                m.scale.x = m.scale.y = m.scale.z = r
                m.color = ColorRGBA(r=1.0, g=0.75, b=0.1, a=0.95)
                ma.markers.append(m)
            # link cylinders (z-aligned via quaternion)
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                d = b - a
                length = float(np.linalg.norm(d))
                if length < 1e-4:
                    continue
                m = Marker()
                m.header = header
                m.ns = 'arm_model'
                m.id = mid
                mid += 1
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                mid_p = 0.5 * (a + b)
                m.pose.position.x = float(mid_p[0])
                m.pose.position.y = float(mid_p[1])
                m.pose.position.z = float(mid_p[2])
                qx, qy, qz, qw = _quat_align_z(d / length)
                m.pose.orientation.x = qx
                m.pose.orientation.y = qy
                m.pose.orientation.z = qz
                m.pose.orientation.w = qw
                m.scale.x = m.scale.y = 0.045 if i < 3 else 0.035
                m.scale.z = length
                m.color = ColorRGBA(r=0.95, g=0.8, b=0.15, a=0.75)
                ma.markers.append(m)
            # tip highlight
            m = Marker()
            m.header = header
            m.ns = 'arm_model'
            m.id = mid
            mid += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(tip[0])
            m.pose.position.y = float(tip[1])
            m.pose.position.z = float(tip[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.035
            m.color = ColorRGBA(r=0.1, g=1.0, b=1.0, a=1.0)
            ma.markers.append(m)
        # delete unused arm_model ids (cap)
        for k in range(mid, 40):
            m = Marker()
            m.header = header
            m.ns = 'arm_model'
            m.id = k
            m.action = Marker.DELETE
            ma.markers.append(m)

        # --- Hemisphere outline (workspace) ---
        c = np.asarray(self._vol.cfg.center, dtype=float)
        R = float(self._vol.cfg.radius_m)
        z_min = float(self._vol.cfg.z_min)
        ring = Marker()
        ring.header = header
        ring.ns = 'workspace_hemisphere'
        ring.id = 0
        ring.type = Marker.LINE_LIST
        ring.action = Marker.ADD
        ring.scale.x = 0.006
        ring.color = ColorRGBA(r=0.55, g=0.55, b=0.75, a=0.45)
        ring.pose.orientation.w = 1.0
        for z_frac in (0.2, 0.55):
            z = z_min + z_frac * (c[2] + R - z_min)
            rr = math.sqrt(max(0.0, R * R - (z - c[2]) ** 2))
            n = 36
            prev = None
            for i in range(n + 1):
                th = 2.0 * math.pi * i / n
                p = Point(
                    x=float(c[0] + rr * math.cos(th)),
                    y=float(c[1] + rr * math.sin(th)),
                    z=float(z))
                if prev is not None:
                    ring.points.append(prev)
                    ring.points.append(p)
                prev = p
        ma.markers.append(ring)

        # --- Target fruits ---
        for bi, p in enumerate(active[:8]):
            m = Marker()
            m.header = header
            m.ns = 'fruit_active'
            m.id = bi
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.95)
            ma.markers.append(m)
            t = Marker()
            t.header = header
            t.ns = 'fruit_active'
            t.id = 100 + bi
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = p[0]
            t.pose.position.y = p[1]
            t.pose.position.z = p[2] + 0.04
            t.pose.orientation.w = 1.0
            t.scale.z = 0.04
            t.color = ColorRGBA(r=1.0, g=0.8, b=1.0, a=1.0)
            t.text = 'ACTIVE'
            ma.markers.append(t)
        for bi in range(len(active), 8):
            for mid_id in (bi, 100 + bi):
                m = Marker()
                m.header = header
                m.ns = 'fruit_active'
                m.id = mid_id
                m.action = Marker.DELETE
                ma.markers.append(m)

        for bi, p in enumerate(others[:12]):
            m = Marker()
            m.header = header
            m.ns = 'fruit_pending'
            m.id = bi
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.03
            m.color = ColorRGBA(r=1.0, g=0.45, b=0.1, a=0.8)
            ma.markers.append(m)
        for bi in range(len(others), 12):
            m = Marker()
            m.header = header
            m.ns = 'fruit_pending'
            m.id = bi
            m.action = Marker.DELETE
            ma.markers.append(m)

        # Delete old confusing namespaces if previously published
        for ns, nmax in (('base_link_axes', 1), ('active_berry', 16)):
            for i in range(nmax):
                m = Marker()
                m.header = header
                m.ns = ns
                m.id = i if ns != 'active_berry' else i + 1
                m.action = Marker.DELETE
                ma.markers.append(m)

        self._pub_markers.publish(ma)

    def _render_fixed_overlay(
        self,
        T_base_cam: np.ndarray,
        hits_f: int,
        hits_w: int,
        occ_n: int,
        free_n: int,
    ) -> Optional[np.ndarray]:
        bgr = self._fixed_bgr
        K = self._fixed_K
        if bgr is None or K is None:
            return None
        out = bgr.copy()
        h, w = out.shape[:2]
        paint = np.zeros_like(out)
        max_pts = int(self.get_parameter('overlay_max_points').value)
        alpha = float(self.get_parameter('overlay_alpha').value)
        labs = self._vol.labels
        # Draw order: FREE sparse → OCC → EGO → BERRY on top
        colors = {
            FREE: (60, 200, 60),
            OCCUPIED: (40, 40, 255),
            EGO: (0, 200, 255),
            BERRY: (255, 0, 255),
        }
        drawn = 0
        for lab, color in colors.items():
            idxs = np.argwhere(labs == lab)
            if idxs.size == 0:
                continue
            stride = 1
            if lab == FREE:
                stride = max(1, len(idxs) // max(1, max_pts // 5))
            elif len(idxs) > max_pts // 2:
                stride = max(1, len(idxs) // (max_pts // 2))
            for i, j, k in idxs[::stride]:
                if drawn >= max_pts:
                    break
                xyz = self._vol.idx_to_world(int(i), int(j), int(k))
                uv = project_base_to_uv(xyz, K, T_base_cam)
                if uv is None:
                    continue
                u, v = int(round(uv[0])), int(round(uv[1]))
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(paint, (u, v), 2 if lab != FREE else 1, color, -1)
                    drawn += 1
        cv2.addWeighted(paint, alpha, out, 1.0 - alpha * 0.35, 0, out)
        cv2.rectangle(out, (0, 0), (w, 48), (20, 20, 20), -1)
        cv2.putText(
            out,
            f'OCC(red) EGO(yel) BERRY(mag) FREE(grn)  '
            f'occ={occ_n} free={free_n} F={hits_f} W={hits_w} drawn={drawn}',
            (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        return out


def main() -> None:
    rclpy.init()
    node = OccupancyMapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
