#!/usr/bin/env python3
"""Lift DINOv3 masks + RGB-D into z-sliced SceneObjects and RViz prisms.

Geometry is stacked XY polygons (detection silhouette per base_z bin).
Does not publish spheres / capsules / OBBs as the visual model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from picking_perception.z_slice_geometry import (
    SEM_BERRY,
    SEM_BRANCH,
    SEM_EGO,
    SEM_RESIDUAL,
    SEM_RIGID,
    FuseReport,
    fuse_world,
    instances_for_lift,
    lift_scene,
    prism_triangles,
)

try:
    from picking_msgs.msg import SceneSlicedObject, SceneSlicedObjectArray, SceneZSlice
except ImportError:
    SceneSlicedObject = None  # type: ignore
    SceneSlicedObjectArray = None  # type: ignore
    SceneZSlice = None  # type: ignore


COLORS = {
    SEM_BERRY: ColorRGBA(r=0.92, g=0.15, b=0.88, a=0.72),
    SEM_BRANCH: ColorRGBA(r=0.12, g=0.72, b=0.22, a=0.62),
    SEM_RIGID: ColorRGBA(r=0.90, g=0.18, b=0.12, a=0.48),
    SEM_EGO: ColorRGBA(r=0.10, g=0.78, b=0.88, a=0.55),
    SEM_RESIDUAL: ColorRGBA(r=0.72, g=0.72, b=0.22, a=0.28),
}
NS = {
    SEM_BERRY: 'slice_berry',
    SEM_BRANCH: 'slice_branch',
    SEM_RIGID: 'slice_rigid',
    SEM_EGO: 'slice_ego',
    SEM_RESIDUAL: 'slice_residual',
}


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
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _decode_sem(msg: Image) -> Optional[np.ndarray]:
    h, w = int(msg.height), int(msg.width)
    if msg.encoding in ('mono8', '8UC1'):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w).copy()
    if msg.encoding in ('mono16', '16UC1'):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).copy()
    return None


def _xyz_cloud(pts: np.ndarray, header: Header) -> PointCloud2:
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = np.asarray(pts, dtype=np.float32).tobytes()
    return msg


class _Cam:
    def __init__(self) -> None:
        self.depth = None
        self.K = None
        self.sem = None
        self.inst = None


class SemanticObstacleNode(Node):
    def __init__(self) -> None:
        super().__init__('semantic_obstacle_node')
        self.declare_parameter('depth_topic', '/camera_fixed/depth/image_raw')
        self.declare_parameter('info_topic', '/camera_fixed/color/camera_info')
        self.declare_parameter('camera_frame', 'camera_fixed_color_optical_frame')
        self.declare_parameter('wrist_depth_topic', '/camera_wrist/depth/image_raw')
        self.declare_parameter('wrist_info_topic', '/camera_wrist/color/camera_info')
        self.declare_parameter('wrist_frame', 'camera_wrist_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_hz', 3.0)
        self.declare_parameter('max_items', 80)
        self.declare_parameter('include_residual', True)
        self.declare_parameter('publish_cloud', False)
        self.declare_parameter('fruit_id', -1)
        self._fixed = _Cam()
        self._wrist = _Cam()
        self._fruit_id = -1
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self.create_subscription(
            Image, self.get_parameter('depth_topic').value,
            lambda m: self._on_depth(self._fixed, m), qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.get_parameter('info_topic').value,
            lambda m: self._on_info(self._fixed, m), qos_profile_sensor_data)
        self.create_subscription(
            Image, '/perception/scene_seg/semantic',
            lambda m: self._on_sem(self._fixed, m), qos_profile_sensor_data)
        self.create_subscription(
            Image, '/perception/scene_seg/instances',
            lambda m: self._on_inst(self._fixed, m), qos_profile_sensor_data)
        self.create_subscription(
            Image, self.get_parameter('wrist_depth_topic').value,
            lambda m: self._on_depth(self._wrist, m), qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.get_parameter('wrist_info_topic').value,
            lambda m: self._on_info(self._wrist, m), qos_profile_sensor_data)
        self.create_subscription(
            Image, '/perception/scene_seg/wrist/semantic',
            lambda m: self._on_sem(self._wrist, m), qos_profile_sensor_data)
        self.create_subscription(
            Image, '/perception/scene_seg/wrist/instances',
            lambda m: self._on_inst(self._wrist, m), qos_profile_sensor_data)
        try:
            from picking_msgs.msg import PlannerInput
            self.create_subscription(PlannerInput, '/planning/planner_input', self._on_plan, 10)
        except Exception:
            pass
        self._pub_obj = None
        if SceneSlicedObjectArray is not None:
            self._pub_obj = self.create_publisher(
                SceneSlicedObjectArray, '/perception/scene_objects', 10)
        self._pub_mk = self.create_publisher(
            MarkerArray, '/planning/viz/scene_slices', 10)
        self._pub_mk_wrist = self.create_publisher(
            MarkerArray, '/planning/viz/scene_slices_wrist', 10)
        self._pub_cloud = self.create_publisher(
            PointCloud2, '/planning/viz/scene_slice_cloud', 5)
        hz = max(float(self.get_parameter('publish_hz').value), 0.2)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(f'scene z-slices + dual-view fuse @ {hz:.1f} Hz')

    def _on_plan(self, msg) -> None:
        try:
            self._fruit_id = int(msg.task_context.fruit_id)
        except Exception:
            pass

    def _on_depth(self, cam: _Cam, msg: Image) -> None:
        try:
            cam.depth = _image_to_depth(msg)
        except Exception as exc:
            self.get_logger().warn(f'depth: {exc}', throttle_duration_sec=5.0)

    def _on_info(self, cam: _Cam, msg: CameraInfo) -> None:
        cam.K = _K_from_info(msg)

    def _on_sem(self, cam: _Cam, msg: Image) -> None:
        arr = _decode_sem(msg)
        if arr is not None:
            cam.sem = arr.astype(np.uint8)

    def _on_inst(self, cam: _Cam, msg: Image) -> None:
        arr = _decode_sem(msg)
        if arr is not None:
            cam.inst = arr.astype(np.uint16)

    def _T(self, frame: str) -> Optional[np.ndarray]:
        try:
            tf = self._tf.lookup_transform(
                str(self.get_parameter('base_frame').value), frame, rclpy.time.Time())
            return _T_from_tf(tf)
        except Exception:
            return None

    def _lift(self, cam: _Cam, frame: str, source: str):
        if cam.depth is None or cam.K is None or cam.sem is None:
            return None, None
        T = self._T(frame)
        if T is None:
            return None, None
        inst = cam.inst
        if inst is None or inst.shape != cam.sem.shape:
            inst = instances_for_lift(cam.sem, hm=None)
        objs = lift_scene(
            cam.sem, inst, cam.depth, cam.K, T,
            include_residual=bool(self.get_parameter('include_residual').value),
            max_objects=int(self.get_parameter('max_items').value),
            source=source)
        return objs, T

    def _tick(self) -> None:
        fixed, _Tf = self._lift(
            self._fixed, str(self.get_parameter('camera_frame').value), 'fixed')
        if not fixed:
            return
        wrist, Tw = self._lift(
            self._wrist, str(self.get_parameter('wrist_frame').value), 'wrist')
        look = None if Tw is None else Tw[:3, 3]
        fid = int(self.get_parameter('fruit_id').value)
        if fid < 0:
            fid = int(self._fruit_id)
        fused, report = fuse_world(
            fixed, wrist or [], fruit_id=fid, look_xyz=look)
        stamp = self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id='base_link')
        if self._pub_obj is not None:
            self._pub_obj.publish(_objects_msg(fused, header))
        self._pub_mk.publish(_slice_markers(fused, header, wrist or [], report))
        self._pub_mk_wrist.publish(_wrist_overlay_markers(wrist or [], header, fused, report))
        if bool(self.get_parameter('publish_cloud').value) and fused:
            cloud = np.concatenate(
                [o.pts_xyz[:: max(1, o.n_pts // 2000)] for o in fused], axis=0)
            self._pub_cloud.publish(_xyz_cloud(cloud, header))
        n_w = 0 if not wrist else sum(1 for o in wrist if o.class_id == SEM_BERRY)
        self.get_logger().info(
            'fuse merged=%d extra=%d active=%d vis_w=%s Δ=%.1fmm minpair=%.1fmm wrist_b=%d reason=%s' % (
                report.n_merged_struct, report.n_wrist_extra,
                report.active_id, report.associated,
                float(report.dist_m) * 1000.0 if report.dist_m >= 0 else -1.0,
                float(report.min_pair_m) * 1000.0 if report.min_pair_m >= 0 else -1.0,
                n_w, report.reject_reason or '-'),
            throttle_duration_sec=2.0)


def _objects_msg(objs, header: Header):
    arr = SceneSlicedObjectArray()
    arr.header = header
    items = []
    n_sl = 0
    for o in objs:
        m = SceneSlicedObject()
        m.id = int(o.id)
        m.class_id = int(o.class_id)
        m.n_pts = int(o.n_pts)
        m.xyz_std = float(o.xyz_std)
        m.centroid_xyz = [float(v) for v in o.centroid_xyz]
        m.source = str(o.source)
        if hasattr(m, 'visible_wrist'):
            m.visible_wrist = bool(o.visible_wrist)
        if hasattr(m, 'pick_role'):
            m.pick_role = int(o.pick_role)
        if hasattr(m, 'fuse_delta_m'):
            m.fuse_delta_m = float(o.fuse_delta_m)
        slices = []
        for sl in o.slices:
            s = SceneZSlice()
            s.z_min = float(sl.z_min)
            s.z_max = float(sl.z_max)
            s.xy = np.asarray(sl.xy, dtype=np.float32).reshape(-1).tolist()
            slices.append(s)
            n_sl += 1
        m.slices = slices
        items.append(m)
    arr.objects = items
    arr.n_objects = len(items)
    arr.n_slices = n_sl
    return arr


def _pt(xyz: np.ndarray) -> Point:
    return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


def _slice_markers(objs, header: Header, wrist_objs, report) -> MarkerArray:
    arr = MarkerArray()
    delete = Marker()
    delete.action = Marker.DELETEALL
    delete.header = header
    arr.markers.append(delete)
    grouped: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    for o in objs:
        tris, edges = prism_triangles(o)
        bucket = grouped.setdefault(int(o.class_id), ([], []))
        if tris.shape[0]:
            bucket[0].append(tris)
        if edges.shape[0]:
            bucket[1].append(edges)
    mid = 1
    for cls, (tri_list, edge_list) in grouped.items():
        color = COLORS.get(cls, COLORS[SEM_RIGID])
        ns = NS.get(cls, 'slice_other')
        fill = Marker()
        fill.header = header
        fill.ns = ns
        fill.id = mid
        mid += 1
        fill.type = Marker.TRIANGLE_LIST
        fill.action = Marker.ADD
        fill.pose.orientation.w = 1.0
        fill.scale.x = fill.scale.y = fill.scale.z = 1.0
        fill.color = color
        if tri_list:
            verts = np.concatenate(tri_list, axis=0)
            fill.points = [_pt(p) for p in verts]
        arr.markers.append(fill)
        line = Marker()
        line.header = header
        line.ns = ns + '_edge'
        line.id = mid
        mid += 1
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.0015
        line.color = ColorRGBA(
            r=min(1.0, color.r + 0.15), g=min(1.0, color.g + 0.15),
            b=min(1.0, color.b + 0.15), a=0.95)
        if edge_list:
            ev = np.concatenate(edge_list, axis=0)
            line.points = [_pt(p) for p in ev]
        arr.markers.append(line)
    hud, _ = _fuse_hud_markers(objs, wrist_objs, report, header, mid)
    arr.markers.extend(hud)
    return arr


WRIST_COLORS = {
    SEM_BERRY: ColorRGBA(r=1.0, g=0.42, b=0.05, a=0.55),
    SEM_BRANCH: ColorRGBA(r=0.15, g=1.0, b=0.45, a=0.45),
    SEM_RIGID: ColorRGBA(r=1.0, g=0.35, b=0.75, a=0.40),
    SEM_EGO: ColorRGBA(r=0.55, g=0.85, b=1.0, a=0.35),
    SEM_RESIDUAL: ColorRGBA(r=1.0, g=1.0, b=0.85, a=0.22),
}


def _append_prisms(arr: MarkerArray, objs, header: Header, ns_prefix: str, colors, mid: int) -> int:
    grouped: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    for o in objs:
        tris, edges = prism_triangles(o)
        bucket = grouped.setdefault(int(o.class_id), ([], []))
        if tris.shape[0]:
            bucket[0].append(tris)
        if edges.shape[0]:
            bucket[1].append(edges)
    for cls, (tri_list, edge_list) in grouped.items():
        color = colors.get(cls, COLORS[SEM_RIGID])
        fill = Marker()
        fill.header = header
        fill.ns = f'{ns_prefix}{NS.get(cls, "other")}'
        fill.id = mid
        mid += 1
        fill.type = Marker.TRIANGLE_LIST
        fill.action = Marker.ADD
        fill.pose.orientation.w = 1.0
        fill.scale.x = fill.scale.y = fill.scale.z = 1.0
        fill.color = color
        if tri_list:
            fill.points = [_pt(p) for p in np.concatenate(tri_list, axis=0)]
        arr.markers.append(fill)
        line = Marker()
        line.header = header
        line.ns = fill.ns + '_edge'
        line.id = mid
        mid += 1
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.0025
        line.color = ColorRGBA(r=1.0, g=0.55, b=0.05, a=0.95)
        if edge_list:
            line.points = [_pt(p) for p in np.concatenate(edge_list, axis=0)]
        arr.markers.append(line)
    return mid


def _wrist_overlay_markers(wrist_objs, header: Header, fused, report) -> MarkerArray:
    arr = MarkerArray()
    delete = Marker()
    delete.action = Marker.DELETEALL
    delete.header = header
    arr.markers.append(delete)
    mid = _append_prisms(arr, wrist_objs, header, 'wrist_', WRIST_COLORS, 1)
    berries = [o for o in wrist_objs if int(o.class_id) == SEM_BERRY]
    if berries:
        sph = Marker()
        sph.header = header
        sph.ns = 'wrist_berry_centers'
        sph.id = mid
        mid += 1
        sph.type = Marker.SPHERE_LIST
        sph.action = Marker.ADD
        sph.pose.orientation.w = 1.0
        sph.scale.x = sph.scale.y = sph.scale.z = 0.014
        sph.color = ColorRGBA(r=1.0, g=0.45, b=0.0, a=1.0)
        sph.points = [_pt(np.asarray(o.centroid_xyz)) for o in berries]
        arr.markers.append(sph)
    hud, _ = _fuse_hud_markers(fused, wrist_objs, report, header, mid)
    arr.markers.extend(hud)
    return arr


def _fuse_hud_markers(objs, wrist_objs, report, header: Header, mid: int):
    out = []
    active = next((o for o in objs if int(o.pick_role) == 1), None)
    if active is None:
        return out, mid
    a_fixed = np.asarray(
        active.centroid_fixed_xyz if active.centroid_fixed_xyz is not None else active.centroid_xyz)
    if report.associated:
        a_wrist = np.asarray(active.centroid_xyz)
    elif report.dist_m >= 0.0:
        a_wrist = a_fixed + np.asarray(report.delta_xyz, dtype=np.float64)
    else:
        a_wrist = a_fixed
    sph_f = Marker()
    sph_f.header = header
    sph_f.ns = 'fuse_active'
    sph_f.id = mid
    mid += 1
    sph_f.type = Marker.SPHERE
    sph_f.action = Marker.ADD
    sph_f.pose.position.x, sph_f.pose.position.y, sph_f.pose.position.z = [float(v) for v in a_fixed]
    sph_f.pose.orientation.w = 1.0
    sph_f.scale.x = sph_f.scale.y = sph_f.scale.z = 0.016
    sph_f.color = ColorRGBA(r=0.95, g=0.15, b=0.9, a=1.0)
    out.append(sph_f)
    sph_w = Marker()
    sph_w.header = header
    sph_w.ns = 'fuse_active'
    sph_w.id = mid
    mid += 1
    sph_w.type = Marker.SPHERE
    sph_w.action = Marker.ADD
    sph_w.pose.position.x, sph_w.pose.position.y, sph_w.pose.position.z = [float(v) for v in a_wrist]
    sph_w.pose.orientation.w = 1.0
    sph_w.scale.x = sph_w.scale.y = sph_w.scale.z = 0.020
    sph_w.color = ColorRGBA(r=1.0, g=0.45, b=0.0, a=1.0)
    out.append(sph_w)
    if float(np.linalg.norm(a_wrist - a_fixed)) > 0.004:
        arrw = Marker()
        arrw.header = header
        arrw.ns = 'fuse_active'
        arrw.id = mid
        mid += 1
        arrw.type = Marker.ARROW
        arrw.action = Marker.ADD
        arrw.pose.orientation.w = 1.0
        arrw.scale.x = 0.006
        arrw.scale.y = 0.012
        arrw.scale.z = 0.012
        if report.associated:
            arrw.color = ColorRGBA(r=1.0, g=0.55, b=0.05, a=1.0)
        else:
            arrw.color = ColorRGBA(r=1.0, g=0.15, b=0.1, a=1.0)
        arrw.points = [_pt(a_fixed), _pt(a_wrist)]
        out.append(arrw)
    txt = Marker()
    txt.header = header
    txt.ns = 'fuse_active'
    txt.id = mid
    mid += 1
    txt.type = Marker.TEXT_VIEW_FACING
    txt.action = Marker.ADD
    txt.pose.position.x = float(a_wrist[0])
    txt.pose.position.y = float(a_wrist[1])
    txt.pose.position.z = float(a_wrist[2] + 0.05)
    txt.pose.orientation.w = 1.0
    txt.scale.z = 0.028
    txt.color = ColorRGBA(r=1.0, g=0.85, b=0.2, a=1.0)
    dmm = max(0.0, float(report.dist_m)) * 1000.0 if report.dist_m >= 0 else -1.0
    pmm = max(0.0, float(report.min_pair_m)) * 1000.0 if report.min_pair_m >= 0 else -1.0
    if report.associated:
        txt.text = 'FUSED id=%d  Δ=%.0fmm' % (report.active_id, dmm)
    else:
        txt.text = 'HOLD id=%d  Δ=%.0fmm min=%.0fmm %s' % (
            report.active_id, dmm, pmm, report.reject_reason or 'no wrist')
    out.append(txt)
    return out, mid


def main() -> None:
    rclpy.init()
    node = SemanticObstacleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
