"""Fine detection using wrist RGB-D + FoundationPose."""

from __future__ import annotations

import os

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from picking_msgs.msg import DetectedBerry
from picking_msgs.srv import TriggerFineDetection
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from picking_perception.berry_tracker import BerryTracker
from picking_perception.foundation_pose_wrapper import FoundationPoseWrapper
from picking_perception.segmentation import filter_berry_mask, segment_blueberry_hsv
from picking_perception.yolo_berry_detector import YoloBerryDetector


def _image_to_rgb(msg: Image) -> np.ndarray:
    """Decode sensor_msgs/Image without cv_bridge (avoids NumPy 1.x ABI clash in conda)."""
    enc = msg.encoding.lower()
    if enc in ('rgb8',):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if enc in ('bgr8',):
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return bgr[:, :, ::-1].copy()
    raise ValueError(f'unsupported color encoding: {msg.encoding}')


def _image_to_depth_m(msg: Image) -> np.ndarray:
    enc = msg.encoding
    if enc in ('32FC1', '32FC'):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    if enc in ('16UC1', 'mono16'):
        depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return depth_mm.astype(np.float32) / 1000.0
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    return arr.astype(np.float32)


def _matrix_from_tf(transform) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def _pose_to_msg(T: np.ndarray, header) -> PoseStamped:
    ps = PoseStamped()
    ps.header = header
    ps.pose.position.x = float(T[0, 3])
    ps.pose.position.y = float(T[1, 3])
    ps.pose.position.z = float(T[2, 3])
    ps.pose.orientation.w = 1.0
    return ps


class FineDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('fine_detector_node')
        self.declare_parameter('mesh_path', '')
        self.declare_parameter(
            'foundation_pose_root', '/home/ziwei/piper_x_dev/FoundationPose')
        self.declare_parameter('score_threshold', 0.3)
        self.declare_parameter('max_detections', 8)
        self.declare_parameter('mask_source', 'yolo')
        self.declare_parameter('yolo_model', 'yoloe-11s-seg-pf.pt')
        self.declare_parameter('yolo_classes', 'blueberry,blueberries,berry')
        self.declare_parameter('yolo_conf_threshold', 0.22)
        self.declare_parameter('yolo_min_area_px', 120)
        self.declare_parameter('yolo_max_area_frac', 0.04)
        self.declare_parameter('yolo_min_circularity', 0.45)
        self.declare_parameter('yolo_max_aspect_ratio', 2.0)
        self.declare_parameter('yolo_open_vocab', True)
        self.declare_parameter('enable_tracking', True)
        self.declare_parameter('berry_diameter_m', 0.015)
        self.declare_parameter('depth_min_m', 0.03)
        self.declare_parameter('depth_max_m', 1.8)
        self.declare_parameter('track_lost_max_frames', 25)

        mesh_path = self.get_parameter('mesh_path').value
        if not mesh_path:
            from ament_index_python.packages import get_package_share_directory
            mesh_path = os.path.join(
                get_package_share_directory('picking_perception'), 'meshes', 'blueberry.obj')

        self._fp = FoundationPoseWrapper(
            mesh_path=mesh_path,
            foundation_pose_root=self.get_parameter('foundation_pose_root').value,
        )
        if self._fp.ready:
            self.get_logger().info(f'FoundationPose ready (mesh={mesh_path})')
        else:
            self.get_logger().error(
                'FoundationPose NOT ready — run: bash scripts/setup_foundationpose_env.sh '
                'and start node via scripts/run_fine_detector_node.sh')

        self._mask_source = str(self.get_parameter('mask_source').value).lower()
        self._enable_tracking = bool(self.get_parameter('enable_tracking').value)
        self._yolo: YoloBerryDetector | None = None
        if self._mask_source == 'yolo':
            prompts = [
                p.strip() for p in str(self.get_parameter('yolo_classes').value).split(',') if p.strip()]
            self._yolo = YoloBerryDetector(
                model_name=str(self.get_parameter('yolo_model').value),
                class_prompts=prompts,
                conf_threshold=float(self.get_parameter('yolo_conf_threshold').value),
                min_area_px=int(self.get_parameter('yolo_min_area_px').value),
                max_area_frac=float(self.get_parameter('yolo_max_area_frac').value),
                min_circularity=float(self.get_parameter('yolo_min_circularity').value),
                max_aspect_ratio=float(self.get_parameter('yolo_max_aspect_ratio').value),
                max_detections=int(self.get_parameter('max_detections').value),
                open_vocab=bool(self.get_parameter('yolo_open_vocab').value),
            )
            if self._yolo.ready:
                self.get_logger().info(f'YOLO mask source ready prompts={prompts}')
            else:
                self.get_logger().warn('YOLO unavailable — falling back to HSV masks')
                self._mask_source = 'hsv'

        self._tracker = BerryTracker(
            berry_diameter_m=float(self.get_parameter('berry_diameter_m').value),
            depth_min_m=float(self.get_parameter('depth_min_m').value),
            depth_max_m=float(self.get_parameter('depth_max_m').value),
            track_lost_max_frames=int(self.get_parameter('track_lost_max_frames').value),
        )

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_wrist_color_optical_frame')
        self._base_frame = self.get_parameter('base_frame').value
        self._camera_frame = self.get_parameter('camera_frame').value

        self._rgb = None
        self._depth = None
        self._k = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(Image, '/camera_wrist/color/image_raw', self._on_rgb, 1)
        self.create_subscription(Image, '/camera_wrist/depth/image_raw', self._on_depth, 1)
        self.create_subscription(CameraInfo, '/camera_wrist/color/camera_info', self._on_info, 1)
        self.create_service(TriggerFineDetection, 'trigger_fine_detection', self._on_detect)

    def _on_rgb(self, msg: Image) -> None:
        try:
            self._rgb = _image_to_rgb(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc))

    def _on_depth(self, msg: Image) -> None:
        try:
            self._depth = _image_to_depth_m(msg)
        except Exception as exc:
            self.get_logger().warn(f'depth decode failed: {exc}')

    def _on_info(self, msg: CameraInfo) -> None:
        if self._k is None:
            self._k = np.array(msg.k).reshape(3, 3)

    def _build_tracked(self, rgb, depth, k, reset_lock: bool):
        if self._yolo is None or not self._yolo.ready:
            return [], -1, 'YOLO not ready'
        yolo_dets = self._yolo.detect(rgb)
        return self._tracker.process(
            yolo_dets, rgb, depth, k, self._fp, reset_lock=reset_lock)

    def _build_hsv_fp(self, rgb, depth, k):
        mask = segment_blueberry_hsv(rgb)
        mask = filter_berry_mask(mask)
        if mask.sum() < 100:
            return [], 'Mask too small (no round berry-colored blobs)'
        fp_dets = self._fp.detect_all(rgb, depth, k, mask)
        if not fp_dets:
            z_m = self._tracker.mono_depth_from_mask(mask, k)
            u, v = int(k[0, 2]), int(k[1, 2])
            if mask.sum() > 0:
                ys, xs = np.where(mask > 0)
                u, v = int(xs.mean()), int(ys.mean())
            x, y, z = self._tracker.uv_to_cam_xyz(u, v, z_m, k)
            T = np.eye(4)
            T[0, 3], T[1, 3], T[2, 3] = x, y, z
            return [(T, 0.4)], 'HSV mono fallback (depth invalid)'
        return fp_dets, ''

    def _on_detect(self, req, res):
        res.locked_berry_index = -1
        if not self._fp.ready:
            res.success = False
            res.message = 'FoundationPose not initialized (see setup_foundationpose_env.sh)'
            return res

        if self._rgb is None or self._depth is None or self._k is None:
            res.success = False
            res.message = 'Camera data not ready'
            return res

        lock_idx = -1
        track_msg = ''
        if self._mask_source == 'yolo' and self._enable_tracking:
            tracked, lock_idx, track_msg = self._build_tracked(
                self._rgb, self._depth, self._k, reset_lock=bool(req.reset_lock))
            if not tracked:
                res.success = False
                res.message = track_msg
                return res
            detections = [(t.pose_cam, t.confidence) for t in tracked]
        elif self._mask_source == 'yolo':
            yolo_dets = self._yolo.detect(self._rgb) if self._yolo else []
            detections = []
            for det in yolo_dets:
                for pose_cam, score in self._fp.detect_all(self._rgb, self._depth, self._k, det.mask):
                    detections.append((pose_cam, float(score) * det.confidence))
            if not detections:
                res.success = False
                res.message = 'YOLO+FP found no poses'
                return res
        else:
            detections, err = self._build_hsv_fp(self._rgb, self._depth, self._k)
            if not detections:
                res.success = False
                res.message = err
                return res

        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, self._camera_frame, rclpy.time.Time())
            T_base_cam = _matrix_from_tf(tf)
            out_frame = self._base_frame
        except Exception as exc:
            self.get_logger().warn(
                f'TF {self._base_frame}->{self._camera_frame} failed ({exc}); '
                f'returning poses in {self._camera_frame}')
            T_base_cam = None
            out_frame = self._camera_frame

        header = PoseStamped().header
        header.frame_id = out_frame
        header.stamp = self.get_clock().now().to_msg()

        berries = []
        for pose_cam, score in detections:
            T_out = T_base_cam @ pose_cam if T_base_cam is not None else pose_cam
            b = DetectedBerry()
            b.header = header
            b.confidence = float(score)
            b.pose = _pose_to_msg(T_out, header)
            berries.append(b)

        if lock_idx < 0 and berries and not track_msg:
            lock_idx = int(min(
                range(len(berries)),
                key=lambda i: berries[i].pose.pose.position.z,
            ))
        elif lock_idx < 0 and len(berries) == 1:
            lock_idx = 0

        res.detected_berries = berries
        res.locked_berry_index = lock_idx
        res.success = True
        src = self._mask_source.upper()
        n = len(berries)
        base = f'{n} berries ({src})'
        if track_msg:
            base = f'{base} | {track_msg}'
        if T_base_cam is None:
            res.message = f'{base} frame={out_frame}'
        else:
            res.message = base
        return res


def main() -> None:
    rclpy.init()
    node = FineDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
