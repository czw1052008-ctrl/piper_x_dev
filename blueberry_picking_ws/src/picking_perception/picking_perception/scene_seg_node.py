"""RGB semantic segmentation node: DINOv3-seg → semantic + instance maps + contours."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from picking_perception.berry_instances import (
    contours_from_instances,
    overlay_instance_contours,
)
from picking_perception.dinov3_seg import (
    CLASS_NAMES,
    colorize_semantic,
    load_checkpoint,
    maps_from_logits,
    unpack_seg,
)


def _decode_rgb(msg: Image) -> Optional[np.ndarray]:
    enc = msg.encoding.lower()
    h, w = int(msg.height), int(msg.width)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == 'rgb8':
        return raw.reshape(h, w, 3).copy()
    if enc == 'bgr8':
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()
    return None


def _mono8(sem: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = int(sem.shape[0]), int(sem.shape[1])
    msg.encoding = 'mono8'
    msg.step = msg.width
    msg.data = np.ascontiguousarray(sem, dtype=np.uint8).tobytes()
    return msg


def _mono16(inst: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = int(inst.shape[0]), int(inst.shape[1])
    msg.encoding = 'mono16'
    msg.step = msg.width * 2
    msg.data = np.ascontiguousarray(inst, dtype=np.uint16).tobytes()
    return msg


def _rgb8(rgb: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = int(rgb.shape[0]), int(rgb.shape[1])
    msg.encoding = 'rgb8'
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return msg


def _default_ckpt(ws: str) -> str:
    for name in ('dinov3-yolo-hm', 'dinov3-human-37', 'dinov3-p2-inst', 'dinov3-overfit-10'):
        p = os.path.join(ws, 'runs', 'seg', name, 'best.pt')
        if os.path.isfile(p):
            return p
    return os.path.join(ws, 'runs', 'seg', 'dinov3-overfit-10', 'best.pt')


class SceneSegNode(Node):
    def __init__(self) -> None:
        super().__init__('scene_seg_node')
        self.declare_parameter('image_topic', '/camera_fixed/color/image_raw')
        self.declare_parameter('camera_frame', 'camera_fixed_color_optical_frame')
        self.declare_parameter('wrist_image_topic', '/camera_wrist/color/image_raw')
        self.declare_parameter('wrist_camera_frame', 'camera_wrist_color_optical_frame')
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('publish_hz', 4.0)
        self.declare_parameter('input_size', 448)
        self._rgb = None
        self._rgb_wrist = None
        self._net = None
        self._kind = 'none'
        self._has_berry_hm = False
        ckpt = str(self.get_parameter('checkpoint').value).strip()
        if not ckpt:
            ws = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            ckpt = _default_ckpt(ws)
        if os.path.isfile(ckpt):
            try:
                self._net, self._kind = load_checkpoint(ckpt, device='cpu')
                self._has_berry_hm = bool(getattr(self._net, 'has_berry_hm', False))
                self.get_logger().info(
                    f'scene_seg loaded {ckpt} kind={self._kind} berry_hm={self._has_berry_hm}')
            except Exception as exc:
                self.get_logger().error(f'checkpoint failed: {exc}')
        else:
            self.get_logger().warn(
                f'no scene_seg checkpoint at {ckpt} — publishing empty maps')

        self.create_subscription(
            Image, str(self.get_parameter('image_topic').value),
            self._on_rgb, qos_profile_sensor_data)
        wrist_topic = str(self.get_parameter('wrist_image_topic').value).strip()
        if wrist_topic:
            self.create_subscription(
                Image, wrist_topic, self._on_rgb_wrist, qos_profile_sensor_data)
        self._pub_sem = self.create_publisher(
            Image, '/perception/scene_seg/semantic', qos_profile_sensor_data)
        self._pub_inst = self.create_publisher(
            Image, '/perception/scene_seg/instances', qos_profile_sensor_data)
        self._pub_viz = self.create_publisher(
            Image, '/perception/scene_seg/viz', qos_profile_sensor_data)
        self._pub_w_sem = self.create_publisher(
            Image, '/perception/scene_seg/wrist/semantic', qos_profile_sensor_data)
        self._pub_w_inst = self.create_publisher(
            Image, '/perception/scene_seg/wrist/instances', qos_profile_sensor_data)
        self._pub_w_viz = self.create_publisher(
            Image, '/perception/scene_seg/wrist/viz', qos_profile_sensor_data)
        self._pub_inst2d = None
        try:
            from picking_msgs.msg import SceneInstance2DArray
            self._pub_inst2d = self.create_publisher(
                SceneInstance2DArray, '/perception/scene_seg/instances_2d', 10)
        except Exception as exc:
            self.get_logger().warn(f'instances_2d topic skipped: {exc}')
        hz = max(float(self.get_parameter('publish_hz').value), 0.2)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(
            f'scene_seg {CLASS_NAMES} @ {hz:.1f} Hz image='
            f'{self.get_parameter("image_topic").value} wrist='
            f'{self.get_parameter("wrist_image_topic").value}')

    def _on_rgb(self, msg: Image) -> None:
        rgb = _decode_rgb(msg)
        if rgb is not None:
            self._rgb = (rgb, msg.header)

    def _on_rgb_wrist(self, msg: Image) -> None:
        rgb = _decode_rgb(msg)
        if rgb is not None:
            self._rgb_wrist = (rgb, msg.header)

    def _infer(self, rgb: np.ndarray):
        import cv2
        import torch
        size = int(self.get_parameter('input_size').value)
        h, w = rgb.shape[:2]
        inp = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            logits, hm = unpack_seg(self._net(x))
        return maps_from_logits(
            logits, hm, (h, w), use_heatmap=self._has_berry_hm)

    def _publish_instances_2d(self, contours, stamp, frame: str) -> None:
        if self._pub_inst2d is None:
            return
        from picking_msgs.msg import SceneInstance2D, SceneInstance2DArray
        arr = SceneInstance2DArray()
        arr.header = Header(stamp=stamp, frame_id=frame)
        items = []
        for c in contours:
            m = SceneInstance2D()
            m.id = int(c.id)
            m.class_id = int(c.class_id)
            m.score = 1.0
            m.polygon_uv = [float(v) for v in c.polygon_uv.reshape(-1)]
            m.bbox_uv = [float(v) for v in c.bbox_uv]
            m.area_px = int(c.area_px)
            items.append(m)
        arr.instances = items
        self._pub_inst2d.publish(arr)

    def _publish_maps(self, rgb, header, frame, pubs) -> int:
        stamp = header.stamp
        if self._net is None:
            sem = np.zeros(rgb.shape[:2], dtype=np.uint8)
            inst = np.zeros(rgb.shape[:2], dtype=np.uint16)
        else:
            sem, inst, _hm, _circles = self._infer(rgb)
        contours = contours_from_instances(sem, inst)
        pubs[0].publish(_mono8(sem, stamp, frame))
        pubs[1].publish(_mono16(inst, stamp, frame))
        overlay = rgb.copy()
        color = colorize_semantic(sem)
        vis = (0.55 * overlay.astype(np.float32) + 0.45 * color.astype(np.float32))
        vis = overlay_instance_contours(
            np.clip(vis, 0, 255).astype(np.uint8), inst, contours, rgb=True)
        pubs[2].publish(_rgb8(vis, stamp, frame))
        if pubs[0] is self._pub_sem:
            self._publish_instances_2d(contours, stamp, frame)
        return len(contours)

    def _tick(self) -> None:
        n_f = n_w = -1
        if self._rgb is not None:
            rgb, header = self._rgb
            n_f = self._publish_maps(
                rgb, header, str(self.get_parameter('camera_frame').value),
                (self._pub_sem, self._pub_inst, self._pub_viz))
        if self._rgb_wrist is not None:
            rgb, header = self._rgb_wrist
            n_w = self._publish_maps(
                rgb, header, str(self.get_parameter('wrist_camera_frame').value),
                (self._pub_w_sem, self._pub_w_inst, self._pub_w_viz))
        if n_f >= 0 or n_w >= 0:
            self.get_logger().info(
                f'scene_seg fixed_inst={n_f} wrist_inst={n_w}',
                throttle_duration_sec=4.0)


def main() -> None:
    rclpy.init()
    node = SceneSegNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
