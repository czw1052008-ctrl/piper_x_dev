#!/usr/bin/env python3
"""Publish colorized wrist depth for rqt during teleop.

Subscribes: /camera_wrist/depth/image_raw
Publishes:  /perception/wrist/depth_viz  (rgb8 jet colormap, metres clipped)
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def _depth_to_m(msg: Image) -> np.ndarray:
    enc = msg.encoding
    if enc in ('32FC1', '32FC'):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    if enc in ('16UC1', 'mono16'):
        mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return mm.astype(np.float32) / 1000.0
    raise ValueError(f'unsupported depth encoding: {enc}')


class DepthVizNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('wrist_depth_viz')
        self._min_m = args.min_m
        self._max_m = args.max_m
        self._pub = self.create_publisher(Image, args.out_topic, qos_profile_sensor_data)
        self.create_subscription(Image, args.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.get_logger().info(
            f'depth viz {args.depth_topic} -> {args.out_topic} '
            f'clip=[{self._min_m},{self._max_m}] m')

    def _on_depth(self, msg: Image) -> None:
        try:
            depth = _depth_to_m(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=2.0)
            return
        valid = np.isfinite(depth) & (depth > self._min_m) & (depth < self._max_m)
        norm = np.zeros(depth.shape, dtype=np.uint8)
        if valid.any():
            clipped = np.clip(depth, self._min_m, self._max_m)
            scaled = (clipped - self._min_m) / max(self._max_m - self._min_m, 1e-6)
            norm = (scaled * 255.0).astype(np.uint8)
            norm[~valid] = 0
        color_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        color_bgr[~valid] = (0, 0, 0)
        rgb = color_bgr[:, :, ::-1]
        out = Image()
        out.header = msg.header
        out.height, out.width = rgb.shape[:2]
        out.encoding = 'rgb8'
        out.is_bigendian = False
        out.step = out.width * 3
        out.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        self._pub.publish(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth-topic', default='/camera_wrist/depth/image_raw')
    parser.add_argument('--out-topic', default='/perception/wrist/depth_viz')
    parser.add_argument('--min-m', type=float, default=0.05)
    parser.add_argument('--max-m', type=float, default=1.2)
    args = parser.parse_args()
    rclpy.init()
    node = DepthVizNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
