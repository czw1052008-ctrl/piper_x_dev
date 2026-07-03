#!/usr/bin/env python3
"""Grab one RGB + depth frame from DaBai wrist camera and save to disk (no cv_bridge)."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    cv2 = None


def _image_to_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc == 'rgb8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if enc == 'bgr8':
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return bgr[:, :, ::-1].copy()
    raise ValueError(f'unsupported color encoding: {msg.encoding}')


def _image_to_depth_m(msg: Image) -> tuple[np.ndarray, np.ndarray]:
    """Return depth in meters and uint16 visualization (mm clipped)."""
    if msg.encoding in ('32FC1', '32FC'):
        depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    elif msg.encoding in ('16UC1', 'mono16'):
        depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        depth_m = depth_mm.astype(np.float32) / 1000.0
    else:
        raise ValueError(f'unsupported depth encoding: {msg.encoding}')
    vis = np.clip(depth_m * 1000.0, 0, 3000).astype(np.uint16)
    return depth_m, vis


def _save_png_rgb(path: str, rgb: np.ndarray) -> None:
    if cv2 is not None:
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return
    from PIL import Image
    Image.fromarray(rgb).save(path)


def _save_png_gray16(path: str, arr: np.ndarray) -> None:
    if cv2 is not None:
        cv2.imwrite(path, arr)
        return
    from PIL import Image
    Image.fromarray(arr).save(path)


class CaptureOnce(Node):
    def __init__(self, color_topic: str, depth_topic: str, out_dir: str, rgb_only: bool = False) -> None:
        super().__init__('capture_camera_frame')
        self._out_dir = out_dir
        self._rgb_only = rgb_only
        self._rgb_msg: Image | None = None
        self._depth_msg: Image | None = None
        self.create_subscription(Image, color_topic, self._on_rgb, 1)
        if not rgb_only:
            self.create_subscription(Image, depth_topic, self._on_depth, 1)
        self._color_topic = color_topic
        self._depth_topic = depth_topic

    def _on_rgb(self, msg: Image) -> None:
        self._rgb_msg = msg

    def _on_depth(self, msg: Image) -> None:
        self._depth_msg = msg

    def ready(self) -> bool:
        if self._rgb_msg is None:
            return False
        return self._rgb_only or self._depth_msg is not None

    def save(self) -> tuple[str, str]:
        os.makedirs(self._out_dir, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        rgb_path = os.path.join(self._out_dir, f'color_{stamp}.png')
        depth_path = os.path.join(self._out_dir, f'depth_{stamp}.png')
        meta_path = os.path.join(self._out_dir, f'meta_{stamp}.txt')

        rgb = _image_to_rgb(self._rgb_msg)
        _save_png_rgb(rgb_path, rgb)
        if self._rgb_only:
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write(f'color_topic={self._color_topic}\n')
                f.write(f'size={self._rgb_msg.width}x{self._rgb_msg.height}\n')
            return rgb_path, ''

        depth_m, depth_vis = _image_to_depth_m(self._depth_msg)
        _save_png_gray16(depth_path, depth_vis)

        cy, cx = depth_m.shape[0] // 2, depth_m.shape[1] // 2
        center_d = float(depth_m[cy, cx]) if np.isfinite(depth_m[cy, cx]) else -1.0
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(f'color_topic={self._color_topic}\n')
            f.write(f'depth_topic={self._depth_topic}\n')
            f.write(f'color_encoding={self._rgb_msg.encoding}\n')
            f.write(f'depth_encoding={self._depth_msg.encoding}\n')
            f.write(f'size={self._rgb_msg.width}x{self._rgb_msg.height}\n')
            f.write(f'center_depth_m={center_d:.4f}\n')
        return rgb_path, depth_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color-topic', default='/camera_wrist/color/image_raw')
    parser.add_argument('--depth-topic', default='/camera_wrist/depth/image_raw')
    parser.add_argument('--out-dir', default='log/real_robot/capture')
    parser.add_argument('--timeout', type=float, default=15.0)
    parser.add_argument('--rgb-only', action='store_true',
                        help='Save RGB only (for annotation; depth not required)')
    args = parser.parse_args()

    rclpy.init()
    node = CaptureOnce(args.color_topic, args.depth_topic, args.out_dir, rgb_only=args.rgb_only)
    t0 = time.time()
    try:
        while rclpy.ok() and not node.ready() and (time.time() - t0) < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not node.ready():
            what = 'RGB' if args.rgb_only else 'RGB+depth'
            print(f'ERROR: timeout waiting for {what}', file=sys.stderr)
            print('  Start camera: bash scripts/real_robot_bringup.sh --camera-only', file=sys.stderr)
            return 1
        rgb_path, depth_path = node.save()
        print(f'captured RGB:   {rgb_path}')
        if depth_path:
            print(f'captured depth: {depth_path}')
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
