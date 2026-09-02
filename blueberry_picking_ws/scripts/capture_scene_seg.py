#!/usr/bin/env python3
"""Save synced fixed+wrist RGB-D frames for scene_seg labeling. Does not move the arm.

Usage:
  python3 scripts/capture_scene_seg.py
  # Enter or 's' + Enter to save; 'q' to quit.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_OUT = os.path.join(ROOT, 'datasets', 'scene_seg', 'raw')


def _decode_color(msg: Image) -> Optional[np.ndarray]:
    enc = msg.encoding.lower()
    h, w = int(msg.height), int(msg.width)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == 'rgb8':
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()
    if enc == 'bgr8':
        return raw.reshape(h, w, 3).copy()
    return None


def _decode_depth_u16(msg: Image) -> Optional[np.ndarray]:
    enc = msg.encoding
    h, w = int(msg.height), int(msg.width)
    if enc in ('16UC1', 'mono16'):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).copy()
    if enc in ('32FC1', '32FC'):
        d = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        mm = np.clip(np.nan_to_num(d, nan=0.0) * 1000.0, 0, 65535)
        return mm.astype(np.uint16)
    return None


class CaptureSceneSeg(Node):
    def __init__(self, out_dir: str) -> None:
        super().__init__('capture_scene_seg')
        self._out = out_dir
        os.makedirs(self._out, exist_ok=True)
        self._fixed_bgr = None
        self._fixed_depth = None
        self._wrist_bgr = None
        self._wrist_depth = None
        self.create_subscription(
            Image, '/camera_fixed/color/image_raw', self._on_fc, qos_profile_sensor_data)
        self.create_subscription(
            Image, '/camera_fixed/depth/image_raw', self._on_fd, qos_profile_sensor_data)
        self.create_subscription(
            Image, '/camera_wrist/color/image_raw', self._on_wc, qos_profile_sensor_data)
        self.create_subscription(
            Image, '/camera_wrist/depth/image_raw', self._on_wd, qos_profile_sensor_data)

    def _on_fc(self, msg: Image) -> None:
        self._fixed_bgr = _decode_color(msg)

    def _on_fd(self, msg: Image) -> None:
        self._fixed_depth = _decode_depth_u16(msg)

    def _on_wc(self, msg: Image) -> None:
        self._wrist_bgr = _decode_color(msg)

    def _on_wd(self, msg: Image) -> None:
        self._wrist_depth = _decode_depth_u16(msg)

    def ready(self) -> bool:
        return all(x is not None for x in (
            self._fixed_bgr, self._fixed_depth, self._wrist_bgr, self._wrist_depth))

    def save(self) -> Optional[str]:
        if not self.ready():
            return None
        import cv2
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        pairs = (
            ('fixed_color', self._fixed_bgr),
            ('fixed_depth', self._fixed_depth),
            ('wrist_color', self._wrist_bgr),
            ('wrist_depth', self._wrist_depth),
        )
        for name, img in pairs:
            path = os.path.join(self._out, f'{stamp}_{name}.png')
            cv2.imwrite(path, img)
        return stamp


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    args = parser.parse_args()
    rclpy.init()
    node = CaptureSceneSeg(args.out)
    print(f'[capture_scene_seg] saving to {args.out}')
    print('  Enter / s = save frame; q = quit. Does not move the arm.')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            # Non-blocking-ish: user hits enter in this process.
            import select, sys
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip().lower()
                if line in ('q', 'quit'):
                    break
                if line in ('', 's', 'save'):
                    if not node.ready():
                        print('  waiting for all four topics…')
                        continue
                    stamp = node.save()
                    print(f'  saved {stamp}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
