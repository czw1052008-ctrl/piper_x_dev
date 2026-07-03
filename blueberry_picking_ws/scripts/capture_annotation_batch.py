#!/usr/bin/env python3
"""Batch-capture wrist camera RGB frames for YOLO annotation (single ROS session)."""

from __future__ import annotations

import argparse
import os
import re
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


def _save_png_rgb(path: str, rgb: np.ndarray) -> None:
    if cv2 is None:
        raise RuntimeError('opencv-python (cv2) required to save PNG')
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


class BatchCapture(Node):
    def __init__(self, color_topic: str) -> None:
        super().__init__('capture_annotation_batch')
        self._rgb_msg: Image | None = None
        self.create_subscription(Image, color_topic, self._on_rgb, 1)

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
        """Spin until a frame newer than after_stamp arrives."""
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._rgb_msg is None:
                continue
            if after_stamp is None or self._stamp_key(self._rgb_msg) > after_stamp:
                return True
        return False

    def spin_for(self, duration_sec: float) -> None:
        """Keep receiving camera frames while waiting between captures."""
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < duration_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

    def grab_rgb(self) -> np.ndarray:
        if self._rgb_msg is None:
            raise RuntimeError('no RGB frame')
        return _image_to_rgb(self._rgb_msg)


def _preflight(color_topic: str, timeout: float) -> bool:
    import subprocess
    try:
        out = subprocess.run(
            ['ros2', 'topic', 'list'],
            capture_output=True, text=True, timeout=10, check=False)
        if color_topic not in out.stdout:
            print(f'ERROR: topic {color_topic} not found.', file=sys.stderr)
            print('  Start camera first:', file=sys.stderr)
            print('    bash scripts/real_robot_bringup.sh --camera-only', file=sys.stderr)
            return False
    except Exception as exc:
        print(f'WARN: ros2 topic list failed: {exc}', file=sys.stderr)
    return True


def _resolve_start_index(out_dir: str, prefix: str, start_index: int) -> int:
    if start_index > 0:
        return start_index
    pat = re.compile(rf'^{re.escape(prefix)}frame_(\d+)\.png$', re.IGNORECASE)
    max_idx = 0
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            m = pat.match(name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color-topic', default='/camera_wrist/color/image_raw')
    parser.add_argument('--out-dir', default='datasets/blueberry/images')
    parser.add_argument('--prefix', default='', help='Optional filename prefix, e.g. seg1_')
    parser.add_argument('--start-index', type=int, default=0,
                        help='First frame index (0 = auto-continue in out-dir)')
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--interval', type=float, default=2.0)
    parser.add_argument('--startup-timeout', type=float, default=60.0)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    start_idx = _resolve_start_index(out_dir, args.prefix, args.start_index)

    if not _preflight(args.color_topic, args.startup_timeout):
        return 1

    rclpy.init()
    node = BatchCapture(args.color_topic)
    try:
        print(f'[capture] Waiting for {args.color_topic} (up to {args.startup_timeout:.0f}s) ...')
        if not node.wait_rgb(args.startup_timeout):
            print('ERROR: no RGB frames — is the camera running?', file=sys.stderr)
            print('  bash scripts/real_robot_bringup.sh --camera-only', file=sys.stderr)
            return 1
        print(f'[capture] Camera OK. Saving {args.count} frames -> {out_dir}')
        if args.prefix:
            print(f'[capture] prefix={args.prefix}')
        print(f'[capture] numbering from frame_{start_idx:04d}')

        ok = 0
        last_stamp: tuple[int, int] | None = None
        for n in range(args.count):
            i = start_idx + n
            if not node.wait_new_rgb(last_stamp, 5.0):
                print(f'[capture] WARN: frame {n + 1}/{args.count} skipped (no new RGB)', file=sys.stderr)
                node.spin_for(args.interval)
                continue
            rgb = node.grab_rgb()
            last_stamp = BatchCapture._stamp_key(node._rgb_msg)
            path = os.path.join(out_dir, f'{args.prefix}frame_{i:04d}.png')
            _save_png_rgb(path, rgb)
            ok += 1
            print(f'[capture] {n + 1}/{args.count} -> {path}')
            if n + 1 < args.count:
                node.spin_for(args.interval)

        print(f'[capture] Done: {ok}/{args.count} saved')
        return 0 if ok > 0 else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
