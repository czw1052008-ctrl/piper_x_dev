#!/usr/bin/env python3
"""Save HSV mask debug images from live /camera_wrist/color/image_raw."""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

_WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_WS_ROOT, 'src', 'picking_perception'))

from picking_perception.segmentation import (  # noqa: E402
    filter_berry_mask,
    iter_berry_blobs,
    segment_blueberry_hsv,
)


def _to_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc == 'rgb8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if enc == 'bgr8':
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return bgr[:, :, ::-1].copy()
    raise ValueError(msg.encoding)


class Snap(Node):
    def __init__(self) -> None:
        super().__init__('debug_perception_mask')
        self._rgb: np.ndarray | None = None
        self.create_subscription(Image, '/camera_wrist/color/image_raw', self._on_rgb, 1)

    def _on_rgb(self, msg: Image) -> None:
        self._rgb = _to_rgb(msg)


def main() -> int:
    out_dir = os.path.join(_WS_ROOT, 'log', 'real_robot', 'capture')
    os.makedirs(out_dir, exist_ok=True)

    rclpy.init()
    node = Snap()
    rgb = None
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.2)
        if node._rgb is not None:
            rgb = node._rgb
            break
    node.destroy_node()
    rclpy.shutdown()

    if rgb is None:
        print('[debug] No RGB from /camera_wrist/color/image_raw')
        return 1

    raw = segment_blueberry_hsv(rgb)
    filt = filter_berry_mask(raw)
    blobs = list(iter_berry_blobs(raw))

    overlay = rgb.copy()
    for b in blobs:
        cv2.rectangle(overlay, (b.x0, b.y0), (b.x1 - 1, b.y1 - 1), (0, 255, 255), 1)
        cv2.putText(
            overlay,
            f'a={b.area_px} c={b.circularity:.2f}',
            (b.x0, max(b.y0 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(os.path.join(out_dir, 'debug_rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, 'debug_raw_mask.png'), raw * 255)
    cv2.imwrite(os.path.join(out_dir, 'debug_filt_mask.png'), filt * 255)
    cv2.imwrite(os.path.join(out_dir, 'debug_overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print('[debug] HSV mask diagnostic')
    print(f'  raw_mask_px={int(raw.sum())}  filtered_px={int(filt.sum())}  blobs={len(blobs)}')
    print(f'  saved: {out_dir}/debug_overlay.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
