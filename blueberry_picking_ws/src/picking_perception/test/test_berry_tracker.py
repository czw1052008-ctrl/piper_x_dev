"""Tests for berry close-range tracking."""

from __future__ import annotations

import numpy as np

from picking_perception.berry_tracker import BerryTracker
from picking_perception.yolo_berry_detector import YoloDetection


class _FakeFP:
    def detect_all(self, rgb, depth, k, mask):
        return []


def test_mono_depth_when_no_valid_depth_pixels():
    k = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    tracker = BerryTracker(berry_diameter_m=0.015, depth_min_m=0.03)
    depth = np.zeros((480, 640), dtype=np.float32)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:230, 300:330] = 1
    z = tracker.mono_depth_from_mask(mask, k)
    assert 0.03 <= z <= 1.8


def test_track_acquire_and_lock():
    k = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[210:220, 310:320] = 0.25
    mask1 = np.zeros((480, 640), dtype=np.uint8)
    mask1[200:230, 300:330] = 1
    mask2 = np.zeros((480, 640), dtype=np.uint8)
    mask2[180:210, 280:310] = 1
    dets = [
        YoloDetection(mask1, 0.8, (300, 200, 330, 230)),
        YoloDetection(mask2, 0.7, (280, 180, 310, 210)),
    ]
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker = BerryTracker()
    berries, lock, msg = tracker.process(dets, rgb, depth, k, _FakeFP())
    assert len(berries) == 2
    assert lock >= 0
    assert 'acquire' in msg

    depth2 = np.zeros_like(depth)
    berries2, lock2, msg2 = tracker.process(dets, rgb, depth2, k, _FakeFP())
    assert lock2 >= 0
    assert berries2[lock2].mode in ('mono', 'coast', 'fp')
