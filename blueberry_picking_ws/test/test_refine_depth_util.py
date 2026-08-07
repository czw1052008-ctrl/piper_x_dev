"""Tests for probe-tri mono chord depth (scheme G)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src' / 'picking_perception'))

from refine_depth_util import (  # noqa: E402
    apply_mono_chord_scale,
    chord_scale_berry,
    z_mono_near_reproject_uv,
)
from picking_perception.berry_tracker import BerryTracker  # noqa: E402

QA_DIR = ROOT / 'log/real_robot/qa/20260806_155054'
GT_DIST_M = 0.25


def test_chord_scale_midpoint():
    tcp = (0.0, 0.0, 0.0)
    berry = (0.0, 0.2, 0.0)
    out = chord_scale_berry(berry, tcp, 0.5)
    assert out == pytest.approx((0.0, 0.1, 0.0))


def test_apply_mono_chord_scale_rejects_out_of_range():
    berry, meta = apply_mono_chord_scale(
        (0.06, 0.44, 0.46),
        (0.07, 0.15, 0.32),
        z_cam_m=0.38,
        z_mono_m=0.10,
        scale_min=0.45,
        scale_max=1.05,
    )
    assert berry is None
    assert meta['rejected'] == 'scale_out_of_range'


def test_mono_depth_near_uv_prefers_closer_pixels():
    k = np.array([[550.0, 0, 320.0], [0, 550.0, 240.0], [0, 0, 1.0]])
    tracker = BerryTracker(berry_diameter_m=0.015, depth_min_m=0.03)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[160:200, 300:360] = 1
    mask[200:260, 300:360] = 1
    z_full = tracker.mono_depth_from_mask(mask, k)
    z_near = tracker.mono_depth_near_uv(mask, 330.0, 175.0, k, radius_px=30.0)
    assert z_near > z_full


@pytest.mark.skipif(
    not (QA_DIR / 'center_depth_qa.json').is_file(),
    reason='QA session 20260806_155054 not present',
)
def test_scheme_g_on_saved_session():
    with open(QA_DIR / 'center_depth_qa.json', encoding='utf-8') as f:
        qa = json.load(f)
    wrist = QA_DIR / 'center_depth_qa_wrist.png'
    if not wrist.is_file():
        pytest.skip('center_depth_qa_wrist.png missing')

    rgb = cv2.cvtColor(cv2.imread(str(wrist)), cv2.COLOR_BGR2RGB)
    berry_uv = qa['berry_uv']
    z_cam = float(qa['z_cam_m'])
    tcp = tuple(qa['tcp_base'])
    berry_tri = tuple(qa['berry_base'])

    z_mono, meta = z_mono_near_reproject_uv(rgb, float(berry_uv[0]), float(berry_uv[1]))
    assert z_mono is not None, meta
    berry_new, scale_meta = apply_mono_chord_scale(
        berry_tri,
        tcp,
        z_cam_m=z_cam,
        z_mono_m=float(z_mono),
    )
    assert berry_new is not None, scale_meta
    dist = math.sqrt(sum((berry_new[i] - tcp[i]) ** 2 for i in range(3)))
    baseline = float(qa['dist_cup_m'])
    assert dist < baseline
    assert abs(dist - GT_DIST_M) < abs(baseline - GT_DIST_M)
    assert abs(dist - GT_DIST_M) < 0.04
