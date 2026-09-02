"""Berry circles gated by semantic cluster (no watershed on the blob)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'picking_perception'))

from picking_perception.berry_instances import SEM_BERRY  # noqa: E402
from picking_perception.instance_gt import (  # noqa: E402
    BerryCenter,
    BerryCircle,
    circles_from_heatmap,
    centers_from_yolo_det,
    heatmap_from_centers,
    instances_from_circles,
    instances_from_heatmap,
    maps_from_yolo_seg,
    peaks_from_heatmap,
)


def test_yolo_txt_two_berries_not_one_cc():
    import tempfile
    h, w = 40, 80
    lines = [
        '0 0.20 0.40 0.28 0.40 0.28 0.60 0.20 0.60',
        '0 0.70 0.40 0.80 0.40 0.80 0.60 0.70 0.60',
        '1 0.05 0.10 0.95 0.10 0.95 0.14 0.05 0.14',
    ]
    with tempfile.TemporaryDirectory() as td:
        txt = Path(td) / 'a.txt'
        txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        sem, inst, centers = maps_from_yolo_seg(txt, h, w)
    assert len(centers) == 2, f'expected 2 berry instances, got {len(centers)}'
    n_berry = len([i for i in np.unique(inst) if i > 0])
    assert n_berry == 2
    assert int((sem == 1).sum()) > 0
    assert int((sem == 2).sum()) > 0


def test_heatmap_circles_inside_cluster_not_voronoi():
    h, w = 48, 80
    yy, xx = np.ogrid[:h, :w]
    blob = ((yy - 24) ** 2) / (10.0 ** 2) + ((xx - 38) ** 2) / (30.0 ** 2) <= 1.0
    sem = np.zeros((h, w), dtype=np.uint8)
    sem[blob] = SEM_BERRY
    centers = [
        BerryCenter(x=28.0, y=24.0, sigma=4.0, area_px=80),
        BerryCenter(x=48.0, y=24.0, sigma=4.0, area_px=80),
    ]
    hm = heatmap_from_centers(h, w, centers)
    peaks = peaks_from_heatmap(hm, min_score=0.3, min_dist_px=6)
    assert len(peaks) >= 2, f'peaks={peaks}'
    circles = circles_from_heatmap(sem, hm, min_score=0.3, min_dist_px=6)
    assert len(circles) >= 2
    inst = instances_from_heatmap(sem, hm, min_score=0.3, min_dist_px=6)
    berry_ids = [i for i in np.unique(inst[sem == SEM_BERRY]) if i > 0]
    assert len(berry_ids) >= 2
    leftover = (sem == SEM_BERRY) & (inst == 0)
    assert int(leftover.sum()) > 0, 'cluster remainder must not be assigned to a fruit'


def test_yolo_det_box_is_center_not_cluster():
    import tempfile
    h, w = 100, 200
    lines = [
        '0 0.25 0.40 0.10 0.12',
        '0 0.70 0.55 0.08 0.10',
        '1 0.50 0.50 0.40 0.40',  # cluster class skipped
        '0 0.50 0.50 0.50 0.50',  # too big → skip
    ]
    with tempfile.TemporaryDirectory() as td:
        txt = Path(td) / 'b.txt'
        txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        centers = centers_from_yolo_det(txt, h, w)
    assert len(centers) == 2
    assert abs(centers[0].x - 50.0) < 1.0
    assert abs(centers[0].y - 40.0) < 1.0


def test_peak_outside_cluster_dropped():
    h, w = 32, 48
    sem = np.zeros((h, w), dtype=np.uint8)
    sem[8:24, 8:24] = SEM_BERRY
    hm = np.zeros((h, w), dtype=np.float32)
    hm[2, 40] = 1.0
    circles = circles_from_heatmap(sem, hm, min_score=0.1, min_dist_px=4)
    assert circles == []


def test_no_peaks_no_fake_fruit_on_cluster():
    h, w = 32, 48
    yy, xx = np.ogrid[:h, :w]
    sem = np.zeros((h, w), dtype=np.uint8)
    sem[(xx - 16) ** 2 + (yy - 16) ** 2 <= 8 ** 2] = SEM_BERRY
    hm = np.zeros((h, w), dtype=np.float32)
    inst = instances_from_heatmap(sem, hm)
    berry_ids = [i for i in np.unique(inst[sem == SEM_BERRY]) if i > 0]
    assert berry_ids == [], 'no heatmap peak → no fruit id; cluster stays semantic only'


def test_center_radius_from_mask():
    from picking_perception.instance_gt import center_radius_from_mask
    m = np.zeros((40, 40), dtype=np.uint8)
    yy, xx = np.ogrid[:40, :40]
    m[(xx - 20) ** 2 + (yy - 20) ** 2 <= 6 ** 2] = 1
    cr = center_radius_from_mask(m)
    assert cr is not None
    cx, cy, r = cr
    assert abs(cx - 20) < 1 and abs(cy - 20) < 1
    assert 5.0 < r < 8.0


def test_overlapping_circles_nearest_center():
    h, w = 40, 40
    sem = np.ones((h, w), dtype=np.uint8)
    circles = [
        BerryCircle(id=1, x=12.0, y=20.0, r=10.0, score=1.0),
        BerryCircle(id=2, x=28.0, y=20.0, r=10.0, score=1.0),
    ]
    inst = instances_from_circles(sem, circles)
    assert inst[20, 12] == 1
    assert inst[20, 28] == 2
    assert inst[20, 20] in (1, 2)
