"""P1: per-berry instance split (2D watershed + 3D cluster) and contours."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'picking_perception'))
sys.path.insert(0, str(ROOT / 'scripts'))

from picking_perception.berry_instances import (  # noqa: E402
    SEM_BERRY,
    cluster_xyz,
    contours_from_instances,
    instances_from_semantic,
    refine_berry_instances_3d,
    split_berry_mask_2d,
)
from semantic_primitives import KIND_BERRY, primitives_from_maps  # noqa: E402


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def test_two_touching_disks_watershed_helper_splits():
    h, w = 48, 80
    mask = np.zeros((h, w), dtype=bool)
    mask |= _disk(h, w, 24, 28, 9)
    mask |= _disk(h, w, 24, 48, 9)
    lab = split_berry_mask_2d(mask, min_peak_dist_px=6, min_peak_dt=2.0)
    n = len([i for i in np.unique(lab) if i > 0])
    assert n >= 2, f'expected >=2 berries, got {n}'


def test_instances_without_heatmap_keeps_berry_cc():
    h, w = 48, 80
    sem = np.zeros((h, w), dtype=np.uint8)
    sem[_disk(h, w, 24, 20, 7)] = SEM_BERRY
    sem[_disk(h, w, 24, 60, 7)] = SEM_BERRY
    inst = instances_from_semantic(sem)
    n_b = len([i for i in np.unique(inst) if i > 0])
    assert n_b == 2
    contours = contours_from_instances(sem, inst)
    berries = [c for c in contours if c.class_id == SEM_BERRY]
    assert len(berries) == 2
    for c in berries:
        assert len(c.polygon_uv) >= 3
        assert c.area_px >= 8


def test_branch_stays_one_cc():
    sem = np.zeros((20, 40), dtype=np.uint8)
    sem[8:12, 5:35] = 2
    inst = instances_from_semantic(sem)
    ids = [i for i in np.unique(inst) if i > 0]
    assert len(ids) == 1


def test_3d_cluster_splits_one_mask_two_clouds():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.003, size=(30, 3))
    b = rng.normal(0, 0.003, size=(30, 3)) + np.array([0.05, 0.0, 0.0])
    lab = cluster_xyz(np.vstack([a, b]), eps_m=0.016, min_pts=4)
    n = len([c for c in np.unique(lab) if c >= 0])
    assert n == 2


def test_refine_3d_splits_same_id_two_blobs():
    """Two disjoint 2D berries wrongly sharing id=1 → 3D gap splits them."""
    h, w = 24, 48
    sem = np.zeros((h, w), dtype=np.uint8)
    inst = np.zeros((h, w), dtype=np.uint16)
    depth = np.full((h, w), 0.40, dtype=np.float64)
    sem[8:16, 6:14] = SEM_BERRY
    sem[8:16, 34:42] = SEM_BERRY
    inst[sem == SEM_BERRY] = 1
    K = np.array([[200.0, 0.0, 24.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    split = refine_berry_instances_3d(sem, inst, depth, K, T, stride=1)
    n = len([i for i in np.unique(split) if i > 0])
    assert n >= 2, f'3D split expected >=2, got {n}'


def test_refine_3d_splits_elongated_bridge():
    """Same-depth bridge is one Euclidean cluster; extent k-means still splits."""
    h, w = 24, 48
    sem = np.zeros((h, w), dtype=np.uint8)
    inst = np.zeros((h, w), dtype=np.uint16)
    depth = np.full((h, w), 0.40, dtype=np.float64)
    sem[8:16, 6:14] = SEM_BERRY
    sem[8:16, 34:42] = SEM_BERRY
    sem[11:13, 14:34] = SEM_BERRY
    inst[sem == SEM_BERRY] = 1
    K = np.array([[200.0, 0.0, 24.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    split = refine_berry_instances_3d(sem, inst, depth, K, T, stride=1)
    n = len([i for i in np.unique(split) if i > 0])
    assert n >= 2, f'extent split expected >=2, got {n}'


def test_primitives_two_berries_not_one_big_sphere():
    h, w = 24, 48
    sem = np.zeros((h, w), dtype=np.uint8)
    inst = np.zeros((h, w), dtype=np.uint16)
    depth = np.full((h, w), 0.40, dtype=np.float64)
    sem[8:16, 6:14] = SEM_BERRY
    sem[8:16, 34:42] = SEM_BERRY
    inst[sem == SEM_BERRY] = 1
    K = np.array([[200.0, 0.0, 24.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    prims = primitives_from_maps(sem, inst, depth, K, T, min_px=4)
    berries = [p for p in prims if p.kind == KIND_BERRY]
    assert len(berries) >= 2
    for p in berries:
        assert p.scale[0] <= 0.021
