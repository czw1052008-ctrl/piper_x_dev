"""P3 z-slice polygons preserve silhouette; no sphere fit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'picking_perception'))

from picking_perception.z_slice_geometry import (  # noqa: E402
    SEM_BERRY,
    SEM_EGO,
    SEM_RESIDUAL,
    SEM_RIGID,
    instances_for_lift,
    lift_scene,
    object_from_points,
    objects_from_points,
    point_hits_object,
    polygon_from_xy,
    prism_triangles,
    residual_mask,
    slices_from_points,
)


def test_square_xy_polygon_not_circle():
    rng = np.random.default_rng(0)
    xs = rng.uniform(0.10, 0.18, size=400)
    ys = rng.uniform(0.00, 0.08, size=400)
    xy = np.stack([xs, ys], axis=1)
    poly = polygon_from_xy(xy, cell_m=0.004)
    assert poly is not None and len(poly) >= 4
    # bounding box of contour should match the square, not a disk
    span = poly.max(0) - poly.min(0)
    assert 0.07 < span[0] < 0.10
    assert 0.07 < span[1] < 0.10


def test_z_slices_stack_along_height():
    xs, ys, zs = np.meshgrid(
        np.linspace(0.20, 0.28, 12),
        np.linspace(0.00, 0.06, 10),
        np.linspace(0.10, 0.18, 16),
        indexing='xy')
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    slices = slices_from_points(pts, class_id=SEM_RIGID, dz=0.01)
    assert len(slices) >= 6
    zs_mid = [0.5 * (s.z_min + s.z_max) for s in slices]
    assert max(zs_mid) - min(zs_mid) > 0.05
    obj = object_from_points(1, SEM_RIGID, pts)
    assert obj is not None
    assert point_hits_object((0.24, 0.03, 0.14), obj)
    assert not point_hits_object((0.40, 0.03, 0.14), obj)
    tris, edges = prism_triangles(obj)
    assert tris.shape[0] >= 9
    assert edges.shape[0] >= 6


def test_rigid_keeps_vertical_wall_against_floor():
    """Floor-dominated rigid CC must not MAD-reject pot walls (world z)."""
    xs, ys = np.meshgrid(np.linspace(0.00, 0.20, 40), np.linspace(0.30, 0.50, 40))
    floor = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    wx, wz = np.meshgrid(np.linspace(0.06, 0.14, 24), np.linspace(0.04, 0.18, 20))
    wall = np.stack([wx.ravel(), np.full(wx.size, 0.40), wz.ravel()], axis=1)
    obj = object_from_points(3, SEM_RIGID, np.concatenate([floor, wall], axis=0))
    assert obj is not None
    assert max(s.z_max for s in obj.slices) > 0.12
    assert point_hits_object((0.10, 0.40, 0.12), obj)
    fly = np.array([[0.10, 0.40, 0.90]] * 8, dtype=np.float64)
    mixed = np.concatenate([obj.pts_xyz, fly], axis=0)
    sparse_tail = object_from_points(4, SEM_RIGID, mixed)
    assert sparse_tail is not None
    assert max(s.z_max for s in sparse_tail.slices) < 0.5


def test_disconnected_3d_clusters_become_separate_objects():
    """Same 2D instance, two 3D blobs far apart → two objects, not floating slices."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.004, size=(80, 3)) + np.array([-0.02, 0.60, 0.22])
    b = rng.normal(0, 0.004, size=(80, 3)) + np.array([0.49, 0.88, 0.22])
    objs = objects_from_points(1, SEM_BERRY, np.concatenate([a, b], axis=0))
    assert len(objs) == 2
    xs = sorted(o.centroid_xyz[0] for o in objs)
    assert xs[0] < 0.1 and xs[1] > 0.35
    for o in objs:
        cents = []
        for sl in o.slices:
            cents.append(sl.xy.mean(axis=0))
        cents = np.asarray(cents)
        assert float(np.linalg.norm(cents.max(0) - cents.min(0))) < 0.08


def test_floor_and_wall_stay_one_rigid():
    xs, ys = np.meshgrid(np.linspace(0.00, 0.20, 40), np.linspace(0.30, 0.50, 40))
    floor = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    wx, wz = np.meshgrid(np.linspace(0.06, 0.14, 24), np.linspace(0.00, 0.18, 24))
    wall = np.stack([wx.ravel(), np.full(wx.size, 0.40), wz.ravel()], axis=1)
    objs = objects_from_points(3, SEM_RIGID, np.concatenate([floor, wall], axis=0))
    assert len(objs) == 1
    assert max(s.z_max for s in objs[0].slices) > 0.12


def test_lift_four_classes_and_residual():
    h, w = 40, 48
    sem = np.zeros((h, w), dtype=np.uint8)
    depth = np.full((h, w), 0.45, dtype=np.float32)
    # berry, branch, rigid, ego blobs
    sem[4:10, 4:12] = SEM_BERRY
    sem[20:24, 4:8] = 2
    sem[8:18, 30:46] = 3
    sem[28:38, 20:40] = SEM_EGO
    depth[28:38, 1:14] = 0.50  # leftover valid depth = residual
    inst = instances_for_lift(sem, hm=None)
    classes = {int(np.bincount(sem[inst == i]).argmax()) for i in np.unique(inst) if i > 0}
    assert SEM_BERRY in classes and 2 in classes and 3 in classes and SEM_EGO in classes
    K = np.array([[200.0, 0.0, 24.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[2, 3] = 0.0
    objs = lift_scene(sem, inst, depth, K, T, include_residual=True)
    kinds = {o.class_id for o in objs}
    assert SEM_BERRY in kinds
    assert SEM_RIGID in kinds
    assert SEM_EGO in kinds
    for o in objs:
        assert o.slices, f'class {o.class_id} has no slices'
        for sl in o.slices:
            assert sl.xy.shape[0] >= 3


def test_residual_drops_near_clip_mode_keeps_far_scene():
    """Empty FOV filled with sensor-near depth must not lift as residual above the camera."""
    h, w = 40, 48
    sem = np.zeros((h, w), dtype=np.uint8)
    depth = np.full((h, w), 0.85, dtype=np.float32)
    depth[:12, :] = 0.12  # empty upper FOV → near-clip pile
    resid = residual_mask(sem, depth)
    assert int(np.sum(resid[:12, :])) == 0
    assert int(np.sum(resid[12:, :])) > 100
    K = np.array([[200.0, 0.0, 24.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    inst = np.zeros((h, w), dtype=np.uint16)
    objs = lift_scene(sem, inst, depth, K, T, include_residual=True)
    resid_objs = [o for o in objs if o.class_id == SEM_RESIDUAL]
    assert resid_objs
    for o in resid_objs:
        assert float(np.median(o.pts_xyz[:, 2])) > 0.4


def test_residual_unimodal_close_scene_kept():
    """Wrist close-up: residual all at one range is not a clip pile."""
    h, w = 32, 32
    sem = np.zeros((h, w), dtype=np.uint8)
    depth = np.full((h, w), 0.22, dtype=np.float32)
    resid = residual_mask(sem, depth)
    assert int(np.sum(resid)) == h * w
