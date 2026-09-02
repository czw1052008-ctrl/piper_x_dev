"""P3 dual-view fusion: keep both cameras; union only where detections overlap."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'picking_perception'))

from picking_perception.z_slice_geometry import (  # noqa: E402
    SEM_BERRY,
    SEM_RIGID,
    SlicedObject,
    associate_wrist_berries,
    fuse_world,
    object_from_points,
    pick_active_berry,
)


def _berry(iid, xyz, n=40, source='fixed', sigma=0.010):
    rng = np.random.default_rng(int(iid) + 7)
    c = np.asarray(xyz, dtype=np.float64)
    pts = c + rng.normal(0.0, sigma, size=(n, 3))
    return SlicedObject(
        id=iid, class_id=SEM_BERRY, pts_xyz=pts,
        centroid_xyz=(float(c[0]), float(c[1]), float(c[2])),
        n_pts=n, source=source)


def test_pick_active_nearest_to_look():
    a = _berry(1, (0.4, 0.2, 0.2))
    b = _berry(2, (0.5, 0.3, 0.2))
    pick = pick_active_berry([a, b], look_xyz=(0.49, 0.29, 0.2))
    assert pick is not None and pick.id == 2
    pick2 = pick_active_berry([a, b], fruit_id=1, look_xyz=(0.49, 0.29, 0.2))
    assert pick2.id == 1


def test_associate_gate():
    f1 = _berry(1, (0.40, 0.20, 0.25))
    f2 = _berry(2, (0.55, 0.20, 0.25))
    w = _berry(9, (0.42, 0.21, 0.25))
    pairs = associate_wrist_berries([f1, f2], [w])
    assert len(pairs) == 1
    assert pairs[0][0].id == 1
    far = _berry(9, (0.80, 0.20, 0.25))
    assert associate_wrist_berries([f1], [far]) == []


def test_overlap_unions_same_berry_keeps_other_objects():
    f_a = _berry(1, (0.40, 0.20, 0.25))
    f_b = _berry(2, (0.70, 0.40, 0.25))
    rigid = SlicedObject(
        id=8, class_id=SEM_RIGID, pts_xyz=np.zeros((4, 3)),
        centroid_xyz=(0.1, 0.0, 0.1), n_pts=4, source='fixed')
    w = _berry(9, (0.415, 0.20, 0.25), source='wrist')
    fused, rep = fuse_world([f_a, f_b, rigid], [w], fruit_id=1)
    berries = [o for o in fused if o.class_id == SEM_BERRY]
    assert len(berries) == 2
    by_id = {o.id: o for o in fused}
    assert by_id[1].source == 'fused'
    assert by_id[1].visible_wrist
    assert by_id[1].n_pts == 80
    assert by_id[2].source == 'fixed'
    assert by_id[8].class_id == SEM_RIGID
    assert rep.n_merged_struct == 1
    assert rep.n_wrist_extra == 0


def test_nonoverlap_wrist_berry_is_kept():
    f_a = _berry(1, (0.40, 0.20, 0.25))
    w = _berry(9, (0.55, 0.20, 0.25), source='wrist')
    fused, rep = fuse_world([f_a], [w], fruit_id=1)
    berries = [o for o in fused if o.class_id == SEM_BERRY]
    assert len(berries) == 2
    sources = {o.source for o in berries}
    assert 'fixed' in sources and 'wrist' in sources
    assert rep.n_wrist_extra == 1
    fixed_kept = next(o for o in fused if o.source == 'fixed')
    assert abs(fixed_kept.centroid_xyz[0] - 0.40) < 0.03


def test_active_label_does_not_drop_wrist_map():
    f_near_cam = _berry(1, (0.40, 0.20, 0.25))
    f_best = _berry(2, (0.55, 0.20, 0.25))
    w = _berry(9, (0.56, 0.20, 0.25), source='wrist')
    fused, rep = fuse_world(
        [f_near_cam, f_best], [w], fruit_id=-1, look_xyz=(0.40, 0.20, 0.25))
    berries = [o for o in fused if o.class_id == SEM_BERRY]
    assert len(berries) == 2
    fused_b = next(o for o in fused if o.source == 'fused')
    assert fused_b.id == 2
    assert rep.active_id == 1
    assert not rep.associated


def test_wrist_only_far_berry_kept_with_near_merge():
    f_a = _berry(1, (0.40, 0.20, 0.25))
    f_near = _berry(2, (0.48, 0.20, 0.25))
    w_near = _berry(8, (0.49, 0.20, 0.25), source='wrist')
    w_far = _berry(9, (0.70, 0.20, 0.25), source='wrist')
    fused, _rep = fuse_world(
        [f_a, f_near], [w_near, w_far], fruit_id=1)
    berries = [o for o in fused if o.class_id == SEM_BERRY]
    assert len(berries) == 3
    assert any(o.source == 'wrist' for o in berries)
    assert any(o.id == 2 and o.source == 'fused' for o in fused)


def _floor_patch(iid, x0, x1, y0, y1, source='fixed'):
    xs = np.linspace(x0, x1, 40)
    ys = np.linspace(y0, y1, 40)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    obj = object_from_points(iid, SEM_RIGID, pts)
    assert obj is not None
    obj.source = source
    return obj


def test_fuse_unions_rigid_halves():
    berry = _berry(1, (0.4, 0.55, 0.25))
    left = _floor_patch(9, -0.12, 0.00, 0.30, 0.50, 'fixed')
    right = _floor_patch(20, 0.00, 0.12, 0.30, 0.50, 'wrist')
    fused, rep = fuse_world([berry, left], [right], fruit_id=1)
    rigid = next(o for o in fused if o.class_id == SEM_RIGID)
    xs = rigid.pts_xyz[:, 0]
    assert xs.min() < -0.08
    assert xs.max() > 0.08
    assert rigid.source == 'fused'
    assert rep.n_merged_struct == 1
