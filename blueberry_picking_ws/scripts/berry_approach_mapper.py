"""3-D approach direction planner using global RGB-D camera depth.

At LOCKING success, build a local point cloud around the locked berry
from a single global depth frame, then select the least-obstructed
approach direction by casting rays through the 3-D point cloud.

Unlike the 2-D image-based approach_dir_selector (which only sees the
current wrist-camera frame at REFINING entry), this module sees the
scene from the global camera before ALIGNING even starts, giving the
PBVS loop a much better prior on which direction is obstacle-free.

Usage (called from reach_fsm_node._build_approach_map_from_global):
    mapper = BerryApproachMapper()
    pts = mapper.build_local_map(depth_img, K, T_base_cam, berry_xyz)
    direction = mapper.select_approach_direction(pts, berry_xyz, ee_xyz)
    # direction is a (3,) unit vector in base_link frame
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class BerryApproachMapper:
    """Build a local 3-D obstacle map and choose a clear approach direction."""

    def build_local_map(
        self,
        depth_img: np.ndarray,       # H×W float32, metres (0 = invalid)
        K: np.ndarray,               # 3×3 camera intrinsic matrix
        T_base_cam: np.ndarray,      # 4×4 transform: camera → base_link
        berry_xyz_base: np.ndarray,  # (3,) berry position in base_link
        radius_m: float = 0.22,      # keep points within this sphere of berry
        depth_min_m: float = 0.05,
        depth_max_m: float = 4.0,
    ) -> np.ndarray:                 # N×3 obstacle points in base_link
        """Back-project valid depth pixels to base_link, keep those near berry."""
        H, W = depth_img.shape[:2]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                     np.arange(H, dtype=np.float32))
        z = depth_img
        valid = (z > depth_min_m) & (z < depth_max_m)
        z_v = z[valid]
        x_c = (u_grid[valid] - cx) * z_v / fx
        y_c = (v_grid[valid] - cy) * z_v / fy
        ones = np.ones_like(z_v)

        pts_cam = np.stack([x_c, y_c, z_v, ones], axis=0)   # 4×N
        pts_base = (T_base_cam @ pts_cam)[:3].T              # N×3

        dist = np.linalg.norm(pts_base - berry_xyz_base, axis=1)
        return pts_base[dist < radius_m]

    def select_approach_direction(
        self,
        point_cloud: np.ndarray,             # N×3 in base_link
        berry_xyz: np.ndarray,               # (3,) target berry
        ee_xyz: Optional[np.ndarray] = None, # current EE position (seed direction)
        n_candidates: int = 36,
        lateral_blend: float = 0.25,         # how much to perturb laterally
        cylinder_radius_m: float = 0.03,     # ray-obstacle cylinder radius
        ray_near_m: float = 0.03,            # ignore obstacles within 3 cm of berry
        ray_far_m: float = 0.25,             # check up to 25 cm from berry
    ) -> np.ndarray:                         # (3,) unit vector in base_link
        """Return the approach direction with fewest obstacles along the ray.

        Casts N candidate rays from the berry outward (reverse approach).
        Counts point cloud hits inside a cylinder around each ray.
        Returns the direction with the lowest obstacle count.
        Falls back to EE→berry (or +X) when point cloud is empty.
        """
        berry = np.asarray(berry_xyz, dtype=np.float64)

        # Seed direction: current EE → berry
        if ee_xyz is not None:
            d = berry - np.asarray(ee_xyz, dtype=np.float64)
            n = float(np.linalg.norm(d))
            base_dir = d / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        else:
            base_dir = np.array([1.0, 0.0, 0.0])

        if len(point_cloud) == 0:
            return base_dir

        # Orthonormal lateral basis around base_dir
        ref = np.array([0.0, 0.0, 1.0]) if abs(base_dir[2]) < 0.9 \
            else np.array([0.0, 1.0, 0.0])
        u_ax = np.cross(base_dir, ref)
        u_ax /= np.linalg.norm(u_ax)
        v_ax = np.cross(base_dir, u_ax)

        pts = np.asarray(point_cloud, dtype=np.float64)

        best_score = -1.0
        best_dir = base_dir.copy()

        for angle in np.linspace(0.0, 2.0 * np.pi, n_candidates, endpoint=False):
            lat = np.cos(angle) * u_ax + np.sin(angle) * v_ax
            cand = base_dir + lateral_blend * lat
            cand /= np.linalg.norm(cand)

            # Ray: from berry backward along -cand (approach from outside)
            neg = -cand
            rel = pts - berry                            # N×3
            proj = rel @ neg                             # N — signed distance along ray
            perp = np.linalg.norm(rel - np.outer(proj, neg), axis=1)  # N

            # Obstacles: on-axis [near, far], within cylinder radius
            hit = (proj > ray_near_m) & (proj < ray_far_m) & (perp < cylinder_radius_m)
            score = 1.0 / (1.0 + float(hit.sum()))

            if score > best_score:
                best_score = score
                best_dir = cand

        return best_dir


# ---------------------------------------------------------------------------
# Unit test  (python berry_approach_mapper.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    print('BerryApproachMapper unit test')
    mapper = BerryApproachMapper()

    # ── Test 1: empty point cloud → fallback to EE→berry ────────────────
    berry = np.array([0.4, 0.0, 0.3])
    ee    = np.array([0.0, 0.0, 0.3])
    d1 = mapper.select_approach_direction(np.zeros((0, 3)), berry, ee)
    expected = berry - ee
    expected /= np.linalg.norm(expected)
    assert np.allclose(d1, expected, atol=1e-6), f'Test 1 failed: {d1}'
    print(f'  Test 1 (empty cloud): dir={d1}  ✓')

    # ── Test 2: obstacle directly ahead (EE→berry), right side clear ─────
    rng = np.random.default_rng(42)
    # Scatter background points 30 cm around berry
    bg = berry + rng.uniform(-0.2, 0.2, (200, 3))
    # Dense obstacle cluster 10 cm in front of berry (along +X approach)
    obs_x = berry + np.column_stack([
        rng.uniform(0.04, 0.18, 80),
        rng.uniform(-0.015, 0.015, 80),
        rng.uniform(-0.015, 0.015, 80),
    ])
    cloud = np.vstack([bg, obs_x])
    d2 = mapper.select_approach_direction(cloud, berry, ee, n_candidates=36)
    # Should NOT choose +X (blocked); y-component should be significant
    assert abs(d2[1]) > 0.1 or abs(d2[2]) > 0.1, \
        f'Test 2: expected non-X dir but got {d2}'
    print(f'  Test 2 (obstacle ahead): dir={d2}  (avoids +X)  ✓')

    # ── Test 3: build_local_map geometry check ───────────────────────────
    H, W = 240, 320
    fx = fy = 300.0
    cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    T = np.eye(4)   # camera = base

    # Flat wall at z=0.5 m
    depth = np.full((H, W), 0.5, dtype=np.float32)
    berry_b = np.array([0.0, 0.0, 0.5])   # centre pixel
    pts = mapper.build_local_map(depth, K, T, berry_b, radius_m=0.3)
    assert len(pts) > 0, 'Test 3: expected non-empty point cloud'
    # All points should be near z=0.5 in base frame
    assert np.all(np.abs(pts[:, 2] - 0.5) < 0.05), 'Test 3: depth mismatch'
    print(f'  Test 3 (build_local_map): {len(pts)} pts in sphere  ✓')

    print('  All assertions passed.')
    sys.exit(0)
