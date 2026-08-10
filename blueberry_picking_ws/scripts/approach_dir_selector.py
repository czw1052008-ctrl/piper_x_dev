"""Select the least-obstructed approach direction for PBVS final approach.

Analyses the wrist depth image around the target berry and returns the unit
vector (in CAMERA frame) that passes through the fewest obstacles.

Usage:
    direction_cam = select_approach_direction(
        depth_img, target_uv, target_depth_m, camera_K)
    # direction_cam is a (3,) unit vector in camera frame (+Z = forward)
    # Caller transforms to base frame:
    #   direction_base = T_base_cam[:3,:3] @ direction_cam
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def select_approach_direction(
    depth_img: np.ndarray,        # H×W float32, metres (0 = invalid)
    target_uv: Tuple[int, int],   # (u_col, v_row) of target berry centre
    target_depth_m: float,        # estimated depth of berry
    camera_K: np.ndarray,         # 3×3 intrinsic matrix
    n_candidates: int = 36,
    cone_radius_px: int = 20,     # lateral offset of sampling circle centre
    sample_radius_px: int = 8,    # radius of the per-candidate sampling disc
    obstacle_margin_m: float = 0.03,   # pixels closer than target−margin = obstacle
    lateral_blend: float = 0.25,  # how much to offset the direction vector
) -> np.ndarray:
    """Return unit vector in camera frame for the best approach direction.

    Falls back to straight-ahead [0,0,1] if the depth image gives no useful
    signal or all candidates are equally obstructed.
    """
    H, W = depth_img.shape[:2]
    u0, v0 = int(target_uv[0]), int(target_uv[1])
    fx = float(camera_K[0, 0])
    fy = float(camera_K[1, 1])

    obstacle_threshold = target_depth_m - obstacle_margin_m

    best_score = -1.0
    best_angle = 0.0

    angles = np.linspace(0.0, 2.0 * np.pi, n_candidates, endpoint=False)

    for angle in angles:
        # Centre of the sampling disc, laterally offset from the target.
        cu = u0 + cone_radius_px * np.cos(angle)
        cv = v0 + cone_radius_px * np.sin(angle)

        # Collect depth samples in a small disc around (cu, cv).
        depths = _sample_disc(depth_img, cu, cv, sample_radius_px, H, W)

        if len(depths) == 0:
            # No pixels in this region → treat as clear.
            score = 1.0
        else:
            obstacle_count = int(np.sum(depths < obstacle_threshold))
            score = 1.0 - obstacle_count / len(depths)

        if score > best_score:
            best_score = score
            best_angle = angle

    # If every direction is equally obstructed (dense cluster), go straight.
    if best_score < 0.05:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Build a 3-D direction vector in camera frame.
    # Start from forward [0,0,1] and blend in a small lateral component.
    du = np.cos(best_angle) / fx   # pixel → normalised camera x
    dv = np.sin(best_angle) / fy   # pixel → normalised camera y
    lateral = np.array([du, dv, 0.0], dtype=np.float64)
    direction = np.array([0.0, 0.0, 1.0]) + lateral_blend * lateral
    direction /= np.linalg.norm(direction)
    return direction


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_disc(
    depth_img: np.ndarray,
    cu: float, cv: float,
    radius: int,
    H: int, W: int,
) -> np.ndarray:
    """Return valid (>0) depth values within a disc of given radius."""
    r = int(radius)
    u_min = max(0, int(cu) - r)
    u_max = min(W - 1, int(cu) + r)
    v_min = max(0, int(cv) - r)
    v_max = min(H - 1, int(cv) + r)
    if u_min > u_max or v_min > v_max:
        return np.array([])

    patch = depth_img[v_min:v_max + 1, u_min:u_max + 1]

    # Circular mask within bounding box.
    uu = np.arange(u_min, u_max + 1, dtype=np.float32) - cu
    vv = np.arange(v_min, v_max + 1, dtype=np.float32) - cv
    UU, VV = np.meshgrid(uu, vv)
    circle = (UU ** 2 + VV ** 2) <= float(radius) ** 2

    valid = patch[circle & (patch > 0)]
    return valid


# ---------------------------------------------------------------------------
# Standalone test  (python approach_dir_selector.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    print('ApproachDirSelector unit test')

    H, W = 480, 640
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)

    # Scenario 1: clear path straight ahead (no obstacles)
    depth_clear = np.full((H, W), 0.30, dtype=np.float32)
    target_uv = (320, 240)
    target_z = 0.18
    d = select_approach_direction(depth_clear, target_uv, target_z, K)
    assert d[2] > 0.9, f'Expected mostly-forward direction, got {d}'
    print(f'  Test 1 (clear): direction={d}  ✓')

    # Scenario 2: obstacle directly ahead, right side clear
    depth_blocked = np.full((H, W), 0.30, dtype=np.float32)
    # Place an obstacle (closer than berry) in a patch above the target.
    depth_blocked[200:260, 300:360] = 0.10   # 10 cm — obstacle
    target_uv2 = (330, 230)
    target_z2 = 0.18
    d2 = select_approach_direction(depth_blocked, target_uv2, target_z2, K,
                                   cone_radius_px=25)
    print(f'  Test 2 (obstacle above): direction={d2}  (should avoid blocked side)')
    assert d2[2] > 0.7, f'Direction not forward enough: {d2}'

    # Scenario 3: all directions blocked → fallback straight
    depth_all_blocked = np.full((H, W), 0.05, dtype=np.float32)
    d3 = select_approach_direction(depth_all_blocked, (320, 240), 0.18, K)
    assert np.allclose(d3, [0, 0, 1]), f'Expected fallback [0,0,1], got {d3}'
    print(f'  Test 3 (all blocked → fallback): direction={d3}  ✓')

    print('  All assertions passed.')
    sys.exit(0)
