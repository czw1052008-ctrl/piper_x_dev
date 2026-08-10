#!/usr/bin/env python3
"""ALIGNING: agent-estimated joint targets (position control) + coarse-orientation judge."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Primary protocol actions (agent estimates magnitudes via joints_deg / delta_deg).
ALIGN_ACTIONS = (
    'set_joints',   # position command with estimated joint targets or deltas
    'coarse_ok',    # hold pose; coarse facing looks right → enter wrist fine control
    'done',         # alias of coarse_ok
    # Legacy named deltas (discouraged; mapped with small fixed steps only as fallback)
    'whole_arm_right',
    'whole_arm_left',
    'yaw_left',
    'yaw_right',
    'pitch_down',
    'pitch_up',
    'yaw_left_pitch_down',
    'yaw_right_pitch_down',
    'yaw_left_pitch_up',
    'yaw_right_pitch_up',
    'face_plant_yaw',
)

JOINT_KEYS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')

# Mild lean helpers when estimating a one-shot pose toward the plant (degrees).
DEFAULT_LEAN_J2_DEG = -12.0
DEFAULT_LEAN_J3_DEG = 15.0

# Fixed mono: plant_uv − ee_uv in align observation (pixels).
DEFAULT_FIXED_TOL_X_PX = 90.0
DEFAULT_FIXED_TOL_Y_PX = 120.0
FIXED_J1_GAIN_DEG_PER_PX = 0.035
FIXED_J2_GAIN_DEG_PER_PX = 0.018
FIXED_J3_GAIN_DEG_PER_PX = -0.022
FIXED_J5_GAIN_DEG_PER_PX = 0.012
MAX_FIXED_CORRECTION_DEG = 12.0
# Reach pose when fixed mono shows plant below EE (wrist can see plant); see qa align judge snaps.
REACH_J2_DEG = 12.0
REACH_J3_DEG = -8.0
DEFAULT_EE_ANGLE_MAX_DEG = 48.0
DEFAULT_J2_REACH_MAX_DEG = 35.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def norm_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def build_observation(
    *,
    joint_positions: Sequence[float],
    plant_yaw: float,
    plant_xyz: Sequence[float],
    ee_xyz: Sequence[float],
    fine_visible: bool,
    fine_confidence: float,
    fine_u: float,
    fine_v: float,
    fine_width: int,
    fine_height: int,
    fixed_has_view: bool = False,
    fixed_dx_px: float = 0.0,
    fixed_dy_px: float = 0.0,
    fixed_plant_uv: Optional[Sequence[float]] = None,
    fixed_ee_uv: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    j1 = float(joint_positions[0]) if len(joint_positions) > 0 else 0.0
    j2 = float(joint_positions[1]) if len(joint_positions) > 1 else 0.0
    j3 = float(joint_positions[2]) if len(joint_positions) > 2 else 0.0
    j4 = float(joint_positions[3]) if len(joint_positions) > 3 else 0.0
    j5 = float(joint_positions[4]) if len(joint_positions) > 4 else 0.0
    j6 = float(joint_positions[5]) if len(joint_positions) > 5 else 0.0
    yaw_error = norm_angle(float(plant_yaw) - j1)

    dx = float(plant_xyz[0]) - float(ee_xyz[0])
    dy = float(plant_xyz[1]) - float(ee_xyz[1])
    dz = float(plant_xyz[2]) - float(ee_xyz[2])
    horiz = math.hypot(dx, dy)
    heading_x = math.cos(j1)
    heading_y = math.sin(j1)
    if horiz > 1e-6:
        plant_dir_x = dx / horiz
        plant_dir_y = dy / horiz
        cos_ang = max(-1.0, min(1.0, heading_x * plant_dir_x + heading_y * plant_dir_y))
        ee_target_angle_deg = math.degrees(math.acos(cos_ang))
    else:
        ee_target_angle_deg = 0.0

    cx = fine_width * 0.5
    cy = fine_height * 0.5
    return {
        'joint1': j1,
        'joint2': j2,
        'joint3': j3,
        'joint4': j4,
        'joint5': j5,
        'joint6': j6,
        'joint1_deg': math.degrees(j1),
        'joint2_deg': math.degrees(j2),
        'joint3_deg': math.degrees(j3),
        'joint5_deg': math.degrees(j5),
        'plant_yaw': float(plant_yaw),
        'plant_yaw_deg': math.degrees(float(plant_yaw)),
        'yaw_error': yaw_error,
        'yaw_error_deg': math.degrees(yaw_error),
        'ee_target_angle_deg': float(ee_target_angle_deg),
        'plant_x': float(plant_xyz[0]),
        'plant_y': float(plant_xyz[1]),
        'plant_z': float(plant_xyz[2]),
        'ee_x': float(ee_xyz[0]),
        'ee_y': float(ee_xyz[1]),
        'ee_z': float(ee_xyz[2]),
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'horiz_dist': horiz,
        'fine_visible': 1.0 if fine_visible else 0.0,
        'fine_confidence': float(fine_confidence),
        'fine_u': float(fine_u),
        'fine_v': float(fine_v),
        'fine_cx': float(cx),
        'fine_cy': float(cy),
        'fine_du': float(fine_u - cx),
        'fine_dv': float(fine_v - cy),
        'fixed_has_view': 1.0 if fixed_has_view else 0.0,
        'fixed_dx_px': float(fixed_dx_px),
        'fixed_dy_px': float(fixed_dy_px),
        'fixed_plant_u': float(fixed_plant_uv[0]) if fixed_plant_uv is not None else float('nan'),
        'fixed_plant_v': float(fixed_plant_uv[1]) if fixed_plant_uv is not None else float('nan'),
        'fixed_ee_u': float(fixed_ee_uv[0]) if fixed_ee_uv is not None else float('nan'),
        'fixed_ee_v': float(fixed_ee_uv[1]) if fixed_ee_uv is not None else float('nan'),
    }


def is_target_visible(
    obs: Dict[str, float],
    *,
    min_conf: Optional[float] = None,
    fine_visible_conf: Optional[float] = None,
) -> bool:
    conf = fine_visible_conf if fine_visible_conf is not None else (
        min_conf if min_conf is not None else 0.25)
    return float(obs.get('fine_visible', 0.0)) >= 1.0 and float(obs.get('fine_confidence', 0.0)) >= conf


def fixed_mono_error_px(obs: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """Plant minus EE in fixed mono image (px). None when projection unavailable."""
    if float(obs.get('fixed_has_view', 0.0)) < 0.5:
        return None
    return float(obs.get('fixed_dx_px', 0.0)), float(obs.get('fixed_dy_px', 0.0))


def fixed_mono_aligned(
    obs: Dict[str, float],
    *,
    tol_x_px: float = DEFAULT_FIXED_TOL_X_PX,
    tol_y_px: float = DEFAULT_FIXED_TOL_Y_PX,
) -> bool:
    err = fixed_mono_error_px(obs)
    if err is None:
        return False
    dx, dy = err
    return abs(dx) <= tol_x_px and abs(dy) <= tol_y_px


def _deg_map_from_decision(decision: Dict[str, object], key: str) -> Dict[str, float]:
    raw = decision.get(key)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def estimate_joint_targets_deg(
    obs: Dict[str, float],
    *,
    pitch_target_rad: float = -0.75,
    joint1_limit_deg: float = 150.0,
    joint5_min_rad: float = -1.4,
    joint5_max_rad: float = 0.2,
    lean_j2_deg: float = DEFAULT_LEAN_J2_DEG,
    lean_j3_deg: float = DEFAULT_LEAN_J3_DEG,
) -> Dict[str, float]:
    """
    Heuristic absolute joint targets (degrees).

    Primary: fixed mono plant↔EE pixel error drives j1/j2/j3/j5 corrections.
    Fallback (no fixed projection): plant_yaw on j1 + mild lean when yaw far.
    """
    j1 = float(obs.get('joint1', 0.0))
    j2 = float(obs.get('joint2', 0.0))
    j3 = float(obs.get('joint3', 0.0))
    j5 = float(obs.get('joint5', 0.0))
    plant_yaw = float(obs.get('plant_yaw', j1))
    yaw_err = norm_angle(plant_yaw - j1)

    j5_min_deg = math.degrees(joint5_min_rad)
    j5_max_deg = math.degrees(joint5_max_rad)
    j5_tgt = _clamp(math.degrees(pitch_target_rad), j5_min_deg, j5_max_deg)

    err = fixed_mono_error_px(obs)
    if err is not None:
        dx, dy = err
        cap = MAX_FIXED_CORRECTION_DEG
        j1_tgt = math.degrees(j1) + _clamp(-dx * FIXED_J1_GAIN_DEG_PER_PX, -cap, cap)
        # plant below EE in fixed mono (dy>0) → reach out (lower j2, unfold j3), not raise arm.
        j2_tgt = math.degrees(j2) - _clamp(dy * FIXED_J2_GAIN_DEG_PER_PX, -cap, cap)
        j3_tgt = math.degrees(j3) + _clamp(dy * abs(FIXED_J3_GAIN_DEG_PER_PX), -cap, cap)
        j5_tgt = _clamp(
            j5_tgt - _clamp(dy * FIXED_J5_GAIN_DEG_PER_PX, -cap * 0.5, cap * 0.5),
            j5_min_deg,
            j5_max_deg,
        )
        if math.degrees(j2) > 40.0 and dy > 60.0:
            blend = min(0.55, dy / 350.0)
            j2_tgt = (1.0 - blend) * j2_tgt + blend * REACH_J2_DEG
            j3_tgt = (1.0 - blend) * j3_tgt + blend * REACH_J3_DEG
    else:
        j1_tgt = _clamp(math.degrees(plant_yaw), -joint1_limit_deg, joint1_limit_deg)
        if abs(math.degrees(yaw_err)) > 8.0:
            j2_tgt = math.degrees(j2) + lean_j2_deg
            j3_tgt = math.degrees(j3) + lean_j3_deg
        else:
            j2_tgt = math.degrees(j2)
            j3_tgt = math.degrees(j3)

    j1_tgt = _clamp(j1_tgt, -joint1_limit_deg, joint1_limit_deg)
    return {
        'joint1': j1_tgt,
        'joint2': _clamp(j2_tgt, -100.0, 100.0),
        'joint3': _clamp(j3_tgt, -100.0, 100.0),
        'joint5': j5_tgt,
    }


def align_entry_ready(
    obs: Dict[str, float],
    *,
    fine_visible_conf: float = 0.25,
    yaw_deadband_deg: float = 8.0,
    pitch_target_rad: float = -0.75,
    pitch_deadband_rad: float = 0.12,
    fixed_tol_x_px: float = DEFAULT_FIXED_TOL_X_PX,
    fixed_tol_y_px: float = DEFAULT_FIXED_TOL_Y_PX,
    ee_angle_max_deg: float = DEFAULT_EE_ANGLE_MAX_DEG,
    j2_reach_max_deg: float = DEFAULT_J2_REACH_MAX_DEG,
) -> Tuple[bool, str]:
    """ALIGN done when wrist sees plant, or fixed-mono + reach pose (EE toward plant, arm extended)."""
    if is_target_visible(obs, fine_visible_conf=fine_visible_conf):
        return True, 'wrist RGB sees plant'
    fixed_err = fixed_mono_error_px(obs)
    fixed_ok = fixed_mono_aligned(
        obs, tol_x_px=fixed_tol_x_px, tol_y_px=fixed_tol_y_px)
    j1 = float(obs.get('joint1', 0.0))
    j5 = float(obs.get('joint5', 0.0))
    plant_yaw = float(obs.get('plant_yaw', j1))
    yaw_err_deg = abs(math.degrees(
        float(obs['yaw_error']) if 'yaw_error' in obs else norm_angle(plant_yaw - j1)))
    pitch_err = pitch_target_rad - j5
    yaw_pitch_ok = yaw_err_deg <= yaw_deadband_deg and abs(pitch_err) <= pitch_deadband_rad
    ee_angle = float(obs.get('ee_target_angle_deg', 180.0))
    j2_deg = float(obs.get('joint2_deg', math.degrees(float(obs.get('joint2', 0.0)))))
    reach_ok = ee_angle <= ee_angle_max_deg and j2_deg <= j2_reach_max_deg
    if fixed_ok and yaw_pitch_ok and reach_ok:
        return True, (
            f'fixed-mono aligned + reach pose (ee_angle={ee_angle:.0f}deg j2={j2_deg:.0f}deg)')
    if fixed_err is not None:
        return False, (
            f'need reach: dx={fixed_err[0]:.0f} dy={fixed_err[1]:.0f}px '
            f'ee_angle={ee_angle:.0f}deg j2={j2_deg:.0f}deg wrist_vis=0')
    return False, f'no fixed-mono view; ee_angle={ee_angle:.0f}deg j2={j2_deg:.0f}deg'


def decide_action(
    obs: Dict[str, float],
    *,
    failed_actions: Optional[Iterable[str]] = None,
    yaw_deadband_deg: float = 8.0,
    pitch_target_rad: float = -0.75,
    pitch_deadband_rad: float = 0.12,
    fine_visible_conf: float = 0.25,
    ee_angle_trigger_deg: float = 20.0,
    phase: str = 'command',
    allow_done_without_visible: bool = False,
    fixed_tol_x_px: float = DEFAULT_FIXED_TOL_X_PX,
    fixed_tol_y_px: float = DEFAULT_FIXED_TOL_Y_PX,
) -> Dict[str, object]:
    """
    Heuristic / agent-fallback decision.

    phase=command → set_joints from fixed-mono until wrist can see plant (or reach pose OK).
    phase=judge   → coarse_ok only when align_entry_ready; else set_joints.
    """
    del failed_actions, allow_done_without_visible
    fixed_err = fixed_mono_error_px(obs)
    fixed_ok = fixed_mono_aligned(
        obs, tol_x_px=fixed_tol_x_px, tol_y_px=fixed_tol_y_px)
    j1 = float(obs.get('joint1', 0.0))
    j5 = float(obs.get('joint5', 0.0))
    plant_yaw = float(obs.get('plant_yaw', j1))
    yaw_err = float(obs['yaw_error']) if 'yaw_error' in obs else norm_angle(plant_yaw - j1)
    yaw_err_deg = abs(math.degrees(yaw_err))
    pitch_err = pitch_target_rad - j5
    ee_angle = float(obs.get('ee_target_angle_deg', 180.0))
    j2_deg = float(obs.get('joint2_deg', math.degrees(float(obs.get('joint2', 0.0)))))
    wrist_ok = is_target_visible(obs, fine_visible_conf=fine_visible_conf)
    ready, ready_reason = align_entry_ready(
        obs,
        fine_visible_conf=fine_visible_conf,
        yaw_deadband_deg=yaw_deadband_deg,
        pitch_target_rad=pitch_target_rad,
        pitch_deadband_rad=pitch_deadband_rad,
        fixed_tol_x_px=fixed_tol_x_px,
        fixed_tol_y_px=fixed_tol_y_px,
        ee_angle_max_deg=max(ee_angle_trigger_deg, DEFAULT_EE_ANGLE_MAX_DEG),
        j2_reach_max_deg=DEFAULT_J2_REACH_MAX_DEG,
    )

    meta: Dict[str, object] = {
        'phase': phase,
        'yaw_error_deg': math.degrees(yaw_err),
        'pitch_error_rad': pitch_err,
        'ee_target_angle_deg': ee_angle,
        'joint2_deg': j2_deg,
        'plant_yaw_deg': math.degrees(plant_yaw),
        'joint1_deg': math.degrees(j1),
        'joint5_deg': math.degrees(j5),
        'fixed_ok': fixed_ok,
        'wrist_visible': wrist_ok,
        'fixed_dx_px': fixed_err[0] if fixed_err is not None else None,
        'fixed_dy_px': fixed_err[1] if fixed_err is not None else None,
        'source': 'heuristic',
    }

    if phase == 'judge':
        if ready:
            return {
                **meta,
                'action': 'coarse_ok',
                'reason': f'held pose: {ready_reason} → REFINING',
            }
        targets = estimate_joint_targets_deg(obs, pitch_target_rad=pitch_target_rad)
        reason = ready_reason if fixed_err is None else (
            f'{ready_reason} → j1={targets["joint1"]:.1f} j2={targets["joint2"]:.1f} '
            f'j3={targets["joint3"]:.1f} j5={targets["joint5"]:.1f}deg'
        )
        return {**meta, 'action': 'set_joints', 'joints_deg': targets, 'reason': reason}

    # command phase
    if ready:
        return {
            **meta,
            'action': 'coarse_ok',
            'reason': f'align entry ready: {ready_reason}',
        }

    targets = estimate_joint_targets_deg(obs, pitch_target_rad=pitch_target_rad)
    if fixed_err is None:
        reason = (
            f'estimate pose (no fixed view): j1→{targets["joint1"]:.1f}deg '
            f'j5→{targets["joint5"]:.1f}deg'
        )
    else:
        reason = (
            f'fixed-mono dx={fixed_err[0]:.0f} dy={fixed_err[1]:.0f}px → '
            f'j1={targets["joint1"]:.1f} j2={targets["joint2"]:.1f} '
            f'j3={targets["joint3"]:.1f} j5={targets["joint5"]:.1f}deg'
        )
    return {**meta, 'action': 'set_joints', 'joints_deg': targets, 'reason': reason}


def inverse_action(action: str) -> str:
    # Position-control path does not auto-inverse; restore is explicit only.
    mapping = {
        'set_joints': 'restore_joints',
        'coarse_ok': 'done',
        'done': 'done',
        'whole_arm_right': 'restore_joints',
        'whole_arm_left': 'restore_joints',
        'yaw_left': 'yaw_right',
        'yaw_right': 'yaw_left',
        'pitch_down': 'pitch_up',
        'pitch_up': 'pitch_down',
        'yaw_left_pitch_down': 'yaw_right_pitch_up',
        'yaw_right_pitch_down': 'yaw_left_pitch_up',
        'yaw_left_pitch_up': 'yaw_right_pitch_down',
        'yaw_right_pitch_up': 'yaw_left_pitch_down',
        'face_plant_yaw': 'restore_joints',
        'restore_joints': 'done',
    }
    return mapping.get(action, 'done')


def apply_joint_command(
    joints: Sequence[float],
    decision: Dict[str, object],
    *,
    joint1_limit_deg: float = 150.0,
    joint5_min_rad: float = -1.4,
    joint5_max_rad: float = 0.2,
    joint2_min_rad: float = -1.8,
    joint2_max_rad: float = 1.8,
    joint3_min_rad: float = -1.8,
    joint3_max_rad: float = 1.8,
    joint6_limit_rad: float = 0.0,
    restore_joints: Optional[Sequence[float]] = None,
) -> List[float]:
    """Apply set_joints (absolute joints_deg or relative delta_deg) as position targets."""
    action = str(decision.get('action', '')).strip().lower()
    if action == 'restore_joints' and restore_joints is not None:
        return [float(v) for v in restore_joints]

    target = [float(v) for v in joints]
    while len(target) < 6:
        target.append(0.0)

    abs_deg = _deg_map_from_decision(decision, 'joints_deg')
    delta_deg = _deg_map_from_decision(decision, 'delta_deg')

    if abs_deg:
        for name, idx in (('joint1', 0), ('joint2', 1), ('joint3', 2),
                          ('joint4', 3), ('joint5', 4), ('joint6', 5)):
            if name in abs_deg:
                target[idx] = math.radians(abs_deg[name])
    elif delta_deg:
        for name, idx in (('joint1', 0), ('joint2', 1), ('joint3', 2),
                          ('joint4', 3), ('joint5', 4), ('joint6', 5)):
            if name in delta_deg:
                target[idx] += math.radians(delta_deg[name])
    else:
        # Legacy named action fallback — small tip steps only (no big fixed whole-arm).
        return apply_action_to_joints(
            action, target, None,
            yaw_step_deg=10.0,
            pitch_step_deg=8.0,
            joint1_limit_deg=joint1_limit_deg,
            joint5_min_rad=joint5_min_rad,
            joint5_max_rad=joint5_max_rad,
            joint6_limit_rad=joint6_limit_rad,
        )

    limit = math.radians(joint1_limit_deg)
    j6_lim = abs(float(joint6_limit_rad))
    target[0] = _clamp(target[0], -limit, limit)
    target[1] = _clamp(target[1], joint2_min_rad, joint2_max_rad)
    target[2] = _clamp(target[2], joint3_min_rad, joint3_max_rad)
    target[4] = _clamp(target[4], joint5_min_rad, joint5_max_rad)
    if j6_lim <= 1e-9:
        target[5] = 0.0
    else:
        target[5] = _clamp(target[5], -j6_lim, j6_lim)
    return target


def apply_action_to_joints(
    action: str,
    joints: Sequence[float],
    obs: Optional[Dict[str, float]] = None,
    *,
    yaw_step_deg: float = 12.0,
    pitch_step_deg: float = 10.0,
    joint1_limit_deg: float = 150.0,
    joint5_min_rad: float = -1.4,
    joint5_max_rad: float = 0.2,
    plant_yaw: Optional[float] = None,
    face_plant_frac: float = 0.85,
    face_plant_min_yaw_deg: float = 25.0,
    face_plant_pitch_mult: float = 1.6,
    whole_arm_j1_deg: float = 20.0,
    whole_arm_j2_deg: float = DEFAULT_LEAN_J2_DEG,
    whole_arm_j3_deg: float = DEFAULT_LEAN_J3_DEG,
    whole_arm_j5_deg: float = 20.0,
    joint2_min_rad: float = -1.8,
    joint2_max_rad: float = 1.8,
    joint3_min_rad: float = -1.8,
    joint3_max_rad: float = 1.8,
    joint6_limit_rad: float = 0.0,
    fixed_dx_px: Optional[float] = None,
    fixed_has_view: Optional[bool] = None,
    restore_joints: Optional[Sequence[float]] = None,
) -> List[float]:
    """Legacy named-action applicator. Prefer apply_joint_command + set_joints."""
    del face_plant_frac, face_plant_min_yaw_deg, face_plant_pitch_mult, fixed_dx_px, fixed_has_view
    obs = obs or {}
    if plant_yaw is None:
        plant_yaw = float(obs.get('plant_yaw', joints[0] if joints else 0.0))

    if action == 'set_joints':
        # Should be routed via apply_joint_command; no-op keep pose.
        return [float(v) for v in joints]

    if action == 'restore_joints' and restore_joints is not None:
        return [float(v) for v in restore_joints]

    # Preferred legacy: face plant by absolute plant_yaw (not fixed overshoot step)
    if action in ('face_plant_yaw', 'whole_arm_right', 'whole_arm_left'):
        decision = {
            'action': 'set_joints',
            'joints_deg': estimate_joint_targets_deg(
                {
                    **obs,
                    'joint1': float(joints[0]) if joints else 0.0,
                    'joint2': float(joints[1]) if len(joints) > 1 else 0.0,
                    'joint3': float(joints[2]) if len(joints) > 2 else 0.0,
                    'joint5': float(joints[4]) if len(joints) > 4 else 0.0,
                    'plant_yaw': float(plant_yaw),
                },
            ),
        }
        # whole_arm_* only used for side hint when plant_yaw missing — still absolute aim
        return apply_joint_command(
            joints, decision,
            joint1_limit_deg=joint1_limit_deg,
            joint5_min_rad=joint5_min_rad,
            joint5_max_rad=joint5_max_rad,
            joint2_min_rad=joint2_min_rad,
            joint2_max_rad=joint2_max_rad,
            joint3_min_rad=joint3_min_rad,
            joint3_max_rad=joint3_max_rad,
        )

    target = [float(v) for v in joints]
    while len(target) < 6:
        target.append(0.0)
    yaw_step = math.radians(yaw_step_deg)
    pitch_step = math.radians(pitch_step_deg)
    if action == 'yaw_left':
        target[0] += yaw_step
    elif action == 'yaw_right':
        target[0] -= yaw_step
    elif action == 'pitch_down':
        target[4] -= pitch_step
    elif action == 'pitch_up':
        target[4] += pitch_step
    elif action == 'yaw_left_pitch_down':
        target[0] += yaw_step
        target[4] -= pitch_step
    elif action == 'yaw_right_pitch_down':
        target[0] -= yaw_step
        target[4] -= pitch_step
    elif action == 'yaw_left_pitch_up':
        target[0] += yaw_step
        target[4] += pitch_step
    elif action == 'yaw_right_pitch_up':
        target[0] -= yaw_step
        target[4] += pitch_step

    limit = math.radians(joint1_limit_deg)
    j6_lim = abs(float(joint6_limit_rad))
    target[0] = _clamp(target[0], -limit, limit)
    target[1] = _clamp(target[1], joint2_min_rad, joint2_max_rad)
    target[2] = _clamp(target[2], joint3_min_rad, joint3_max_rad)
    target[4] = _clamp(target[4], joint5_min_rad, joint5_max_rad)
    # joint6_limit_rad==0 → lock at 0; >0 → clamp to ±limit
    if j6_lim <= 1e-9:
        target[5] = 0.0
    else:
        target[5] = _clamp(target[5], -j6_lim, j6_lim)
    return target


def score_observation(obs: Dict[str, float], *, pitch_target_rad: float = -0.75) -> float:
    pitch_bonus = max(0.0, pitch_target_rad - float(obs.get('joint5', 0.0))) * 0.5
    yaw_penalty = abs(math.degrees(float(obs.get('yaw_error', 0.0)))) * 0.15
    angle_penalty = float(obs.get('ee_target_angle_deg', 180.0)) * 0.08
    fixed_err = fixed_mono_error_px(obs)
    fixed_penalty = 0.0
    if fixed_err is not None:
        fixed_penalty = (abs(fixed_err[0]) + abs(fixed_err[1])) * 0.04
    return pitch_bonus - yaw_penalty - angle_penalty - fixed_penalty


def verify_step(
    before: Dict[str, float],
    after: Dict[str, float],
    *,
    accept_score_margin: float = 0.15,
    min_joint_move_deg: float = 0.8,
    min_ee_angle_gain_deg: float = 3.0,
    min_fixed_err_improve_px: float = 20.0,
) -> Tuple[bool, Dict[str, object]]:
    """Numeric helper kept for logging; FSM now prefers agent coarse_ok after hold."""
    before_score = score_observation(before)
    after_score = score_observation(after)
    score_delta = after_score - before_score
    j1_delta_deg = abs(math.degrees(after.get('joint1', 0.0) - before.get('joint1', 0.0)))
    j2_delta_deg = abs(math.degrees(after.get('joint2', 0.0) - before.get('joint2', 0.0)))
    j3_delta_deg = abs(math.degrees(after.get('joint3', 0.0) - before.get('joint3', 0.0)))
    j5_delta_deg = abs(math.degrees(after.get('joint5', 0.0) - before.get('joint5', 0.0)))
    ee_gain = before.get('ee_target_angle_deg', 180.0) - after.get('ee_target_angle_deg', 180.0)
    moved = max(j1_delta_deg, j2_delta_deg, j3_delta_deg, j5_delta_deg) >= min_joint_move_deg
    yaw_after = abs(math.degrees(float(after.get('yaw_error', 0.0))))
    yaw_before = abs(math.degrees(float(before.get('yaw_error', 0.0))))

    fixed_before = fixed_mono_error_px(before)
    fixed_after = fixed_mono_error_px(after)
    fixed_improved = False
    if fixed_before is not None and fixed_after is not None:
        err_b = abs(fixed_before[0]) + abs(fixed_before[1])
        err_a = abs(fixed_after[0]) + abs(fixed_after[1])
        fixed_improved = err_a + min_fixed_err_improve_px <= err_b

    accepted = bool(
        fixed_mono_aligned(after)
        or fixed_improved
        or score_delta > accept_score_margin
        or (moved and ee_gain >= min_ee_angle_gain_deg)
        or (moved and yaw_after + 1.0 < yaw_before)
    )
    reason = 'fixed-mono aligned' if fixed_mono_aligned(after) else (
        'fixed-mono improved' if fixed_improved else (
            'improved observation' if accepted else 'no observable improvement'))
    return accepted, {
        'accepted': accepted,
        'before_score': before_score,
        'after_score': after_score,
        'score_delta': score_delta,
        'j1_delta_deg': j1_delta_deg,
        'j2_delta_deg': j2_delta_deg,
        'j3_delta_deg': j3_delta_deg,
        'j5_delta_deg': j5_delta_deg,
        'ee_angle_gain_deg': ee_gain,
        'moved': moved,
        'reason': reason,
    }


# ---------------------------------------------------------------------------
# VLM-backed decision (pbvs-vlm-reach-v2 branch)
# ---------------------------------------------------------------------------

_vlm_client = None   # module-level singleton, initialised on first call


def decide_action_vlm(
    obs: Dict[str, object],
    global_img,   # np.ndarray RGB, target circled in red
    wrist_img,    # np.ndarray RGB
    phase: str = 'command',
    **kwargs,
) -> Dict[str, object]:
    """VLM-backed alignment decision via Claude vision API.

    Falls back to decide_action() on any API or parse error so the robot
    never stalls.  Set --action-judge-source=vlm to activate.
    """
    global _vlm_client
    if _vlm_client is None:
        try:
            from vlm_align_client import VLMAlignClient
            _vlm_client = VLMAlignClient()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f'VLMAlignClient init failed ({exc}); using heuristic for this session')
            _vlm_client = False   # sentinel: don't retry import

    if not _vlm_client:
        return decide_action(obs, phase=phase, **kwargs)

    try:
        return _vlm_client.decide(global_img, wrist_img, obs, phase=phase)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            f'decide_action_vlm error ({exc}), falling back to heuristic')
        return decide_action(obs, phase=phase, **kwargs)


# ---------------------------------------------------------------------------
# BC policy decision (pbvs-vlm-reach-v2 Goal 4)
# ---------------------------------------------------------------------------

_bc_policy = None   # BCAlignPolicyInference singleton


def decide_action_bc(
    obs: Dict[str, object],
    global_img,    # np.ndarray RGB
    wrist_img,     # np.ndarray RGB
    model_path: str = '',
    phase: str = 'command',
    fail_count: int = 0,
    **kwargs,
) -> Dict[str, object]:
    """BC policy decision; VLM fallback when fail_count > 2, heuristic on error.

    Activate with --align-judge-mode bc --bc-model-path <path>.
    """
    global _bc_policy

    # After repeated failures let VLM try to recover.
    if fail_count > 2:
        import logging
        logging.getLogger(__name__).info(
            f'BC: fail_count={fail_count} > 2 → VLM recovery')
        return decide_action_vlm(obs, global_img, wrist_img, phase=phase, **kwargs)

    if _bc_policy is None:
        if not model_path or not __import__('os').path.exists(model_path):
            import logging
            logging.getLogger(__name__).warning(
                f'BC model not found at {model_path!r}; falling back to heuristic')
            return decide_action(obs, phase=phase, **kwargs)
        try:
            from bc_align_policy import BCAlignPolicyInference
            _bc_policy = BCAlignPolicyInference(model_path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f'BC model load failed ({exc}); using heuristic')
            return decide_action(obs, phase=phase, **kwargs)

    try:
        return _bc_policy.decide(global_img, wrist_img, obs, phase=phase)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            f'BC inference error ({exc}); using heuristic')
        return decide_action(obs, phase=phase, **kwargs)
