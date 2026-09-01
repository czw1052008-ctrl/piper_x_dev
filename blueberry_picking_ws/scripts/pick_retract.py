#!/usr/bin/env python3
"""Retract cup along +surface_normal after PBVS touch (pull berry off stem)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from pick_motion import ARM_JOINTS, read_current_joints, send_joint_trajectory

SCRIPTS_DIR = Path(__file__).resolve().parent


def load_surface_normal(qa_session_dir: Path) -> Optional[np.ndarray]:
    arrive = qa_session_dir / 'pbvs_direct_arrive.json'
    if not arrive.is_file():
        return None
    with open(arrive, 'r', encoding='utf-8') as f:
        data = json.load(f)
    n = data.get('n_base')
    if not n or len(n) < 3:
        return None
    n_a = np.asarray(n, dtype=np.float64).flatten()[:3]
    nn = float(np.linalg.norm(n_a))
    if nn < 1e-9:
        return None
    return n_a / nn


def retract_along_normal(
    node,
    n_base: Sequence[float],
    *,
    retract_m: float = 0.05,
    traj_s: float = 0.8,
    settle_s: float = 0.2,
) -> dict:
    """Move EE +retract_m along outward surface normal; keep orientation."""
    from piper_position_ik import fk_xyz, position_ik_keep_orient, position_ik_keep_orient_chunked

    joints = read_current_joints(node)
    if joints is None:
        return {'ok': False, 'error': 'no_joint_states'}

    tcp = fk_xyz(joints)
    if tcp is None:
        return {'ok': False, 'error': 'fk_failed'}

    n_a = np.asarray(n_base, dtype=np.float64).flatten()[:3]
    nn = float(np.linalg.norm(n_a))
    if nn < 1e-9:
        return {'ok': False, 'error': 'bad_normal'}
    n_a = n_a / nn

    d_base = n_a * float(retract_m)
    tgt = tuple((np.asarray(tcp, dtype=np.float64) + d_base).tolist())

    ik_joints = position_ik_keep_orient(tgt, joints)
    ik_method = 'keep_orient'
    if ik_joints is None:
        ik_joints = position_ik_keep_orient_chunked(tgt, joints)
        ik_method = 'keep_orient_chunked'
    if ik_joints is None:
        return {'ok': False, 'error': 'ik_failed', 'tgt': list(tgt), 'n_base': n_a.tolist()}

    ok = send_joint_trajectory(node, ik_joints, traj_s=traj_s, settle_s=settle_s)
    result = {
        'ok': ok,
        'retract_m': float(retract_m),
        'traj_s': float(traj_s),
        'n_base': n_a.tolist(),
        'd_base': d_base.tolist(),
        'tcp_start': list(tcp),
        'tcp_goal': list(tgt),
        'ik_method': ik_method,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    return result


def retract_from_qa_session(
    node,
    qa_session_dir: Path,
    *,
    retract_m: float = 0.05,
    traj_s: float = 0.8,
) -> dict:
    n = load_surface_normal(qa_session_dir)
    if n is None:
        return {'ok': False, 'error': f'no n_base in {qa_session_dir}/pbvs_direct_arrive.json'}
    result = retract_along_normal(node, n, retract_m=retract_m, traj_s=traj_s)
    out = qa_session_dir / 'pick_retract_arrive.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    return result
