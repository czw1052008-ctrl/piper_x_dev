from __future__ import annotations

import json
import math
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from align_judge import (  # noqa: E402
    apply_joint_command,
    decide_action,
    estimate_joint_targets_deg,
    inverse_action,
    is_target_visible,
    norm_angle,
    score_observation,
    verify_step,
)
from align_vlm_decider import run as run_decider  # noqa: E402


def test_norm_angle_wraps():
    assert abs(norm_angle(3.5) + 2.7831853071795862) < 1e-9
    assert abs(norm_angle(-3.5) - 2.7831853071795862) < 1e-9


def test_command_phase_estimates_absolute_set_joints():
    obs = {
        'joint1': 0.0,
        'joint2': 0.0,
        'joint3': 0.0,
        'yaw_error': 0.4,
        'joint5': 0.0,
        'fine_visible': 0.0,
        'fine_confidence': 0.0,
        'ee_target_angle_deg': 90.0,
        'plant_yaw': 0.4,
    }
    out = decide_action(obs, phase='command')
    assert out['action'] == 'set_joints'
    assert 'joints_deg' in out
    # Aim joint1 at plant_yaw (~22.9deg), not a fixed ±35 overshoot step
    assert abs(out['joints_deg']['joint1'] - math.degrees(0.4)) < 0.5


def test_set_joints_locks_joint6_at_zero():
    out = apply_joint_command(
        [0.0] * 6,
        {'action': 'set_joints', 'joints_deg': {'joint6': 45.0}},
    )
    assert abs(out[5]) < 1e-9
    out2 = apply_joint_command(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        {'action': 'set_joints', 'joints_deg': {'joint1': 10.0}},
    )
    assert abs(out2[5]) < 1e-9


def test_judge_phase_coarse_ok_when_wrist_visible():
    obs = {
        'joint1': 0.4,
        'yaw_error': 0.02,
        'joint5': -0.75,
        'fine_visible': 1.0,
        'fine_confidence': 0.6,
        'ee_target_angle_deg': 46.0,
        'joint2_deg': 6.0,
        'plant_yaw': 0.42,
        'fixed_has_view': 1.0,
        'fixed_dx_px': -70.0,
        'fixed_dy_px': 300.0,
    }
    out = decide_action(obs, phase='judge', yaw_deadband_deg=8.0)
    assert out['action'] == 'coarse_ok'


def test_judge_phase_rejects_folded_arm_even_if_yaw_ok():
    obs = {
        'joint1': 0.4,
        'joint2': math.radians(98.0),
        'yaw_error': 0.02,
        'joint5': -0.75,
        'fine_visible': 0.0,
        'fine_confidence': 0.0,
        'joint2_deg': 98.0,
        'ee_target_angle_deg': 87.0,
        'plant_yaw': 0.42,
        'fixed_has_view': 1.0,
        'fixed_dx_px': -40.0,
        'fixed_dy_px': 301.0,
    }
    out = decide_action(obs, phase='judge', yaw_deadband_deg=8.0)
    assert out['action'] == 'set_joints'
    assert out['joints_deg']['joint2'] < 98.0


def test_judge_phase_reaims_when_off():
    obs = {
        'joint1': 0.0,
        'joint2': 0.0,
        'joint3': 0.0,
        'yaw_error': 0.5,
        'joint5': 0.0,
        'fine_visible': 0.0,
        'fine_confidence': 0.0,
        'ee_target_angle_deg': 60.0,
        'plant_yaw': 0.5,
    }
    out = decide_action(obs, phase='judge')
    assert out['action'] == 'set_joints'
    assert abs(out['joints_deg']['joint1'] - math.degrees(0.5)) < 0.5


def test_visible_target_coarse_ok_at_judge():
    obs = {
        'fine_visible': 1.0,
        'fine_confidence': 0.8,
        'joint1': 0.0,
        'plant_yaw': 0.5,
        'yaw_error': 0.5,
        'joint2_deg': 8.0,
        'ee_target_angle_deg': 40.0,
        'fixed_has_view': 1.0,
        'fixed_dx_px': 200.0,
        'fixed_dy_px': 250.0,
    }
    assert is_target_visible(obs, fine_visible_conf=0.2)
    out = decide_action(obs, phase='judge')
    assert out['action'] == 'coarse_ok'


def test_command_still_set_joints_when_arm_folded():
    obs = {
        'fine_visible': 0.0,
        'fine_confidence': 0.0,
        'joint1': 0.0,
        'plant_yaw': 0.5,
        'yaw_error': 0.5,
        'fixed_has_view': 1.0,
        'fixed_dx_px': 200.0,
        'fixed_dy_px': 250.0,
        'joint2_deg': 95.0,
        'ee_target_angle_deg': 85.0,
    }
    out = decide_action(obs, phase='command')
    assert out['action'] == 'set_joints'


def test_estimate_targets_use_fixed_mono_pixels():
    obs = {
        'joint1': math.radians(28.6),
        'joint2': math.radians(97.6),
        'joint3': math.radians(-86.4),
        'joint5': math.radians(-43.0),
        'plant_yaw': math.radians(28.5),
        'fixed_has_view': 1.0,
        'fixed_dx_px': -101.0,
        'fixed_dy_px': 301.0,
    }
    t = estimate_joint_targets_deg(obs, pitch_target_rad=-0.75)
    assert t['joint1'] > 28.6
    assert t['joint2'] < 97.6
    assert t['joint3'] > -86.4


def test_apply_joint_command_absolute():
    joints = [0.1, 0.0, 0.0, 0.0, -0.2, 0.0]
    target = apply_joint_command(
        joints,
        {'action': 'set_joints', 'joints_deg': {'joint1': 25.0, 'joint5': -40.0}},
    )
    assert abs(math.degrees(target[0]) - 25.0) < 0.2
    assert abs(math.degrees(target[4]) + 40.0) < 0.2


def test_apply_joint_command_delta():
    joints = [0.1, 0.0, 0.0, 0.0, -0.2, 0.0]
    target = apply_joint_command(
        joints,
        {'action': 'set_joints', 'delta_deg': {'joint1': 10.0, 'joint5': -15.0}},
    )
    assert abs(math.degrees(target[0] - joints[0]) - 10.0) < 0.2
    assert abs(math.degrees(target[4] - joints[4]) + 15.0) < 0.2


def test_estimate_targets_follow_plant_yaw():
    obs = {'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0, 'joint5': 0.0, 'plant_yaw': 0.45}
    t = estimate_joint_targets_deg(obs, pitch_target_rad=-0.75)
    assert abs(t['joint1'] - math.degrees(0.45)) < 0.2
    assert abs(t['joint5'] - math.degrees(-0.75)) < 0.2


def test_inverse_set_joints_is_restore():
    assert inverse_action('set_joints') == 'restore_joints'
    assert inverse_action('coarse_ok') == 'done'


def test_score_prefers_lower_yaw():
    bad = {'fine_visible': 0.0, 'fine_confidence': 0.0, 'yaw_error': 0.5, 'joint5': -0.5, 'ee_target_angle_deg': 80.0}
    good = {'fine_visible': 0.0, 'fine_confidence': 0.0, 'yaw_error': 0.1, 'joint5': -0.5, 'ee_target_angle_deg': 20.0}
    assert score_observation(good) > score_observation(bad)


def test_verify_accepts_fixed_mono_improvement():
    before = {
        'fine_visible': 0.0, 'fine_confidence': 0.0, 'joint1': 0.0, 'joint5': -0.2,
        'yaw_error': 0.4, 'ee_target_angle_deg': 40.0,
        'fixed_has_view': 1.0, 'fixed_dx_px': -120.0, 'fixed_dy_px': 280.0,
    }
    after = {
        'fine_visible': 0.0, 'fine_confidence': 0.0, 'joint1': 0.35, 'joint5': -0.7,
        'yaw_error': 0.05, 'ee_target_angle_deg': 15.0,
        'fixed_has_view': 1.0, 'fixed_dx_px': -60.0, 'fixed_dy_px': 180.0,
    }
    accepted, verdict = verify_step(before, after)
    assert accepted
    assert verdict['reason'] == 'fixed-mono improved'


def test_verify_accepts_yaw_improvement():
    before = {
        'fine_visible': 0.0, 'fine_confidence': 0.0, 'joint1': 0.0, 'joint5': -0.2,
        'yaw_error': 0.4, 'ee_target_angle_deg': 40.0,
    }
    after = {
        'fine_visible': 0.0, 'fine_confidence': 0.0, 'joint1': 0.35, 'joint5': -0.7,
        'yaw_error': 0.05, 'ee_target_angle_deg': 15.0,
    }
    accepted, verdict = verify_step(before, after)
    assert accepted
    assert verdict['moved']


def test_align_vlm_decider_once_writes_set_joints(tmp_path: Path):
    qa_dir = tmp_path / 'qa'
    session = qa_dir / '20260804_160500'
    session.mkdir(parents=True)
    req = session / 'align_00_request.json'
    req.write_text(json.dumps({
        'step_idx': 0,
        'phase': 'command',
        'failed_actions': [],
        'observation': {
            'joint1': 0.0,
            'joint2': 0.0,
            'joint3': 0.0,
            'yaw_error': 0.6,
            'joint5': 0.0,
            'fine_visible': 0.0,
            'fine_confidence': 0.0,
            'ee_target_angle_deg': 90.0,
            'plant_yaw': 0.6,
        },
    }), encoding='utf-8')
    args = Namespace(
        qa_dir=str(qa_dir),
        session_dir=str(session),
        provider='heuristic',
        poll_s=0.01,
        once=True,
        yaw_deadband_deg=8.0,
        pitch_target_rad=-0.75,
        fine_visible_conf=0.2,
        ee_angle_trigger_deg=20.0,
    )
    assert run_decider(args) == 0
    out = json.loads((session / 'align_decision.json').read_text(encoding='utf-8'))
    assert out['step_idx'] == 0
    assert out['action'] == 'set_joints'
    assert 'joints_deg' in out
