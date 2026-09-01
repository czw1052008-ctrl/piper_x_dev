#!/usr/bin/env python3
"""Offline RL smoke: random trajectory search vs teacher on PlannerSimEnv.

Not a full trainer — verifies reward loop + obstacle collision signal.
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np

from picking_msgs.msg import PlannerTaskContext
from planner_sim_env import PlannerSimEnv, planner_input_from_dict, load_planner_inputs_from_bag
from tool_trajectory_utils import build_tool_trajectory_4s


def random_perturbed_traj(env: PlannerSimEnv, *, scale_m: float, seed: int):
    tip0 = np.asarray(
        __import__('piper_position_ik', fromlist=['tip_xyz']).tip_xyz(
            env.planner_input.ego.q_rad,
            tip_offset_link6=env.tip_offset_link6),
        dtype=np.float64)
    goal, axis = env.goal_tip_and_axis()
    rng = np.random.default_rng(seed)
    offset = rng.normal(0.0, scale_m, size=3)
    goal_p = goal + offset
    return build_tool_trajectory_4s(
        tip0, goal_p, axis,
        move_duration_s=1.0,
        execute_until_index=env.execute_until_index,
    )


def run_smoke(env: PlannerSimEnv, *, n_trials: int, seed: int) -> None:
    teacher = env.evaluate_teacher()
    print(f'teacher: reward={teacher.reward:.3f} ok={teacher.ok} {teacher.detail}')

    best_r = -1e9
    best_detail = ''
    rng = np.random.default_rng(seed)
    for i in range(n_trials):
        scale = float(rng.uniform(0.005, 0.04))
        traj = random_perturbed_traj(env, scale_m=scale, seed=int(rng.integers(1e9)))
        res = env.rollout_from_tool_trajectory(traj)
        if res.reward > best_r:
            best_r = res.reward
            best_detail = res.detail
    print(f'random search n={n_trials}: best_reward={best_r:.3f} ({best_detail})')
    print('RL smoke OK — reward loop functional')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', default='')
    parser.add_argument('--trials', type=int, default=48)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.bag:
        msgs = load_planner_inputs_from_bag(args.bag, max_msgs=5)
        if not msgs:
            print('empty bag')
            return 1
        env = PlannerSimEnv(msgs[0])
    else:
        env = PlannerSimEnv(planner_input_from_dict({
            'cluster': [0.32, 0.05, 0.28],
            'berry': [0.30, 0.06, 0.27],
            'fsm_state': PlannerTaskContext.FSM_APPROACH_FRUIT,
            'fruit_id': 1,
            'obstacles': [
                {'position': [0.29, 0.04, 0.265], 'radius_m': 0.045},
                {'position': [0.31, 0.08, 0.275], 'radius_m': 0.04},
            ],
        }))

    run_smoke(env, n_trials=args.trials, seed=args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
