"""Synthetic alignment data generator.

Moves the arm to random joint configurations and runs ALIGNING
(with VLM or heuristic as teacher) to collect diverse training episodes.

The AlignDataCollector in reach_fsm_node saves each episode automatically.

Requires: reach_fsm_node running with --collect-data enabled.

Usage:
    # Terminal 1: launch the FSM with data collection enabled
    python reach_fsm_node.py --collect-data --align-judge-mode vlm ...

    # Terminal 2: run the generator
    python generate_align_data.py \
        --n-episodes 500 \
        --teacher vlm \
        --joint1-range -60 60 \
        --joint2-range -30 10 \
        --joint3-range 0 40 \
        --joint5-range -60 -10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from typing import List, Optional, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)

# FSM state constants (must match reach_fsm_node).
_FSM_SUCCESS_STATES = {'REFINING', 'WAIT_CONFIRM'}
_FSM_FAIL_STATES    = {'ERROR', 'IDLE'}
_TIMEOUT_S = 60.0

# Safe workspace guards (metres from base).
_MIN_EE_HEIGHT_M = 0.05   # don't let EE go below 5 cm
_MIN_DIST_FROM_BASE_M = 0.15


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def sample_random_joints(
    j1_range: Tuple[float, float],
    j2_range: Tuple[float, float],
    j3_range: Tuple[float, float],
    j5_range: Tuple[float, float],
) -> dict:
    """Sample a random joint configuration (degrees)."""
    return {
        'joint1': random.uniform(*j1_range),
        'joint2': random.uniform(*j2_range),
        'joint3': random.uniform(*j3_range),
        'joint4': 0.0,
        'joint5': random.uniform(*j5_range),
        'joint6': 0.0,
    }


class AlignDataGenerator:
    """ROS2-based generator: moves arm, triggers FSM, waits for result."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._node = None
        self._fsm_state: str = 'IDLE'
        self._consecutive_errors = 0

        self._init_ros()

    def _init_ros(self) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        from sensor_msgs.msg import JointState

        rclpy.init()
        self._node = rclpy.create_node('align_data_generator')

        # Subscribe to FSM state topic.
        self._node.create_subscription(
            String,
            '/reach/fsm_state',
            self._fsm_state_cb,
            10,
        )

        # Publisher for FSM commands.
        self._cmd_pub = self._node.create_publisher(String, '/reach/command', 10)

        # Publisher for arm joint targets (position mode).
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        self._traj_pub = self._node.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10,
        )
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self._String = String

        _LOG.info('AlignDataGenerator ROS2 node initialised')

    def _fsm_state_cb(self, msg) -> None:
        self._fsm_state = msg.data.strip().upper()

    def _wait_for_state(
        self,
        target_states: set,
        timeout_s: float = _TIMEOUT_S,
    ) -> str:
        import rclpy
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._fsm_state in target_states:
                return self._fsm_state
        return 'TIMEOUT'

    def _send_cmd(self, cmd: str) -> None:
        msg = self._String()
        msg.data = cmd
        self._cmd_pub.publish(msg)

    def _move_to_joints(self, joints_deg: dict, duration_s: float = 3.0) -> None:
        """Send a joint trajectory command to move arm to specified angles."""
        from builtin_interfaces.msg import Duration as RosDuration

        traj = self._JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        pt = self._JointTrajectoryPoint()
        pt.positions = [
            math.radians(joints_deg.get(f'joint{i}', 0.0))
            for i in range(1, 7)
        ]
        pt.time_from_start.sec = int(duration_s)
        pt.time_from_start.nanosec = int((duration_s % 1) * 1e9)
        traj.points = [pt]

        self._traj_pub.publish(traj)
        time.sleep(duration_s + 0.5)   # wait for motion to complete

    def run(self) -> None:
        args = self._args
        j1_range = (args.joint1_range[0], args.joint1_range[1])
        j2_range = (args.joint2_range[0], args.joint2_range[1])
        j3_range = (args.joint3_range[0], args.joint3_range[1])
        j5_range = (args.joint5_range[0], args.joint5_range[1])

        success_count = 0
        fail_count    = 0

        for ep_idx in range(args.n_episodes):
            print(f'\n[{ep_idx+1}/{args.n_episodes}] success={success_count} fail={fail_count}')

            if self._consecutive_errors >= 3:
                print('ERROR: 3 consecutive FSM errors — pausing for operator.')
                input('  Press Enter to continue, Ctrl-C to abort.')
                self._consecutive_errors = 0

            # 1. Reset FSM to IDLE.
            self._send_cmd('reset')
            final = self._wait_for_state({'IDLE'}, timeout_s=15.0)
            if final != 'IDLE':
                _LOG.warning(f'Could not reach IDLE (state={self._fsm_state}); skipping')
                fail_count += 1
                self._consecutive_errors += 1
                continue

            # 2. Move arm to random start.
            joints = sample_random_joints(j1_range, j2_range, j3_range, j5_range)
            print(f'  Random joints: {", ".join(f"{k}={v:.1f}°" for k,v in joints.items() if k != "joint4" and k != "joint6")}')
            self._move_to_joints(joints, duration_s=args.move_duration_s)

            # 3. Trigger lock + align cycle (FSM needs a target berry locked first).
            self._send_cmd('start')
            final = self._wait_for_state(
                _FSM_SUCCESS_STATES | _FSM_FAIL_STATES | {'LOCKING', 'ALIGNING'},
                timeout_s=10.0,
            )
            if final in ('ERROR', 'TIMEOUT'):
                _LOG.warning(f'Lock failed (state={final}); skipping episode')
                fail_count += 1
                self._consecutive_errors += 1
                continue

            # 4. Wait for ALIGNING to complete.
            final = self._wait_for_state(
                _FSM_SUCCESS_STATES | {'ERROR'},
                timeout_s=_TIMEOUT_S,
            )

            if final in _FSM_SUCCESS_STATES:
                print(f'  → SUCCESS (state={final})')
                success_count += 1
                self._consecutive_errors = 0
            else:
                print(f'  → FAIL (state={final})')
                fail_count += 1
                self._consecutive_errors += 1

        print(f'\nGeneration complete: {success_count} success / {fail_count} fail '
              f'out of {args.n_episodes} episodes.')
        print(f'Success rate: {success_count/max(1,args.n_episodes)*100:.1f}%')

        import rclpy
        self._node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser(description='Generate synthetic alignment training data')
    ap.add_argument('--n-episodes',   type=int,   default=200,
                    help='Number of episodes to collect')
    ap.add_argument('--teacher',      choices=('vlm', 'heuristic'), default='vlm',
                    help='Decision source for ALIGNING (must match FSM launch args)')
    ap.add_argument('--joint1-range', type=float, nargs=2, default=[-60.0,  60.0])
    ap.add_argument('--joint2-range', type=float, nargs=2, default=[-30.0,  10.0])
    ap.add_argument('--joint3-range', type=float, nargs=2, default=[  0.0,  40.0])
    ap.add_argument('--joint5-range', type=float, nargs=2, default=[-60.0, -10.0])
    ap.add_argument('--move-duration-s', type=float, default=3.0,
                    help='Seconds to allow for each random-start motion')
    args = ap.parse_args()

    gen = AlignDataGenerator(args)
    gen.run()
