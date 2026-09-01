#!/usr/bin/env python3
"""Shared ROS2 motion helpers for the pick-cycle orchestrator."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_ENTRY_POSE = SCRIPTS_DIR.parent / 'log' / 'real_robot' / 'refine_entry_pose.json'


class ReachClient:
    """Thin wrapper around /reach/cmd and /reach/status."""

    def __init__(self, node) -> None:
        from std_msgs.msg import String

        self._node = node
        self._String = String
        self._reach_state = 'IDLE'
        self._cmd_pub = node.create_publisher(String, '/reach/cmd', 10)
        node.create_subscription(String, '/reach/status', self._on_status, 10)

    def _on_status(self, msg) -> None:
        status = msg.data.strip()
        self._reach_state = status.split(':', 1)[0].strip().upper()

    @property
    def reach_state(self) -> str:
        return self._reach_state

    def spin(self, secs: float) -> None:
        import rclpy

        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def send_cmd(self, cmd: str, *, repeat: int = 5) -> None:
        msg = self._String()
        msg.data = cmd
        for _ in range(repeat):
            self._cmd_pub.publish(msg)
            self.spin(0.08)

    def wait_reach_state(self, targets: Set[str], timeout_s: float) -> str:
        import rclpy

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._reach_state in targets:
                return self._reach_state
        return 'TIMEOUT'

    def restore_entry_pose(
        self,
        path: Path = DEFAULT_ENTRY_POSE,
        *,
        traj_s: float = 2.0,
    ) -> bool:
        cmd = [
            'python3', str(SCRIPTS_DIR / 'refine_entry_pose.py'),
            'restore', '--path', str(path),
            '--traj-s', str(traj_s),
            '--skip-if-within-deg', '3.0',
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.returncode != 0 and proc.stderr:
            print(proc.stderr.rstrip())
        return proc.returncode == 0

    def start_refine(self) -> None:
        self.send_cmd('start_refine', repeat=8)

    def next_fruit(self) -> None:
        self.send_cmd('next_fruit', repeat=8)

    def abort(self) -> None:
        self.send_cmd('abort', repeat=8)

    def confirm_reset_home(self) -> bool:
        self.send_cmd('confirm_reset', repeat=8)
        state = self.wait_reach_state({'IDLE', 'RESETTING'}, timeout_s=8.0)
        if state == 'RESETTING':
            state = self.wait_reach_state({'IDLE'}, timeout_s=20.0)
        return state == 'IDLE'


def read_current_joints(node, timeout_s: float = 4.0) -> Optional[List[float]]:
    from sensor_msgs.msg import JointState

    got: dict = {'js': None}

    def cb(msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(j in name_to_pos for j in ARM_JOINTS):
            got['js'] = [float(name_to_pos[j]) for j in ARM_JOINTS]

    sub = node.create_subscription(JointState, '/feedback/joint_states', cb, 10)
    import rclpy

    t0 = time.time()
    while got['js'] is None and time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_subscription(sub)
    return got['js']


def send_joint_trajectory(
    node,
    target: Sequence[float],
    *,
    traj_s: float = 1.0,
    settle_s: float = 0.3,
) -> bool:
    """Send FollowJointTrajectory to /arm_controller/follow_joint_trajectory."""
    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    client = ActionClient(node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
    if not client.wait_for_server(timeout_sec=5.0):
        return False

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINTS)
    pt = JointTrajectoryPoint()
    pt.positions = [float(v) for v in target[:6]]
    pt.time_from_start.sec = int(traj_s)
    pt.time_from_start.nanosec = int((traj_s % 1.0) * 1e9)
    goal.trajectory.points = [pt]

    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=traj_s + 5.0)
    if not fut.done():
        return False
    gh = fut.result()
    if gh is None or not gh.accepted:
        return False
    res_fut = gh.get_result_async()
    rclpy.spin_until_future_complete(node, res_fut, timeout_sec=traj_s + settle_s + 5.0)
    time.sleep(settle_s)
    return True
