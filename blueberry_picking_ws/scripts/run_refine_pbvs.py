#!/usr/bin/env python3
"""Standalone REFINING validator for the PBVS+KF pipeline (pbvs-vlm-reach-v2).

Replaces _run_refine_contact.sh for the new PBVS path.
The old pipeline needed REFINING_WAIT_NEAR + confirm_near.
PBVS handles approach fully autonomously — no human confirmation required.

Prerequisites (Ubuntu, AGX robot):
  1. Arm bringup running (agx_arm_ros)
  2. fine_detector_node running (see _run_refine_pbvs.sh to start everything)
  3. reach_fsm_node running with --use-pbvs  (launched by _run_refine_pbvs.sh)
  4. refine_entry_pose.json exists:
       python scripts/refine_entry_pose.py save   # arm must be at a good ALIGN exit pose

Usage:
  # Single validation run (restores entry pose, sends start_refine, monitors)
  python scripts/run_refine_pbvs.py

  # Repeat N times (stress-test / data collection)
  python scripts/run_refine_pbvs.py --loops 5

  # Skip pose restore (arm already at entry pose)
  python scripts/run_refine_pbvs.py --no-restore

  # Collect BC training data for each loop (FSM must have --collect-data)
  python scripts/run_refine_pbvs.py --loops 10 --collect-data-dir data/align_episodes

PBVS state progression (printed live):
  REFINING (INIT) → SERVO → APPROACH → WAIT_CONFIRM (contact ✓)
                                     └→ ERROR (something wrong)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import Optional

_LOG = logging.getLogger(__name__)

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_WS_DIR = os.path.abspath(os.path.join(_SCRIPTS_DIR, '..'))

# FSM commands / states
_CMD_ABORT        = 'abort'
_CMD_CONFIRM_RESET = 'confirm_reset'
_CMD_START_REFINE = 'start_refine'
_STATE_IDLE       = 'IDLE'
_STATE_RESETTING  = 'RESETTING'
_STATE_REFINING   = 'REFINING'
_STATE_DONE       = 'WAIT_CONFIRM'
_STATE_ERROR      = 'ERROR'
_TERMINAL_STATES  = {_STATE_DONE, _STATE_ERROR, _STATE_IDLE}


# ---------------------------------------------------------------------------
# ROS2 helpers
# ---------------------------------------------------------------------------

class RefineRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._fsm_state = _STATE_IDLE
        self._node = None
        self._fine_det_rate = 0.0
        self._fine_det_count = 0
        self._fine_det_t0 = time.time()

        self._init_ros()

    def _init_ros(self) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        from picking_msgs.msg import DetectedBerryArray

        rclpy.init()
        self._node = Node('run_refine_pbvs')
        self._String = String

        # FSM publishes state on /reach/status (e.g. "REFINING" or "ERROR:…").
        # /reach/fsm_state is not published by reach_fsm_node.
        self._node.create_subscription(
            String, '/reach/status', self._on_status, 10)

        # Subscribe to fine detector to show detection rate.
        try:
            self._node.create_subscription(
                DetectedBerryArray, '/perception/fine/detections',
                self._on_fine_det, 10)
        except Exception:
            pass  # picking_msgs may not be available in all envs

        # Command publisher.
        self._cmd_pub = self._node.create_publisher(String, '/reach/cmd', 10)

        _LOG.info('RefineRunner ROS2 node ready')

    def _on_status(self, msg) -> None:
        status = msg.data.strip()
        # Status payload is "<STATE>" or "<STATE>:<error detail>".
        state = status.split(':', 1)[0].strip().upper()
        if state:
            self._fsm_state = state
        # Print FSM status lines that contain PBVS keywords.
        if any(k in status for k in ('PBVS', 'SERVO', 'APPROACH', 'probe',
                                      'DONE', 'contact', 'cup_dist', 'KF',
                                      'ERROR')):
            print(f'  [status] {status}')

    def _on_fine_det(self, msg) -> None:
        self._fine_det_count += 1

    def _spin(self, secs: float) -> None:
        import rclpy
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def _send_cmd(self, cmd: str, repeat: int = 5) -> None:
        msg = self._String()
        msg.data = cmd
        for _ in range(repeat):
            self._cmd_pub.publish(msg)
            self._spin(0.1)

    def _wait_for_state(self, target: set, timeout_s: float) -> str:
        import rclpy
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._fsm_state in target:
                return self._fsm_state
        return 'TIMEOUT'

    # ------------------------------------------------------------------

    def confirm_reset_arm(self) -> bool:
        """Home the arm when the previous shot ended at WAIT_CONFIRM / ERROR."""
        self._spin(0.4)
        st = self._fsm_state
        if st not in (_STATE_DONE, _STATE_ERROR):
            return True
        print(f'  FSM: {st} → confirm_reset (arm → home) …')
        self._send_cmd(_CMD_CONFIRM_RESET, repeat=8)
        state = self._wait_for_state({_STATE_IDLE, _STATE_RESETTING}, timeout_s=6.0)
        if state == _STATE_RESETTING:
            state = self._wait_for_state({_STATE_IDLE}, timeout_s=30.0)
        if state != _STATE_IDLE:
            print(f'[warn] confirm_reset did not reach IDLE (state={state})')
            return False
        print('  FSM: IDLE ✓ (after confirm_reset)')
        return True

    def restore_entry_pose(self) -> bool:
        pose_path = os.path.join(_WS_DIR, 'log', 'real_robot', 'refine_entry_pose.json')
        if not os.path.exists(pose_path):
            print(f'[warn] Entry pose not found: {pose_path}')
            print('       Run:  python scripts/refine_entry_pose.py save')
            return False

        print(f'Restoring entry pose from {pose_path} …')
        ret = subprocess.run(
            [sys.executable, os.path.join(_SCRIPTS_DIR, 'refine_entry_pose.py'),
             'restore', '--traj-s', '4.0', '--settle-s', '1.5'],
            cwd=_WS_DIR,
        )
        if ret.returncode != 0:
            print(f'[warn] refine_entry_pose restore exited {ret.returncode}')
            return False
        print('Entry pose restored.')
        return True

    def enable_arm(self) -> None:
        import rclpy
        from std_srvs.srv import SetBool
        for srv_name in ('/enable_agx_arm', '/control_enable'):
            cli = self._node.create_client(SetBool, srv_name)
            if cli.wait_for_service(timeout_sec=3.0):
                req = SetBool.Request()
                req.data = True
                fut = cli.call_async(req)
                rclpy.spin_until_future_complete(self._node, fut, timeout_sec=5.0)
                _LOG.info(f'{srv_name} → {fut.result()}')
            else:
                _LOG.warning(f'{srv_name} service not available')

    def run_one(self, loop_idx: int) -> bool:
        print(f'\n{"─"*60}')
        print(f'Loop {loop_idx} — prepare for REFINING …')

        self._spin(0.3)

        # 1. Home arm when previous loop ended at contact (WAIT_CONFIRM) or ERROR.
        if not self._args.no_reset:
            if not self.confirm_reset_arm():
                if not self._args.ignore_pose_error:
                    return False
        else:
            print('  (skip confirm_reset — --no-reset)')

        # 2. FSM → IDLE (abort only if still active; confirm_reset already yields IDLE).
        if self._fsm_state not in (_STATE_IDLE, _STATE_DONE, _STATE_ERROR):
            print(f'  FSM: {self._fsm_state} → abort …')
            self._send_cmd(_CMD_ABORT)
            state = self._wait_for_state({_STATE_IDLE}, timeout_s=10.0)
            if state != _STATE_IDLE:
                print(f'[fail] Could not reach IDLE (state={self._fsm_state})')
                return False
        print(f'  FSM: IDLE ✓')

        # 3. Restore entry pose (ALIGN exit / refine entry).
        if not self._args.no_restore:
            if not self.restore_entry_pose():
                if not self._args.ignore_pose_error:
                    return False
            self._spin(0.5)
        else:
            print('  (skip restore — --no-restore)')

        # 4. Send start_refine.
        print('Sending start_refine …')
        self._send_cmd(_CMD_START_REFINE, repeat=6)
        self._spin(0.5)

        state = self._wait_for_state(
            {_STATE_REFINING, _STATE_DONE, _STATE_ERROR}, timeout_s=8.0)
        if state != _STATE_REFINING:
            print(f'[fail] FSM did not enter REFINING (state={state})')
            return False
        print(f'  FSM: REFINING ✓  — PBVS running …')

        # 4. Monitor PBVS progression.
        t0 = time.time()
        prev_state = _STATE_REFINING
        self._fine_det_count = 0
        self._fine_det_t0 = time.time()

        while True:
            import rclpy
            rclpy.spin_once(self._node, timeout_sec=0.1)
            elapsed = time.time() - t0

            if self._fsm_state != prev_state:
                print(f'  t={elapsed:.1f}s  FSM: {prev_state} → {self._fsm_state}')
                prev_state = self._fsm_state

            if self._fsm_state in _TERMINAL_STATES:
                break

            if elapsed > self._args.timeout_s:
                print(f'[timeout] REFINING did not finish in {self._args.timeout_s}s')
                self._send_cmd(_CMD_ABORT)
                return False

            # Periodic progress print.
            if int(elapsed) % 5 == 0 and elapsed > 0:
                det_rate = self._fine_det_count / max(1e-3, time.time() - self._fine_det_t0)
                print(f'  t={elapsed:.0f}s  state={self._fsm_state}  '
                      f'det={det_rate:.1f}Hz', end='\r', flush=True)

        print()  # newline after \r progress
        success = self._fsm_state == _STATE_DONE
        marker = '✓' if success else '✗'
        print(f'  {marker} Loop {loop_idx} → {self._fsm_state}  '
              f'({time.time()-t0:.1f}s  '
              f'{self._fine_det_count} det frames)')
        return success

    def run(self) -> None:
        args = self._args

        # Enable arm once.
        if not args.no_enable:
            self.enable_arm()
            self._spin(0.5)

        success_count = 0
        fail_count    = 0

        for loop in range(1, args.loops + 1):
            ok = self.run_one(loop)
            if ok:
                success_count += 1
            else:
                fail_count += 1

        print(f'\n{"═"*60}')
        print(f'Done: {success_count}/{args.loops} success '
              f'({success_count/(args.loops)*100:.0f}%)')

        # PBVS per-tick QA replay (log/real_robot/qa/<session>/pbvs_replay.html).
        qa_dir = os.path.join(_WS_DIR, 'log', 'real_robot', 'qa')
        if os.path.isdir(qa_dir):
            sessions = sorted(
                d for d in os.listdir(qa_dir)
                if os.path.isdir(os.path.join(qa_dir, d))
            )
            for sess in reversed(sessions):
                html = os.path.join(qa_dir, sess, 'pbvs_replay.html')
                jsonl = os.path.join(qa_dir, sess, 'pbvs_stream.jsonl')
                if os.path.isfile(html):
                    print(f'PBVS QA replay : {html}')
                    break
                if os.path.isfile(jsonl):
                    from pbvs_qa_recorder import render_pbvs_replay_html
                    render_pbvs_replay_html(os.path.join(qa_dir, sess), html)
                    print(f'PBVS QA replay : {html} (generated from jsonl)')
                    break

        # Point user at the collected episode HTML.
        viz_dir = os.path.join(
            getattr(args, 'collect_data_dir', 'data/align_episodes'), 'viz')
        if os.path.isdir(viz_dir):
            htmls = sorted(
                f for f in os.listdir(viz_dir) if f.endswith('.html') and f != 'summary.html'
            )
            if htmls:
                print(f'Latest report : {os.path.join(viz_dir, htmls[-1])}')
                print(f'All episodes  : {os.path.join(viz_dir, "summary.html")}')

        import rclpy
        self._node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING,
                        format='%(levelname)s %(name)s: %(message)s')

    ap = argparse.ArgumentParser(
        description='Standalone PBVS REFINING validator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--loops', type=int, default=1,
                    help='How many REFINE cycles to run (default 1)')
    ap.add_argument('--timeout-s', type=float, default=120.0,
                    help='Per-loop timeout in seconds (default 120)')
    ap.add_argument('--no-restore', action='store_true',
                    help='Skip restoring the entry pose (arm already in position)')
    ap.add_argument('--no-reset', action='store_true',
                    help='Skip confirm_reset before restore (arm not at contact pose)')
    ap.add_argument('--no-enable', action='store_true',
                    help='Skip arm enable service calls')
    ap.add_argument('--ignore-pose-error', action='store_true',
                    help='Continue even if refine_entry_pose restore fails')
    ap.add_argument('--collect-data-dir', default='data/align_episodes',
                    help='Data dir to look for episode HTML after run')
    args = ap.parse_args()

    runner = RefineRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
