#!/usr/bin/env python3
"""Top-level pick-cycle orchestrator (PBVS mainline).

State flow:
  HOME → CLUSTER_ALIGN → CLUSTER_READY
  → per fruit: RESTORE_ENTRY → TOUCH → CONFIRM_TOUCH → RETRACT → SUCTION_OFF → RESTORE_ENTRY
  → MISSION_DONE → HOME

Human in the loop:
  - s: suction on (during touch)
  - y/f: touch ok / fail
  - e: suction off (safe drop after retract)

Test jump-in:
  --test-config config/pick_test_fruit_task.json
"""

from __future__ import annotations

import argparse
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
WS_DIR = SCRIPTS_DIR.parent
DEFAULT_ENTRY_POSE = WS_DIR / 'log' / 'real_robot' / 'refine_entry_pose.json'
DEFAULT_QA_DIR = WS_DIR / 'log' / 'real_robot' / 'qa'


class PickState(str, Enum):
    IDLE = 'IDLE'
    CLUSTER_PLAN = 'CLUSTER_PLAN'
    CLUSTER_ALIGN_MOVE = 'CLUSTER_ALIGN_MOVE'
    FRUIT_RESTORE_ENTRY = 'FRUIT_RESTORE_ENTRY'
    FRUIT_TOUCH = 'FRUIT_TOUCH'
    FRUIT_WAIT_REACH = 'FRUIT_WAIT_REACH'
    FRUIT_WAIT_TOUCH_CONFIRM = 'FRUIT_WAIT_TOUCH_CONFIRM'
    FRUIT_RETRACT = 'FRUIT_RETRACT'
    FRUIT_WAIT_SUCTION_OFF = 'FRUIT_WAIT_SUCTION_OFF'
    FRUIT_ADVANCE = 'FRUIT_ADVANCE'
    MISSION_DONE = 'MISSION_DONE'
    ERROR_ABORT = 'ERROR_ABORT'


class PickCycleFsm:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._state = PickState.IDLE
        self._fruit_index = 0
        self._fruit_total = 0
        self._mission_id = time.strftime('%Y%m%d_%H%M%S')
        self._reach_qa_session: Optional[str] = None
        self._suction_on = False
        self._touch_confirm: Optional[str] = None
        self._cluster_plan = None
        self._phase_t0 = time.time()
        self._touch_started = False

        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Bool, String
        from picking_msgs.msg import DetectedBerryArray

        rclpy.init()
        self._node = Node('pick_cycle_fsm')
        self._String = String

        from pick_motion import ReachClient
        from pick_session_recorder import PickSessionRecorder
        from picking_msgs.msg import PlannerTaskContext

        self._PlannerTaskContext = PlannerTaskContext
        self._reach = ReachClient(self._node)
        self._recorder = PickSessionRecorder(Path(args.failure_dir))
        self._global_berries: List = []

        self._status_pub = self._node.create_publisher(String, '/pick/status', 10)
        self._task_ctx_pub = self._node.create_publisher(
            PlannerTaskContext, '/planning/task_context', 10)
        self._node.create_subscription(String, '/pick/cmd', self._on_pick_cmd, 10)
        self._node.create_subscription(Bool, '/pick/suction_state', self._on_suction, 10)
        self._node.create_subscription(String, '/pick/touch_confirm', self._on_touch_confirm, 10)
        self._node.create_subscription(
            DetectedBerryArray, '/perception/global/berries', self._on_global, 10)

        self._node.create_timer(0.1, self._tick)
        self._set_state(PickState.IDLE)
        self._node.get_logger().info('pick_cycle_fsm ready')

        if args.test_config:
            self._apply_test_config(Path(args.test_config))

    def _apply_test_config(self, path: Path) -> None:
        from cluster_align_planner import ClusterPlan, load_align_joints_from_json, load_test_config
        from fruit_queue import FruitTarget

        cfg = load_test_config(path)
        skip_to = cfg.get('skip_to', 'fruit_task')
        entry = Path(cfg.get('entry_pose', str(DEFAULT_ENTRY_POSE)))
        self._args.entry_pose = str(entry)

        if skip_to != 'fruit_task':
            self._abort(f'test_config skip_to={skip_to!r} not supported yet')
            return

        joints = load_align_joints_from_json(entry)
        if joints is None:
            self._abort('test_config: missing entry pose joints')
            return

        n_fruits = int(cfg.get('fruit_count', 1))
        fake_targets = [
            FruitTarget(
                index=i, track_id=i, confidence=1.0,
                base_xyz=(0.3, 0.1 + i * 0.02, 0.2), dist_m=0.3 + i * 0.02,
            )
            for i in range(n_fruits)
        ]
        self._cluster_plan = ClusterPlan(
            cluster_index=0,
            cluster_center=(0.3, 0.1, 0.2),
            cluster_score=1.0,
            align_joints_rad=joints,
            fruit_targets=fake_targets,
            source=f'test_config:{path.name}',
        )
        self._fruit_total = len(fake_targets)
        self._fruit_index = int(cfg.get('start_fruit_index', 0))
        self._set_state(PickState.FRUIT_RESTORE_ENTRY)
        self._node.get_logger().info(
            f'test_config skip_to=fruit_task fruits={self._fruit_total} '
            f'start_index={self._fruit_index}')

    def _set_state(self, state: PickState, detail: str = '') -> None:
        self._state = state
        self._phase_t0 = time.time()
        payload = state.value if not detail else f'{state.value}:{detail}'
        msg = self._String()
        msg.data = payload
        self._status_pub.publish(msg)
        self._publish_task_context()
        self._node.get_logger().info(f'pick → {payload}')

    def _publish_task_context(self) -> None:
        ctx = self._PlannerTaskContext()
        st = self._state
        if st in (PickState.CLUSTER_PLAN, PickState.CLUSTER_ALIGN_MOVE):
            ctx.fsm_state = self._PlannerTaskContext.FSM_CLUSTER_ALIGN
        elif st in (
            PickState.FRUIT_RESTORE_ENTRY,
            PickState.FRUIT_TOUCH,
            PickState.FRUIT_WAIT_REACH,
            PickState.FRUIT_WAIT_TOUCH_CONFIRM,
        ):
            ctx.fsm_state = self._PlannerTaskContext.FSM_APPROACH_FRUIT
        elif st == PickState.FRUIT_RETRACT:
            ctx.fsm_state = self._PlannerTaskContext.FSM_RETRACT
        elif st == PickState.IDLE:
            ctx.fsm_state = self._PlannerTaskContext.FSM_IDLE
        else:
            ctx.fsm_state = self._PlannerTaskContext.FSM_HOLD

        cluster_id = -1
        if self._cluster_plan is not None:
            cluster_id = int(self._cluster_plan.cluster_index)
        ctx.cluster_id = cluster_id

        fruit_id = -1
        if self._cluster_plan is not None and self._fruit_index < len(
                self._cluster_plan.fruit_targets):
            fruit_id = int(self._cluster_plan.fruit_targets[self._fruit_index].track_id)
        ctx.fruit_id = fruit_id
        self._task_ctx_pub.publish(ctx)

    def _on_pick_cmd(self, msg) -> None:
        cmd = msg.data.strip().lower()
        if cmd == 'start_mission':
            if self._state in (PickState.IDLE, PickState.ERROR_ABORT):
                self._mission_id = time.strftime('%Y%m%d_%H%M%S')
                self._fruit_index = 0
                self._set_state(PickState.CLUSTER_PLAN)
        elif cmd == 'retry_fruit':
            if self._state == PickState.ERROR_ABORT:
                self._set_state(PickState.FRUIT_RESTORE_ENTRY)
        elif cmd == 'abort':
            self._abort('operator abort')

    def _on_suction(self, msg) -> None:
        self._suction_on = bool(msg.data)

    def _on_touch_confirm(self, msg) -> None:
        self._touch_confirm = msg.data.strip().lower()

    def _on_global(self, msg) -> None:
        self._global_berries = list(msg.berries)

    def _qa_session_dir(self) -> Optional[Path]:
        if not self._reach_qa_session:
            return None
        d = Path(self._args.qa_dir) / self._reach_qa_session
        return d if d.is_dir() else None

    def _latest_qa_session(self) -> Optional[Path]:
        qa_root = Path(self._args.qa_dir)
        if not qa_root.is_dir():
            return None
        sessions = sorted(
            [p for p in qa_root.iterdir() if p.is_dir()],
            key=lambda p: p.name,
            reverse=True,
        )
        for s in sessions:
            if (s / 'pbvs_direct_arrive.json').is_file():
                return s
        return sessions[0] if sessions else None

    def _abort(self, reason: str) -> None:
        self._reach.abort()
        qa = self._qa_session_dir() or self._latest_qa_session()
        fail_dir = self._recorder.record_failure(
            stage=self._state.value,
            reason=reason,
            qa_session_dir=qa,
            extra={'fruit_index': self._fruit_index, 'mission_id': self._mission_id},
        )
        self._set_state(PickState.ERROR_ABORT, reason)
        self._node.get_logger().error(f'ABORT: {reason} → recorded {fail_dir}')

    def _tick(self) -> None:
        handlers = {
            PickState.CLUSTER_PLAN: self._tick_cluster_plan,
            PickState.CLUSTER_ALIGN_MOVE: self._tick_cluster_align_move,
            PickState.FRUIT_RESTORE_ENTRY: self._tick_restore_entry,
            PickState.FRUIT_TOUCH: self._tick_fruit_touch,
            PickState.FRUIT_WAIT_REACH: self._tick_wait_reach,
            PickState.FRUIT_WAIT_TOUCH_CONFIRM: self._tick_wait_touch_confirm,
            PickState.FRUIT_RETRACT: self._tick_retract,
            PickState.FRUIT_WAIT_SUCTION_OFF: self._tick_wait_suction_off,
            PickState.FRUIT_ADVANCE: self._tick_advance_fruit,
            PickState.MISSION_DONE: self._tick_mission_done,
        }
        fn = handlers.get(self._state)
        if fn is not None:
            fn()

    def _tick_cluster_plan(self) -> None:
        from cluster_align_planner import plan_cluster_align

        align_dec = Path(self._args.align_decision) if self._args.align_decision else None
        plan = plan_cluster_align(
            self._global_berries,
            entry_pose_path=Path(self._args.entry_pose),
            align_decision_path=align_dec,
        )
        if plan is None:
            self._abort('cluster_plan: no global berries or missing entry_pose JSON')
            return
        self._cluster_plan = plan
        self._fruit_total = len(plan.fruit_targets)
        self._node.get_logger().info(
            f'cluster idx={plan.cluster_index} score={plan.cluster_score:.2f} '
            f'fruits={self._fruit_total} source={plan.source}')
        self._set_state(PickState.CLUSTER_ALIGN_MOVE)

    def _tick_cluster_align_move(self) -> None:
        from pick_motion import send_joint_trajectory

        if self._cluster_plan is None:
            self._abort('cluster_align: no plan')
            return
        if time.time() - self._phase_t0 < 0.2:
            return
        ok = send_joint_trajectory(
            self._node,
            self._cluster_plan.align_joints_rad,
            traj_s=float(self._args.cluster_align_traj_s),
            settle_s=0.5,
        )
        if not ok:
            self._abort('cluster_align: trajectory failed')
            return
        self._set_state(PickState.FRUIT_RESTORE_ENTRY)

    def _tick_restore_entry(self) -> None:
        if time.time() - self._phase_t0 < 0.15:
            return
        ok = self._reach.restore_entry_pose(
            Path(self._args.entry_pose),
            traj_s=float(self._args.entry_restore_traj_s),
        )
        if not ok:
            self._abort('restore_entry failed')
            return
        self._touch_confirm = None
        self._suction_on = False
        self._touch_started = False
        self._set_state(PickState.FRUIT_TOUCH)

    def _tick_fruit_touch(self) -> None:
        if time.time() - self._phase_t0 < 0.2:
            return
        self._recorder.start_fruit(
            mission_id=self._mission_id, fruit_index=self._fruit_index)
        print(
            f'\n=== Fruit {self._fruit_index + 1}/{self._fruit_total}: '
            f'press S (suction ON) — PBVS touch starting ===\n',
            flush=True,
        )
        self._reach.start_refine()
        self._touch_started = True
        self._set_state(PickState.FRUIT_WAIT_REACH)

    def _tick_wait_reach(self) -> None:
        if self._reach.reach_state == 'ERROR':
            self._abort('reach_fsm ERROR during touch')
            return
        if not self._suction_on and time.time() - self._phase_t0 > float(self._args.suction_on_timeout_s):
            self._abort('suction_on timeout — press S when touch starts')
            return
        if time.time() - self._phase_t0 > float(self._args.touch_timeout_s):
            self._abort('touch timeout — no WAIT_CONFIRM from reach_fsm')
            return
        if self._reach.reach_state == 'WAIT_CONFIRM':
            latest = self._latest_qa_session()
            if latest is not None:
                self._reach_qa_session = latest.name
            print('\n=== Touch done — press Y (ok) or F (fail) ===\n', flush=True)
            self._set_state(PickState.FRUIT_WAIT_TOUCH_CONFIRM)

    def _tick_wait_touch_confirm(self) -> None:
        if self._touch_confirm == 'fail':
            self._abort('operator marked touch FAIL')
            return
        if self._touch_confirm != 'ok':
            return
        self._set_state(PickState.FRUIT_RETRACT)

    def _tick_retract(self) -> None:
        if time.time() - self._phase_t0 < 0.15:
            return
        from pick_retract import retract_from_qa_session

        qa = self._qa_session_dir() or self._latest_qa_session()
        if qa is None:
            self._abort('retract: no QA session with pbvs_direct_arrive.json')
            return

        result = retract_from_qa_session(
            self._node,
            qa,
            retract_m=float(self._args.retract_m),
            traj_s=float(self._args.retract_traj_s),
        )
        if not result.get('ok'):
            self._abort(f'retract failed: {result.get("error", "unknown")}')
            return

        print('\n=== Safe drop pose — press E (suction OFF) ===\n', flush=True)
        self._set_state(PickState.FRUIT_WAIT_SUCTION_OFF)

    def _tick_wait_suction_off(self) -> None:
        if self._suction_on:
            if time.time() - self._phase_t0 > float(self._args.suction_off_timeout_s):
                self._abort('suction_off timeout — press E after berry drops')
            return
        self._recorder.record_success(
            stage='fruit_complete',
            extra={'fruit_index': self._fruit_index, 'qa_session': self._reach_qa_session},
        )
        self._set_state(PickState.FRUIT_ADVANCE)

    def _tick_advance_fruit(self) -> None:
        if time.time() - self._phase_t0 < 0.15:
            return
        self._reach.next_fruit()
        self._reach.spin(0.5)
        self._fruit_index += 1
        self._touch_confirm = None
        if self._fruit_index >= self._fruit_total:
            self._set_state(PickState.MISSION_DONE)
        else:
            self._set_state(PickState.FRUIT_RESTORE_ENTRY)

    def _tick_mission_done(self) -> None:
        if time.time() - self._phase_t0 < 0.2:
            return
        self._reach.confirm_reset_home()
        self._fruit_index = 0
        self._fruit_total = 0
        self._cluster_plan = None
        self._set_state(PickState.IDLE, 'mission_done')

    def spin(self) -> None:
        import rclpy
        try:
            rclpy.spin(self._node)
        except KeyboardInterrupt:
            pass
        finally:
            self._node.destroy_node()
            rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry-pose', default=str(DEFAULT_ENTRY_POSE))
    parser.add_argument('--align-decision', default='')
    parser.add_argument('--qa-dir', default=str(DEFAULT_QA_DIR))
    parser.add_argument('--failure-dir', default=str(WS_DIR / 'log' / 'real_robot' / 'pick_failures'))
    parser.add_argument('--cluster-align-traj-s', type=float, default=4.0)
    parser.add_argument('--entry-restore-traj-s', type=float, default=2.0)
    parser.add_argument('--retract-m', type=float, default=0.05)
    parser.add_argument('--retract-traj-s', type=float, default=0.8)
    parser.add_argument('--touch-timeout-s', type=float, default=90.0)
    parser.add_argument('--suction-on-timeout-s', type=float, default=30.0)
    parser.add_argument('--suction-off-timeout-s', type=float, default=60.0)
    parser.add_argument('--test-config', default='')
    args = parser.parse_args()

    fsm = PickCycleFsm(args)
    print('\n' + '=' * 72)
    print('  Pick cycle FSM running.')
    print('  ros2 topic pub --once /pick/cmd std_msgs/String "{data: start_mission}"')
    print('  Keyboard: python3 scripts/suction_keyboard_sim.py  (s/e/y/f)')
    print('=' * 72 + '\n')
    fsm.spin()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
