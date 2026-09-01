#!/usr/bin/env python3
"""One-shot: set AGX gripper max stroke to 0.1 m (100 mm Piper-X gripper).

Run while arm stack is STOPPED (exclusive CAN access):
  bash scripts/real_robot_shutdown.sh --no-disable
  python3 scripts/gripper_init_range.py
  bash scripts/real_robot_bringup.sh --no-wait
"""
from __future__ import annotations

import sys
import time

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config


def main() -> int:
    channel = sys.argv[1] if len(sys.argv) > 1 else 'can0'
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=PiperFW.DEFAULT,
        channel=channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    ef = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    if ef is None:
        print('init_effector failed', file=sys.stderr)
        return 1
    ok = ef.set_gripper_teaching_pendant_param(max_range_config=0.1, timeout=2.0)
    print(f'set max_range_config=0.1 -> {ok}')
    param = ef.get_gripper_teaching_pendant_param(timeout=2.0)
    if param is not None:
        print(f'read back max_range_config={param.msg.max_range_config}')
    print('open test...')
    ef.move_gripper_m(0.08, force=2.0)
    time.sleep(3.0)
    st = ef.get_gripper_status()
    if st is not None:
        print(f'width={st.msg.value} force={st.msg.force}')
    print('close test...')
    ef.move_gripper_m(0.0, force=2.0)
    time.sleep(2.0)
    st = ef.get_gripper_status()
    if st is not None:
        print(f'width={st.msg.value} force={st.msg.force}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
