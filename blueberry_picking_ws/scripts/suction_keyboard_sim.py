#!/usr/bin/env python3
"""Keyboard simulator for suction on/off during pick cycle.

Keys (terminal must be focused):
  s  suction ON  → publishes /pick/suction_state true
  e  suction OFF → publishes /pick/suction_state false
  y  touch OK    → publishes /pick/touch_confirm ok
  f  touch FAIL  → publishes /pick/touch_confirm fail
  q  quit
"""

from __future__ import annotations

import argparse
import sys
import termios
import tty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, String

    rclpy.init()
    node = Node('suction_keyboard_sim')
    suction_pub = node.create_publisher(Bool, '/pick/suction_state', 10)
    touch_pub = node.create_publisher(String, '/pick/touch_confirm', 10)

    suction_on = False

    def publish_suction(on: bool) -> None:
        nonlocal suction_on
        suction_on = on
        msg = Bool()
        msg.data = on
        suction_pub.publish(msg)
        print(f'  → suction {"ON" if on else "OFF"}', flush=True)

    def publish_touch(result: str) -> None:
        msg = String()
        msg.data = result
        touch_pub.publish(msg)
        print(f'  → touch_confirm={result}', flush=True)

    print('Suction keyboard sim ready:')
    print('  s=ON  e=OFF  y=touch_ok  f=touch_fail  q=quit')
    print('')

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            ch = sys.stdin.read(1)
            if not ch:
                continue
            c = ch.lower()
            if c == 'q':
                break
            elif c == 's':
                publish_suction(True)
            elif c == 'e':
                publish_suction(False)
            elif c == 'y':
                publish_touch('ok')
            elif c == 'f':
                publish_touch('fail')
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
