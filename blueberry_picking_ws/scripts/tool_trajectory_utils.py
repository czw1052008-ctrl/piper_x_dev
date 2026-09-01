"""Build /planning/tool_trajectory_4s from tip start→goal + approach axis."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from builtin_interfaces.msg import Duration
from std_msgs.msg import Header

from picking_msgs.msg import ToolTrajectory4s, ToolWaypoint6d

HORIZON_S = 4.0
DT_S = 0.25
N_WAYPOINTS = 16  # 0..15


def _duration(t_s: float) -> Duration:
    t = max(0.0, float(t_s))
    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    d = Duration()
    d.sec = sec
    d.nanosec = nanosec
    return d


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return a / n


def build_tool_trajectory_4s(
    tip_start: Sequence[float],
    tip_goal: Sequence[float],
    approach_axis: Sequence[float],
    *,
    header: Optional[Header] = None,
    seq: int = 0,
    move_duration_s: float = 1.0,
    horizon_s: float = HORIZON_S,
    dt_s: float = DT_S,
    execute_until_index: Optional[int] = None,
    reference: str = 'tool_tip',
) -> ToolTrajectory4s:
    """Linear tip interp start→goal over ``move_duration_s``, then hold.

    ``execute_until_index`` defaults to covering the move (capped at 15), so a
    PBVS oneshot can be consumed in one executor window.
    """
    p0 = np.asarray(tip_start, dtype=float).reshape(3)
    p1 = np.asarray(tip_goal, dtype=float).reshape(3)
    axis = _unit(approach_axis)
    move_s = float(max(1e-3, min(float(horizon_s), float(move_duration_s))))
    n_wp = int(round(float(horizon_s) / float(dt_s)))
    n_wp = max(1, min(N_WAYPOINTS, n_wp))

    if execute_until_index is None:
        exec_i = int(round(move_s / float(dt_s)))
        exec_i = max(1, min(n_wp - 1, exec_i))
    else:
        exec_i = int(max(0, min(n_wp - 1, int(execute_until_index))))

    msg = ToolTrajectory4s()
    if header is not None:
        msg.header = header
    else:
        msg.header = Header()
        msg.header.frame_id = 'base_link'
    msg.seq = int(seq)
    msg.horizon_s = float(horizon_s)
    msg.dt_s = float(dt_s)
    msg.execute_until_index = int(exec_i)
    msg.reference = str(reference)

    waypoints = []
    for k in range(n_wp):
        t = k * float(dt_s)
        if t >= move_s:
            alpha = 1.0
        else:
            alpha = t / move_s
        tip = (1.0 - alpha) * p0 + alpha * p1
        wp = ToolWaypoint6d()
        wp.position = [float(tip[0]), float(tip[1]), float(tip[2])]
        wp.approach_axis = [float(axis[0]), float(axis[1]), float(axis[2])]
        wp.time_from_start = _duration(t)
        waypoints.append(wp)
    msg.waypoints = waypoints
    return msg


def tip_axis_from_lookat(
    tip: Sequence[float], lookat: Sequence[float],
) -> Tuple[float, float, float]:
    """Approach axis = unit(lookat − tip); fallback +Z if coincident."""
    d = np.asarray(lookat, dtype=float).reshape(3) - np.asarray(tip, dtype=float).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return (0.0, 0.0, 1.0)
    u = d / n
    return (float(u[0]), float(u[1]), float(u[2]))
