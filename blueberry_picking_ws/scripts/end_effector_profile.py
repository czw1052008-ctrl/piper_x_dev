"""Load end-effector profiles from config/end_effector_profiles.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

WS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROFILES_PATH = WS_DIR / 'config' / 'end_effector_profiles.yaml'


@dataclass(frozen=True)
class EndEffectorProfile:
    name: str
    tool_type_id: int
    tip_offset_link6: Tuple[float, float, float]
    approach_standoff_m: float
    cup_radius_m: float
    mode: str = 'suction'  # suction | gripper
    pre_grasp_width_m: float = 0.0
    grasp_width_m: float = 0.0
    grasp_force_n: float = 1.0
    press_in_m: float = 0.0
    retract_m: float = 0.05

    @property
    def is_gripper(self) -> bool:
        return str(self.mode).lower() == 'gripper' or int(self.tool_type_id) == 1

    def to_planner_msg_fields(self) -> Dict[str, Any]:
        return {
            'tool_type_id': self.tool_type_id,
            'profile_name': self.name,
            'tip_offset_link6': list(self.tip_offset_link6),
            'approach_standoff_m': self.approach_standoff_m,
            'cup_radius_m': self.cup_radius_m,
        }

    def contact_tip_goal(
        self,
        berry_xyz: Tuple[float, float, float] | list,
        approach_axis: Tuple[float, float, float] | list,
    ) -> Tuple[float, float, float]:
        """Tip target at grasp: berry + approach * press_in (gripper) or berry (suction)."""
        import math
        ax, ay, az = (float(approach_axis[0]), float(approach_axis[1]), float(approach_axis[2]))
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-9:
            ax, ay, az, n = 0.0, 0.0, 1.0, 1.0
        ax, ay, az = ax / n, ay / n, az / n
        d = float(self.press_in_m) if self.is_gripper else 0.0
        return (
            float(berry_xyz[0]) + ax * d,
            float(berry_xyz[1]) + ay * d,
            float(berry_xyz[2]) + az * d,
        )


def load_profiles(path: Path = DEFAULT_PROFILES_PATH) -> Dict[str, EndEffectorProfile]:
    if yaml is None:
        raise RuntimeError('PyYAML required: pip install pyyaml')
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    out: Dict[str, EndEffectorProfile] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        tip = cfg.get('tip_offset_link6', [0.0, 0.0, 0.0])
        out[name] = EndEffectorProfile(
            name=str(name),
            tool_type_id=int(cfg.get('tool_type_id', 0)),
            tip_offset_link6=(float(tip[0]), float(tip[1]), float(tip[2])),
            approach_standoff_m=float(cfg.get('approach_standoff_m', 0.003)),
            cup_radius_m=float(cfg.get('cup_radius_m', 0.012)),
            mode=str(cfg.get('mode', 'suction')),
            pre_grasp_width_m=float(cfg.get('pre_grasp_width_m', 0.0)),
            grasp_width_m=float(cfg.get('grasp_width_m', 0.0)),
            grasp_force_n=float(cfg.get('grasp_force_n', 1.0)),
            press_in_m=float(cfg.get('press_in_m', 0.0)),
            retract_m=float(cfg.get('retract_m', 0.05)),
        )
    return out


def load_profile(name: str, path: Path = DEFAULT_PROFILES_PATH) -> EndEffectorProfile:
    profiles = load_profiles(path)
    if name not in profiles:
        raise KeyError(f'unknown end-effector profile {name!r}; available={list(profiles)}')
    return profiles[name]
