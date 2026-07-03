"""Shared helpers for MoveIt launch files."""

import os
import subprocess
import sys

import yaml
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command, FindExecutable

_PICKING_LAUNCH = os.path.join(
    get_package_share_directory('picking_description'), 'launch')
if _PICKING_LAUNCH not in sys.path:
    sys.path.insert(0, _PICKING_LAUNCH)
from urdf_utils import prepare_urdf_for_gz  # noqa: E402


def get_xacro_path(end_effector_mode: str) -> str:
    return os.path.join(
        get_package_share_directory('picking_description'),
        'urdf',
        f'piper_with_{end_effector_mode}.urdf.xacro',
    )


def load_urdf(end_effector_mode: str, use_gz: bool = False) -> str:
    xacro_path = get_xacro_path(end_effector_mode)
    gz_flag = 'true' if use_gz else 'false'
    result = subprocess.run(
        ['xacro', xacro_path, f'use_gz:={gz_flag}'],
        capture_output=True,
        text=True,
        check=True,
    )
    urdf = result.stdout
    if use_gz:
        urdf = prepare_urdf_for_gz(urdf)
    return urdf


def robot_description_command(end_effector_mode: str, use_gz: bool = False) -> Command:
    return Command([
        FindExecutable(name='xacro'), ' ',
        get_xacro_path(end_effector_mode),
        ' use_gz:=', 'true' if use_gz else 'false',
    ])


def srdf_path(end_effector_mode: str) -> str:
    return os.path.join(
        get_package_share_directory('picking_moveit_config'),
        'config',
        f'piper_{end_effector_mode}.srdf',
    )


def config_path(filename: str) -> str:
    return os.path.join(
        get_package_share_directory('picking_moveit_config'),
        'config',
        filename,
    )


def load_config(filename: str) -> dict:
    with open(config_path(filename), encoding='utf-8') as f:
        return yaml.safe_load(f)


def move_group_parameters(end_effector_mode: str, use_sim_time: bool, use_gz: bool = False) -> list:
    """MoveIt move_group node parameters (dict form, not raw params-file paths)."""
    ompl = load_config('ompl_planning.yaml')
    return [
        {'robot_description_kinematics': load_config('kinematics.yaml')},
        {'robot_description_planning': load_config('joint_limits.yaml')},
        {
            'planning_pipelines': ['ompl'],
            'default_planning_pipeline': 'ompl',
            'ompl': ompl,
        },
        load_config('moveit_controllers.yaml'),
        {
            'publish_robot_description_semantic': True,
            'allow_trajectory_execution': True,
            'trajectory_execution': {
                'allowed_start_tolerance': 0.08,
            },
            'publish_planning_scene': True,
            'publish_geometry_updates': True,
            'publish_state_updates': True,
            'publish_transforms_updates': not use_gz,
            'use_sim_time': use_sim_time,
        },
    ]
