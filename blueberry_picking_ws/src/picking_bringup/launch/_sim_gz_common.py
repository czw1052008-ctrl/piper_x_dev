"""Shared helpers for Gazebo Harmonic simulation launches."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node

_MOVEIT_LAUNCH = os.path.join(
    get_package_share_directory('picking_moveit_config'), 'launch')
if _MOVEIT_LAUNCH not in sys.path:
    sys.path.insert(0, _MOVEIT_LAUNCH)


def pick_action_server_node(end_effector_mode: str) -> Node:
    """MoveIt pick server with embedded URDF (gz RSP publishes /robot_description)."""
    from _moveit_launch import load_config, load_urdf, srdf_path

    task_pkg = get_package_share_directory('picking_task')
    with open(srdf_path(end_effector_mode), encoding='utf-8') as f:
        semantic = f.read()
    gdb_prefix = None
    if os.environ.get('PICK_GDB'):
        gdb_script = os.path.join(
            get_package_share_directory('picking_bringup'), 'scripts', 'run_under_gdb.sh')
        gdb_prefix = [gdb_script]
    return Node(
        package='picking_task',
        executable='pick_action_server',
        name='pick_action_server',
        output='screen',
        prefix=gdb_prefix,
        parameters=[{
            'robot_description': load_urdf(end_effector_mode, use_gz=True),
            'robot_description_semantic': semantic,
            'robot_description_kinematics': load_config('kinematics.yaml'),
            'robot_description_planning': load_config('joint_limits.yaml'),
            'bt_xml_suction': os.path.join(task_pkg, 'bt_xml', 'pick_suction.xml'),
            'bt_xml_vibration': os.path.join(task_pkg, 'bt_xml', 'pick_vibration.xml'),
            'use_moveit': True,
            'use_sim_time': True,
        }],
    )


def _sanitize_ld_library_path() -> str:
    """Drop conda lib dirs so gz_ros2_control plugins load reliably."""
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    parts = []
    for entry in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep):
        if not entry:
            continue
        if conda_prefix and entry.startswith(conda_prefix):
            continue
        parts.append(entry)
    ros_lib = '/opt/ros/jazzy/lib'
    if ros_lib not in parts:
        parts.insert(0, ros_lib)
    return os.pathsep.join(parts)


def kill_stale_nodes_action() -> ExecuteProcess:
    """Run stale-node cleanup before the rest of the launch graph starts."""
    script = os.path.join(
        get_package_share_directory('picking_bringup'), 'scripts', 'kill_stale_nodes.sh')
    return ExecuteProcess(cmd=['bash', script], output='screen', shell=False)


def _dedupe_path_list(paths: list[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in paths:
        if not entry or entry in seen:
            continue
        seen.add(entry)
        ordered.append(entry)
    return os.pathsep.join(ordered)


def gz_sim_env_actions():
    install_prefix = os.path.abspath(
        os.path.join(get_package_share_directory('picking_bringup'), '..'))
    resource_paths = _dedupe_path_list([
        os.path.join(install_prefix, 'agx_arm_description', 'share'),
        os.path.join(install_prefix, 'picking_description', 'share'),
        '/opt/ros/jazzy/share',
    ])
    plugin_paths = _dedupe_path_list(['/opt/ros/jazzy/lib'])
    return [
        # WSL2: SHM transport causes port lock failures; keep builtin UDP for discovery.
        SetEnvironmentVariable(name='RMW_FASTRTPS_USE_SHM', value='0'),
        SetEnvironmentVariable(name='ROS_LOCALHOST_ONLY', value=''),
        SetEnvironmentVariable(name='OMP_NUM_THREADS', value='2'),
        SetEnvironmentVariable(
            name='LD_LIBRARY_PATH',
            value=_sanitize_ld_library_path(),
        ),
        SetEnvironmentVariable(
            name='GZ_SIM_SYSTEM_PLUGIN_PATH',
            value=plugin_paths,
        ),
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=resource_paths,
        ),
    ]
