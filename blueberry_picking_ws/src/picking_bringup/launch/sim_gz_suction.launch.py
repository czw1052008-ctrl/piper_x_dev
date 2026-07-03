import os
import sys

sys.path.append(os.path.dirname(__file__))

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from _sim_gz_common import gz_sim_env_actions, pick_action_server_node


def generate_launch_description():
    pkg = get_package_share_directory
    return LaunchDescription([
        *gz_sim_env_actions(),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg('picking_description'), 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={'end_effector_mode': 'suction'}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg('picking_moveit_config'), 'launch', 'move_group.launch.py')
            ),
            launch_arguments={
                'end_effector_mode': 'suction',
                'use_gz': 'true',
                'use_sim_time': 'true',
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg('picking_perception'), 'launch', 'perception.launch.py')
            ),
            launch_arguments={
                'global_perception': 'gz',
                'fine_perception': 'foundation_pose',
                'end_effector_mode': 'suction',
                'use_sim_time': 'true',
                'use_fake_perception': 'false',
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg('picking_grasp'), 'launch', 'grasp_planner.launch.py')
            ),
            launch_arguments={'end_effector_mode': 'suction'}.items(),
        ),
        pick_action_server_node('suction'),
        TimerAction(
            period=18.0,
            actions=[
                Node(
                    package='picking_bringup',
                    executable='send_pick_goal',
                    parameters=[{
                        'end_effector_mode': 'suction',
                        'max_retries': 3,
                        'controller_wait_sec': 90.0,
                        'use_sim_time': True,
                    }],
                ),
            ],
        ),
    ])
