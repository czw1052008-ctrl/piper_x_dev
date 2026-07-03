import os
import sys

sys.path.append(os.path.dirname(__file__))

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from _sim_gz_common import gz_sim_env_actions, pick_action_server_node

sys.path.append(os.path.join(get_package_share_directory('picking_moveit_config'), 'launch'))
from _moveit_launch import load_urdf  # noqa: E402


def generate_launch_description():
    pkg = get_package_share_directory
    fine_perception = LaunchConfiguration('fine_perception')
    return LaunchDescription([
        DeclareLaunchArgument(
            'fine_perception',
            default_value='foundation_pose',
            description='gz (GT from link poses) | foundation_pose',
        ),
        DeclareLaunchArgument('auto_pick', default_value='true'),
        *gz_sim_env_actions(),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='picking_bringup',
                    executable='robot_description_publisher',
                    output='screen',
                    parameters=[{
                        'robot_description': load_urdf('vibration', use_gz=True),
                        'use_sim_time': True,
                    }],
                ),
            ],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg('picking_description'), 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={'end_effector_mode': 'vibration'}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg('picking_moveit_config'), 'launch', 'move_group.launch.py')
            ),
            launch_arguments={
                'end_effector_mode': 'vibration',
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
                'fine_perception': fine_perception,
                'end_effector_mode': 'vibration',
                'use_sim_time': 'true',
                'use_fake_perception': 'false',
            }.items(),
        ),
        TimerAction(
            period=5.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(pkg('picking_grasp'), 'launch', 'grasp_planner.launch.py')
                    ),
                    launch_arguments={
                        'end_effector_mode': 'vibration',
                        'use_sim_time': 'true',
                    }.items(),
                ),
            ],
        ),
        TimerAction(
            period=8.0,
            actions=[pick_action_server_node('vibration')],
        ),
        TimerAction(
            period=12.0,
            actions=[
                Node(
                    package='picking_bringup',
                    executable='send_pick_goal',
                    parameters=[{
                        'end_effector_mode': 'vibration',
                        'max_retries': 3,
                        'controller_wait_sec': 90.0,
                        'goal_timeout_sec': 120.0,
                        'use_sim_time': True,
                    }],
                    condition=IfCondition(LaunchConfiguration('auto_pick')),
                ),
            ],
        ),
    ])
