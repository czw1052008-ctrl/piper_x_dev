import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg('picking_moveit_config'), 'launch', 'move_group.launch.py')
            ),
            launch_arguments={'end_effector_mode': 'suction', 'use_gz': 'false'}.items(),
        ),
        Node(
            package='picking_bringup',
            executable='static_fake_perception_node',
            parameters=[{'end_effector_mode': 'suction'}],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg('picking_grasp'), 'launch', 'grasp_planner.launch.py')
            ),
            launch_arguments={'end_effector_mode': 'suction'}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg('picking_task'), 'launch', 'task_server.launch.py')
            ),
            launch_arguments={'use_moveit': 'true'}.items(),
        ),
        TimerAction(
            period=30.0,
            actions=[
                Node(
                    package='picking_bringup',
                    executable='send_pick_goal',
                    parameters=[{'end_effector_mode': 'suction', 'max_retries': 3}],
                ),
            ],
        ),
    ])
