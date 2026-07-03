import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='picking_bringup',
            executable='static_fake_perception_node',
            parameters=[{'end_effector_mode': 'vibration'}],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('picking_grasp'), 'launch', 'grasp_planner.launch.py')
            ),
            launch_arguments={'end_effector_mode': 'vibration'}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('picking_task'), 'launch', 'task_server.launch.py')
            ),
        ),
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='picking_bringup',
                    executable='send_pick_goal',
                    parameters=[{'end_effector_mode': 'vibration', 'max_retries': 3}],
                ),
            ],
        ),
    ])
