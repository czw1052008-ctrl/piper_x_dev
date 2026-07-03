import os
import sys

sys.path.append(os.path.dirname(__file__))

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from _sim_gz_common import gz_sim_env_actions

sys.path.append(os.path.join(get_package_share_directory('picking_moveit_config'), 'launch'))
from _moveit_launch import load_urdf  # noqa: E402


def generate_launch_description():
    pkg = get_package_share_directory
    desc_pkg = pkg('picking_description')
    marker_sdf = os.path.join(desc_pkg, 'models', 'contact_marker', 'model.sdf')

    return LaunchDescription([
        DeclareLaunchArgument('stream_rate_hz', default_value='10.0'),
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
                os.path.join(desc_pkg, 'launch', 'gz_sim.launch.py')
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
                'fine_perception': 'gz',
                'end_effector_mode': 'vibration',
                'use_sim_time': 'true',
                'use_fake_perception': 'false',
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg('picking_grasp'), 'launch', 'grasp_planner.launch.py')
            ),
            launch_arguments={
                'end_effector_mode': 'vibration',
                'use_sim_time': 'true',
            }.items(),
        ),
        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package='ros_gz_sim',
                    executable='create',
                    arguments=[
                        '-name', 'contact_marker',
                        '-file', marker_sdf,
                        '-x', '0.316', '-y', '-0.083', '-z', '0.74',
                    ],
                    output='screen',
                ),
            ],
        ),
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='picking_perception',
                    executable='contact_gz_visualizer',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                ),
            ],
        ),
        # Teleop: run in a separate terminal (needs interactive TTY):
        #   ros2 run picking_bringup link6_teleop_node --ros-args -p use_sim_time:=true
    ])
