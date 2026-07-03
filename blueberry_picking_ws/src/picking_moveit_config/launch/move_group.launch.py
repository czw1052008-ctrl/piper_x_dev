import os
import sys

sys.path.append(os.path.dirname(__file__))

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

from _moveit_launch import (
    config_path,
    load_urdf,
    move_group_parameters,
    robot_description_command,
    srdf_path,
)


def _setup(context, *args, **kwargs):
    mode = LaunchConfiguration('end_effector_mode').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context) == 'true'
    use_gz = LaunchConfiguration('use_gz').perform(context) == 'true'

    robot_description = {
        'robot_description': (
            load_urdf(mode, use_gz=True) if use_gz else ParameterValue(
                robot_description_command(mode, use_gz=use_gz), value_type=str)
        ),
    }

    with open(srdf_path(mode), encoding='utf-8') as f:
        robot_description_semantic = {'robot_description_semantic': f.read()}

    common_params = [
        robot_description,
        robot_description_semantic,
        *move_group_parameters(mode, use_sim_time, use_gz),
    ]

    nodes = []
    if not use_gz:
        nodes.append(
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                output='screen',
                parameters=[robot_description, {'use_sim_time': use_sim_time}],
            ),
        )
    nodes.append(
        TimerAction(
            period=5.0 if use_gz else 0.0,
            actions=[
                Node(
                    package='moveit_ros_move_group',
                    executable='move_group',
                    output='screen',
                    arguments=['--ros-args', '--log-level', 'tf2_buffer:=ERROR'],
                    parameters=common_params,
                ),
            ],
        ) if use_gz else Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            arguments=['--ros-args', '--log-level', 'tf2_buffer:=ERROR'],
            parameters=common_params,
        ),
    )

    if not use_gz:
        nodes.extend([
            Node(
                package='controller_manager',
                executable='ros2_control_node',
                output='screen',
                parameters=[
                    config_path(
                        'ros2_controllers_vibration.yaml'
                        if mode == 'vibration' else 'ros2_controllers.yaml'
                    ),
                    robot_description,
                    {'use_sim_time': use_sim_time},
                ],
            ),
            TimerAction(
                period=3.0,
                actions=[
                    Node(
                        package='controller_manager',
                        executable='spawner',
                        arguments=[
                            'joint_state_broadcaster',
                            'arm_controller',
                            *(
                                ['vibration_motor_controller']
                                if mode == 'vibration' else []
                            ),
                            '--controller-manager', '/controller_manager',
                        ],
                    ),
                ],
            ),
        ])
    else:
        spawn_script = os.path.join(
            get_package_share_directory('picking_bringup'),
            'scripts',
            'spawn_gz_controllers.sh',
        )
        nodes.append(
            TimerAction(
                period=6.0,
                actions=[
                    ExecuteProcess(
                        cmd=['bash', spawn_script, mode],
                        output='screen',
                        shell=False,
                    ),
                ],
            ),
        )

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('end_effector_mode', default_value='suction'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('use_gz', default_value='false'),
        OpaqueFunction(function=_setup),
    ])
