import os
import subprocess
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from urdf_utils import prepare_urdf_for_gz  # noqa: E402


def _load_urdf(end_effector_mode: str) -> str:
    xacro_path = os.path.join(
        get_package_share_directory('picking_description'),
        'urdf',
        f'piper_with_{end_effector_mode}.urdf.xacro',
    )
    result = subprocess.run(
        ['xacro', xacro_path, 'use_gz:=true'],
        capture_output=True,
        text=True,
        check=True,
    )
    return prepare_urdf_for_gz(result.stdout)


def _setup(context, *args, **kwargs):
    mode = context.launch_configurations['end_effector_mode']
    desc_pkg = get_package_share_directory('picking_description')
    world = os.path.join(desc_pkg, 'worlds', 'blueberry_picking.sdf')
    urdf_xml = _load_urdf(mode)
    robot_description = {
        'robot_description': urdf_xml,
        'use_sim_time': True,
    }

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
            ]),
            launch_arguments={'gz_args': f'-r {world}', 'on_exit_shutdown': 'true'}.items(),
        ),
        # Bridge /clock immediately; delay other bridges until sim time is flowing.
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='clock_bridge',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
            ros_arguments=['--param', 'use_sim_time:=false'],
        ),
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package='ros_gz_bridge',
                    executable='parameter_bridge',
                    name='gz_sensor_bridge',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                    arguments=[
                        '/model/blueberry_plant/pose_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                        '/camera_wrist/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                        '/camera_wrist/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                        '/camera_wrist/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                    ],
                ),
            ],
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    arguments=[
                        '--x', '0', '--y', '0', '--z', '0.1',
                        '--roll', '0', '--pitch', '0', '--yaw', '0',
                        '--frame-id', 'world', '--child-frame-id', 'base_link',
                    ],
                    parameters=[{'use_sim_time': True}],
                ),
                Node(
                    package='ros_gz_sim',
                    executable='create',
                    output='screen',
                    parameters=[{
                        'name': 'piper',
                        'string': urdf_xml,
                        'x': 0.0,
                        'y': 0.0,
                        'z': 0.1,
                    }],
                ),
            ],
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    output='screen',
                    parameters=[robot_description],
                ),
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('end_effector_mode', default_value='suction'),
        OpaqueFunction(function=_setup),
    ])
