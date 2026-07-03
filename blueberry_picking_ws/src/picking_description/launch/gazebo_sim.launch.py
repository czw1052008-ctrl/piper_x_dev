import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _launch_setup(context, *args, **kwargs):
    mode = LaunchConfiguration('end_effector_mode').perform(context)
    desc_pkg = get_package_share_directory('picking_description')
    xacro_file = os.path.join(desc_pkg, 'urdf', f'piper_with_{mode}.urdf.xacro')

    robot_description = Command([
        FindExecutable(name='xacro'), ' ', xacro_file, ' use_gazebo:=true',
    ])

    world = os.path.join(desc_pkg, 'worlds', 'blueberry_picking.world')
    models_path = os.path.join(desc_pkg, 'models')

    return [
        SetEnvironmentVariable(
            name='GAZEBO_MODEL_PATH',
            value=models_path + ':' + os.environ.get('GAZEBO_MODEL_PATH', ''),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
            ]),
            launch_arguments={'world': world, 'verbose': 'false'}.items(),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'use_sim_time': True,
                'robot_description': ParameterValue(robot_description, value_type=str),
            }],
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'piper', '-topic', 'robot_description', '-x', '0', '-y', '0', '-z', '0.1'],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[
                os.path.join(
                    get_package_share_directory('picking_moveit_config'), 'config', 'ros2_controllers.yaml'),
                {'use_sim_time': True},
            ],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['arm_controller', '--controller-manager', '/controller_manager'],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('end_effector_mode', default_value='suction'),
        OpaqueFunction(function=_launch_setup),
    ])
