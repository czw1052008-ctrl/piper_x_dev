import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    global_mode = LaunchConfiguration('global_perception').perform(context)
    fine_mode = LaunchConfiguration('fine_perception').perform(context)
    legacy_fake = LaunchConfiguration('use_fake_perception').perform(context) == 'true'
    mode = LaunchConfiguration('end_effector_mode').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context) == 'true'

    if legacy_fake and global_mode == 'static':
        global_mode = 'static'
        if fine_mode == 'gz':
            fine_mode = 'gz'

    pkg = get_package_share_directory('picking_perception')
    fp_config = os.path.join(pkg, 'config', 'foundation_pose.yaml')
    nodes = []

    if global_mode == 'gz':
        nodes.append(
            Node(
                package='picking_perception',
                executable='fake_perception_node',
                name='fake_perception_node',
                parameters=[{
                    'end_effector_mode': mode,
                    'provide_fine_detection': fine_mode == 'gz',
                    'stream_topics': True,
                    'stream_rate_hz': 10.0,
                    'gz_pose_topic': '/model/blueberry_plant/pose_static',
                    'use_sim_time': use_sim_time,
                }],
            ),
        )

    if fine_mode == 'foundation_pose':
        nodes.append(
            Node(
                package='picking_perception',
                executable='fine_detector_node',
                parameters=[fp_config, {'use_sim_time': use_sim_time}],
            ),
        )

    if global_mode == 'camera':
        nodes.append(
            Node(
                package='picking_perception',
                executable='global_detector_node',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
        )

    if not nodes and legacy_fake:
        nodes.append(
            Node(
                package='picking_perception',
                executable='fake_perception_node',
                name='fake_perception_node',
                parameters=[{
                    'end_effector_mode': mode,
                    'provide_fine_detection': True,
                    'use_sim_time': use_sim_time,
                }],
            ),
        )

    if not nodes and not legacy_fake:
        nodes.extend([
            Node(
                package='picking_perception',
                executable='global_detector_node',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
            Node(
                package='picking_perception',
                executable='fine_detector_node',
                parameters=[fp_config, {'use_sim_time': use_sim_time}],
            ),
        ])

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_fake_perception', default_value='true'),
        DeclareLaunchArgument(
            'global_perception',
            default_value='static',
            description='static | gz | camera',
        ),
        DeclareLaunchArgument(
            'fine_perception',
            default_value='gz',
            description='gz | foundation_pose',
        ),
        DeclareLaunchArgument('end_effector_mode', default_value='suction'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=_setup),
    ])
