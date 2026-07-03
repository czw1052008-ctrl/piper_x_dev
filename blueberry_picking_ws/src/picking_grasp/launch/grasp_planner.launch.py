from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def _setup(context, *args, **kwargs):
    mode = LaunchConfiguration('end_effector_mode').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context) == 'true'

    stream_topics = LaunchConfiguration('stream_topics').perform(context) == 'true'
    params = [{
        'end_effector_mode': mode,
        'use_sim_time': use_sim_time,
        'stream_topics': stream_topics,
    }]
    pkg = get_package_share_directory('picking_grasp')
    if mode == 'vibration':
        config = os.path.join(pkg, 'config', 'vibration_params.yaml')
        params.insert(0, config)
    elif mode == 'suction':
        config = os.path.join(pkg, 'config', 'suction_params.yaml')
        params.insert(0, config)

    return [
        Node(
            package='picking_grasp',
            executable='grasp_planner_node',
            name='grasp_planner_node',
            output='screen',
            parameters=params,
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('end_effector_mode', default_value='suction'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('stream_topics', default_value='false'),
        OpaqueFunction(function=_setup),
    ])
