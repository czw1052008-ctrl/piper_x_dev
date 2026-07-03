import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('picking_task')
    return LaunchDescription([
        DeclareLaunchArgument('end_effector_mode', default_value='suction'),
        DeclareLaunchArgument('use_moveit', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(
            package='picking_task',
            executable='pick_action_server',
            name='pick_action_server',
            parameters=[{
                'bt_xml_suction': os.path.join(pkg, 'bt_xml', 'pick_suction.xml'),
                'bt_xml_vibration': os.path.join(pkg, 'bt_xml', 'pick_vibration.xml'),
                'use_moveit': LaunchConfiguration('use_moveit'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
        ),
    ])
