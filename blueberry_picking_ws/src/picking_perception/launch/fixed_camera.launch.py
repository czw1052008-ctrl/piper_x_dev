"""Launch fixed USB mono camera as /camera_fixed/image_raw + static TF to base_link."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch(context, *args, **kwargs):
    device = LaunchConfiguration('video_device').perform(context)
    frame_id = LaunchConfiguration('frame_id').perform(context)
    camera_name = LaunchConfiguration('camera_name').perform(context)

    cam = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        namespace=camera_name,
        name='driver',
        parameters=[{
            'video_device': device,
            'image_size': [1280, 720],
            'pixel_format': 'YUYV',
            'camera_frame_id': frame_id,
        }],
        output='screen',
    )

    tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=f'{camera_name}_tf',
        arguments=[
            '--x', LaunchConfiguration('tx').perform(context),
            '--y', LaunchConfiguration('ty').perform(context),
            '--z', LaunchConfiguration('tz').perform(context),
            '--qx', LaunchConfiguration('qx').perform(context),
            '--qy', LaunchConfiguration('qy').perform(context),
            '--qz', LaunchConfiguration('qz').perform(context),
            '--qw', LaunchConfiguration('qw').perform(context),
            '--frame-id', 'base_link',
            '--child-frame-id', frame_id,
        ],
        output='screen',
    )
    return [cam, tf]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument('camera_name', default_value='camera_fixed'),
        DeclareLaunchArgument('frame_id', default_value='camera_fixed_optical_frame'),
        DeclareLaunchArgument('tx', default_value='0.45'),
        DeclareLaunchArgument('ty', default_value='0.0'),
        DeclareLaunchArgument('tz', default_value='0.55'),
        DeclareLaunchArgument('qx', default_value='0.0'),
        DeclareLaunchArgument('qy', default_value='0.7071'),
        DeclareLaunchArgument('qz', default_value='0.0'),
        DeclareLaunchArgument('qw', default_value='0.7071'),
        OpaqueFunction(function=_launch),
    ])
