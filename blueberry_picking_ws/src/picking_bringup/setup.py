from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'picking_bringup'
_scripts = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'launch', 'debug'), glob('launch/debug/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'scripts'),
         glob(os.path.join(_scripts, 'kill_stale_nodes.sh')) +
         glob(os.path.join(_scripts, 'run_under_gdb.sh')) +
         glob(os.path.join(_scripts, 'run_with_timeout.sh')) +
         glob(os.path.join(_scripts, 'spawn_gz_controllers.sh')) +
         glob(os.path.join(_scripts, 'preflight_gz_stack.sh')) +
         glob(os.path.join(_scripts, 'run_gz_vibration.sh'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ziwei',
    maintainer_email='dev@example.com',
    description='Top-level launch files for blueberry picking',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'static_fake_perception_node = picking_bringup.static_fake_perception_node:main',
            'send_pick_goal = picking_bringup.send_pick_goal:main',
            'link6_teleop_node = picking_bringup.link6_teleop_node:main',
            'robot_description_publisher = picking_bringup.robot_description_publisher:main',
        ],
    },
)
