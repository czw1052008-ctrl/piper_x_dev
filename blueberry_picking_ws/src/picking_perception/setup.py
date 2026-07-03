from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'picking_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ziwei',
    maintainer_email='dev@example.com',
    description='Perception nodes for blueberry picking',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'global_detector_node = picking_perception.global_detector_node:main',
            'fine_detector_node = picking_perception.fine_detector_node:main',
            'fake_perception_node = picking_perception.fake_perception_node:main',
            'contact_gz_visualizer = picking_perception.contact_gz_visualizer:main',
        ],
    },
)
