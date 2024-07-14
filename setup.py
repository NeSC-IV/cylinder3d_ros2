from setuptools import setup
from glob import glob
import os
package_name = 'cylinder3d_ros2'
submodules = 'cylinder3d_ros2/submodules'
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oliver',
    maintainer_email='olivercjm@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "cylinder3d_ros2 = cylinder3d_ros2.cylinder3d_ros2:main"
        ],
    },
)
