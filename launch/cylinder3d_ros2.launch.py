import os
from ament_index_python import get_package_share_directory,get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import GroupAction,ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    return LaunchDescription(
        [
            ExecuteProcess(cmd=['/home/oliver/.conda/envs/slam2/bin/python', os.path.join(os.path.dirname(os.path.dirname(get_package_prefix('cylinder3d_ros2'))), 'src', 'cylinder3d_ros2', 'cylinder3d_ros2','cylinder3d_ros2.py')] ),
        ]
    )