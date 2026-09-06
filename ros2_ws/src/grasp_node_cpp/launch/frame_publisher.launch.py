"""The live frame-store driver in a process of its own.

Paired with grasp_node.launch.py. composed.launch.py does not use it: there the
same component is loaded into the consumer's container instead, which is the
only arrangement in which the depth image never reaches the RMW.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
import yaml

# Read from config/ rather than written out in each action: topics.yaml owns
# the remap table and qos.yaml owns the profiles, and a launch file that
# repeated either would be a second source of truth that drifts the first time
# a namespace changes. Launch files are loaded by path rather than as a
# package, so the three of them repeat these lines rather than sharing a module
# that would not be importable.
SHARE = get_package_share_directory('grasp_node_cpp')
QOS_FILE = os.path.join(SHARE, 'config', 'qos.yaml')


def remappings(node):
    """Remap rules for one node, in the (from, to) pairs launch wants."""
    with open(os.path.join(SHARE, 'config', 'topics.yaml'), encoding='utf-8') as handle:
        return list(yaml.safe_load(handle)[node].items())


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('params_file', default_value=QOS_FILE),
        DeclareLaunchArgument(
            'autostart', default_value='true',
            description='configure and activate as soon as the node is up'),
    ]

    node = LifecycleNode(
        package='grasp_node_cpp',
        executable='frame_publisher',
        name='frame_publisher',
        namespace='',
        parameters=[QOS_FILE, LaunchConfiguration('params_file')],
        remappings=remappings('frame_publisher'),
        autostart=LaunchConfiguration('autostart'),
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription(arguments + [node])
