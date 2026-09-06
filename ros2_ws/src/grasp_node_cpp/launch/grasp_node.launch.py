"""The rclcpp grasp node in a process of its own.

This is the inter-process arm of the comparison: the depth image crosses a
process boundary, is serialised on one side and deserialised on the other, and
that cost lands in end_to_end_ns. composed.launch.py is the other arm, and the
difference between the two is one of the results this repository reports.
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
    # params_file carries config_path, chain_path, ransac_table_path and
    # joint_names, and docker/run_ros_benchmark.sh generates it from
    # assets/pipeline_config.json. It defaults to the QoS file only so the
    # launch file can be run bare: the node then refuses to configure, loudly,
    # because config_path is unset, which is the behaviour that is wanted.
    arguments = [
        DeclareLaunchArgument('params_file', default_value=QOS_FILE),
        DeclareLaunchArgument(
            'autostart', default_value='true',
            description='configure and activate as soon as the node is up'),
    ]

    node = LifecycleNode(
        package='grasp_node_cpp',
        executable='grasp_node',
        name='grasp_node',
        namespace='',
        # QoS first, then the run's own parameters, so a single run can
        # override a profile without editing the file both packages share.
        parameters=[QOS_FILE, LaunchConfiguration('params_file')],
        remappings=remappings('grasp_node'),
        autostart=LaunchConfiguration('autostart'),
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription(arguments + [node])
