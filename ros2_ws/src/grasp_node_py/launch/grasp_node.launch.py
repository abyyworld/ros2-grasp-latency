"""The rclpy grasp node in a process of its own.

There is no composed variant here and there cannot be: a component container
loads C++ shared libraries, and rclpy implements no intra-process transport, so
the Python node only ever has the inter-process arm. The C++ node is launched
the same way for that arm, from grasp_node_cpp/launch/grasp_node.launch.py, so
the two are compared on the same footing.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
import yaml

# Read from config/ rather than written out in the action: topics.yaml owns the
# remap table and qos.yaml owns the profiles. Both files are byte-identical to
# the pair in grasp_node_cpp, and docker/run_ros_benchmark.sh refuses to start
# a run unless they still are: two nodes on different QoS would be measuring a
# DDS policy difference and reporting it as a language difference.
SHARE = get_package_share_directory('grasp_node_py')
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
        package='grasp_node_py',
        executable='grasp_node',
        name='grasp_node',
        namespace='',
        parameters=[QOS_FILE, LaunchConfiguration('params_file')],
        remappings=remappings('grasp_node'),
        autostart=LaunchConfiguration('autostart'),
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription(arguments + [node])
