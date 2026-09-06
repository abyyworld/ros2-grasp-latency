"""Driver and consumer in one process, with intra-process delivery on or off.

This is the composed arm of the experiment, and the reason both nodes are
components. Loading them into one container is what makes intra-process
delivery possible at all; the `intra_process` argument is what decides whether
it is used. Everything else about the two runs is identical, which is the point
of putting them behind one launch file with a flag rather than behind two files
that could drift apart:

    ros2 launch grasp_node_cpp composed.launch.py intra_process:=false
    ros2 launch grasp_node_cpp composed.launch.py intra_process:=true

With the flag off, a 640x480 depth frame is serialised to CDR, handed to
Fast DDS, and deserialised again, all inside one process. With it on, and with
exactly one subscriber, rclcpp moves the publisher's unique_ptr straight into
the subscription queue: no serialisation, no RMW, no copy of the pixel buffer.
The difference between the two end_to_end_ns distributions is what that costs,
and it is charged to the transport rather than to either language.

The container is the single-threaded one on purpose. component_container_mt
would let the driver's timer and the consumer's callback run on separate
threads, which on a four-core machine is a different experiment.

Neither node is transitioned here. launch_ros drives lifecycle transitions by
emitting events at LifecycleNode *actions*, and a component loaded into a
container is not one, so docker/run_ros_benchmark.sh calls
`ros2 lifecycle set` instead, in an order it controls: the consumer is active
before the driver publishes its first frame.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
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
            'intra_process', default_value='true',
            description='hand the depth image over as a pointer instead of '
                        'serialising it through the RMW'),
    ]

    # A bare LaunchConfiguration would reach the container as the string
    # 'true', and use_intra_process_comms is read as a bool. ParameterValue
    # with an explicit value_type is what coerces it.
    intra_process = [{
        'use_intra_process_comms': ParameterValue(
            LaunchConfiguration('intra_process'), value_type=bool),
    }]

    parameters = [QOS_FILE, LaunchConfiguration('params_file')]

    container = ComposableNodeContainer(
        name='grasp_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='grasp_node_cpp',
                plugin='grasp_node_cpp::GraspNode',
                name='grasp_node',
                parameters=parameters,
                remappings=remappings('grasp_node'),
                extra_arguments=intra_process,
            ),
            ComposableNode(
                package='grasp_node_cpp',
                plugin='grasp_node_cpp::FramePublisher',
                name='frame_publisher',
                parameters=parameters,
                remappings=remappings('frame_publisher'),
                extra_arguments=intra_process,
            ),
        ],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription(arguments + [container])
