"""rclpy front end for the Python grasp pipeline.

The measurement boundaries mirror grasp_node_cpp exactly, because a comparison
between the two is only worth anything if both stop the clock in the same place:

  compute_ns    starts at subscription-callback entry, before any validation,
                and stops immediately before JointTrajectory.publish(). It
                includes turning ``msg.data`` into NumPy views and building the
                outgoing message -- costs the in-process benchmark never pays,
                and precisely the rclpy overhead this repository is measuring.
  end_to_end_ns is the node clock at that same instant minus the depth header
                stamp, so it also carries DDS transport, CDR deserialisation,
                executor dispatch and rclpy's C-to-Python message conversion.

Nothing in the callback opens a file, parses configuration or draws a random
number. The pipeline and the outgoing messages are built once, in __init__.
"""

import array
import gc
import time

import numpy as np
import rclpy
from grasp_core import GraspPipeline
from grasp_msgs.msg import GraspLatency
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = 7
DEPTH_ENCODING = '16UC1'
DEPTH_BYTES = 2

# Index order of GraspLatency.stage_ns, which is the key order docs/FORMATS.md
# fixes for stage_ns and the order of the STAGE_* constants in the message.
STAGE_KEYS = (
    'decode', 'deproject', 'transform_crop', 'plane',
    'cluster', 'grasp', 'ik', 'traj',
)


def _stage_ns(stage):
    """Read the core's per-stage breakdown as a list in STAGE_KEYS order.

    Called after the measured region has closed, so its cost is not attributed
    to the pipeline. Accepts either a mapping or an attribute-bearing object,
    since the core is free to return whichever reads better on its side.
    """
    if isinstance(stage, dict):
        return [int(stage[k]) for k in STAGE_KEYS]
    return [int(getattr(stage, k)) for k in STAGE_KEYS]


class GraspNode(Node):

    def __init__(self):
        super().__init__('grasp_node')

        # Declared with a type and no default, so an unset one raises
        # ParameterUninitializedException on read instead of silently starting a
        # node that measures the wrong thing.
        self.declare_parameter('config_path', Parameter.Type.STRING)
        self.declare_parameter('chain_path', Parameter.Type.STRING)
        self.declare_parameter('ransac_table_path', Parameter.Type.STRING)
        # Joint names are a constant of the experiment and live in
        # assets/pipeline_config.json under trajectory.joint_names; taking them
        # as a parameter keeps this node from growing a second reader of that
        # file, and docker/run_ros_benchmark.sh feeds both nodes from it.
        self.declare_parameter('joint_names', Parameter.Type.STRING_ARRAY)
        self.declare_parameter('latency_queue_depth', 256)

        self._joint_names = list(self.get_parameter('joint_names').value)
        if len(self._joint_names) != JOINTS:
            raise RuntimeError(
                'joint_names must list exactly 7 joints, sourced from '
                'assets/pipeline_config.json:trajectory.joint_names')

        self._pipeline = GraspPipeline(
            self.get_parameter('config_path').value,
            self.get_parameter('chain_path').value,
            self.get_parameter('ransac_table_path').value)

        self._traj = JointTrajectory()
        self._traj.joint_names = self._joint_names
        self._latency = GraspLatency()
        self._seq = 0
        self._rgb = None
        # Writable NumPy views onto the array.array buffers the outgoing
        # trajectory message already owns; see _fill_trajectory.
        self._pos = []
        self._vel = []
        self._acc = []

        self._traj_pub = self.create_publisher(
            JointTrajectory, '/grasp/joint_trajectory',
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE))

        # Reliable and deep: a lost latency sample is a hole in the dataset,
        # whereas a dropped depth frame is a legitimate measurement outcome.
        self._latency_pub = self.create_publisher(
            GraspLatency, '/grasp/latency',
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=max(1, int(self.get_parameter('latency_queue_depth').value)),
                reliability=QoSReliabilityPolicy.RELIABLE))

        # No message_filters synchroniser: grasp_core takes fx, fy, cx, cy from
        # pipeline_config.json rescaled by the image width, exactly as the
        # in-process benchmark does, so CameraInfo is not needed per frame. It
        # is latched once to confirm the stream agrees with the config, and the
        # subscription is then destroyed so it costs nothing thereafter.
        self._info_sub = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info',
            self._on_camera_info, qos_profile_sensor_data)

        self._depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw',
            self._on_depth, qos_profile_sensor_data)

    def _on_camera_info(self, msg):
        self.get_logger().info(
            f'camera_info latched: {msg.width}x{msg.height} '
            f'fx={msg.k[0]:.4f} fy={msg.k[4]:.4f} cx={msg.k[2]:.4f} cy={msg.k[5]:.4f} '
            '(grasp_core takes its own intrinsics from pipeline_config.json; a '
            'mismatch here means the bag and the config disagree)')
        self.destroy_subscription(self._info_sub)
        self._info_sub = None

    def _on_depth(self, msg):
        t_enter = time.perf_counter_ns()

        width = msg.width
        height = msg.height
        expected = width * height * DEPTH_BYTES

        if (msg.encoding != DEPTH_ENCODING or msg.is_bigendian
                or msg.step != width * DEPTH_BYTES or len(msg.data) != expected):
            self.get_logger().error(
                f"rejecting depth frame: encoding='{msg.encoding}' "
                f'is_bigendian={msg.is_bigendian} step={msg.step} '
                f'size={len(msg.data)}, expected {DEPTH_ENCODING!r}, 0, '
                f'{width * DEPTH_BYTES}, {expected}',
                throttle_duration_sec=2.0)
            return

        if self._rgb is None or self._rgb.shape[:2] != (height, width):
            # The depth topic is the only input carried over ROS, but the core's
            # signature takes an RGB plane because no stage of the algorithm
            # reads it (docs/ALGORITHM.md S0-S7). Allocated once, on the first
            # frame, which the analysis discards as warm-up.
            self._rgb = np.zeros((height, width, 3), dtype=np.uint8)

        # rclpy hands over an array.array('B'); frombuffer wraps it without
        # copying, which is the honest equivalent of the C++ pointer cast. The
        # cost of doing so lands inside compute_ns, where it belongs.
        depth = np.frombuffer(msg.data, dtype='<u2').reshape(height, width)

        result = self._pipeline.run(depth, self._rgb)

        self._fill_trajectory(result)

        compute_ns = time.perf_counter_ns() - t_enter
        t_publish = self.get_clock().now()

        self._traj_pub.publish(self._traj)

        latency = self._latency
        latency.stamp = msg.header.stamp
        latency.seq = self._seq
        self._seq += 1
        latency.compute_ns = compute_ns
        e2e = t_publish.nanoseconds - Time.from_msg(msg.header.stamp).nanoseconds
        latency.end_to_end_ns = e2e if e2e > 0 else 0
        # Fixed-size uint64 arrays arrive from rosidl_generator_py as NumPy
        # arrays; writing through the slice reuses them instead of allocating.
        latency.stage_ns[:] = _stage_ns(result.stage_ns)
        stats = gc.get_stats()
        latency.gc_collections[:] = (
            stats[0]['collections'], stats[1]['collections'],
            stats[2]['collections'])
        latency.graspable = bool(result.graspable)
        latency.converged = bool(result.converged)
        self._latency_pub.publish(latency)

    def _fill_trajectory(self, result):
        """Copy the core's waypoints into the reused JointTrajectory message.

        The waypoint count is a config constant, so the JointTrajectoryPoint
        objects and their array buffers are built on the first frame only.
        """
        points = self._traj.points
        if len(points) != len(result.trajectory):
            points = []
            self._pos, self._vel, self._acc = [], [], []
            for _ in range(len(result.trajectory)):
                p = JointTrajectoryPoint()
                p.positions = array.array('d', [0.0] * JOINTS)
                p.velocities = array.array('d', [0.0] * JOINTS)
                p.accelerations = array.array('d', [0.0] * JOINTS)
                # Unbounded float64 sequences are array.array('d') in rclpy, and
                # array.array exposes a writable buffer, so a NumPy view over it
                # writes straight into the message. Assigning through the
                # generated setter instead would build a fresh array per field
                # per waypoint: 60 allocations a frame at 20 waypoints.
                self._pos.append(np.frombuffer(p.positions, dtype=np.float64))
                self._vel.append(np.frombuffer(p.velocities, dtype=np.float64))
                self._acc.append(np.frombuffer(p.accelerations, dtype=np.float64))
                points.append(p)
            self._traj.points = points

        self._traj.header.stamp = self.get_clock().now().to_msg()
        for k, src in enumerate(result.trajectory):
            self._pos[k][:] = src.position
            self._vel[k][:] = src.velocity
            self._acc[k][:] = src.acceleration
            sec = int(src.time_from_start)
            stamp = points[k].time_from_start
            stamp.sec = sec
            stamp.nanosec = int(round((src.time_from_start - sec) * 1e9))


def main(args=None):
    rclpy.init(args=args)
    node = GraspNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
