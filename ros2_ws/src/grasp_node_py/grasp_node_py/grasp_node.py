"""rclpy front end for the Python grasp pipeline, as a managed node.

The measurement boundaries mirror grasp_node_cpp exactly, because a comparison
between the two is only worth anything if both stop the clock in the same place:

  compute_ns    starts at subscription-callback entry, before any validation,
                and stops immediately before JointTrajectory.publish(). It
                includes turning ``msg.data`` into NumPy views and building the
                outgoing message: costs the in-process benchmark never pays,
                and precisely the rclpy overhead this repository is measuring.
  end_to_end_ns is the node clock at that same instant minus the depth header
                stamp, so it also carries DDS transport, CDR deserialisation,
                executor dispatch and rclpy's C-to-Python message conversion.

Nothing in the callback opens a file, parses configuration or draws a random
number. The pipeline and the outgoing messages are built in on_configure.

This node has no composed variant, and that absence is a result rather than an
omission. Component containers load C++ shared libraries, and rclpy implements
no intra-process transport at all, so the zero-copy path the C++ node is
measured on is not available to a Python node at any price. docs/ROS2.md says
what that is worth in microseconds.
"""

import array
import gc
import time

import numpy as np
import rclpy
from grasp_msgs.msg import GraspLatency
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
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

RELIABILITY = {
    'reliable': QoSReliabilityPolicy.RELIABLE,
    'best_effort': QoSReliabilityPolicy.BEST_EFFORT,
}
DURABILITY = {
    'volatile': QoSDurabilityPolicy.VOLATILE,
    'transient_local': QoSDurabilityPolicy.TRANSIENT_LOCAL,
}

# Defaults reproduce config/qos.yaml, so a node started without the file
# behaves the way the recorded runs did rather than differently.
QOS_DEFAULTS = {
    'depth': ('best_effort', 'volatile', 5),
    'camera_info': ('best_effort', 'volatile', 5),
    'trajectory': ('reliable', 'volatile', 1),
    'latency': ('reliable', 'volatile', 256),
}


def _stage_ns(stage):
    """Read the core's per-stage breakdown as a list in STAGE_KEYS order.

    Called after the measured region has closed, so its cost is not attributed
    to the pipeline. Accepts either a mapping or an attribute-bearing object,
    since the core is free to return whichever reads better on its side.
    """
    if isinstance(stage, dict):
        return [int(stage[k]) for k in STAGE_KEYS]
    return [int(getattr(stage, k)) for k in STAGE_KEYS]


class GraspNode(LifecycleNode):
    """The rclpy half of the experiment.

    Construction declares parameters and nothing else. The pipeline reads three
    files and sizes every scratch array, which is tens of milliseconds that
    must not be attributable to a frame and must not happen before the executor
    exists, so it happens in on_configure and is given back in on_cleanup.
    """

    def __init__(self):
        super().__init__('grasp_node')

        self.declare_parameter('config_path', '')
        self.declare_parameter('chain_path', '')
        self.declare_parameter('ransac_table_path', '')
        # Joint names are a constant of the experiment and live in
        # assets/pipeline_config.json under trajectory.joint_names; taking them
        # as a parameter keeps this node from growing a second reader of that
        # file, and docker/run_ros_benchmark.sh feeds both nodes from it.
        # Declared with a type and no default so that an unset one is an error
        # rather than an empty list that quietly measures the wrong thing.
        self.declare_parameter('joint_names', Parameter.Type.STRING_ARRAY)

        for key, (reliability, durability, depth) in QOS_DEFAULTS.items():
            self.declare_parameter(f'qos.{key}.reliability', reliability)
            self.declare_parameter(f'qos.{key}.durability', durability)
            self.declare_parameter(f'qos.{key}.history_depth', depth)

        self.declare_parameter('smoke_test', False)

        self._pipeline = None
        self._joint_names = []
        self._traj = None
        self._latency = None
        self._seq = 0
        self._rgb = None
        self._pos = []
        self._vel = []
        self._acc = []
        self._traj_pub = None
        self._latency_pub = None
        self._info_sub = None
        self._depth_sub = None

    @property
    def smoke_test(self):
        return bool(self.get_parameter('smoke_test').value)

    def configurable(self):
        """Whether on_configure has any chance of succeeding.

        The smoke path asks this so it can walk the transition square when the
        parameters happen to be present and skip it when they are not.
        """
        return bool(self.get_parameter('config_path').value)

    def _qos(self, key):
        reliability = self.get_parameter(f'qos.{key}.reliability').value
        durability = self.get_parameter(f'qos.{key}.durability').value
        depth = int(self.get_parameter(f'qos.{key}.history_depth').value)
        if reliability not in RELIABILITY:
            raise ValueError(
                f"qos.{key}.reliability must be one of {sorted(RELIABILITY)}, "
                f'got {reliability!r}')
        if durability not in DURABILITY:
            raise ValueError(
                f"qos.{key}.durability must be one of {sorted(DURABILITY)}, "
                f'got {durability!r}')
        if depth < 1:
            raise ValueError(f'qos.{key}.history_depth must be at least 1, got {depth}')
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=depth,
            reliability=RELIABILITY[reliability], durability=DURABILITY[durability])

    def on_configure(self, state):
        try:
            joint_names = list(self.get_parameter('joint_names').value)
        except ParameterUninitializedException:
            self.get_logger().error(
                'joint_names is unset; it is sourced from '
                'assets/pipeline_config.json:trajectory.joint_names')
            return TransitionCallbackReturn.FAILURE

        config_path = self.get_parameter('config_path').value
        chain_path = self.get_parameter('chain_path').value
        ransac_table_path = self.get_parameter('ransac_table_path').value
        if not (config_path and chain_path and ransac_table_path):
            self.get_logger().error(
                'config_path, chain_path and ransac_table_path are all required; '
                'refusing to configure rather than measuring a pipeline built '
                'from defaults')
            return TransitionCallbackReturn.FAILURE
        if len(joint_names) != JOINTS:
            self.get_logger().error(
                f'joint_names lists {len(joint_names)} joints, expected {JOINTS}, '
                'sourced from assets/pipeline_config.json:trajectory.joint_names')
            return TransitionCallbackReturn.FAILURE

        try:
            # Imported here and not at module scope so that a node started with
            # smoke_test:=true proves the ROS half works even where the
            # repository's Python core is not on the path, which is exactly the
            # situation in CI.
            from grasp_core import GraspPipeline

            self._pipeline = GraspPipeline(config_path, chain_path, ransac_table_path)
            self._traj_pub = self.create_lifecycle_publisher(
                JointTrajectory, 'joint_trajectory', self._qos('trajectory'))
            # Reliable and deep: a lost latency sample is a hole in the dataset,
            # whereas a dropped depth frame is a legitimate measurement outcome.
            self._latency_pub = self.create_lifecycle_publisher(
                GraspLatency, 'latency', self._qos('latency'))
        except Exception as error:  # noqa: BLE001 - reported as a failed transition
            self.get_logger().error(f'configure failed: {error}')
            self._release()
            return TransitionCallbackReturn.FAILURE

        self._joint_names = joint_names
        self._traj = JointTrajectory()
        self._traj.joint_names = joint_names
        self._latency = GraspLatency()
        self._seq = 0
        self.get_logger().info('configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        # The base implementation flips every managed entity, which is both
        # publishers, to activated. Skipping it would leave publish() dropping
        # messages with a warning.
        result = super().on_activate(state)
        if result != TransitionCallbackReturn.SUCCESS:
            return result

        # Subscriptions live only while active, rather than being created in
        # on_configure and guarded by a state check. An inactive node that
        # still ran the pipeline would burn a core it does not need and would
        # put its own scheduling noise into the active node's tail.
        #
        # No message_filters synchroniser: grasp_core takes fx, fy, cx, cy from
        # pipeline_config.json rescaled by the image width, exactly as the
        # in-process benchmark does, so CameraInfo is not needed per frame. It
        # is latched once to confirm the stream agrees with the config, and the
        # subscription is then destroyed so it costs nothing thereafter.
        self._info_sub = self.create_subscription(
            CameraInfo, 'depth/camera_info', self._on_camera_info, self._qos('camera_info'))
        self._depth_sub = self.create_subscription(
            Image, 'depth/image_raw', self._on_depth, self._qos('depth'))
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._drop_subscriptions()
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        self._release()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._release()
        return TransitionCallbackReturn.SUCCESS

    def _drop_subscriptions(self):
        for sub in (self._depth_sub, self._info_sub):
            if sub is not None:
                self.destroy_subscription(sub)
        self._depth_sub = None
        self._info_sub = None

    def _release(self):
        """Give back everything on_configure took.

        A configure/cleanup cycle that leaves a publisher behind would still
        appear to work, and would then be found later as a node that answers on
        a topic it has no business answering on.
        """
        self._drop_subscriptions()
        for pub in (self._traj_pub, self._latency_pub):
            if pub is not None:
                self.destroy_lifecycle_publisher(pub)
        self._traj_pub = None
        self._latency_pub = None
        self._pipeline = None
        self._traj = None
        self._latency = None
        self._rgb = None
        self._pos = []
        self._vel = []
        self._acc = []
        self._joint_names = []

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
        latency.plane_found = bool(result.plane_found)
        latency.graspable = bool(result.graspable)
        latency.converged = bool(result.converged)
        latency.points = int(result.points)
        latency.cluster_points = int(result.cluster_points)
        latency.ik_iterations = int(result.iterations)
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


def _walk_transitions(node):
    """Configure, activate, deactivate, clean up, checking each landed.

    Each step runs only if the previous one succeeded: asking the state machine
    for a transition it does not currently offer raises.
    """
    steps = (
        ('configure', node.trigger_configure),
        ('activate', node.trigger_activate),
        ('deactivate', node.trigger_deactivate),
        ('cleanup', node.trigger_cleanup),
    )
    for name, trigger in steps:
        if trigger() != TransitionCallbackReturn.SUCCESS:
            node.get_logger().error(f'smoke test: {name} did not succeed')
            return 1
    node.get_logger().info(
        'smoke test: configure, activate, deactivate, cleanup all held')
    return 0


def run_smoke_test(node, timeout_s=10.0):
    """Prove the node is alive without a bag, a frame store or a pipeline.

    CI cannot supply any of those, so the most it can prove is that this
    package imports under the interpreter the image ships, reaches the graph
    and answers for its own state. That is exactly the class of failure a ROS
    package written without a ROS installation is likely to have, which is what
    makes the check worth its wall time. When the pipeline parameters happen to
    be set as well, the full transition square is walked too.
    """
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    wanted = {
        f'{node.get_fully_qualified_name()}/change_state',
        f'{node.get_fully_qualified_name()}/get_state',
    }

    deadline = time.monotonic() + timeout_s
    alive = False
    while rclpy.ok() and not alive and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
        alive = wanted <= {name for name, _ in node.get_service_names_and_types()}

    try:
        if not alive:
            node.get_logger().error(
                f'smoke test: {sorted(wanted)} did not appear in the graph within '
                f'{timeout_s:.0f} s, so the lifecycle interface is not up')
            return 1
        node.get_logger().info('smoke test: alive, lifecycle interface up')
        if not node.configurable():
            node.get_logger().info(
                'smoke test: config_path is unset, so the transition cycle was skipped')
            return 0
        return _walk_transitions(node)
    finally:
        executor.remove_node(node)


def main(args=None):
    rclpy.init(args=args)
    node = GraspNode()
    status = 0
    try:
        if node.smoke_test:
            status = run_smoke_test(node)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:  # noqa: BLE001 - a crash must exit non-zero, not abort
        node.get_logger().error(f'grasp_node failed: {error}')
        status = 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return status


if __name__ == '__main__':
    raise SystemExit(main())
