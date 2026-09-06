# One record per depth frame processed by a grasp node. Both the rclcpp and the
# rclpy node publish this and nothing else on /grasp/latency, so the two
# implementations are compared on identical evidence.
#
# The clock boundaries are the point of this message:
#
#   stamp          header stamp of the depth image, set by whoever published the
#                  frame. Under `ros2 bag play` this is the stamp baked into the
#                  bag at record time and not a live clock reading. Read
#                  docs/ROS2.md before believing end_to_end_ns.
#   end_to_end_ns  node clock at the instant the JointTrajectory is handed to
#                  the publisher, minus stamp. Covers DDS transport,
#                  deserialisation, executor dispatch and compute.
#   compute_ns     monotonic elapsed time from subscription-callback entry to
#                  that same instant. The node's own cost, transport excluded.
#
# end_to_end_ns - compute_ns is everything ROS 2 adds ahead of the callback,
# which is exactly what the in-process benchmark cannot see.

uint8 STAGE_DECODE=0
uint8 STAGE_DEPROJECT=1
uint8 STAGE_TRANSFORM_CROP=2
uint8 STAGE_PLANE=3
uint8 STAGE_CLUSTER=4
uint8 STAGE_GRASP=5
uint8 STAGE_IK=6
uint8 STAGE_TRAJ=7

builtin_interfaces/Time stamp

# Frames this node has processed, from zero. ROS 2 dropped Header.seq, and the
# analysis needs a gap-detectable counter to tell a slow node from a lossy one.
uint32 seq

uint64 end_to_end_ns
uint64 compute_ns

# The core's own per-stage breakdown, indexed by the STAGE_* constants above.
# The order is the key order docs/FORMATS.md fixes for stage_ns.
uint64[8] stage_ns

# Cumulative CPython collections per generation, sampled after the measured
# region closes. Always zero from the C++ node.
uint64[3] gc_collections

bool graspable
bool converged
