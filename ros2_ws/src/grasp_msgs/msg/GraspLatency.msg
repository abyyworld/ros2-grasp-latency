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

# Frames this node has processed, from zero. ROS 2 dropped Header.seq, so there
# is no per-message number coming in and this is the node's own count going out.
#
# Read what that does and does not detect. A gap here means a latency message
# was lost between this node and the recorder. A frame the node never received,
# because the queue overflowed under best-effort QoS, produces no message and
# therefore no gap: this counter cannot see it. Loss of that kind is measured
# by counting records against the frames the driver published, and the frame a
# record belongs to is recovered from its own point and cluster counts. See
# docs/ROS2.md.
uint32 seq

uint64 end_to_end_ns
uint64 compute_ns

# The core's own per-stage breakdown, indexed by the STAGE_* constants above.
# The order is the key order docs/FORMATS.md fixes for stage_ns.
uint64[8] stage_ns

# Cumulative CPython collections per generation, sampled after the measured
# region closes. Always zero from the C++ node.
uint64[3] gc_collections

# The shape of the frame's answer, so a ROS run's timing JSONL carries the same
# columns docs/FORMATS.md fixes for an in-process run and the analysis needs no
# special case for ROS records. These are outputs of the pipeline, not timings:
# they are here because they are the only way to tell a run that was fast
# because the machine was fast from one that was fast because the plane fit
# gave up early.
bool plane_found
bool graspable
bool converged

# Points entering S3, that is after deprojection and the workspace crop.
uint32 points
uint32 cluster_points

# Damped-least-squares iterations S6 took. Named ik_iterations rather than
# iterations to match the timing JSONL key.
uint32 ik_iterations
