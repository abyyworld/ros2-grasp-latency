# The ROS 2 layer

The in-process benchmark answers "how long does the pipeline take in each
language". It does not answer "how long does a grasp node take", because a node
also pays for DDS transport, CDR serialisation, executor dispatch and, on the
Python side, rclpy's conversion of a C message into Python objects. Those costs
are not incidental. They are most of what people mean when they say a ROS node
is slow, and rclpy pays them differently from rclcpp.

So there is a second, smaller experiment: the same two pipelines behind
`rclcpp` and `rclpy` nodes, on the same topics, with the same QoS, measured at
the same two boundaries, producing the same timing JSONL that
[FORMATS.md](FORMATS.md) defines. `tools/analyse.py` does not need to know
which kind of run it is reading.

ROS 2 cannot be installed in the development container that produced this
repository, so **everything under `ros2_ws/` and `docker/` is built and run
inside the image `docker/Dockerfile` describes, and nowhere else.** See
[What is not verified](#what-is-not-verified) at the end.

## Packages

| Package | Type | What it is |
|---|---|---|
| `grasp_msgs` | `ament_cmake` | One message, `GraspLatency`, published per frame. |
| `grasp_core_cpp` | `ament_cmake` | Shim that re-exports the ROS-independent `grasp_core::grasp_core` target to colcon. No sources. |
| `grasp_node_cpp` | `ament_cmake` | `grasp_node` (rclcpp) and `frame_publisher`. |
| `grasp_node_py` | `ament_python` | `grasp_node` (rclpy). |

The pipeline itself lives outside the workspace, in `cpp/` and `python/`, and
has no ROS dependency of any kind. That is deliberate: the code measured
in-process and the code measured inside a node are the same code, so any
difference between the two runs is the ROS layer and nothing else.

## Topics

```
/camera/depth/image_raw     sensor_msgs/msg/Image          16UC1, SensorDataQoS
/camera/depth/camera_info   sensor_msgs/msg/CameraInfo     SensorDataQoS
/grasp/joint_trajectory     trajectory_msgs/msg/JointTrajectory   reliable, depth 1
/grasp/latency              grasp_msgs/msg/GraspLatency    reliable, depth 256
```

**There is no `message_filters` synchroniser, and that is a decision, not an
omission.** `grasp_core` derives `fx, fy, cx, cy` from `pipeline_config.json`
rescaled by the image width, exactly as the in-process benchmark does, so
`CameraInfo` is not an input to any frame. Both nodes latch it once, log it so
a disagreement between the bag and the config is visible, then destroy the
subscription. An `ApproximateTime` synchroniser would have added a pairing
buffer, a second callback, a policy question about what to do when the two
topics drift, and a queue's worth of latency, all to obtain four numbers that
never change. Depth is subscribed alone.

Sensor QoS on the depth topic is best-effort, which is what a real depth camera
driver offers. A node that cannot keep up therefore *drops* frames rather than
building an unbounded backlog. That is the honest behaviour, and the drop
count is reported. `/grasp/latency` is the opposite: reliable and deep, because
a lost latency sample is a hole in the dataset rather than a measurement.

Neither node copies the image buffer. The C++ node reinterprets
`msg->data.data()` as `const uint16_t*` and hands that pointer to the core; the
Python node wraps `msg.data` (an `array.array('B')`, per
`rosidl_generator_py`) with `np.frombuffer`, which is a view. Both allocate the
outgoing `JointTrajectory` once and refill it in place. The Python node keeps
NumPy views onto the `array.array` buffers the message already owns, because
assigning through the generated setters would build 60 fresh arrays per frame
at 20 waypoints.

## What is measured, and where the clock starts

`GraspLatency` carries two durations, and the difference between them is the
point of the whole exercise.

```
              publish                                        callback
              by driver                                        entry        publish
  ────────────────┬───────────────────────────────────────────────┬────────────┬──────►
                  │  DDS transport, CDR deserialisation,          │  pipeline  │
                  │  executor dispatch, rclpy object conversion   │  + msg     │
                  │                                               │  build     │
  header.stamp ───┘                                               └── compute_ns ──┘
                  └──────────────────── end_to_end_ns ─────────────────────────┘
```

* `compute_ns`: monotonic clock, from subscription-callback entry (before any
  validation) to immediately before `JointTrajectory::publish()`. This is the
  node's own cost. It is *larger* than the in-process `total_ns` for the same
  frame, because it also includes message inspection and building the outgoing
  trajectory, which the in-process benchmark never does.
* `end_to_end_ns`: the node clock at that same instant minus the depth header
  stamp. Everything above, plus everything ROS 2 did before the callback ran.
* `stage_ns[8]`: the core's own S0..S7 breakdown, in the key order
  `FORMATS.md` fixes. Read out *after* the compute clock stops, so
  instrumentation is not attributed to the pipeline.
* `gc_collections[3]`: cumulative CPython collections per generation, sampled
  after the measured region. Always zero from the C++ node. This is what lets a
  Python tail sample be attributed to garbage collection rather than guessed at.

Nothing inside either callback opens a file, parses configuration, or draws a
random number. The pipeline, the RGB scratch plane and both outgoing messages
are built in the constructor.

### The bag playback caveat: read this before quoting end_to_end_ns

`ros2 bag play` republishes the bytes it recorded, header stamp included. The
Jazzy play verb offers `--rate`, `--loop`, `--clock` and `--start-offset`, and
nothing at all that rewrites a field inside a message. So under bag playback

```
end_to_end_ns  =  true latency  +  (playback start - bag time origin)
```

and that second term is a large, unknown constant. **From a `--driver bag` run,
`total_ns` is sound and `end_to_end_ns` is not an absolute latency.** Its
*spread* is still meaningful, because the constant cancels in
`p99 - p50`, so bag runs are the right way to compare the jitter the two
languages induce, and the wrong way to quote a floor.

`--driver live` exists for the floor. `grasp_node_cpp/frame_publisher` reads
the frame store into memory at start-up and publishes it at `deadline_hz` with
`now()` in the header, so `end_to_end_ns` from such a run is the real
wall-clock figure. It costs one extra process on a four-core box; see
[Confounds](#confounds).

## Running it

Build the image from the repository root (the context must be the root, and
`data/` is deliberately not copied into it):

```bash
docker build -f docker/Dockerfile -t grasp-latency-ros .
```

Generate the frame store and the input bag on the host, then bind-mount them:

```bash
python3 harness/make_dataset.py --name table_640x480        # writes data/table_640x480/
docker run --rm -it \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/results:/workspace/results" \
  grasp-latency-ros \
  docker/run_ros_benchmark.sh --driver bag
```

That runs the two nodes **in turn**, never together, because on four cores a
concurrent run would measure the scheduler. It writes

```
results/ros2_cpp.timing.jsonl
results/ros2_py.timing.jsonl
```

in the `FORMATS.md` timing format, with `"transport":"ros2"` marking them as
ROS runs. Feed them to the same analysis as the in-process runs.

For an absolute end-to-end number, no input bag is needed:

```bash
docker run --rm -it -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" \
  grasp-latency-ros docker/run_ros_benchmark.sh --driver live
```

To run one node by hand:

```bash
ros2 run grasp_node_cpp grasp_node --ros-args --params-file /tmp/params.yaml
ros2 run grasp_node_py  grasp_node --ros-args --params-file /tmp/params.yaml
```

`run_ros_benchmark.sh` generates that parameter file from
`assets/pipeline_config.json` and the dataset manifest. Both nodes take
`config_path`, `chain_path`, `ransac_table_path` and `joint_names`; the joint
names are a parameter rather than a second JSON read so that `grasp_core`
stays the only reader of the config in the C++ stack. All four are declared
without defaults: an unset one aborts start-up instead of quietly measuring the
wrong thing.

Converting a recording by hand, on any machine with `rosbags` installed and no
ROS at all:

```bash
python3 docker/latency_bag_to_jsonl.py \
  --bag  /path/to/recorded_bag \
  --msg  ros2_ws/src/grasp_msgs/msg/GraspLatency.msg \
  --out  results/ros2_py.timing.jsonl \
  --impl py --dataset table_640x480 --frame-count 100 --warmup 100 --limit 2000
```

Recording with rosbag2's C++ recorder and converting offline keeps every byte
of file I/O and every Python subscriber away from the node under measurement.
The converter registers `GraspLatency` from the `.msg` file in the workspace,
so it cannot drift from the message it is decoding.

## Confounds

Stated rather than hidden, because they bound what the numbers mean.

* **Four cores, three processes.** Node, driver and recorder compete. The
  recorder is rosbag2's C++ one and handles ~100 small messages a second, so it
  is cheap; the driver is not. Pin them if you care about the tail:
  `taskset -c 0 ros2 run ...` for the node, `taskset -c 1` for the driver.
* **Depth only.** The topic list carries depth. The core's signature takes an
  RGB plane, but no stage of the algorithm reads it (`ALGORITHM.md` S0-S7), so
  both nodes hand it a zeroed buffer allocated once. The ROS runs therefore do
  not pay the RGB half of S0 that the in-process runs pay: a small, constant,
  favourable-to-ROS bias in `stage_ns.decode`.
* **`compute_ns` is not `total_ns`.** It brackets more work. Compare ROS runs
  to ROS runs for the language question, and `end_to_end_ns - compute_ns`
  against zero for the ROS-overhead question.
* **RMW.** `rmw_fastrtps_cpp`, Jazzy's default, with
  `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` so no traffic leaves the host.
  Cyclone DDS will give different transport numbers; set `RMW_IMPLEMENTATION`
  and rerun if that is the question you have.
* **Sequence numbers, not header sequence.** ROS 2 removed `Header.seq`, so
  `GraspLatency.seq` is the node's own counter. A gap in it means the transport
  or the recorder lost a sample, and the converter says so.

## What is not verified

Honest inventory. None of the following was compiled, launched or executed,
because `packages.ros.org` is unreachable from the container this repository
was written in and ROS 2 therefore cannot be installed here.

* No package in `ros2_ws/` has been built. `colcon build` has never run.
* Neither node has been launched, so no topic connection, QoS match or
  parameter load has been observed.
* The Docker image has never been built.
* `run_ros_benchmark.sh` has never been run end to end.

What *was* verified, here, by running it:

* Every `package.xml` parses and validates under `catkin_pkg` (the same parser
  ROS tooling uses) and is well-formed per `xmllint`.
* `GraspLatency.msg` parses under a real ROS 2 IDL parser (`rosbags`), and a
  bag of such messages round-trips through CDR serialisation and back into the
  JSONL that `FORMATS.md` specifies, including the sequence-gap warning.
* Every Python file compiles under `python3 -m py_compile`.
* Both shell scripts are clean under `shellcheck -S style` and `bash -n`.
* Both embedded Python generators in `run_ros_benchmark.sh` were executed
  against the real `pipeline_config.json`; the parameter YAML they emit parses
  and carries the right types.
* Every message type, field name and QoS symbol used here was checked against
  the Jazzy sources rather than recalled, including that
  `sensor_msgs/Image.data` is `array.array('B')` in rclpy while
  `uint64[8]` is a NumPy array, and that the Jazzy `ros2 bag play` verb has no
  timestamp-rewriting option.
