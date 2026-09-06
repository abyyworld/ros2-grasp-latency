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
| `grasp_node_cpp` | `ament_cmake` | `GraspNode` and `FramePublisher`, each built as a component library and wrapped in an executable. `launch/`, `config/`. |
| `grasp_node_py` | `ament_python` | `grasp_node` (rclpy). `launch/`, `config/`. |

Both node packages follow the same layout, and both `config/` directories hold
the same two files:

```
grasp_node_cpp/
  include/grasp_node_cpp/{grasp_node,frame_publisher,qos}.hpp
  src/{grasp_node,frame_publisher}.cpp          component libraries
  src/{grasp_node,frame_publisher}_main.cpp     standalone executables
  launch/{grasp_node,frame_publisher,composed}.launch.py
  config/{qos,topics}.yaml
grasp_node_py/
  grasp_node_py/grasp_node.py
  launch/grasp_node.launch.py
  config/{qos,topics}.yaml
```

The pipeline itself lives outside the workspace, in `cpp/` and `python/`, and
has no ROS dependency of any kind. That is deliberate: the code measured
in-process and the code measured inside a node are the same code, so any
difference between the two runs is the ROS layer and nothing else.

## Topics and QoS, which live in `config/` and not in the source

Neither node names a topic or a QoS policy in its source. It subscribes and
publishes under short relative names, the launch files map those onto the names
the bags and the driver use, and the profiles arrive as ordinary ROS parameters:

| Node name | Remapped to | Type | QoS |
|---|---|---|---|
| `depth/image_raw` | `/camera/depth/image_raw` | `sensor_msgs/msg/Image` | best effort, keep last 5 |
| `depth/camera_info` | `/camera/depth/camera_info` | `sensor_msgs/msg/CameraInfo` | best effort, keep last 5 |
| `joint_trajectory` | `/grasp/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` | reliable, keep last 1 |
| `latency` | `/grasp/latency` | `grasp_msgs/msg/GraspLatency` | reliable, keep last 256 |

`config/topics.yaml` is the left two columns, read by the launch files at
description time. `config/qos.yaml` is the right one, loaded as parameters
(`qos.<topic>.reliability`, `.durability`, `.history_depth`) so that
`ros2 param get` on a running node reports the profile it is actually using
rather than the one the source says it should have.

The two packages must agree on all of it, because a reliability or a queue
depth that differs between them would turn a language comparison into a DDS
policy comparison and would not announce itself as an error, only as a
different number. So both packages carry byte-identical copies and
`docker/run_ros_benchmark.sh` refuses to start a run unless they still match
each other and unless the depth and camera-info entries still match
`assets/pipeline_config.json:ros`, which is what the bag writer used.

Best effort on the depth topic is what a real depth camera driver offers. A
node that cannot keep up therefore *drops* frames rather than building an
unbounded backlog. That is the honest behaviour, and the drop count is
reported. `latency` is the opposite: reliable and deep, because a lost latency
sample is a hole in the dataset rather than a measurement.

**There is no `message_filters` synchroniser, and that is a decision, not an
omission.** `grasp_core` derives `fx, fy, cx, cy` from `pipeline_config.json`
rescaled by the image width, exactly as the in-process benchmark does, so
`CameraInfo` is not an input to any frame. Both nodes latch it once, log it so
a disagreement between the bag and the config is visible, then destroy the
subscription. An `ApproximateTime` synchroniser would have added a pairing
buffer, a second callback, a policy question about what to do when the two
topics drift, and a queue's worth of latency, all to obtain four numbers that
never change. Depth is subscribed alone.

Neither node copies the image buffer. The C++ node reinterprets
`msg->data.data()` as `const uint16_t*` and hands that pointer to the core; the
Python node wraps `msg.data` (an `array.array('B')`, per
`rosidl_generator_py`) with `np.frombuffer`, which is a view. Both allocate the
outgoing `JointTrajectory` once and refill it in place. The Python node keeps
NumPy views onto the `array.array` buffers the message already owns, because
assigning through the generated setters would build 60 fresh arrays per frame
at 20 waypoints.

## Both nodes are managed nodes

`GraspNode` is an `rclcpp_lifecycle::LifecycleNode` and its rclpy counterpart
is an `rclpy.lifecycle.LifecycleNode`. This is not ceremony. Building the
pipeline reads `pipeline_config.json`, the flattened chain and the RANSAC
deviate table, sizes every scratch buffer for the configured resolution and
solves one forward-kinematics call for the wrist reference. That is tens of
milliseconds of work which must not be attributable to a frame, and a plain
node does it in its constructor, before the executor exists, where there is no
instant at which the node can be observed to be ready and no way to report a
failure except by throwing out of `main`.

| Transition | C++ and Python both |
|---|---|
| constructor | declares parameters. Nothing else: no file is opened. |
| `on_configure` | builds the pipeline, creates both lifecycle publishers. Returns `FAILURE`, with a reason on the log, if a path is unset or a file will not load. |
| `on_activate` | activates the publishers through the base implementation, then creates the depth and camera-info subscriptions. |
| `on_deactivate` | destroys the subscriptions, deactivates the publishers. |
| `on_cleanup`, `on_shutdown` | releases the publishers, the pipeline and every buffer. |

Subscriptions are created on activation rather than on configuration and
guarded by a state check, which is the other common shape. An inactive node
that still ran the pipeline would burn a core it does not need, and on four
cores that core is the one the active node wanted.

This is also what removes a blind `sleep 5` from the benchmark script. It polls
`ros2 lifecycle get` until the node reports `active`, so no frame is ever
published at a node that is still parsing JSON, and the warm-up window is not
silently shifted by however long that took on the day.

The cost of the lifecycle machinery inside the measured region is one branch:
`LifecyclePublisher::publish` checks an atomic activation flag before
forwarding to `rclcpp::Publisher::publish`. That branch is inside `compute_ns`
and is paid by every frame, which is correct: it is what a managed node costs.

### `smoke_test`

Both nodes take a boolean parameter `smoke_test`. With it set, the node comes
up, waits for its own `change_state` and `get_state` services to appear in the
graph, confirms it is in `unconfigured`, and exits 0. With it set *and* the
pipeline parameters supplied, it also walks configure, activate, deactivate,
cleanup and checks each landed where it should, which exercises the build and
release paths as well.

That is what CI runs, because CI has no frame store and no bag. What it proves
is narrow and worth stating: the binary links against the RMW the image ships,
reaches the graph, and answers for itself. Those are exactly the failures a ROS
package written on a machine without ROS is likely to have.

The rclpy node imports `grasp_core` inside `on_configure` rather than at module
scope for the same reason: a smoke run has to prove the ROS half works even
where the repository's own Python core is not on `PYTHONPATH`.

## Composition, and what intra-process delivery is worth

`GraspNode` and `FramePublisher` are both registered with
`rclcpp_components_register_nodes`, so both can be loaded into one
`component_container`. `composed.launch.py` does exactly that, and takes one
argument:

```bash
ros2 launch grasp_node_cpp composed.launch.py intra_process:=false
ros2 launch grasp_node_cpp composed.launch.py intra_process:=true
```

Both configurations come from the same launch file with the flag flipped,
deliberately. Two separate files would be two things that can drift, and the
whole value of the comparison rests on the claim that the flag is the only
difference between the two runs.

With the flag off, a 640x480 depth frame is serialised to CDR, handed to
Fast DDS, and deserialised again, all within one process. With it on, and with
exactly one subscriber, rclcpp moves the publisher's `unique_ptr` straight into
the subscription queue: no serialisation, no RMW, no copy of the 614 KB pixel
buffer. `FramePublisher` publishes a `unique_ptr` rather than a const reference
for precisely this reason, and it writes the header stamp *after* taking its
copy out of the frame store, so the copy is outside every latency reported.

The container is the single-threaded `component_container`.
`component_container_mt` would let the driver's timer and the consumer's
callback run on separate threads, which on four cores is a different
experiment.

Three modes are therefore measured, and `harness/analyze.py` separates them by
the `transport` field alone:

| Mode | `transport` | What it is |
|---|---|---|
| `standalone` | `ros2` | Node, driver and recorder in three processes. |
| `composed` | `ros2_composed` | Node and driver in one container, delivery through the RMW. |
| `composed_ipc` | `ros2_composed_ipc` | The same container, intra-process delivery on. |

`ros2_composed` against `ros2` isolates the cost of crossing a process
boundary; `ros2_composed_ipc` against `ros2_composed` isolates the cost of
serialisation and the RMW hop, with the process boundary already gone.

**The Python node has no composed variant and cannot have one.** A component
container loads C++ shared libraries, and rclpy implements no intra-process
transport at all, so this optimisation is not available to a Python node at any
price. That is a result rather than a gap in the experiment, and it is why the
comparison is reported as `cpp [ros2]` against `py [ros2]` for the language
question, with the composed rows standing beside them to say how much of the
C++ node's remaining advantage a Python node could never close.

## What is measured, and where the clock starts

`GraspLatency` carries two durations, and the difference between them is the
point of the whole exercise.

```
        driver                                     callback
        publishes                                  entry              publish
  ----------+------------------------------------------+-----------------+------->
            |  DDS transport, CDR deserialisation,     |  pipeline       |
            |  executor dispatch, rclpy conversion     |  + msg build    |
            |                                          |                 |
  header.stamp                                         |<- compute_ns ->|
            |                                                            |
            |<--------------------- end_to_end_ns --------------------->|
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
* `plane_found`, `graspable`, `converged`, `points`, `cluster_points`,
  `ik_iterations`: what the frame's answer was. These are not timings and they
  are carried for one reason: `FORMATS.md` fixes them as columns of the timing
  record, and without them a ROS run's JSONL would be a shape
  `harness/analyze.py` could not read. They are also the only way to tell a
  frame that was quick because the machine was quick from one that was quick
  because the plane fit gave up early.

Nothing inside either callback opens a file, parses configuration, or draws a
random number. The pipeline, the RGB scratch plane and both outgoing messages
are built in `on_configure`, before the node can be activated and therefore
before a frame can reach it.

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
  docker/run_ros_benchmark.sh --mode all
```

`--mode all` runs four arms **in turn**, never together, because on four cores
a concurrent run would measure the scheduler:

```
results/ros2_cpp.timing.jsonl                the rclcpp node, own process
results/ros2_py.timing.jsonl                 the rclpy node, own process
results/ros2_composed_cpp.timing.jsonl       composed, RMW delivery
results/ros2_composed_ipc_cpp.timing.jsonl   composed, intra-process delivery
```

all in the `FORMATS.md` timing format and all distinguished by their
`transport` field, so the whole set can be handed to `harness/analyze.py` at
once and comes back as four groups.

`--mode all` implies `--driver live`, and so does either composed mode on its
own. That is not a convenience: a bag player is a separate process, so it
cannot be composed away, and comparing a composed run against a bag-driven one
would be comparing two things that differ in two ways. `--mode standalone`
alone still defaults to `--driver bag`, which is the cheaper run and the right
one for the jitter question.

Single arms, for when only one number is wanted:

```bash
docker/run_ros_benchmark.sh --mode standalone --driver bag     # the default
docker/run_ros_benchmark.sh --mode standalone --driver live
docker/run_ros_benchmark.sh --mode composed
docker/run_ros_benchmark.sh --mode composed_ipc
```

Before it starts anything, the script checks that the two packages' `config/`
directories still match each other and that the topic table still matches
`assets/pipeline_config.json:ros`. It then generates the parameter file from
that same config and the dataset manifest, and drives every lifecycle
transition itself, polling `ros2 lifecycle get` rather than sleeping.

To run one node by hand:

```bash
ros2 launch grasp_node_cpp grasp_node.launch.py params_file:=/tmp/params.yaml
ros2 launch grasp_node_py  grasp_node.launch.py params_file:=/tmp/params.yaml
ros2 launch grasp_node_cpp composed.launch.py  params_file:=/tmp/params.yaml \
                                               intra_process:=true
```

The two standalone launch files configure and activate on their own
(`autostart:=false` to stop them). The composed one does not, because
launch_ros drives transitions by emitting events at `LifecycleNode` *actions*
and a component inside a container is not one, so a composed node is
transitioned with `ros2 lifecycle set /grasp_node configure` and then
`activate`.

Both nodes take `config_path`, `chain_path`, `ransac_table_path` and
`joint_names`. The joint names are a parameter rather than a second JSON read
so that `grasp_core` stays the only reader of the config in the C++ stack. An
unset one fails the configure transition with a reason on the log rather than
quietly measuring the wrong thing.

Converting a recording by hand, on any machine with `rosbags` installed and no
ROS at all:

```bash
python3 docker/latency_bag_to_jsonl.py \
  --bag  /path/to/recorded_bag \
  --msg  ros2_ws/src/grasp_msgs/msg/GraspLatency.msg \
  --out  results/ros2_py.timing.jsonl \
  --impl py --transport ros2 \
  --dataset table_640x480 --frame-count 100 --warmup 100 --limit 2000
```

`--transport` is what keeps the four arms apart in the analysis, so it has to
be right: `ros2`, `ros2_composed` or `ros2_composed_ipc`.

Recording with rosbag2's C++ recorder and converting offline keeps every byte
of file I/O and every Python subscriber away from the node under measurement.
The converter registers `GraspLatency` from the `.msg` file in the workspace,
so it cannot drift from the message it is decoding.

## Confounds

Stated rather than hidden, because they bound what the numbers mean.

* **Four cores, three processes.** In the standalone arms the node, the driver
  and the recorder compete. The recorder is rosbag2's C++ one and handles about
  a hundred small messages a second, so it is cheap; the driver is not. Pin
  them if you care about the tail: `taskset -c 0 ros2 launch ...` for the node,
  `taskset -c 1` for the driver. The composed arms are two processes rather
  than three, which is part of what `ros2_composed` against `ros2` measures.
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
* **The recorder is never composed.** `/grasp/latency` leaves the node through
  the RMW in all four arms, because `ros2 bag record` is its own process. So
  the intra-process arm is intra-process on the depth topic only. That is the
  topic the comparison is about, and holding the latency path constant across
  arms is what makes them comparable, but the composed numbers are not "a
  process with no DDS in it".
* **Composition changes the executor, not only the transport.** In a container
  the driver's timer and the consumer's callback share one single-threaded
  executor, so a frame cannot arrive while the previous one is still being
  processed: the driver's own tick is queued behind it. In the standalone arm
  the driver is a separate process and has its own. `ros2_composed` against
  `ros2` therefore carries both effects, which is why `ros2_composed_ipc`
  against `ros2_composed` is the cleaner of the two comparisons.

## What is not verified

Honest inventory. None of the following was compiled, launched or executed,
because `packages.ros.org` is unreachable from the container this repository
was written in and ROS 2 therefore cannot be installed here.

* No package in `ros2_ws/` has been built. `colcon build` has never run, so
  nothing here is known to compile.
* Neither node has been launched. No lifecycle transition has actually fired,
  no topic connection or QoS match has been observed, and no parameter file has
  been loaded by a real node.
* No component has been loaded into a container, and **no intra-process
  delivery has ever happened**. The claim that publishing a `unique_ptr` to a
  single subscriber avoids the copy is read off the rclcpp sources, not
  measured. Until a run exists, this repository quotes no number for it.
* Intra-process delivery and lifecycle nodes are known to interact badly in
  places. rclcpp issue 2721 reports failures shutting a lifecycle node down
  cleanly with intra-process comms enabled, both composed and separate; issue
  2872 records that publishing a `shared_ptr` copies even under `std::move`,
  which is why `FramePublisher` publishes a `unique_ptr`. Neither has been
  reproduced or ruled out here, and if `--mode composed_ipc` turns out to hang
  on teardown, that is where to look first.
* The launch files have never been evaluated by `ros2 launch`. In particular
  `autostart=` on `launch_ros.actions.LifecycleNode` and
  `ParameterValue(..., value_type=bool)` for `use_intra_process_comms` were
  checked against the Jazzy sources and not against a running launch service.
* The Docker image has never been built.
* `run_ros_benchmark.sh` has never been run against a real ROS installation.

What *was* verified, here, by running it:

* Every `package.xml` parses and validates under `catkin_pkg`, the same parser
  ROS tooling uses.
* `GraspLatency.msg`, with the four output fields added for the analysis,
  parses under a real ROS 2 IDL parser (`rosbags`) and round-trips through CDR
  serialisation.
* A synthetic `/grasp/latency` rosbag2 was written and fed through
  `docker/latency_bag_to_jsonl.py`, and the JSONL it produced was then read by
  `harness/analyze.py`, which grouped it under the `transport` tag it was given.
  That is the whole output path, end to end, without ROS.
* `run_ros_benchmark.sh` was run in full, in `--mode all`, against a stub `ros2`
  that answers `launch`, `lifecycle get/set` and `bag record`. All four arms
  completed, the transitions were requested in the intended order, and the four
  timing files came out named and tagged as this document says. That exercises
  the script's logic; it says nothing about the real `ros2` commands underneath.
* Its refusals were exercised too: a composed mode with `--driver bag`, a
  `qos.yaml` that differs between the two packages, and a `topics.yaml` that
  disagrees with `pipeline_config.json:ros` each stop the run with a reason.
* Every Python file compiles under `python3 -m py_compile`, launch files
  included.
* Both shell scripts are clean under `shellcheck -S style` and `bash -n`.
* Every ROS 2 API used here was read out of the Jazzy sources rather than
  recalled: the `LifecycleNodeInterface::CallbackReturn` signatures, that
  `LifecycleNode::on_activate` is itself an override that flips managed
  entities and so must be called from a derived override, that
  `rclcpp_components_register_nodes` requires a shared-library target installed
  to `lib/`, that `NodeFactoryTemplate` needs only a `NodeOptions` constructor
  and `get_node_base_interface`, that `rclpy.lifecycle` exports
  `TransitionCallbackReturn` and `create_lifecycle_publisher`, that
  `sensor_msgs/Image.data` is `array.array('B')` in rclpy while `uint64[8]` is
  a NumPy array, and that the Jazzy `ros2 bag play` verb has no
  timestamp-rewriting option.
* That `rclcpp_components`, `rclcpp_lifecycle`, `launch_ros` and the
  `ros2 lifecycle` verb are all present in `ros:jazzy-ros-base`, by reading
  REP 2001 and the `rosbag2_transport` and `ros2cli_common_extensions`
  manifests. The Dockerfile names them anyway.
