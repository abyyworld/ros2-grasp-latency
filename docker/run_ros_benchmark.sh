#!/usr/bin/env bash
#
# Run the grasp nodes against the same input, record /grasp/latency from each,
# and convert every recording into the timing JSONL that docs/FORMATS.md
# defines, so harness/analyze.py sees ROS runs and in-process runs in exactly
# the same shape.
#
# Nothing here runs two nodes at once: with four cores, a concurrent run would
# have them competing for CPU and the tail would measure the scheduler rather
# than the language.
#
# Usage:
#   docker/run_ros_benchmark.sh [--mode MODE] [--driver bag|live]
#                               [--dataset NAME] [--out DIR]
#
#   --mode standalone       (default) rclcpp and rclpy nodes, each in its own
#                           process, driver in a third. Tagged transport=ros2.
#   --mode composed         rclcpp node and driver in one container, delivery
#                           through the RMW. Tagged transport=ros2_composed.
#   --mode composed_ipc     the same container with intra-process delivery on,
#                           so the depth image is handed over as a pointer.
#                           Tagged transport=ros2_composed_ipc.
#   --mode all              all three, in that order. Requires --driver live.
#
#   --driver bag   `ros2 bag play` the recorded stream. Measures transport,
#                  serialisation and dispatch, but see docs/ROS2.md: rosbag2
#                  replays the header stamp it recorded, so end_to_end_ns from
#                  a bag run carries a constant unknown offset and only
#                  total_ns is directly comparable. The default for
#                  --mode standalone, and not available for the composed modes:
#                  a bag player is a separate process and cannot be composed,
#                  so composing it away is the whole point.
#   --driver live  Publish the frame store from frame_publisher, which stamps
#                  at publish. end_to_end_ns is then an absolute number, and it
#                  is the only driver under which the three modes are
#                  comparable to each other.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE=standalone
DRIVER=""
DATASET=table_640x480
OUT="${REPO}/results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --driver) DRIVER="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${MODE}" in
  standalone|composed|composed_ipc|all) ;;
  *) echo "unknown mode: ${MODE}" >&2; exit 2 ;;
esac

# A composed run has no separate process for the driver to live in, so the bag
# player cannot participate. Defaulting rather than erroring keeps `--mode all`
# a single word to type, and the choice is printed at the end of the run.
if [[ -z "${DRIVER}" ]]; then
  if [[ "${MODE}" == "standalone" ]]; then DRIVER=bag; else DRIVER=live; fi
fi
if [[ "${MODE}" != "standalone" && "${DRIVER}" != "live" ]]; then
  echo "--mode ${MODE} needs --driver live: a bag player is a separate process," \
       "so a composed run cannot be driven by one" >&2
  exit 2
fi

CONFIG="${REPO}/assets/pipeline_config.json"
CHAIN="${REPO}/assets/franka/panda_chain.json"
RANSAC="${REPO}/assets/ransac_uniform.bin"
MSGDEF="${REPO}/ros2_ws/src/grasp_msgs/msg/GraspLatency.msg"
CPP_CONFIG="${REPO}/ros2_ws/src/grasp_node_cpp/config"
PY_CONFIG="${REPO}/ros2_ws/src/grasp_node_py/config"
FRAMES="${REPO}/data/${DATASET}"
INPUT_BAG="${REPO}/data/${DATASET}_bag"

for f in "${CONFIG}" "${CHAIN}" "${RANSAC}" "${MSGDEF}"; do
  [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
[[ -d "${FRAMES}" ]] || { echo "missing frame store ${FRAMES}; run harness/make_dataset.py first" >&2; exit 1; }

# The two packages must offer identical QoS and sit on identical topics, or the
# run measures a DDS policy difference and reports it as a language difference.
# Checked rather than trusted, because the failure is silent.
for name in qos.yaml topics.yaml; do
  if ! cmp -s "${CPP_CONFIG}/${name}" "${PY_CONFIG}/${name}"; then
    echo "grasp_node_cpp/config/${name} and grasp_node_py/config/${name} differ;" \
         "the two nodes would not be comparable" >&2
    exit 1
  fi
done

python3 - "${CONFIG}" "${CPP_CONFIG}/topics.yaml" <<'PY'
import json
import sys

import yaml

ros = json.load(open(sys.argv[1]))['ros']
topics = yaml.safe_load(open(sys.argv[2]))
# The bag writer used these names. A remap table that disagreed would produce a
# node that subscribes to a topic nothing publishes and a run of zero frames.
wanted = {
    'depth/image_raw': ros['depth_topic'],
    'depth/camera_info': ros['camera_info_topic'],
}
bad = [
    f'{node}:{key} is {topics[node][key]!r}, pipeline_config.json says {value!r}'
    for node, table in topics.items()
    for key, value in wanted.items()
    if key in table and table[key] != value
]
if bad:
    raise SystemExit('config/topics.yaml disagrees with pipeline_config.json:ros\n  ' +
                     '\n  '.join(bad))
PY

# Keep DDS off the network entirely. ROS_LOCALHOST_ONLY was removed in Jazzy;
# the discovery range is the supported knob.
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

mkdir -p "${OUT}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# Constants come out of pipeline_config.json, the dataset manifest and the
# packages' own config/; nothing in this script or in either node hard-codes one.
eval "$(python3 - "${CONFIG}" "${FRAMES}/manifest.json" "${CPP_CONFIG}/topics.yaml" <<'PY'
import json
import sys

import yaml

cfg = json.load(open(sys.argv[1]))
man = json.load(open(sys.argv[2]))
topics = yaml.safe_load(open(sys.argv[3]))
b = cfg['benchmark']
print(f"WARMUP={b['warmup_frames']}")
print(f"MEASURED={b['measured_frames']}")
print(f"RATE_HZ={b['deadline_hz']}")
print(f"FRAME_COUNT={man['frame_count']}")
print(f"WIDTH={man['width']}")
print(f"HEIGHT={man['height']}")
print(f"LATENCY_TOPIC={topics['grasp_node']['latency']}")
PY
)"

TOTAL=$(( WARMUP + MEASURED ))
# Enough wall time for the driver to deliver TOTAL frames, plus slack for node
# start-up and for a node that cannot keep up with the deadline rate.
DURATION=$(python3 -c "print(int(${TOTAL} / ${RATE_HZ} * 1.5) + 15)")

PARAMS="${WORK}/params.yaml"
python3 - "${CONFIG}" "${FRAMES}" "${PARAMS}" "${CHAIN}" "${RANSAC}" "${RATE_HZ}" \
         "${FRAME_COUNT}" "${WIDTH}" "${HEIGHT}" "${TOTAL}" <<'PY'
import json
import sys

cfg_path, frames, out, chain, ransac, rate, count, w, h, total = sys.argv[1:11]
cfg = json.load(open(cfg_path))
cam = cfg['camera']
scale = int(w) / cam['reference_width']
names = '\n'.join(f'      - {n}' for n in cfg['trajectory']['joint_names'])
open(out, 'w').write(f"""grasp_node:
  ros__parameters:
    config_path: "{cfg_path}"
    chain_path: "{chain}"
    ransac_table_path: "{ransac}"
    joint_names:
{names}
frame_publisher:
  ros__parameters:
    frames_dir: "{frames}"
    width: {int(w)}
    height: {int(h)}
    frame_count: {int(count)}
    total_frames: {int(total)}
    rate_hz: {float(rate)}
    frame_id: "{cfg['ros']['depth_frame_id']}"
    fx: {cam['fx'] * scale}
    fy: {cam['fy'] * scale}
    cx: {cam['cx'] * scale}
    cy: {cam['cy'] * scale}
""")
PY

if [[ "${DRIVER}" == "bag" && ! -d "${INPUT_BAG}" ]]; then
  echo "missing input bag ${INPUT_BAG}; run harness/make_dataset.py --bag, or use --driver live" >&2
  exit 1
fi

# `ros2 lifecycle get` prints "<state> [<id>]", so a prefix match on the label
# is the whole test. Polling beats sleeping: a node that has not finished
# building its pipeline would otherwise silently miss the first frames and
# shift the warm-up window by an amount nobody recorded.
wait_for_lifecycle() {
  local node="$1" want="$2" limit="${3:-60}" waited=0 state=""
  while (( waited < limit )); do
    state="$(ros2 lifecycle get "${node}" 2>/dev/null || true)"
    if [[ "${state}" == "${want}"* ]]; then
      return 0
    fi
    sleep 1
    waited=$(( waited + 1 ))
  done
  echo "timed out after ${limit}s waiting for ${node} to reach '${want}'" \
       "(last seen: ${state:-unreachable})" >&2
  return 1
}

# Every pid under `root`, root last, listed while the tree is still intact.
#
# The listing has to happen BEFORE anything is killed. Killing a supervisor
# reparents its children to pid 1, so `pgrep -P` on a dead parent returns
# nothing and the children survive as orphans holding their DDS participants,
# which is the exact failure this is meant to prevent. A test with a parent and
# child that both ignore SIGINT and SIGTERM caught that: the parent was reaped
# and the child was not.
descendants_of() {
  local root="$1" child
  for child in $(pgrep -P "${root}" 2>/dev/null); do
    descendants_of "${child}"
  done
  printf '%s\n' "${root}"
}

# A launch that will not stop must not stop the run. `ros2 launch` forwards
# SIGINT to its children and then waits for them, and a lifecycle node that
# declines to leave the active state leaves it waiting forever: the first real
# execution of this script hung here, after the driver had already published
# all 2100 frames and the recording was complete. A bare `wait` therefore
# blocks on a node that logs "ctrl-c ... ignoring", and the recorded data that
# was already on disk never reaches the conversion step.
#
# So escalate on a clock rather than trusting the node, and say which signal it
# took: a run that needed SIGKILL is still a valid measurement of the pipeline,
# but it is not a clean shutdown and the reader should know.
stop() {
  local pid="$1" waited=0 tree

  # Snapshot the tree first: see descendants_of.
  tree="$(descendants_of "${pid}")"

  kill -INT "${pid}" 2>/dev/null || true
  while (( waited < 15 )) && kill -0 "${pid}" 2>/dev/null; do
    sleep 1
    waited=$(( waited + 1 ))
  done

  if kill -0 "${pid}" 2>/dev/null; then
    echo "  pid ${pid} ignored SIGINT for ${waited}s, escalating to SIGTERM" >&2
    kill -TERM "${pid}" 2>/dev/null || true
    waited=0
    while (( waited < 10 )) && kill -0 "${pid}" 2>/dev/null; do
      sleep 1
      waited=$(( waited + 1 ))
    done
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    echo "  pid ${pid} ignored SIGTERM too, sending SIGKILL" >&2
    kill -KILL "${pid}" 2>/dev/null || true
  fi

  wait "${pid}" 2>/dev/null || true

  # `ros2 launch` is a supervisor: SIGKILLing it orphans the nodes it started,
  # and an orphan keeps its DDS participant and its topic, so the next arm
  # would discover a stale publisher and record two nodes at once. Reap this
  # launch's own descendants, found by walking the tree, and nothing else:
  # stop() is called on the recorder before the node, so a pkill by name would
  # SIGKILL a recorder that is still flushing the bag this run exists to
  # produce.
  local orphan
  for orphan in ${tree}; do
    if kill -0 "${orphan}" 2>/dev/null; then
      kill -KILL "${orphan}" 2>/dev/null || true
    fi
  done
}

settle() {
  sleep 5
}

convert() {
  local impl="$1" transport="$2" recbag="$3"
  python3 "${REPO}/docker/latency_bag_to_jsonl.py" \
    --bag "${recbag}" --msg "${MSGDEF}" \
    --out "${OUT}/${transport}_${impl}.timing.jsonl" \
    --impl "${impl}" --transport "${transport}" --dataset "${DATASET}" \
    --frame-count "${FRAME_COUNT}" --warmup "${WARMUP}" --limit "${MEASURED}"
}

record() {
  ros2 bag record --topics "${LATENCY_TOPIC}" -o "$1" \
    --disable-keyboard-controls --storage mcap >/dev/null &
  echo $!
}

run_standalone() {
  local impl="$1" pkg="$2"
  local recbag="${WORK}/${impl}_standalone_latency"
  echo "=== ${impl}: ${pkg}, own process, ${DRIVER} driver, ${TOTAL} frames at ${RATE_HZ} Hz ==="

  ros2 launch "${pkg}" grasp_node.launch.py params_file:="${PARAMS}" &
  local node=$!
  wait_for_lifecycle /grasp_node active

  local rec
  rec="$(record "${recbag}")"
  sleep 2

  if [[ "${DRIVER}" == "live" ]]; then
    timeout -s INT "${DURATION}s" \
      ros2 launch grasp_node_cpp frame_publisher.launch.py params_file:="${PARAMS}" || true
  else
    timeout -s INT "${DURATION}s" \
      ros2 bag play "${INPUT_BAG}" --loop --rate 1.0 --disable-keyboard-controls || true
  fi

  # Give the last frame time through the pipeline and into the recorder before
  # tearing anything down.
  sleep 3
  stop "${rec}"
  stop "${node}"
  # Let discovery forget the node before the next arm starts one with the same
  # name, or wait_for_lifecycle would answer from the corpse.
  settle

  convert "${impl}" ros2 "${recbag}"
}

run_composed() {
  local intra="$1" transport="$2"
  local recbag="${WORK}/cpp_${transport}_latency"
  echo "=== cpp: one container, intra-process comms ${intra}, ${TOTAL} frames at ${RATE_HZ} Hz ==="

  ros2 launch grasp_node_cpp composed.launch.py \
    params_file:="${PARAMS}" intra_process:="${intra}" &
  local container=$!

  # launch_ros drives lifecycle transitions by emitting events at LifecycleNode
  # actions, and a component loaded into a container is not one, so the
  # transitions are requested here instead. The order matters: the consumer is
  # active before the driver publishes anything.
  wait_for_lifecycle /grasp_node unconfigured
  ros2 lifecycle set /grasp_node configure
  wait_for_lifecycle /grasp_node inactive
  ros2 lifecycle set /grasp_node activate
  wait_for_lifecycle /grasp_node active

  local rec
  rec="$(record "${recbag}")"
  sleep 2

  wait_for_lifecycle /frame_publisher unconfigured
  ros2 lifecycle set /frame_publisher configure
  wait_for_lifecycle /frame_publisher inactive
  ros2 lifecycle set /frame_publisher activate
  wait_for_lifecycle /frame_publisher active

  # The driver cancels its own timer after total_frames and then idles, because
  # a component that called rclcpp::shutdown() would take the node under
  # measurement down with it. So the wall clock, not the driver, ends the run.
  sleep "${DURATION}"
  ros2 lifecycle set /frame_publisher deactivate || true

  sleep 3
  stop "${rec}"
  stop "${container}"
  settle

  convert cpp "${transport}" "${recbag}"
}

case "${MODE}" in
  standalone)
    run_standalone cpp grasp_node_cpp
    run_standalone py grasp_node_py
    ;;
  composed)
    run_composed false ros2_composed
    ;;
  composed_ipc)
    run_composed true ros2_composed_ipc
    ;;
  all)
    run_standalone cpp grasp_node_cpp
    run_standalone py grasp_node_py
    run_composed false ros2_composed
    run_composed true ros2_composed_ipc
    ;;
esac

echo
echo "wrote:"
ls -1 "${OUT}"/ros2*.timing.jsonl
echo "rmw=${RMW_IMPLEMENTATION} driver=${DRIVER} mode=${MODE}"
echo "the transport field separates the runs; feed them all to harness/analyze.py at once"
