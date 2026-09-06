#!/usr/bin/env bash
#
# Run the rclcpp node and then the rclpy node against the same input, record
# /grasp/latency from each, and convert both recordings into the timing JSONL
# that docs/FORMATS.md defines, so tools/analyse.py sees ROS runs and
# in-process runs in exactly the same shape.
#
# The two nodes are run in turn, never together: with four cores, a concurrent
# run would have them competing for CPU and the tail would measure the
# scheduler rather than the language.
#
# Usage:
#   docker/run_ros_benchmark.sh [--driver bag|live] [--dataset NAME] [--out DIR]
#
#   --driver bag   `ros2 bag play` the recorded stream (the default). Measures
#                  transport, serialisation and dispatch, but see docs/ROS2.md:
#                  rosbag2 replays the header stamp it recorded, so
#                  end_to_end_ns from a bag run carries a constant unknown
#                  offset and only total_ns is directly comparable.
#   --driver live  Publish the frame store from grasp_node_cpp's frame_publisher,
#                  which stamps at publish. Slower to set up, but end_to_end_ns
#                  is then an absolute number.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRIVER=bag
DATASET=table_640x480
OUT="${REPO}/results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --driver) DRIVER="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

CONFIG="${REPO}/assets/pipeline_config.json"
CHAIN="${REPO}/assets/franka/panda_chain.json"
RANSAC="${REPO}/assets/ransac_uniform.bin"
MSGDEF="${REPO}/ros2_ws/src/grasp_msgs/msg/GraspLatency.msg"
FRAMES="${REPO}/data/${DATASET}"
INPUT_BAG="${REPO}/data/${DATASET}_bag"

for f in "${CONFIG}" "${CHAIN}" "${RANSAC}" "${MSGDEF}"; do
  [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
[[ -d "${FRAMES}" ]] || { echo "missing frame store ${FRAMES}; run harness/make_dataset.py first" >&2; exit 1; }

# Keep DDS off the network entirely. ROS_LOCALHOST_ONLY was removed in Jazzy;
# the discovery range is the supported knob.
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

mkdir -p "${OUT}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# Constants come out of pipeline_config.json and the dataset manifest; nothing
# in this script or in either node hard-codes one.
eval "$(python3 - "${CONFIG}" "${FRAMES}/manifest.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
man = json.load(open(sys.argv[2]))
b = cfg['benchmark']
print(f"WARMUP={b['warmup_frames']}")
print(f"MEASURED={b['measured_frames']}")
print(f"RATE_HZ={b['deadline_hz']}")
print(f"FRAME_COUNT={man['frame_count']}")
print(f"WIDTH={man['width']}")
print(f"HEIGHT={man['height']}")
PY
)"

TOTAL=$(( WARMUP + MEASURED ))
# Enough wall time for the driver to deliver TOTAL frames, plus slack for node
# start-up and for a node that cannot keep up with the deadline rate.
DURATION=$(python3 -c "print(int(${TOTAL} / ${RATE_HZ} * 1.5) + 15)")

PARAMS="${WORK}/params.yaml"
python3 - "${CONFIG}" "${FRAMES}" "${PARAMS}" "${CHAIN}" "${RANSAC}" "${RATE_HZ}" \
         "${FRAME_COUNT}" "${WIDTH}" "${HEIGHT}" "${TOTAL}" <<'PY'
import json, sys
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

run_one() {
  local impl="$1" pkg="$2"
  local recbag="${WORK}/${impl}_latency"
  echo "=== ${impl}: ${pkg} / ${DRIVER} driver, ${TOTAL} frames at ${RATE_HZ} Hz ==="

  ros2 run "${pkg}" grasp_node --ros-args --params-file "${PARAMS}" &
  local node=$!
  # Let discovery settle and the pipeline finish construction before anything
  # is published at it; frames sent during start-up would be dropped and shift
  # the warm-up window.
  sleep 5

  ros2 bag record --topics /grasp/latency -o "${recbag}" \
    --disable-keyboard-controls --storage mcap >/dev/null &
  local rec=$!
  sleep 2

  if [[ "${DRIVER}" == "live" ]]; then
    timeout -s INT "${DURATION}s" \
      ros2 run grasp_node_cpp frame_publisher --ros-args --params-file "${PARAMS}" || true
  else
    timeout -s INT "${DURATION}s" \
      ros2 bag play "${INPUT_BAG}" --loop --rate 1.0 --disable-keyboard-controls || true
  fi

  # Give the last frame time through the pipeline and into the recorder before
  # tearing anything down.
  sleep 3
  kill -INT "${rec}" 2>/dev/null || true
  wait "${rec}" 2>/dev/null || true
  kill -INT "${node}" 2>/dev/null || true
  wait "${node}" 2>/dev/null || true

  python3 "${REPO}/docker/latency_bag_to_jsonl.py" \
    --bag "${recbag}" --msg "${MSGDEF}" \
    --out "${OUT}/ros2_${impl}.timing.jsonl" \
    --impl "${impl}" --dataset "${DATASET}" \
    --frame-count "${FRAME_COUNT}" --warmup "${WARMUP}" --limit "${MEASURED}"
}

run_one cpp grasp_node_cpp
run_one py  grasp_node_py

echo
echo "wrote ${OUT}/ros2_cpp.timing.jsonl and ${OUT}/ros2_py.timing.jsonl"
echo "rmw=${RMW_IMPLEMENTATION} driver=${DRIVER}"
