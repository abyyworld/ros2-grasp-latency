#!/usr/bin/env bash
#
# Everything the project measures, on one machine, in one container.
#
# The repository grew three weaknesses that all have the same root: parts of it
# were measured in different places. This closes them.
#
#   1. The rclpy frame-drop figure, which is the headline, came from a single
#      run. This repeats the ROS arms and reports the spread.
#   2. The in-process table came from an x86 host and the ROS table from an
#      arm64 container, so the two could not be compared. This runs the
#      in-process benchmark here too, on the same silicon as the ROS arms.
#   3. The explanation offered for the in-process result, that NumPy's parity
#      with C++ depends on the BLAS it links, was a hypothesis. This measures
#      the BLAS directly on the shape the pipeline actually issues.
#
# Usage: docker/run_full_comparison.sh [--repeats N] [--frames N]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPEATS=3
FRAMES=2000
DATASET=table_640x480

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeats) REPEATS="$2"; shift 2 ;;
    --frames)  FRAMES="$2";  shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

OUT="${REPO}/results"
mkdir -p "${OUT}"
cd "${REPO}"

echo "=== 1/4  which BLAS is NumPy reaching here ==="
python3 tools/blas_probe.py --out results/blas_probe_container.json

echo
echo "=== 2/4  corpus ==="
[ -d "data/${DATASET}" ] || python3 harness/make_dataset.py \
  --name "${DATASET}" --width 640 --height 480 --frames 100 --out data

echo
echo "=== 3/4  in-process, on this machine, so the ROS table has a local baseline ==="
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build cpp/build -j"$(nproc)" >/dev/null
./cpp/build/bench_pipeline --dataset "data/${DATASET}" --impl cpp \
  --out-timing "${OUT}/inproc_cpp.timing.jsonl" \
  --out-output "${OUT}/inproc_cpp.output.jsonl" --warmup 100 --frames "${FRAMES}"
PYTHONPATH=python python3 python/bench/bench_pipeline.py --dataset "data/${DATASET}" --impl py \
  --out-timing "${OUT}/inproc_py.timing.jsonl" \
  --out-output "${OUT}/inproc_py.output.jsonl" --warmup 100 --frames "${FRAMES}"

echo "--- equivalence gate on this machine ---"
python3 harness/compare_outputs.py \
  "${OUT}/inproc_cpp.output.jsonl" "${OUT}/inproc_py.output.jsonl" \
  --json "${OUT}/equivalence.inproc.json"

echo
echo "=== 4/4  ROS 2 arms, ${REPEATS} repeats, so the drop rate has a spread ==="
for rep in $(seq 1 "${REPEATS}"); do
  echo "--- repeat ${rep} of ${REPEATS} ---"
  docker/run_ros_benchmark.sh --mode all --driver live
  for f in "${OUT}"/ros2*.timing.jsonl; do
    [ -e "${f}" ] || continue
    mv "${f}" "${f%.timing.jsonl}.rep${rep}.timing.jsonl"
  done
done

echo
echo "wrote:"
ls -1 "${OUT}"/inproc_*.timing.jsonl "${OUT}"/ros2*.rep*.timing.jsonl \
      "${OUT}"/blas_probe_container.json 2>/dev/null | sed 's|.*/|  |'
echo
echo "Every figure above came off this one machine. Feed them to"
echo "harness/analyze.py together; the impl and transport fields keep them apart."
