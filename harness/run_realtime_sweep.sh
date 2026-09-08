#!/usr/bin/env bash
#
# The real-time sweep, including the PREEMPT_RT arm this repository cannot run.
#
# The container these results were produced in runs a stock kernel: SCHED_FIFO
# and mlockall are permitted, so the scheduling-policy comparison is real, but
# there is no PREEMPT_RT kernel and one cannot be installed. That arm is
# written here and left unrun rather than estimated, and the script says which
# arms it actually executed so a committed result cannot claim more than it is.
#
# On a host with a PREEMPT_RT kernel this runs the complete matrix. On a stock
# kernel it runs everything except that arm and says so.
#
# Usage: harness/run_realtime_sweep.sh [--dataset NAME] [--rate HZ]
#                                      [--frames N] [--repeats N] [--cpu N]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

DATASET=table_320x240
RATE=30
FRAMES=2000
REPEATS=3
CPU=2
OUT="${REPO}/results/jitter"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --rate)    RATE="$2";    shift 2 ;;
    --frames)  FRAMES="$2";  shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --cpu)     CPU="$2";     shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUT}"

# uname -v carries PREEMPT_RT on a patched kernel. Checked rather than assumed,
# because the difference between the two is the entire point of the arm.
if uname -v | grep -q PREEMPT_RT; then
  KERNEL_IS_RT=yes
else
  KERNEL_IS_RT=no
fi

echo "kernel:        $(uname -r)"
echo "PREEMPT_RT:    ${KERNEL_IS_RT}"
echo "dataset:       ${DATASET} at ${RATE} Hz, ${FRAMES} cycles, ${REPEATS} repeats"
echo "memlock limit: $(ulimit -l) KB"
echo

[ -d "data/${DATASET}" ] || {
  echo "generating ${DATASET}"
  W="${DATASET##*_}"; W="${W%%x*}"
  H="${DATASET##*x}"
  python3 harness/make_dataset.py --name "${DATASET}" --width "${W}" --height "${H}" \
    --frames 100 --out data
}

cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build cpp/build --target bench_jitter -j"$(nproc)" >/dev/null

for rep in $(seq 1 "${REPEATS}"); do
  for policy in other fifo; do
    label="${policy}_rep${rep}"
    echo "--- ${label} ---"
    ./cpp/build/bench_jitter --dataset "data/${DATASET}" \
      --out "${OUT}/${label}.jsonl" --policy "${policy}" --priority 80 \
      --cpu "${CPU}" --rate "${RATE}" --frames "${FRAMES}" --warmup 200 \
      --label "${label}"
  done
done

# Locking memory is only meaningful where the working set fits the limit, so
# this arm is attempted and its own failure is reported rather than hidden: on
# a container with a small RLIMIT_MEMLOCK it is expected to be refused, and
# that refusal is itself worth recording next to the fault counts.
echo "--- mlock arm ---"
./cpp/build/bench_jitter --dataset "data/${DATASET}" \
  --out "${OUT}/fifo_mlock.jsonl" --policy fifo --priority 80 --cpu "${CPU}" \
  --mlock --rate "${RATE}" --frames "${FRAMES}" --warmup 200 --label fifo_mlock

echo
python3 harness/analyze_jitter.py "${OUT}"/*.jsonl --out results/jitter_summary.json

echo
if [ "${KERNEL_IS_RT}" = yes ]; then
  echo "This host runs PREEMPT_RT, so the figures above are the RT arm."
  echo "Compare them against the stock-kernel numbers in docs/REALTIME.md."
else
  echo "This host does not run PREEMPT_RT, so the RT arm was NOT run and no"
  echo "number here describes it. To produce it, run this same script on a host"
  echo "with an RT kernel and commit the result beside the stock-kernel one."
fi
