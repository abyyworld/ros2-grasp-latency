#!/usr/bin/env bash
# The whole experiment, in the order the conclusions depend on.
#
# Corpora, then the C++ build, then both pipelines at every resolution, then
# the equivalence gate, then the control-rate sweep, then the analysis. The
# gate is in the middle rather than at the end on purpose: if the two
# implementations do not compute the same answer, every latency number
# downstream of it is a comparison between two different programs, so the run
# stops there rather than producing a report that looks fine.
#
# Nothing here is quick. At the default 2000 measured frames per run this takes
# tens of minutes, most of it at 1280x720. Override for a smoke pass:
#
#   FRAMES=50 WARMUP=10 RATE_SWEEP_JOBS=20 harness/run_all.sh
#
# Environment:
#   DATASETS          space-separated "name:WIDTHxHEIGHT" pairs
#   FRAMES, WARMUP    per-run frame counts, default from pipeline_config.json
#   RATE_SWEEP_JOBS   jobs per control rate, default from pipeline_config.json
#   RATE_SWEEP_DATASETS  corpus names to sweep, default from pipeline_config.json
#   OUT_DIR           where results land, default results/
#   SKIP_BUILD=1      trust the existing cpp/build
#   SKIP_BENCH=1      reuse the timing and output JSONL already in OUT_DIR and
#                     rerun only the gate, the sweep and the analysis
#   SKIP_ONE_THREAD=1 do not take the extra single-BLAS-thread Python pass

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-results}"
BUILD_DIR="${BUILD_DIR:-cpp/build}"
DATA_DIR="${DATA_DIR:-data}"
CONFIG="assets/pipeline_config.json"

# Four resolutions, because one point does not show how latency scales and the
# crossover against the control budget has to be interpolated between two
# measured resolutions rather than asserted.
DATASETS="${DATASETS:-table_320x240:320x240 table_640x480:640x480 table_848x480:848x480 table_1280x720:1280x720}"

step() { printf '\n==> %s\n' "$*" >&2; }
die() { printf 'run_all: %s\n' "$*" >&2; exit 1; }

command -v python3 >/dev/null || die "python3 is not on PATH"
[ -f "$CONFIG" ] || die "$CONFIG is missing; are you in the repository?"

read -r DEFAULT_WARMUP DEFAULT_FRAMES DEFAULT_JOBS REFERENCE <<EOF
$(python3 -c "
import json
c = json.load(open('$CONFIG'))
print(c['benchmark']['warmup_frames'], c['benchmark']['measured_frames'],
      c['rate_sweep']['jobs_per_rate'], c['analysis']['reference_dataset'])")
EOF
WARMUP="${WARMUP:-$DEFAULT_WARMUP}"
FRAMES="${FRAMES:-$DEFAULT_FRAMES}"
RATE_SWEEP_JOBS="${RATE_SWEEP_JOBS:-$DEFAULT_JOBS}"
STORE_FRAMES="${STORE_FRAMES:-100}"

mkdir -p "$OUT_DIR"

step "corpora"
for entry in $DATASETS; do
  name="${entry%%:*}"
  shape="${entry##*:}"
  width="${shape%%x*}"
  height="${shape##*x}"
  if [ -f "$DATA_DIR/$name/manifest.json" ]; then
    printf '    %s already present\n' "$name" >&2
    continue
  fi
  printf '    generating %s (%sx%s)\n' "$name" "$width" "$height" >&2
  python3 harness/make_dataset.py --name "$name" --width "$width" \
    --height "$height" --frames "$STORE_FRAMES" --out "$DATA_DIR"
done

if [ "${SKIP_BUILD:-0}" = "1" ]; then
  step "build skipped (SKIP_BUILD=1)"
  [ -x "$BUILD_DIR/bench_pipeline" ] || die "$BUILD_DIR/bench_pipeline is not there to skip to"
else
  step "building the C++ core"
  cmake -S cpp -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" -j"$(nproc)"
fi
[ -x "$BUILD_DIR/bench_pipeline" ] || die "$BUILD_DIR/bench_pipeline was not built"

if [ "${SKIP_BENCH:-0}" = "1" ]; then
  step "unit tests skipped with the benchmarks"
else
  step "unit tests"
  (cd "$BUILD_DIR" && ctest --output-on-failure)
  python3 -m pytest tests/ python/tests/ -q
fi

if [ "${SKIP_BENCH:-0}" = "1" ]; then
  step "benchmarks skipped (SKIP_BENCH=1), reusing $OUT_DIR"
  for entry in $DATASETS; do
    name="${entry%%:*}"
    for impl in cpp py; do
      [ -s "$OUT_DIR/$impl.$name.timing.jsonl" ] \
        || die "$OUT_DIR/$impl.$name.timing.jsonl is not there to reuse"
    done
  done
else
step "benchmarks: $FRAMES measured frames after $WARMUP warm-up, per run"
for entry in $DATASETS; do
  name="${entry%%:*}"
  printf '\n    --- %s ---\n' "$name" >&2
  ./"$BUILD_DIR"/bench_pipeline \
    --dataset "$DATA_DIR/$name" \
    --out-timing "$OUT_DIR/cpp.$name.timing.jsonl" \
    --out-output "$OUT_DIR/cpp.$name.output.jsonl" \
    --warmup "$WARMUP" --frames "$FRAMES"
  python3 python/bench/bench_pipeline.py \
    --dataset "$DATA_DIR/$name" \
    --out-timing "$OUT_DIR/py.$name.timing.jsonl" \
    --out-output "$OUT_DIR/py.$name.output.jsonl" \
    --warmup "$WARMUP" --frames "$FRAMES"
done

# NumPy's BLAS takes every core it can find; Eigen here is built without
# OpenMP and takes one. Left alone, the headline is a four-core program
# against a one-core program, which is a fact about the deployment and not
# about the language. One extra pass on the reference corpus with the BLAS
# pinned to a single thread separates the two questions, and the gate then
# checks that pinning it did not change the answer.
reference_benchmarked=0
for entry in $DATASETS; do
  [ "${entry%%:*}" = "$REFERENCE" ] && reference_benchmarked=1
done
# Nothing to pin against if the reference corpus was not benchmarked.
if [ "${SKIP_ONE_THREAD:-0}" = "1" ] || [ "$reference_benchmarked" = "0" ]; then
  printf '    single-BLAS-thread pass skipped\n' >&2
else
  printf '\n    --- %s, BLAS pinned to one thread ---\n' "$REFERENCE" >&2
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 python/bench/bench_pipeline.py \
      --dataset "$DATA_DIR/$REFERENCE" \
      --out-timing "$OUT_DIR/py1t.$REFERENCE.timing.jsonl" \
      --out-output "$OUT_DIR/py1t.$REFERENCE.output.jsonl" \
      --impl py1t --warmup "$WARMUP" --frames "$FRAMES"
fi
fi

step "equivalence gate"
for entry in $DATASETS; do
  name="${entry%%:*}"
  python3 harness/compare_outputs.py \
    "$OUT_DIR/cpp.$name.output.jsonl" "$OUT_DIR/py.$name.output.jsonl" \
    --json "$OUT_DIR/equivalence.$name.json" \
    || die "cpp and py disagree on $name; every latency number below this line would be a comparison between two different programs"
done
if [ -s "$OUT_DIR/py1t.$REFERENCE.output.jsonl" ]; then
  python3 harness/compare_outputs.py \
    "$OUT_DIR/cpp.$REFERENCE.output.jsonl" \
    "$OUT_DIR/py1t.$REFERENCE.output.jsonl" \
    --json "$OUT_DIR/equivalence.$REFERENCE.py1t.json" \
    || die "pinning the BLAS to one thread changed the answer on $REFERENCE"
fi

step "control-rate sweep"
sweep_args=()
for name in ${RATE_SWEEP_DATASETS:-}; do
  sweep_args+=(--dataset "$DATA_DIR/$name")
done
python3 harness/rate_sweep.py --out-dir "$OUT_DIR" \
  --build-dir "$BUILD_DIR" --data-root "$DATA_DIR" \
  --jobs "$RATE_SWEEP_JOBS" "${sweep_args[@]+"${sweep_args[@]}"}"

step "analysis"
python3 harness/analyze.py "$OUT_DIR"/*.timing.jsonl \
  --out-dir "$OUT_DIR" --data-root "$DATA_DIR"

step "done: $OUT_DIR/RESULTS.md"
