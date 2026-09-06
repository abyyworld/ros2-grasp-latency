#!/usr/bin/env bash
#
# stop() has had three bugs in twenty lines, none of which reading found:
#   1. a bare `wait` that hung forever on a launch which ignores SIGINT, which
#      is what stalled the first real run after all 2100 frames were published;
#   2. a pkill by name that would have SIGKILLed the recorder mid-flush, since
#      stop() is called on the recorder before the node;
#   3. reaping descendants after killing the parent, which reparents them to
#      pid 1 and leaves them alive holding their DDS participants.
#
# So it gets a test. This stands in a parent and child that both ignore SIGINT
# and SIGTERM, exactly as `ros2 launch` did, and asserts stop() terminates in
# bounded time with nothing left behind.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(sed -n '/^descendants_of() {/,/^}$/p' "${HERE}/run_ros_benchmark.sh")"
eval "$(sed -n '/^stop() {/,/^}$/p' "${HERE}/run_ros_benchmark.sh")"

bash -c 'trap "" INT TERM; bash -c "trap \"\" INT TERM; sleep 300" & sleep 300' &
parent=$!
sleep 1
child="$(pgrep -P "${parent}" | head -1)"
[ -n "${child}" ] || { echo "FAIL: no child spawned, the test is not testing anything" >&2; exit 1; }

started=$(date +%s)
stop "${parent}" >/dev/null 2>&1
elapsed=$(( $(date +%s) - started ))

# SIGKILL is asynchronous and `kill -0` succeeds on a zombie the reaper has not
# collected yet, so give the tree a moment and check for a live process rather
# than merely a present pid.
sleep 2
alive() {
  local pid="$1" state
  kill -0 "${pid}" 2>/dev/null || return 1
  state="$(ps -o state= -p "${pid}" 2>/dev/null | tr -d ' ')"
  [ -n "${state}" ] && [ "${state#Z}" = "${state}" ]
}

rc=0
if alive "${parent}"; then echo "FAIL: parent ${parent} survived" >&2; kill -9 "${parent}"; rc=1; fi
if alive "${child}"; then echo "FAIL: child ${child} survived as an orphan" >&2; kill -9 "${child}"; rc=1; fi
# INT wait 15 + TERM wait 10, plus slack. A regression to a bare `wait` hangs
# instead of finishing, so the ceiling is the real assertion.
if (( elapsed > 40 )); then echo "FAIL: stop() took ${elapsed}s, expected under 40" >&2; rc=1; fi

[ "${rc}" -eq 0 ] && echo "teardown OK: parent and child both reaped in ${elapsed}s"
exit "${rc}"
