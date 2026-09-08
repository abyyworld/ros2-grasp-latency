# Wire formats

Three formats let a C++ binary, a Python process and an analysis script agree
without any of them depending on the others. All are little-endian; the
benchmark only ever runs on x86-64 and aarch64, and both are LE.

## 1. Frame store: the recorded input

Frames are **not committed**. `harness/make_dataset.py` regenerates them
deterministically from a seed, so a 1.2 GB corpus stays out of the repository
while remaining reproducible byte-for-byte.

```
data/<name>/
  manifest.json
  000000.depth.bin      H*W   uint16, little-endian, millimetres, row-major
  000000.rgb.bin        H*W*3 uint8,  RGB row-major
  000001.depth.bin
  ...
```

`manifest.json`:

```json
{
  "name": "table_640x480",
  "width": 640, "height": 480,
  "fx": 615.0, "fy": 615.0, "cx": 319.5, "cy": 239.5,
  "depth_scale_m": 0.001,
  "seed": 20240906,
  "frame_count": 100,
  "frames": [
    {"id": 0, "depth": "000000.depth.bin", "rgb": "000000.rgb.bin",
     "truth": {"object_center": [0.42, -0.05, 0.03], "yaw": 0.31, "width": 0.045}}
  ]
}
```

`truth` is what the scene generator placed, for sanity-checking the pipeline's
answer. It is **not** used by either implementation and never enters the
measured path.

A benchmark run of `measured_frames` cycles through the frame store, so the
working set is the store's size rather than one frame sitting hot in L2.

## 2. Latency records: `*.timing.jsonl`

One JSON object per line, one line per frame, written **after** the run so file
I/O never lands inside a measurement.

```json
{"impl":"cpp","dataset":"table_640x480","frame":0,"seq":0,
 "stage_ns":{"decode":812,"deproject":1043221,"transform_crop":402118,
             "plane":2210934,"cluster":881204,"grasp":31007,
             "ik":184552,"traj":9930},
 "total_ns":4763779,"points":301244,"cluster_points":892,
 "ik_iterations":14,"plane_found":true,"graspable":true,"converged":true}
```

`seq` counts measured frames (post-warm-up); `frame` indexes into the store.
Python records additionally carry `"gc":{"gen0":n,"gen1":n,"gen2":n}`,
cumulative collection counts sampled at the end of the frame, which is what
lets tail latency be attributed to garbage collection rather than guessed at.

## 3. Jitter records: `results/jitter/*.jsonl`

Written by `cpp/bench/bench_jitter.cpp`, again only after the run. The first
line is a run header and every line after it is one cycle.

```json
{"record":"run","impl":"cpp","label":"fifo_rep2","policy":"fifo",
 "policy_notes":"pinned;fifo;","rate_hz":30,"frames":2000,
 "warmup":200,"dataset":"data/table_320x240","width":320,"height":240,
 "cpu":2,"priority":80,"pretouch":true,"allocations":0,"allocated_bytes":0,
 "minor_faults":0,"major_faults":0,"overruns":0}
{"impl":"cpp","label":"fifo_rep2","policy":"fifo","rate_hz":30,"seq":0,
 "frame":0,"release_jitter_ns":84310,"compute_ns":10418622,
 "slack_ns":22830401,"allocs":0}
```

The header carries what belongs to the window rather than to a cycle:

* `policy_notes` records which of the requested privileges the kernel actually
  granted. A denied `SCHED_FIFO` reads `fifo-denied(Operation not permitted)`,
  so a run that silently degraded to `SCHED_OTHER` cannot be mistaken for a
  null result.
* `allocations`, `minor_faults` and `major_faults` are totals across the
  measured window. `getrusage` is called outside every cycle, so the syscall
  never lands in a region it would then be reported as part of.
* `allocations` and `overruns` are also derivable from the cycle lines, and
  `harness/analyze_jitter.py` refuses to summarise a file where the two
  disagree.

Per cycle, `release_jitter_ns` is `wake - release[k]` against a schedule fixed
in advance, and `slack_ns` is `(release[k] + period) - finish`, negative on an
overrun. `docs/REALTIME.md` says what was measured with them.

## 4. Pipeline outputs: `*.output.jsonl`

The equivalence check reads these. One line per **distinct** frame of the
store, in frame order: the pipeline is deterministic, so a frame the benchmark
cycles past twice is recorded once. These files are not committed
(`results/*.output.jsonl` is gitignored), which is what lets a line carry every
number the gate compares rather than a summary of them.

```json
{"impl":"cpp","frame":0,
 "plane_found":true,"plane":[0.0,0.0,1.0,-0.0021],
 "cluster_points":892,
 "graspable":true,"width":0.0551,
 "tcp":[ ... 16 row-major doubles ... ],
 "converged":true,"iterations":14,
 "q":[0.113,-0.842,-0.097,-2.401,0.058,1.612,0.702],
 "duration_s":1.83,
 "trajectory":[ ... waypoints * (3 * dof + 1) doubles ... ]}
```

`trajectory` is the whole waypoint block, flattened in the order
[ALGORITHM.md](ALGORITHM.md) S7 generates it: for each waypoint in turn, `dof`
positions, then `dof` velocities, then `dof` accelerations, then
`time_from_start`. Its length is `trajectory.waypoints * (3 * dof + 1)` from
the config and the chain, and the gate rejects a file whose lines disagree with
that rather than comparing the part that happens to line up. At the shipped
20 waypoints and 7 joints that is 440 doubles, about 10 kB of a line and about
1 MB of a hundred-frame file.

### Why the block and not a digest of it

This field used to be `traj_checksum`, a SHA-256 over the same values rounded
to nine decimal places. Rounding is an **equality** test on a grid, and the two
implementations are only ever known to agree to a **tolerance**: with about
44,000 waypoint doubles per corpus differing by around 1e-13 against a 1e-9
grid, a value landing within 1e-13 of a grid boundary rounds one way in one
implementation and the other way in the other, and one flipped digit fails the
gate on two answers that agree to twelve orders of magnitude better than
required. That is not a hypothetical: `table_848x480` frame 95 did it, until a
corpus regeneration happened to move the value off the boundary.

A digest also cannot report *how far apart* two runs are, which is the only
thing worth printing about two implementations that will never be bit
identical. Emitting the values costs a megabyte of a file nobody commits and
lets `harness/compare_outputs.py` hold every waypoint to the same 1e-6
tolerance it holds the pose and the joints to, and report the worst deviation
and where it landed.

### Which fields a frame carries

The spec, not the implementation, says when a field means anything.
`harness/compare_outputs.py` gates on exactly this and skips nothing else:

| Field | Defined when |
|---|---|
| `plane_found`, `cluster_points`, `graspable`, `converged` | always; a false or a zero is an answer |
| `plane` | `plane_found` (S3) |
| `width`, `tcp` | `cluster_points > 0` (S5 needs a cluster) |
| `iterations`, `q`, `duration_s`, `trajectory` | always: S6 and S7 run on every frame with a cluster whatever `graspable` says, and S6/S7 state what they hold when there is no cluster (`q_neutral`, 0 iterations, a zero trajectory of zero duration) |

Where a field is not defined, both runners still emit the slot, and what sits
in it is whatever the implementation left there. Nothing may read it.

Doubles are written with 17 significant digits (`%.17g`) so a round trip
through JSON is lossless.
