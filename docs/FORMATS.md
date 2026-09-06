# Wire formats

Three formats let a C++ binary, a Python process and an analysis script agree
without any of them depending on the others. All are little-endian; the
benchmark only ever runs on x86-64 and aarch64, and both are LE.

## 1. Frame store — the recorded input

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

## 2. Latency records — `*.timing.jsonl`

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
Python records additionally carry `"gc":{"gen0":n,"gen1":n,"gen2":n}` —
cumulative collection counts sampled at the end of the frame, which is what
lets tail latency be attributed to garbage collection rather than guessed at.

## 3. Pipeline outputs — `*.output.jsonl`

The equivalence check reads these. One line per frame, in frame order.

```json
{"impl":"cpp","frame":0,
 "plane_found":true,"plane":[0.0,0.0,1.0,-0.0021],
 "graspable":true,"width":0.0551,
 "tcp":[ ... 16 row-major doubles ... ],
 "converged":true,"iterations":14,
 "q":[0.113,-0.842,-0.097,-2.401,0.058,1.612,0.702],
 "duration_s":1.83,
 "traj_checksum":"sha256 of the waypoint block, see below"}
```

`traj_checksum` is the SHA-256 of every waypoint's 21 doubles plus its
`time_from_start`, each serialised as its IEEE-754 bit pattern in little-endian
order, positions then velocities then accelerations then time, waypoint by
waypoint. Comparing 20x22 doubles per frame across 2000 frames as text would
dwarf the useful output; comparing a digest catches any divergence just as
well. Because the two implementations only agree to `1e-6`, the checksum is
taken over values **rounded to 9 decimal places** first.

Doubles are written with 17 significant digits (`%.17g`) so a round trip
through JSON is lossless.
