# How the measurement is made, and why it should be believed

[ALGORITHM.md](ALGORITHM.md) fixes *what* is computed. This document fixes
*how long it took* is arrived at, which is the only claim this repository
actually makes. It ends with a section on what could be wrong with it, written
at the length the problems deserve rather than the length that is comfortable.

The short version: the same recorded bytes go into two implementations of one
frozen spec, the region between the first and last clock read contains nothing
but the pipeline, both implementations must produce the same answer before any
timing is quoted, and every figure is a percentile with the mean printed beside
it so a reader can see how badly the mean would have described it.

---

## 1. What is inside the measured region

The measured region is `S0` entry to `S7` exit, bracketed by nine clock reads:
one before `S0` and one after each of the eight stages.
`total_ns = mark8 - mark0`, and the eight stage durations are the consecutive
differences, so the stages sum exactly to the total by construction and no time
can hide between them.

Inside the region, per frame:

* getting typed arrays out of the raw depth and colour buffers (`S0`),
* deprojection, transform and crop, plane RANSAC and refit, clustering, grasp
  synthesis, damped-least-squares IK, quintic trajectory generation.

Outside the region, and deliberately so:

| Outside | Where it happens instead |
|---|---|
| Reading the frame store from disk | Before the first warm-up frame. The whole corpus is loaded into RAM. |
| Parsing `pipeline_config.json`, the chain and the RANSAC deviate table | Pipeline construction. Determinism rule 5. |
| Allocating every scratch buffer | `Pipeline::reserve` / `GraspPipeline.resize`, called before the first warm-up frame. |
| Formatting and writing the timing and output JSONL | After the loop. Records are held in preallocated arrays until then. |
| Capturing the per-frame answer for the equivalence gate | After the frame's last clock read. |
| Reading the garbage-collection counters | A `gc.callbacks` hook, which fires on collection rather than being polled. |
| Wrapping a frame's answer for the caller | Both implementations return a reference to a result object they own and reuse. |

The Python runner does not disable the garbage collector. An `rclpy` node runs
with it on, so turning it off here would remove exactly the tail this
repository exists to measure. It is counted instead, and section 8 attributes
the tail with the counts.

No I/O, no logging, no string formatting, no configuration parsing and no
random number occurs inside the region in either implementation.

Allocation is where the two honestly differ, and the difference is not swept
up. The C++ pipeline allocates nothing at all in the measured region:
`cpp/tests/test_no_allocation` replaces global `operator new` with a counting
version in the test binary, warms the pipeline, arms the counter and requires a
run of frames to move it by exactly zero. The Python pipeline cannot make that
claim and is not asked to. NumPy allocates a temporary for every expression it
will not let you write in place, and refusing those temporaries would mean
writing something other than the best reasonable NumPy code that
`ALGORITHM.md`'s fairness rule requires. That allocation is a genuine cost of
the thing being measured, so it is inside the region, and section 8 is how its
consequences are attributed rather than assumed.

## 2. The clock, and what reading it costs

C++ uses `clock_gettime(CLOCK_MONOTONIC)`; Python uses
`time.perf_counter_ns()`, which on Linux is `clock_gettime(CLOCK_MONOTONIC)`
with the same vDSO path. Both are monotonic, both are unaffected by NTP steps,
and both read the same underlying counter, so a nanosecond in one file means
the same thing as a nanosecond in the other.

Reading a clock is not free, and nine reads per frame is nine chances to
mismeasure. Both runners calibrate at construction by taking back-to-back reads
and recording the median delta, and stamp the result into every timing record
as `timer_overhead_ns`:

| Implementation | Median cost of one clock read | Nine reads per frame |
|---|---|---|
| C++ | 21 ns | 189 ns |
| Python | 184 ns | 1656 ns |

**The overhead is reported and not subtracted.** Two reasons. It is far below
the resolution of anything being compared: the instrumentation floor is
1656 ns against a Python frame of tens of milliseconds, which is
0.002% of it. And subtracting an estimate from a measurement makes
the measurement depend on the quality of the estimate, which is a worse
property than being a known amount too large. A reader who wants it removed has
the number in every record.

It is worth naming that the Python clock read costs about
9 times the C++ one, and that this difference is charged to
Python inside Python's own measurement. It is a real cost of instrumenting
Python code, and at 1.5 microseconds per frame it does not move
any conclusion here. On a pipeline a hundred times faster it would.

## 3. Warm-up

Every run executes `benchmark.warmup_frames` (100) frames before the first
measured one, cycling the store exactly as the measured loop does. The measured
loop then continues from where warm-up left off, so the first measured frame is
not the first frame of the store sitting hot in cache.

Warm-up is there for five specific things, not as a ritual:

1. **Page faults.** Every scratch buffer is allocated at construction but not
   touched until the first frame. Without warm-up the first measured frame pays
   for hundreds of minor faults on a 1280x720 point cloud.
2. **CPU frequency.** A core that has been idle starts below its sustained
   clock.
3. **Branch predictors and caches.** The RANSAC scoring loop and the IK
   iteration are both branchy.
4. **NumPy dispatch.** The first call into a ufunc or a `scipy.ndimage` routine
   resolves and caches a good deal of machinery.
5. **CPython 3.11's specialising interpreter.** Bytecode is quickened after a
   function has been executed enough times, and the specialised forms are
   measurably faster. Measuring a Python pipeline before it has quickened
   measures the wrong interpreter.

Warm-up does not reset the garbage collector's counters: the baseline is taken
after warm-up so that only collections during measured frames are attributed.

## 4. Why the frame store is cycled rather than one frame reused

A benchmark that runs one frame two thousand times measures that frame from L2.
Cycling a 100-frame store means the working set is the store, which at
640x480 is about 150 MB of depth and colour: too large for any cache on this
machine, which is what a real camera feed also is.

Cycling buys a second thing that matters more. The store's frames differ in
point count after the crop, in cluster size, in how many RANSAC candidates
score well and in how many IK iterations the target needs. That variation is
workload variance, and it belongs in the distribution. A single-frame benchmark
would report a distribution made entirely of scheduler noise and would look far
tighter than the pipeline actually is.

The cost of cycling is that the samples are not independent. At the default
2000 measured frames over a 100-frame store each frame is executed twenty
times, so the *effective* sample size for workload variation is 100, not 2000.
That is why the confidence interval in section 6 resamples frames rather than
samples, and why `RESULTS.md` quotes the wider of the two intervals.

## 5. Why percentiles, and why the mean is printed anyway

Latency distributions are bounded below by the work and unbounded above by
everything that can go wrong, so they are right-skewed. The mean of such a
distribution sits above the median and below the tail and describes neither.
A control loop does not miss a deadline on the mean. It misses on the frame
that took longest.

So every figure in `RESULTS.md` is a percentile: p50, p90, p95, p99, p99.9 and
the maximum. The mean is printed in the same table, beside a `mean/p50` column,
for exactly one purpose: to let a reader see the size of the error they would
have made by quoting it.

Percentiles use NumPy's `nearest` method, which returns an observed sample
rather than interpolating between two neighbours. At n = 2000 the p99.9 falls
between the second and third slowest samples, and interpolating there would
invent a latency that never occurred. A latency that was never measured is not
a latency.

## 6. The confidence interval on p99

A p99 quoted from a few thousand samples with no interval attached invites
being picked apart, and deserves to be. `harness/analyze.py` attaches a
percentile bootstrap, `analysis.bootstrap_resamples` (2000) resamples at
`analysis.bootstrap_confidence` (95%), seeded from `analysis.bootstrap_seed` so
the interval is reproducible.

It computes two intervals and publishes the wider one:

* **i.i.d. bootstrap**: resample the measured samples with replacement. This is
  the textbook interval and it is too narrow here, because twenty samples of
  store frame 7 are not twenty independent draws from the workload.
* **Frame-blocked bootstrap**: resample the *distinct store frames* with
  replacement and keep every repeat of each drawn frame. The resampling unit is
  then the thing that actually varies.

Both are in `summary.json`. The table in `RESULTS.md` shows the blocked one.
The gap between them is a measure of how much the repeat structure was
flattering the naive interval.

## 7. Why both implementations see identical recorded input

The two pipelines read the same bytes, from the same files, in the same order.
Not the same distribution: the same bytes. That removes an entire class of
explanation from any difference in the numbers, and it is why the corpus is a
byte-reproducible frame store rather than a live camera or a re-sampled scene.

Determinism goes further than the input. `ALGORITHM.md` rules 1 to 5 fix
accumulation width, tie-breaking, eigenvector sign, and, critically, forbid any
PRNG inside the measured path: RANSAC's candidate triples come from
`assets/ransac_uniform.bin`, a committed table of 384 float64 deviates, so both
implementations sample the same points on the same iteration of the same frame.
Without that, "C++ found a better plane in fewer effective iterations" would be
a permanent confound on the most expensive stage in the pipeline.

## 8. Attributing the Python tail

Every Python timing record carries the cumulative CPython collection counters
per generation, sampled once per frame through a `gc.callbacks` hook. A
collection during frame *k* shows up as an increase between frames *k-1* and
*k*, so the frames a collection landed on are known rather than guessed at.
Frame 0 has nothing to difference against and is excluded from both sides of
the split.

`harness/analyze.py` then reports, per run: how many collections fired, how
many frames they landed on, p99 over all frames, p99 over frames with no
collection, the share of tail frames that saw one, and the point-biserial
correlation between "a collection fired" and the frame's latency.

If the tail turns out not to be the collector, that is stated plainly and is
itself a finding: it means the Python tail is scheduler noise and allocator
behaviour rather than the one thing everyone assumes it is. See
`results/RESULTS.md` for what this corpus actually showed.

## 9. The equivalence gate

`harness/compare_outputs.py` runs before any latency number is quoted, and CI
runs it as a gate rather than as a report. It compares, per frame of the store:
`plane_found`, the plane `(n, d)`, `graspable`, `width`, the 4x4 TCP pose,
`converged`, the IK iteration count, the joint solution `q`, the trajectory
duration, and the SHA-256 of the whole waypoint block.

The tolerance is `analysis.equivalence_tolerance`, 1e-6 absolute, in metres for
positions, radians for angles and dimensionless for direction cosines. That is
several orders of magnitude tighter than any physical effect and several orders
looser than the ~1e-15 noise that different but equally valid summation orders
produce.

**Bail-outs are compared up to the point where they happened.** A frame that
found no plane has no plane to compare, so the plane vector is skipped, but
both implementations must agree that no plane was found. A frame that was
judged not graspable never ran IK, so its pose, joints and trajectory are
skipped, and both must agree that it was not graspable. Disagreeing about
*where* a frame gave up is a mismatch even when every field that survives the
bail-out matches, because it means the two programs took different paths.

The gate does not print "passed". It prints the worst absolute deviation
observed in every field, over every frame, with the frame and index where it
occurred. A stated deviation is evidence; "passed" is an assertion, and it
looks identical whether the true figure is 1e-15 or 9e-7. Measured over every
corpus in this run:

| field | values compared | worst absolute deviation |
|---|---:|---|
| `plane_found` | 500 | exact, every value |
| `plane` | 2,000 | 2.5e-14 (unit normal, metres) |
| `graspable` | 500 | exact, every value |
| `width` | 500 | 1.4e-15 (metres) |
| `tcp` | 8,000 | 1.4e-13 (metres, direction cosines) |
| `converged` | 500 | exact, every value |
| `iterations` | 500 | exact, every value |
| `q` | 3,500 | 3.5e-13 (radians) |
| `duration_s` | 500 | 2.5e-13 (seconds) |
| `traj_checksum` | 500 | 499 identical, 1 on the rounding grid |

Over `table_320x240`, `table_640x480`, `table_848x480` and `table_1280x720`,
plus the single-BLAS-thread pass on `table_640x480`: five comparisons, 500
frames, tolerance 1e-6.

The one digest difference is `table_848x480` frame 95, and it is the
boundary-straddle case described above rather than a divergence. Two waypoint
values on that frame sit 6.7e-15 from a nine-decimal grid boundary while the
joint solution that generates them agrees to 5.0e-14, which the quintic
amplifies to at most 1.1e-12 inside a waypoint against a grid half-step of
5.0e-10: two thousandths of one step. Nothing can hide in that, and the gate
says so with the numbers rather than either failing or staying quiet.

## 10. The control-rate sweep

`harness/rate_sweep.py` answers the question the per-frame percentiles cannot:
at a given control rate, how often is the answer late, and by how much.

**The scheduler is real and its deadlines are absolute.** Job *k* is released
at `origin + k * period` and is due at `origin + (k+1) * period`, both derived
from a single monotonic origin taken once at the start of the run. The obvious
alternative, sleeping for `period - work` after each job, accumulates the
sleep's own overshoot: a few tens of microseconds per iteration becomes tens of
milliseconds of drift over a few hundred jobs, and the drift is then reported
as jitter. That is a measurement destroying itself.

**Overrun is not queued away.** If job *k* finishes after job *k+1* was due,
job *k+1* starts immediately and late. That is what a best-effort
`SensorDataQoS` depth subscriber does with a backlog it cannot clear
(see [ROS2.md](ROS2.md)), and it is why the sweep reports two latencies: the
`compute` time of the pipeline, and the `response` time from release to finish,
which carries the backlog and is what a downstream controller waits for.

The scheduler is not free either. At a rate where nothing is backlogged, the
difference between `response` and `compute` is the whole of it: waking from
`time.sleep` a little after the release instant, plus reading the job's result
out of the worker. Measured at 10 Hz on this machine it is
0.15 to 0.25 ms per job, dominated by sleep wake-up rather than by
anything either implementation does.

**Both implementations run under the same Python scheduler**, so the
scheduler's own cost and its own noise are charged to both equally. The C++
pipeline is reached through a C ABI shim that `rate_sweep.py` generates and
compiles against the same `libgrasp_core.a` that `bench_pipeline` links, using
compile flags read out of the CMake build tree so the shim cannot drift from
the library. The latency attributed to a job is the pipeline's own `total_ns`,
measured by the same instrumentation as the in-process benchmark, so the
`ctypes` transition is not inside it. The transition is measured separately and
recorded in `rate_sweep.json` metadata (370 ns per call on this
machine, an upper bound because the calibration loop's own Python overhead is
inside it), because it does sit between the scheduler's release and the
pipeline's first instruction and a reader is entitled to know its size.

## 11. The IK null-space sweep

`ALGORITHM.md` records that `ik.nullspace_gain` is 0 because the term did not
earn its place, and points here for the measurement. `ik.damping` is swept
against `ik.nullspace_gain` over the grasp poses S5 produces on
`table_640x480`, which are the targets the pipeline actually asks for rather
than reachable poses drawn at random, seeding every solve from `q_neutral`.
`tools/fill_method_placeholders.py --measure-nullspace` reruns it and writes
`results/nullspace_sweep.json`, which is what the table below is generated
from.

| `nullspace_gain` | converged | median iterations | max iterations | worst position error | worst orientation error |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 of 100 | 9 | 36 | 0.494 mm | 0.3 mrad |
| 0.01 | 100 of 100 | 9 | 96 | 0.500 mm | 0.2 mrad |
| 0.02 | 95 of 100 | 10 | 100 | 0.804 mm | 0.3 mrad |
| 0.05 | 0 of 100 | 100 | 100 | 1.902 mm | 0.7 mrad |
| 0.1 | 0 of 100 | 100 | 100 | 2.867 mm | 1.4 mrad |
| 0.2 | 0 of 100 | 100 | 100 | 5.690 mm | 2.9 mrad |
| 0.5 | 0 of 100 | 100 | 100 | 13.811 mm | 6.9 mrad |

At gain 0 the solver converges on 100 of 100 targets in a median of 9
iterations, with a worst-case position error of 0.494 mm. That reproduces the
two figures `ALGORITHM.md` quotes.

Raising the gain buys nothing and costs convergence. The term is `I - J^T (J
J^T + lambda^2 I)^-1 J`, which is not a null-space projector while `lambda >
0`, so the posture bias leaks into task space instead of staying in the
redundant degree of freedom. By gain 0.1 the solver reaches tolerance on 0 of
100 targets and burns all 100 iterations on every frame.

**One correction to `ALGORITHM.md`.** It records that at gain 0.1 the
orientation error parks at about 198 mrad. That is not what this sweep finds:
at gain 0.1 the orientation error stays inside 1.4 mrad, comfortably under the
5 mrad tolerance, and what fails is *position*, with a worst case of 2.87 mm
against a 0.5 mm tolerance. The likely explanation is that the 198 mrad figure
predates the S5 wrist fold, which changed which of two equivalent grasp frames
is demanded and so changed where the solver stalls. The conclusion is unchanged
and the diagnosis is not, so the number should be corrected rather than
repeated.

A genuine pseudo-inverse projector does converge, and costs an SVD per
iteration on a stage that runs every frame. Since every frame is seeded from
`q_neutral` the solution already sits near the neutral posture, so the term had
nothing left to buy and the default is 0.

Reproduce it by writing a copy of `assets/pipeline_config.json` with a
different `ik.nullspace_gain`, constructing `GraspPipeline` against that copy,
running the 100 frames of the corpus and comparing `IKSolver.forward(q)`
against the S5 target pose.

---

# Threats to validity

None of these are hypothetical and none are cheap to dismiss. They are ordered
roughly by how much they could move the headline number.

## T1. The pipeline's shape decides the answer, and the shape was chosen once

This is the largest threat and it is not a caveat, it is the main result's
boundary. `deproject.stride` is 1 and `plane.iterations` is 128, so the RANSAC
plane stage dominates the frame at every resolution, and that stage is a large
blocked array reduction in both languages. A pipeline dominated by one big
vectorised reduction is close to the best case for NumPy, because almost all of
the wall time is spent inside compiled loops that both implementations reach.

Change the constants and the ratio changes. Raise `deproject.stride` to 4 and
the point count drops sixteenfold, the plane stage stops dominating, and the
per-frame Python interpreter overhead in the small-matrix stages (grasp, IK,
trajectory) becomes a much larger share. The per-stage table in `RESULTS.md`
makes this visible on purpose: the stages where the ratio is smallest are the
big-array ones and the stages where it is tens of times are the ones that
iterate. **The single headline ratio is a property of this pipeline at these
constants, and the per-stage table is the transferable result.**

## T2. Four shared vCPUs, no pinning, no isolation

The machine in [TOOLCHAIN.md](TOOLCHAIN.md) is a four-vCPU shared instance.
There is no CPU pinning, no `isolcpus`, no `nohz_full`, no real-time priority
and no control over what else the host is doing. Every tail figure therefore
contains scheduler noise that has nothing to do with either language.

The consequence is specific: **p99, p99.9 and max are upper bounds for this
stack, not clean measurements of the pipeline.** p50 is comparatively safe,
because the median is not where preemption lands. There is no obvious reason
the noise should favour one implementation, and it is charged to both runs
identically, but "no obvious reason" is not a control. The sweep records the
load average at the end of every run so a row taken under load can be spotted
rather than trusted, and `RESULTS.md` reports the load average observed at
analysis time.

A reproduction on a pinned, isolated core with the governor at `performance`
would give tighter tails for both. It would not necessarily give the same
ratio, because the two implementations have different sensitivities to
preemption: the Python process has more resident state to have evicted.

**The published run was not taken on an idle machine, and how far that moves
the numbers was measured rather than waved at.** The load average recorded in
`summary.json` says what the box was carrying. To size the effect the whole
benchmark was run twice, once with a neighbouring workload on the box and once
with the box to itself:

| run | p50 under load | p50 quiet | p99 under load | p99 quiet | p99 inflation |
|---|---:|---:|---:|---:|---:|
| `cpp.table_320x240` | 13.0 ms | 13.0 ms | 17.4 ms | 14.7 ms | 1.18x |
| `py.table_320x240` | 22.1 ms | 22.7 ms | 27.0 ms | 28.0 ms | 0.96x |
| `cpp.table_640x480` | 52.8 ms | 58.0 ms | 61.9 ms | 64.5 ms | 0.96x |
| `py.table_640x480` | 89.0 ms | 86.9 ms | 232.3 ms | 107.2 ms | 2.17x |
| `cpp.table_848x480` | 78.9 ms | 78.9 ms | 87.0 ms | 91.9 ms | 0.95x |
| `py.table_848x480` | 126.9 ms | 124.2 ms | 277.7 ms | 158.8 ms | 1.75x |
| `cpp.table_1280x720` | 180.7 ms | 180.6 ms | 204.0 ms | 207.2 ms | 0.98x |
| `py.table_1280x720` | 289.0 ms | 294.8 ms | 383.7 ms | 388.4 ms | 0.99x |

The first run of this experiment shared the box with another workload at a load
average of 2 to 3.5; the published run had it to itself. p50 barely noticed,
moving between 0.91x and 1.02x, and the direction is not even consistent, so a
few percent of that is ordinary run-to-run variation rather than the neighbour.
p99 noticed a great deal: `py.table_640x480` reported 232 ms under load against
107 ms quiet, an inflation of 2.17x, and it would have been published as a
language result. That is the whole of this threat, measured on this machine:
**the median is a property of the pipeline and the tail is partly a property of
the box.** Every figure in `RESULTS.md` comes from the quiet run.

## T3. ROS 2 is absent from this machine, so the transport is not measured here

`packages.ros.org` returns 403 through this container's egress proxy, so ROS 2
cannot be installed and has not been. Everything in `results/` is the
**in-process** pipeline: no DDS, no CDR serialisation, no executor dispatch, no
`rclpy` message conversion.

Those are not incidental costs. They are most of what people mean when they say
a ROS node is slow, and `rclpy` pays them very differently from `rclcpp`: the
Python node converts a C message into Python objects on every callback, which
the C++ node does not. So the ratio measured here is a **lower bound on the
language cost of a ROS 2 grasp node**, and possibly a loose one.

The ROS layer exists (`ros2_ws/`, `docker/`) and is designed to emit the same
timing format, but as [ROS2.md](ROS2.md) states without hedging, it has never
been built or run: `colcon build` has not executed, neither node has been
launched, and the Docker image has not been assembled. CI builds the workspace
and smoke-tests both nodes on every commit, which proves they compile and
start, and proves nothing about their latency. Any ROS number will be produced
on a host with Docker or it will not be produced.

## T4. This measures "Python plus NumPy", not "interpreted against compiled"

The Python implementation is deliberately the best reasonable NumPy and SciPy
code, not a transliteration of the C++. `ALGORITHM.md` specifies stage outputs
and forbids specifying loop structure precisely so that each side can be
idiomatic. Forcing scalar Python loops where a competent engineer would
vectorise would have manufactured a large, meaningless ratio.

The consequence is that the comparison is between **one compiled implementation
and one implementation that spends most of its time inside a different compiled
library**. Where a stage is one call over a large array, the "Python" number is
mostly NumPy's compiled loops and, in the plane scoring and the IK solve,
OpenBLAS; the gap there is small. Where a stage iterates over 6x7 Jacobians in
the interpreter, the gap is large. Neither of those is a statement about
CPython's execution speed in general. There is no `numba`, no Cython and no C
extension of our own, because the question is what an ordinary `rclpy` node
costs, and an ordinary `rclpy` node has NumPy and nothing else.

There is a sharper version of this worry: NumPy here is linked against
`scipy-openblas` built with `MAX_THREADS=64`, and the plane stage's block
scoring is a real `dgemm`. If OpenBLAS chose to thread it, the Python run would
be using several cores where Eigen's single-threaded kernels use one, and the
comparison would be between one core and four. That is checkable rather than
arguable, so it was checked:

| run | cores used | wall p50 | CPU seconds for the same work |
|---|---:|---:|---:|
| C++, Eigen, no OpenMP | 0.99 | 52.8 ms | 26 s |
| Python, BLAS threads unset | 3.87 | 82.0 ms | 152 s |
| Python, `OPENBLAS_NUM_THREADS=1` | 1.00 | 89.4 ms | 41 s |

OpenBLAS does thread: the Python process ran at 3.87 cores against C++'s 0.99.
What it does not do is convert that into latency. Pinning it to one thread
costs 9% of wall-clock p50 (89.4 ms against 82.0 ms) and saves 111 CPU seconds
out of 152 over the same 400 frames. The matrices here are too small for four
threads to beat their own synchronisation, so most of those cores are spinning.

Two consequences, and they point in opposite directions. The latency ratio in
`RESULTS.md` is **not** an artefact of Python being handed more cores: the
`py1t` row is within a few percent of the `py` row, so the comparison survives
being made per core. But the *cost* ratio is far worse than the latency ratio,
and on a four-core robot controller running a planner and a driver alongside
the perception node, burning 3.9 cores to save 9% of one node's latency is the
wrong trade. A deployment should set `OPENBLAS_NUM_THREADS=1` and take the
`py1t` number.

## T5. The corpus is graspable on 100% of frames by construction

`harness/make_dataset.py` guarantees that every frame contains exactly one
unambiguously largest, reachable, graspable object: footprints are separated by
`scene.objects.min_gap_m` so 1 cm voxels cannot merge two of them, the primary
object beats the runner-up in footprint area by `primary_area_ratio`, and
placement is constrained to a region where a top-down Panda grasp is reachable.

Therefore **a bail-out rate of zero in `RESULTS.md` is a property of the
generator, not a measured result about the pipeline.** It says nothing about
how often a real scene defeats plane fitting, clustering or IK. It was done on
purpose: a frame that bails out at S3 or S5 is a legitimate latency sample but
a *shorter* pipeline, and a corpus full of them would have measured a mixture
of two different amounts of work and reported the mixing ratio as latency. The
price is that the failure path is exercised by unit tests and not by the
benchmark.

## T6. The depth renderer is an analytic ray-caster, not a sensor

Depth comes from casting one ray per pixel against a plane, boxes and cylinders
in closed form, then applying an axial noise model
(`sigma(z) = 0.0015 z^2` metres) and an incidence-angle dropout model. That
gives exact ground truth and byte-reproducibility, which a rasteriser could not
on a machine with no GPU.

What it does not give is a real sensor. There is no multipath, no
interreflection,
no flying-pixel artefact at depth discontinuities, no exposure-dependent noise
floor, no per-unit calibration error, no rolling shutter and no temporal
correlation between frames. Real structured-light and time-of-flight depth is
messier in ways that change the *point count* surviving the crop and the
*inlier count* RANSAC finds, and both of those are directly proportional to the
cost of the dominant stage. A real sensor would plausibly move the absolute
latencies. It is less likely to move the C++ against Python ratio, since both
implementations would see the same messier cloud, but that is an argument, not
a measurement.

## T7. One machine, one compiler, one BLAS, one libc

Every number was produced by g++ 13.3 at `-O2 -DNDEBUG` with Eigen 3.4 and
CPython 3.11 with NumPy 2.4 on one x86-64 instance. That is a point, not a
curve.

Specific things that are unmeasured and could move the ratio: clang against
gcc; `-O3` and `-march=native` against the pinned `-O2` (the C++ side is the
one with headroom here, so the pinned flags are the conservative choice for the
C++ number and the generous one for Python); a different BLAS, or a BLAS with
threading enabled, since the Python side reaches BLAS and the C++ side reaches
Eigen's own kernels; aarch64, where NumPy's and Eigen's vectorisation stories
differ; and CPython 3.12 or 3.13, which changed the specialising interpreter
again. Nothing here should be quoted as "Python costs Nx" without the machine
attached.

## T8. Effective sample size is 100 frames, not 2000 samples

Covered in sections 4 and 6 and repeated here because it is easy to skim past:
2000 measured samples come from 100 distinct frames executed twenty times each.
For questions about scheduler noise, n is 2000. For questions about how the
pipeline behaves on a scene, n is 100. The frame-blocked bootstrap is the only
interval that respects the difference, and it is the one published.

## T9. The rate sweep drives C++ through ctypes, not through a C++ main loop

The sweep uses one Python scheduler for both implementations so that release
times, sleep behaviour and scheduler noise are identical. The cost is that the
C++ pipeline is entered through a `ctypes` call rather than from a C++ loop.
The pipeline's own instrumentation is inside the call, so the reported
`compute` latency excludes the transition; the `response` latency includes it.
At 370 ns per call against a C++ frame of milliseconds this is far
below the resolution of any conclusion drawn from it, but it is a difference
between how the two implementations are reached and it is not zero.

A C++ scheduler driving the C++ pipeline would remove it and introduce a
larger problem: two different scheduler implementations, with different
sleep-precision behaviour, whose difference would be indistinguishable from the
language difference the sweep is trying to isolate.

## T10. The equivalence gate bounds agreement, not correctness

The gate proves the two implementations compute the same thing to 1e-6. It does
not prove either of them computes the *right* thing. Correctness against the
spec is the job of the unit tests and of the URDF-against-MJCF kinematics
agreement test, which checks the forward kinematics against MuJoCo's
independent implementation of the same Panda model. Two implementations of a
wrong spec would sail through the gate together.
