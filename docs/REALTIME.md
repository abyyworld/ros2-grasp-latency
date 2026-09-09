# Real-time behaviour

`analyze.py` answers how long a frame takes. A control loop is not bought by
that number. It is bought by whether the answer arrives on the same phase every
period, by what the worst period does, and by whether the hot path can be
relied on not to allocate or fault. Those are different questions, and the rest
of this repository did not answer them.

Everything below is 2000 cycles at 30 Hz on `table_320x240`, pinned to one CPU,
six alternating repeats per policy, and every figure is read back out of
`results/jitter/` by `harness/analyze_jitter.py`.

## What jitter is measured against

Releases are absolute offsets from one monotonic origin:

```
release[k] = origin + k * period
clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, release[k])
```

Not `sleep(period)` after finishing. A loop written that way folds its own
compute time and its own wake-up latency back into the schedule, so it drifts,
and what it then measures is its own drift rather than the system's jitter.
With absolute deadlines the schedule is fixed in advance and the measurement is
the difference between it and reality:

* **Release jitter**: `wake - release[k]`. The scheduler's contribution, with
  the pipeline removed from it.
* **Slack**: `(release[k] + period) - finish`. Negative is an overrun.

`cpp/bench/bench_jitter.cpp` records both per cycle, along with the number of
allocations that cycle caused.

## Allocation in the hot path

The claim is that the measured path performs no heap allocation in steady
state. It is checked by replacing global `operator new` in the test binary,
never in the library, so the code under test is byte-identical to what the
benchmark runs.

| what | allocations |
|---|---|
| 32 frames at 640x480 | **0** |
| 32 frames at 320x240 | **0** |
| 32 frames taking `Result` **by value** | **32** |

The third row is a positive control and it is the reason the first two mean
anything. Every assertion here is a zero, and a broken counter also reports
zero, so without a case that must trip the instrument the test is equally
consistent with a clean pipeline and a dead one.

It also puts a number on an interface decision. `Pipeline::run` returns
`const Result &`. Copying that result reallocates the trajectory's storage,
once per frame, forever: the reference in the signature is worth exactly one
heap allocation per frame.

Across all fourteen jitter runs, 28000 measured cycles, the counter moved zero
times.

## Page faults, and the instrument's own hot path

Minor and major faults are read with `getrusage` outside every measured window,
so the syscall never lands inside a cycle it would then be reported as part of.

The first honest run of this did not report zero. It reported **19 minor
faults** over 2000 cycles, and the cause was the recorder, not the pipeline.
`bench_jitter` writes one record per cycle into a buffer it had `reserve()`d
and never touched, so the kernel handed over those pages on first write, inside
the measured window, where a reader would have charged them to the pipeline.

Buffer size and fault count line up exactly. Each record is 56 bytes:

| cycles | records | pages of records | minor faults observed |
|---:|---:|---:|---:|
| 2000 | 112 KB | 27.3 | **19** |
| 8000 | 448 KB | 109.4 | **109** |

The 8000-cycle run allocates its buffer past glibc's mmap threshold, so every
one of its 109 pages is fresh and every one faults. The 2000-cycle buffer comes
off a heap that was already partly resident, so it faults fewer times than its
page count. The scaling is the recorder's, not the workload's, which is what
identifies it.

The buffer is now pre-touched before the window opens, and `--no-pretouch` runs
the identical loop without that. Both are in the sweep:

| run | minor faults | major faults |
|---|---:|---:|
| twelve paired runs, pre-touched | **0** | **0** |
| `fifo_mlock`, pre-touched | **0** | **0** |
| `other_no_pretouch`, the control | **19** | 0 |

So the hot path touches no new memory once it is warm, which is the property
`mlockall` exists to guarantee and which this pipeline reaches without needing
it. The control is what makes that a measurement rather than an assertion: the
same binary, one flag apart, gives 19.

`mlockall` is available in this container but `RLIMIT_MEMLOCK` is 8 MB, below
the resident frame store, so it is run as its own arm. Locked, jitter p50 is
98.9 us against a `SCHED_FIFO` median of 97.2 us: no effect worth reporting on
a run that was already fault-free, which is what a fault-free run predicts.

## Worst-case execution time, and what causes it

Compute time over six repeats per policy. The percentiles are medians across
those repeats; the last row is not, because the median of six maxima is not a
worst case. It is the worst single cycle any of the six runs saw.

| | `SCHED_OTHER` | `SCHED_FIFO` |
|---|---:|---:|
| p50, median of runs | 9.90 ms | 9.85 ms |
| p99, median of runs | 12.50 ms | 12.55 ms |
| worst cycle, worst of the six runs | 18.98 ms | 27.69 ms |
| worst cycle, median of the six runs | 18.20 ms | 18.31 ms |

A maximum on its own is not a result, so the excess is attributed rather than
quoted. The loop cycles over 100 stored frames many times each, so the spread
of the per-frame medians is what the workload itself can account for.

**Frame content explains a band of 1.37 to 1.76 ms**, in every one of the
twelve runs. The worst cycle in each run sits **5.38 to 17.21 ms above its own
frame's median**, and in all fourteen runs every one of the ten slowest cycles
allocated zero times.

So the algorithm's own worst case is bounded tightly and the tail is not the
algorithm. Taking each run's worst cycle against the cheapest frame's median,
the share of that excess which frame content cannot account for is **86 to 98
percent** across the twelve paired runs. The rest is environmental: preemption,
cache eviction, and whatever else shares the vCPU.
That is the argument for CPU isolation and a preemptible kernel, and it is a
measurement here rather than a belief.

## SCHED_FIFO against SCHED_OTHER

Six repeats of each policy, alternating so neither policy owns a quieter
stretch of the machine. Release jitter in microseconds.

| run | p50 | p99 | max | overruns |
|---|---:|---:|---:|---:|
| `SCHED_OTHER` rep 1 | 165.5 | 580.9 | 9232 | 0 |
| `SCHED_OTHER` rep 2 | 153.1 | 724.1 | 37561 | 1 |
| `SCHED_OTHER` rep 3 | 156.2 | 289.9 | 2415 | 0 |
| `SCHED_OTHER` rep 4 | 151.0 | 510.4 | 5422 | 0 |
| `SCHED_OTHER` rep 5 | 146.0 | 493.6 | 22444 | 0 |
| `SCHED_OTHER` rep 6 | 148.7 | 362.0 | 22278 | 1 |
| `SCHED_FIFO` rep 1 | 98.4 | 396.4 | 20510 | 0 |
| `SCHED_FIFO` rep 2 | 95.9 | 227.9 | 21083 | 1 |
| `SCHED_FIFO` rep 3 | 103.7 | 230.1 | 26002 | 1 |
| `SCHED_FIFO` rep 4 | 100.1 | 225.0 | 4764 | 0 |
| `SCHED_FIFO` rep 5 | 88.3 | 206.6 | 19424 | 0 |
| `SCHED_FIFO` rep 6 | 91.0 | 247.4 | 42105 | 1 |

Medians across repeats, so one unlucky run cannot decide it:

| | `SCHED_OTHER` | `SCHED_FIFO` | improvement |
|---|---:|---:|---:|
| jitter p50 | 152.0 us | 97.2 us | **1.56x** |
| jitter p99 | 502.0 us | 229.0 us | **2.19x** |
| jitter max | 15.8 ms | 20.8 ms | **0.76x, none** |
| overruns in 12000 cycles | 2 | 3 | none |

**Real-time priority buys the median and the ninety-ninth percentile, and does
nothing at all for the maximum.**

The p50 result is the firm one: every `SCHED_FIFO` run beats every
`SCHED_OTHER` run, with no overlap between the ranges (88.3 to 103.7 against
146.0 to 165.5).

The p99 result is real but softer, and the ranges touch: `SCHED_FIFO` rep 1 at
396.4 us is worse than `SCHED_OTHER` rep 3 at 289.9 us. It is also the figure
that made this run six repeats instead of three. Reps 1 to 3 alone give a
median of 230.1 against 580.9, which is 2.52x; all six give 2.19x; and an
earlier three-repeat matrix on this same machine put the two policies level.
Three repeats do not settle a percentile on a shared vCPU. Six do better and
are still worth reading with the per-run spread beside them, which is why the
whole table is printed rather than only its medians.

The maximum is where the argument ends. Worst-case release jitter is 2.4 ms to
42.1 ms against a 33.3 ms period, in both policies, with the largest single
value in the `SCHED_FIFO` column: the loop occasionally wakes up **one to two
entire periods late**, and no guest scheduling policy prevents it, because the
thing being preempted is the whole virtual machine rather than the process
inside it. `SCHED_FIFO` decides which runnable thread the guest kernel picks;
it has no say in whether the guest is running at all.

That agrees with the worst-case attribution above, which found most of the
compute excess to be environmental. Both point the same way from different
directions: on this machine the algorithm is not the limit, and the remedy is a
machine that is not shared, not a flag. A loop whose budget is tight in the
ordinary case gains from the policy. Anyone quoting it as insurance against the
worst period on shared hardware would be quoting something this data does not
show.

## Determinism

A real-time claim is worth little if the answer depends on how the process was
scheduled or built. Every row below is an artifact in `results/` or a test in
the suite.

| the answer does not change across | evidence | agreement |
|---|---|---|
| the two implementations, x86 | `equivalence.table_640x480.json` | 4.17e-13, 0 mismatches |
| the two implementations, arm64 | `equivalence.inproc.json` | 2.45e-13, 0 mismatches |
| `-O2`, `-O3`, `-O3 -march=native` | `compiler_flags.json` | three distinct binaries, 1e-14 |
| RANSAC block size, 512 to 131072 | `test_block_size_invariance.py` | bit-identical |
| repeated runs of one binary | `cmp` on the output JSONL | byte-identical |

Nothing in the measured path draws from a PRNG, and the iteration counts are
fixed by the config rather than by convergence alone, which is what makes the
last row achievable at all.

## What this does not establish

**PREEMPT_RT was not run.** The kernel here is stock. `SCHED_FIFO` and
`mlockall` are permitted, so the scheduling-policy comparison above is real,
but a preemptible kernel is a different thing and no number in this repository
describes one. `harness/run_realtime_sweep.sh` runs the full matrix including
that arm, checks `uname` for an RT kernel, and reports plainly when the arm was
skipped.

**This machine is a poor one for jitter.** Four shared vCPUs, no isolation, no
governor exposed, and a documented run-to-run spread of about 9 percent that
was never fully explained (see METHOD.md T2). Absolute jitter figures from here
describe the container more than the pipeline. What survives is the comparison
between two policies measured back to back on the same machine, and the
attribution above, which is a ratio between two things measured in the same
run.

**Nothing here is a WCET bound in the formal sense.** It is a maximum observed
over a finite run. A bound would need static analysis or a measurement-based
methodology with a confidence argument, and neither is claimed.

**The pipeline is not a 30 Hz workload on this machine at this resolution.**
Compute p50 is about 9.9 ms against a 33.3 ms period, so the loop has three
times the budget it needs and the jitter figures are measured with the CPU
mostly idle between cycles. A loop running near its deadline would see a
different picture, and `results/rate_sweep.json` is where that question is
asked instead.
