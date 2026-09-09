#!/usr/bin/env python3
"""What a control loop actually buys, which is not the median latency.

`analyze.py` reports how long a frame takes. A servo consuming this output has
a period and a deadline, and what it cares about is whether the answer lands on
the same phase every cycle and what the worst cycle does. Those are different
questions and this answers them:

* **Release jitter**: the gap between when a cycle was scheduled to start and
  when it actually did. This is the scheduler's contribution, with the pipeline
  taken out of it.
* **Slack**: deadline minus finish. Negative is an overrun, and the count of
  those is the number that decides whether the loop holds.
* **Allocation and fault counts**, which are reported as a total rather than a
  distribution because for a hot path the only defensible value is zero.

Percentiles use the nearest observed sample, never an interpolation between two
values that were not measured. A percentile deeper than the sample count can
support is refused rather than printed: with 400 samples the p99.9 *is* the
maximum, and reporting it as though it were a tail estimate invites a
conclusion the data cannot carry.

Usage: analyze_jitter.py results/jitter/*.jsonl [--out results/jitter_summary.json]
"""
import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A percentile is only reported when at least this many samples sit above it,
# so that the figure is not simply the maximum wearing a percentile's name.
MIN_SAMPLES_ABOVE = 5

# The alternating repeats that make up the policy comparison. Anything else in
# the directory is reported on its own line and kept out of the medians.
PAIRED_RUN = re.compile(r"(other|fifo)_rep\d+")


def percentile(values, p):
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def supported(n, p):
    # The tolerance is for the representation, not for the rule: 99.9 is not
    # exact in binary, so 5000 samples at p99.9 evaluates to 4.999999999999449
    # and would be refused for a shortfall of 5e-13 samples.
    return n * (1.0 - p / 100.0) >= MIN_SAMPLES_ABOVE - 1e-9


def load(paths):
    """Return {(policy, label): (header, cycles)}.

    Each file opens with a `record: "run"` line carrying the counts that belong
    to the whole window rather than to a cycle: allocations, page faults, and
    the policy notes saying which of the requested privileges the kernel
    actually granted. A file without one is refused rather than summarised: it
    predates the header and the fault counts for it exist only in a terminal
    someone has since closed.
    """
    runs = {}
    for path in paths:
        header = None
        cycles = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record") == "run":
                header = row
            else:
                cycles.append(row)
        if header is None:
            raise SystemExit(f"{path} has no run header; re-run bench_jitter to produce one")
        runs[(header["policy"], header.get("label", header["policy"]))] = (header, cycles)
    return runs


def attribute_worst_case(rows):
    """Split the worst cycle into what the frame costs and what the machine did.

    A maximum on its own is not a result. The loop cycles over a fixed store,
    so every frame is processed many times, and the spread of the per-frame
    medians is what the workload itself can account for. Whatever the worst
    cycle sits above its own frame's median is not the algorithm: it is
    preemption, cache eviction, or whatever else shares the CPU.

    Allocation counts for the slowest cycles come along because they are the
    obvious alternative explanation, and one that would be the pipeline's
    fault.
    """
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r["compute_ns"] / 1e6)
    frame_medians = {f: statistics.median(v) for f, v in by_frame.items()}

    worst = max(rows, key=lambda r: r["compute_ns"])
    worst_ms = worst["compute_ns"] / 1e6
    slowest = sorted(rows, key=lambda r: r["compute_ns"], reverse=True)[:10]
    return {
        "frames_seen": len(frame_medians),
        "frame_median_min_ms": min(frame_medians.values()),
        "frame_median_max_ms": max(frame_medians.values()),
        "frame_content_band_ms": max(frame_medians.values()) - min(frame_medians.values()),
        "worst_cycle_ms": worst_ms,
        "worst_cycle_frame": worst["frame"],
        "worst_cycle_excess_over_its_frame_median_ms":
            worst_ms - frame_medians[worst["frame"]],
        # Of the gap between the worst cycle and the cheapest frame's median,
        # the share that no difference in frame content can account for. A
        # number near 1 says the tail is the machine.
        "environmental_share_of_worst_excess":
            (worst_ms - frame_medians[worst["frame"]])
            / (worst_ms - min(frame_medians.values())),
        "allocations_in_ten_slowest_cycles": sum(r.get("allocs", 0) for r in slowest),
    }


def summarise(header, rows):
    jitter = [r["release_jitter_ns"] / 1e3 for r in rows]      # microseconds
    compute = [r["compute_ns"] / 1e6 for r in rows]            # milliseconds
    slack = [r["slack_ns"] / 1e6 for r in rows]
    n = len(rows)
    out = {
        "samples": n,
        "jitter_us": {"p50": percentile(jitter, 50), "p99": percentile(jitter, 99),
                      "max": max(jitter), "mean": statistics.fmean(jitter)},
        "compute_ms": {"p50": percentile(compute, 50), "p99": percentile(compute, 99),
                       "max": max(compute)},
        "worst_slack_ms": min(slack),
        "overruns": sum(1 for s in slack if s < 0),
        "allocations": sum(r.get("allocs", 0) for r in rows),
        "minor_faults": header["minor_faults"],
        "major_faults": header["major_faults"],
        "policy_notes": header["policy_notes"],
        "pretouch": header["pretouch"],
        "wcet_attribution": attribute_worst_case(rows),
    }
    # The per-cycle counts and the window count come from two different
    # counters reading the same atomic, so they have to agree. If they do not,
    # one of them is wrong and neither should be published.
    if out["allocations"] != header["allocations"]:
        raise SystemExit(
            f"{header['label']}: cycles account for {out['allocations']} allocations "
            f"but the window counted {header['allocations']}")
    if out["overruns"] != header["overruns"]:
        raise SystemExit(
            f"{header['label']}: cycles show {out['overruns']} overruns "
            f"but the run header says {header['overruns']}")
    out["jitter_us"]["p99_9"] = percentile(jitter, 99.9) if supported(n, 99.9) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="results/jitter_summary.json")
    args = ap.parse_args()

    runs = load(args.paths)
    summary = {f"{policy}|{label}": summarise(header, rows)
               for (policy, label), (header, rows) in sorted(runs.items())}

    print(f"{'run':<18} {'n':>6} {'jitter p50':>11} {'p99':>9} {'p99.9':>9} {'max':>9} "
          f"{'overruns':>9} {'allocs':>7} {'minflt':>7} {'majflt':>7}")
    print(f"{'':<18} {'':>6} {'us':>11} {'us':>9} {'us':>9} {'us':>9}")
    for name, s in summary.items():
        j = s["jitter_us"]
        p999 = f"{j['p99_9']:.1f}" if j["p99_9"] is not None else "n/a"
        print(f"{name.split('|')[1]:<18} {s['samples']:>6} {j['p50']:>11.1f} {j['p99']:>9.1f} "
              f"{p999:>9} {j['max']:>9.1f} {s['overruns']:>9} {s['allocations']:>7} "
              f"{s['minor_faults']:>7} {s['major_faults']:>7}")
    for name, s in summary.items():
        print(f"  {name.split('|')[1]}: {s['policy_notes']}"
              f"{'' if s['pretouch'] else 'not-pretouched;'}")

    print()
    print("worst cycle, split into what the frame costs and what the machine did")
    print(f"{'run':<18} {'worst':>9} {'frame band':>11} {'excess over':>12} {'allocs in':>10}")
    print(f"{'':<18} {'ms':>9} {'ms':>11} {'frame median':>12} {'10 slowest':>10}")
    for name, s in summary.items():
        a = s["wcet_attribution"]
        print(f"{name.split('|')[1]:<18} {a['worst_cycle_ms']:>9.2f} "
              f"{a['frame_content_band_ms']:>11.2f} "
              f"{a['worst_cycle_excess_over_its_frame_median_ms']:>12.2f} "
              f"{a['allocations_in_ten_slowest_cycles']:>10}")

    # The comparison the run exists for: the same machine, two policies. Only
    # the paired repeats are pooled. The mlock arm also runs under SCHED_FIFO
    # and pooling it here would slide a second treatment into one policy's
    # median while the other policy has no counterpart to it.
    by_policy = defaultdict(list)
    for name, s in summary.items():
        policy, label = name.split("|")
        if PAIRED_RUN.fullmatch(label):
            by_policy[policy].append(s)
    if len(by_policy) == 2 and all(len(v) > 1 for v in by_policy.values()):
        print()
        print(f"{'policy':<10} {'jitter p50 median of runs':>26} {'p99 median of runs':>20}")
        medians = {}
        for policy, entries in sorted(by_policy.items()):
            p50 = statistics.median([e["jitter_us"]["p50"] for e in entries])
            p99 = statistics.median([e["jitter_us"]["p99"] for e in entries])
            medians[policy] = (p50, p99)
            print(f"{policy:<10} {p50:>26.1f} {p99:>20.1f}")
        if "other" in medians and "fifo" in medians:
            print()
            print(f"  SCHED_FIFO against SCHED_OTHER: "
                  f"p50 {medians['other'][0] / medians['fifo'][0]:.2f}x better, "
                  f"p99 {medians['other'][1] / medians['fifo'][1]:.2f}x better")
        summary["_policy_medians"] = {k: {"jitter_p50_us": v[0], "jitter_p99_us": v[1]}
                                      for k, v in medians.items()}

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
