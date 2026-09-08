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
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A percentile is only reported when at least this many samples sit above it,
# so that the figure is not simply the maximum wearing a percentile's name.
MIN_SAMPLES_ABOVE = 5


def percentile(values, p):
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def supported(n, p):
    return n * (1.0 - p / 100.0) >= MIN_SAMPLES_ABOVE


def load(paths):
    runs = defaultdict(list)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                runs[(row["policy"], row.get("label", row["policy"]))].append(row)
    return runs


def summarise(rows):
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
    }
    out["jitter_us"]["p99_9"] = percentile(jitter, 99.9) if supported(n, 99.9) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="results/jitter_summary.json")
    args = ap.parse_args()

    runs = load(args.paths)
    summary = {f"{policy}|{label}": summarise(rows) for (policy, label), rows in sorted(runs.items())}

    print(f"{'run':<18} {'n':>6} {'jitter p50':>11} {'p99':>9} {'p99.9':>9} {'max':>9} "
          f"{'overruns':>9} {'allocs':>7}")
    print(f"{'':<18} {'':>6} {'us':>11} {'us':>9} {'us':>9} {'us':>9}")
    for name, s in summary.items():
        j = s["jitter_us"]
        p999 = f"{j['p99_9']:.1f}" if j["p99_9"] is not None else "n/a"
        print(f"{name.split('|')[1]:<18} {s['samples']:>6} {j['p50']:>11.1f} {j['p99']:>9.1f} "
              f"{p999:>9} {j['max']:>9.1f} {s['overruns']:>9} {s['allocations']:>7}")

    # The comparison the run exists for: the same machine, two policies.
    by_policy = defaultdict(list)
    for name, s in summary.items():
        by_policy[name.split("|")[0]].append(s)
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
