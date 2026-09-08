"""The jitter analyser has to refuse what it cannot support.

Three ways a jitter summary could quietly say more than the run does, each one
checked here rather than trusted:

* a percentile deeper than the sample count, where p99.9 of 400 samples is the
  maximum wearing a percentile's name,
* a file whose cycle lines and run header disagree, which means one of the two
  counters is wrong and neither figure should be published,
* a file with no run header at all, which predates the header and whose fault
  counts exist only in a terminal someone has since closed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

import analyze_jitter  # noqa: E402


def header(**overrides) -> dict:
    base = {
        "record": "run", "impl": "cpp", "label": "other_rep1", "policy": "other",
        "policy_notes": "pinned;other;pretouched;", "rate_hz": 30, "frames": 4,
        "warmup": 200, "dataset": "data/table_320x240", "width": 320,
        "height": 240, "cpu": 2, "priority": 80, "pretouch": True,
        "allocations": 0, "allocated_bytes": 0, "minor_faults": 0,
        "major_faults": 0, "overruns": 0,
    }
    base.update(overrides)
    return base


def cycle(seq: int, jitter_ns: int, compute_ns: int, slack_ns: int) -> dict:
    return {"impl": "cpp", "label": "other_rep1", "policy": "other", "rate_hz": 30,
            "seq": seq, "frame": seq, "release_jitter_ns": jitter_ns,
            "compute_ns": compute_ns, "slack_ns": slack_ns, "allocs": 0}


def write(path: Path, head, cycles) -> Path:
    lines = ([] if head is None else [json.dumps(head)]) + [json.dumps(c) for c in cycles]
    path.write_text("\n".join(lines) + "\n")
    return path


FOUR = [cycle(0, 1000, 10_000_000, 23_000_000),
        cycle(1, 2000, 11_000_000, 22_000_000),
        cycle(2, 3000, 12_000_000, 21_000_000),
        cycle(3, 9000, 13_000_000, 20_000_000)]


def test_percentiles_come_from_observed_samples(tmp_path):
    runs = analyze_jitter.load([write(tmp_path / "a.jsonl", header(), FOUR)])
    (head, rows), = runs.values()
    out = analyze_jitter.summarise(head, rows)
    assert out["samples"] == 4
    # 4 samples: 1, 2, 3 and 9 microseconds. p50 is an observed value, never an
    # interpolation between two that were not measured.
    assert out["jitter_us"]["p50"] == 2.0
    assert out["jitter_us"]["max"] == 9.0
    # p99.9 of 4 samples is just the maximum, so it is refused rather than
    # printed with a percentile's authority.
    assert out["jitter_us"]["p99_9"] is None


def test_deep_percentile_is_reported_once_the_samples_support_it():
    assert not analyze_jitter.supported(400, 99.9)
    assert analyze_jitter.supported(5000, 99.9)


def test_a_header_that_disagrees_with_the_cycles_is_refused(tmp_path):
    path = write(tmp_path / "b.jsonl", header(allocations=7), FOUR)
    runs = analyze_jitter.load([path])
    (head, rows), = runs.values()
    with pytest.raises(SystemExit, match="allocations"):
        analyze_jitter.summarise(head, rows)

    overrun = FOUR[:-1] + [cycle(3, 9000, 13_000_000, -1_000_000)]
    path = write(tmp_path / "c.jsonl", header(), overrun)
    runs = analyze_jitter.load([path])
    (head, rows), = runs.values()
    with pytest.raises(SystemExit, match="overruns"):
        analyze_jitter.summarise(head, rows)


def test_a_file_without_a_run_header_is_refused(tmp_path):
    path = write(tmp_path / "d.jsonl", None, FOUR)
    with pytest.raises(SystemExit, match="no run header"):
        analyze_jitter.load([path])


def test_only_the_paired_repeats_enter_the_policy_medians(tmp_path):
    """The mlock and no-pretouch arms also run under a named policy. Pooling
    them into that policy's median would slide a second treatment into one side
    of the comparison while the other side has no counterpart to it."""
    for label, policy in (("other_rep1", "other"), ("other_rep2", "other"),
                          ("fifo_rep1", "fifo"), ("fifo_rep2", "fifo"),
                          ("fifo_mlock", "fifo"), ("other_no_pretouch", "other")):
        cycles = [dict(c, label=label, policy=policy) for c in FOUR]
        if label in ("fifo_mlock", "other_no_pretouch"):
            # Far off the paired runs, so pooling it would move the median.
            cycles = [dict(c, release_jitter_ns=c["release_jitter_ns"] * 1000)
                      for c in cycles]
        write(tmp_path / f"{label}.jsonl",
              header(label=label, policy=policy), cycles)

    out = tmp_path / "summary.json"
    subprocess.run(
        [sys.executable, str(ROOT / "harness" / "analyze_jitter.py"),
         *sorted(str(p) for p in tmp_path.glob("*.jsonl")), "--out", str(out)],
        cwd=ROOT, check=True, capture_output=True)
    summary = json.loads(out.read_text())
    assert set(summary["_policy_medians"]) == {"other", "fifo"}
    for policy in ("other", "fifo"):
        assert summary["_policy_medians"][policy]["jitter_p50_us"] == 2.0


def test_worst_case_is_split_between_the_frame_and_the_machine():
    """Two frames of different cost, each seen twice, and one cycle of the
    cheap frame blown out by something outside the pipeline. The band belongs
    to the frames and the blow-out must not be attributed to them."""
    rows = [
        dict(cycle(0, 100, 10_000_000, 0), frame=0),
        dict(cycle(1, 100, 12_000_000, 0), frame=1),
        dict(cycle(2, 100, 10_000_000, 0), frame=0),
        dict(cycle(3, 100, 12_000_000, 0), frame=1),
        dict(cycle(4, 100, 30_000_000, 0), frame=0),
    ]
    a = analyze_jitter.attribute_worst_case(rows)
    assert a["frames_seen"] == 2
    # Frame 0 medians over 10, 10 and 30 ms is 10; frame 1 is 12.
    assert a["frame_content_band_ms"] == pytest.approx(2.0)
    assert a["worst_cycle_ms"] == pytest.approx(30.0)
    assert a["worst_cycle_frame"] == 0
    # 20 of the 30 ms is not something the workload can account for.
    assert a["worst_cycle_excess_over_its_frame_median_ms"] == pytest.approx(20.0)
    assert a["allocations_in_ten_slowest_cycles"] == 0
