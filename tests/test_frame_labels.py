"""A ROS record's frame index is a claim about identity, so it gets checked.

ROS 2 has no per-message sequence number, so `GraspLatency.seq` counts the
frames a node processed and the converter turns that into a frame index with
`seq % frame_count`. That is the published frame only while nothing is dropped,
and the arms this repository cares about drop plenty. These tests cover what
`harness/analyze.py` does about it: verify the label against the corpus, repair
it from the record's own fingerprint, and refuse a frame-blocked interval on a
row it cannot repair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

import analyze  # noqa: E402

# Frame f is fingerprinted by (points, cluster_points). Frames 3 and 4 share a
# point count, so the pair is what separates them.
TRUTH = {0: (1000, 50), 1: (1100, 60), 2: (1200, 70), 3: (1300, 80), 4: (1300, 90)}


def group(transport: str, frames, points, clusters, dataset="table_test"):
    g = analyze.Group("cpp", dataset, transport)
    g.frame = np.array(frames, dtype=np.int64)
    g.points = np.array(points, dtype=np.int64)
    g.cluster_points = np.array(clusters, dtype=np.int64)
    return g


def inproc(dataset="table_test"):
    order = sorted(TRUTH)
    return group("inproc", order, [TRUTH[f][0] for f in order],
                 [TRUTH[f][1] for f in order], dataset)


def test_an_inproc_run_is_taken_at_its_word():
    g = inproc()
    analyze.check_frame_labels([g])
    assert g.frame_labels["blockable"]
    assert g.frame_labels["agreement"] == 1.0


def test_a_lossless_ros_row_agrees_and_is_left_alone():
    ros = group("ros2", [0, 1, 2], [1000, 1100, 1200], [50, 60, 70])
    analyze.check_frame_labels([inproc(), ros])
    assert ros.frame_labels["agreement"] == 1.0
    assert ros.frame_labels["blockable"]
    assert ros.frame_labels["source"] == "seq modulo the frame count"
    assert ros.frame.tolist() == [0, 1, 2]


def test_a_lossy_ros_row_is_repaired_from_its_own_fingerprint():
    """The node processed frames 2, 0 and 3 but, having dropped everything in
    between, labelled them 0, 1 and 2. The point and cluster counts say what
    they really were."""
    ros = group("ros2", [0, 1, 2], [1200, 1000, 1300], [70, 50, 80])
    analyze.check_frame_labels([inproc(), ros])
    assert ros.frame_labels["agreement"] == 0.0
    assert ros.frame_labels["source"] == "point and cluster fingerprint"
    assert ros.frame_labels["blockable"]
    assert ros.frame.tolist() == [2, 0, 3]
    # The observed agreement is kept: overwriting it with the post-repair 1.0
    # would hide the defect the check exists to report.
    assert ros.frame_labels["agreement_after_repair"] == 1.0


def test_a_row_that_cannot_be_repaired_loses_its_frame_blocked_interval():
    """A fingerprint two frames share resolves to neither. One such record is
    enough to refuse the interval: a block bootstrap on a label that is wrong
    for part of the run is not the interval it claims to be."""
    ambiguous = dict(TRUTH)
    ambiguous[4] = ambiguous[3]          # frames 3 and 4 now indistinguishable
    order = sorted(ambiguous)
    truth_run = group("inproc", order, [ambiguous[f][0] for f in order],
                      [ambiguous[f][1] for f in order])
    ros = group("ros2", [0, 1], [1000, 1300], [50, 80])
    analyze.check_frame_labels([truth_run, ros])
    assert not ros.frame_labels["blockable"]
    assert ros.frame_labels["ambiguous"] == 1
    assert "could not be fully repaired" in ros.frame_labels["note"]


def test_a_corpus_with_no_inproc_run_is_not_guessed_at():
    ros = group("ros2", [0, 1], [1000, 1100], [50, 60], dataset="table_other")
    analyze.check_frame_labels([inproc(), ros])
    assert not ros.frame_labels["blockable"]
    assert ros.frame_labels["agreement"] is None


def test_the_committed_ros_runs_reproduce_the_documented_agreements():
    """The figures docs/ROS2.md quotes, read back off the committed files."""
    import json

    def load(path):
        rows = [json.loads(l) for l in open(ROOT / path) if l.strip()]
        return group(rows[0].get("transport", "inproc"),
                     [r["frame"] for r in rows], [r["points"] for r in rows],
                     [r["cluster_points"] for r in rows],
                     rows[0]["dataset"])

    truth_run = load("results/cpp.table_640x480.timing.jsonl")
    truth_run.transport = "inproc"
    quoted = {"results/ros2_composed_cpp.timing.jsonl": 1.0,
              "results/ros2_cpp.timing.jsonl": 0.003,
              "results/ros2_py.timing.jsonl": 0.010}
    rows = {p: load(p) for p in quoted}
    analyze.check_frame_labels([truth_run, *rows.values()])
    for path, expected in quoted.items():
        assert rows[path].frame_labels["agreement"] == pytest.approx(expected, abs=0.0005), path
        # Repaired or already right, every one of them ends up blockable.
        assert rows[path].frame_labels["blockable"], path
