#!/usr/bin/env python3
"""In-process latency benchmark for the Python pipeline.

Same command line as `cpp/build/bench_pipeline`, same two output files, so
`harness/compare_outputs.py` and the analysis do not need to know which
implementation produced which.

Three things this runner does deliberately:

* The frame store is read into RAM before the first measurement. A benchmark
  that reads a file per frame measures the page cache.
* Records are formatted and written after the loop, never inside it, so no
  measured frame pays for `json.dumps` or a `write`.
* Garbage collection is counted, not disabled. A real rclpy node runs with the
  collector on, so the honest thing is to leave it on and record when it fires.
  The counter is a `gc.callbacks` hook rather than `gc.get_stats()`, because
  `get_stats` builds a list of dicts on every call and the harness must not be
  a meaningful contributor to the allocation pressure it is trying to measure.

The measured frames cycle through the store, so the working set is the store
rather than one frame sitting hot in L2.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'python'))

from grasp_core import (STAGE_KEYS, GraspPipeline,  # noqa: E402
                        calibrate_timer_ns)

SIGNIFICANT = '.17g'
CHECKSUM_DECIMALS = 9
PERCENTILES = (50.0, 95.0, 99.0)
# Percentiles are reported at the nearest observed sample rather than
# interpolated between two: a latency that was never measured is not a latency.
PERCENTILE_METHOD = 'nearest'


def number(value) -> str:
    """A double that survives a JSON round trip, per docs/FORMATS.md."""
    return format(float(value), SIGNIFICANT)


def vector(values) -> str:
    return '[' + ','.join(number(v) for v in values) + ']'


def flag(value) -> str:
    return 'true' if value else 'false'


class FrameStore:
    """The whole corpus in memory, as the bytes a subscriber would receive."""

    def __init__(self, directory: Path):
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.name = manifest['name']
        self.width = manifest['width']
        self.height = manifest['height']
        self.manifest = manifest
        self.frames = [
            ((directory / entry['depth']).read_bytes(),
             (directory / entry['rgb']).read_bytes())
            for entry in manifest['frames']]
        expected = self.width * self.height * 2
        for depth, _ in self.frames:
            if len(depth) != expected:
                raise ValueError(f'{directory} holds a depth frame of '
                                 f'{len(depth)} bytes, expected {expected}')

    def __len__(self) -> int:
        return len(self.frames)


class CollectionCounter:
    """Cumulative CPython collections per generation, per docs/FORMATS.md.

    A `gc.callbacks` hook costs one Python call per collection, where sampling
    `gc.get_stats()` would cost a list and three dicts per frame.
    """

    def __init__(self):
        self.counts = [0, 0, 0]
        gc.callbacks.append(self._on_collect)

    def _on_collect(self, phase, info):
        if phase == 'stop':
            self.counts[info['generation']] += 1

    def detach(self):
        gc.callbacks.remove(self._on_collect)


def trajectory_checksum(result) -> str:
    """SHA-256 over the waypoint block, exactly as docs/FORMATS.md defines it.

    Positions, then velocities, then accelerations, then time, waypoint by
    waypoint, each an IEEE-754 double rounded to nine decimals first, because
    the two implementations only agree to 1e-6.
    """
    count = result.positions.shape[0]
    block = np.empty((count, result.positions.shape[1] * 3 + 1),
                     dtype=np.float64)
    dof = result.positions.shape[1]
    block[:, :dof] = result.positions
    block[:, dof:2 * dof] = result.velocities
    block[:, 2 * dof:3 * dof] = result.accelerations
    block[:, 3 * dof] = result.times
    np.round(block, CHECKSUM_DECIMALS, out=block)
    return hashlib.sha256(block.tobytes()).hexdigest()


def output_record(impl: str, frame: int, result) -> str:
    return (
        '{"impl":"' + impl + '","frame":' + str(frame)
        + ',"plane_found":' + flag(result.plane_found)
        + ',"plane":' + vector(result.plane)
        + ',"graspable":' + flag(result.graspable)
        + ',"width":' + number(result.width)
        + ',"tcp":' + vector(result.tcp.reshape(-1))
        + ',"converged":' + flag(result.converged)
        + ',"iterations":' + str(result.iterations)
        + ',"q":' + vector(result.q)
        + ',"duration_s":' + number(result.duration_s)
        + ',"traj_checksum":"' + trajectory_checksum(result) + '"}')


def timing_record(impl: str, dataset: str, row) -> str:
    (seq, frame, total, points, cluster_points, iterations,
     plane_found, graspable, converged, gen0, gen1, gen2) = row[:12]
    stages = row[12:]
    parts = ','.join(f'"{key}":{value}' for key, value in zip(STAGE_KEYS, stages))
    return (
        '{"impl":"' + impl + '","dataset":"' + dataset + '","frame":'
        + str(frame) + ',"seq":' + str(seq)
        + ',"stage_ns":{' + parts + '}'
        + ',"total_ns":' + str(total)
        + ',"points":' + str(points)
        + ',"cluster_points":' + str(cluster_points)
        + ',"ik_iterations":' + str(iterations)
        + ',"plane_found":' + flag(plane_found)
        + ',"graspable":' + flag(graspable)
        + ',"converged":' + flag(converged)
        + ',"gc":{"gen0":' + str(gen0) + ',"gen1":' + str(gen1)
        + ',"gen2":' + str(gen2) + '}}')


def percentiles(samples: np.ndarray) -> dict:
    return {f'p{int(p)}_ns': float(np.percentile(samples, p,
                                                 method=PERCENTILE_METHOD))
            for p in PERCENTILES}


def main(argv=None) -> int:
    default_config = REPO_ROOT / 'assets' / 'pipeline_config.json'
    config = json.loads(default_config.read_text())
    benchmark = config['benchmark']

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, required=True,
                        help='frame store directory, holding manifest.json')
    parser.add_argument('--out-timing', type=Path, required=True)
    parser.add_argument('--out-output', type=Path, required=True)
    parser.add_argument('--out-meta', type=Path, default=None,
                        help='run metadata; defaults beside --out-timing')
    parser.add_argument('--warmup', type=int, default=benchmark['warmup_frames'])
    parser.add_argument('--frames', type=int, default=benchmark['measured_frames'])
    parser.add_argument('--config', type=Path, default=default_config)
    parser.add_argument('--chain', type=Path,
                        default=REPO_ROOT / 'assets' / 'franka' / 'panda_chain.json')
    parser.add_argument('--ransac-table', type=Path,
                        default=REPO_ROOT / 'assets' / 'ransac_uniform.bin')
    parser.add_argument('--impl', default='py')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    store = FrameStore(args.dataset)
    pipeline = GraspPipeline(str(args.config), str(args.chain),
                             str(args.ransac_table))
    timer = calibrate_timer_ns()

    frames = args.frames
    stages = np.zeros((frames, len(STAGE_KEYS)), dtype=np.int64)
    scalars = np.zeros((frames, 6), dtype=np.int64)
    flags = np.zeros((frames, 3), dtype=np.bool_)
    collections = np.zeros((frames, 3), dtype=np.int64)
    outputs = {}

    counter = CollectionCounter()
    counts = counter.counts
    baseline = None
    width = store.width
    height = store.height
    corpus = store.frames
    total_frames = len(corpus)
    run = pipeline.run
    impl = args.impl

    for index in range(args.warmup):
        depth, colour = corpus[index % total_frames]
        run(depth, colour, width, height)

    baseline = list(counts)
    started = time.time()
    for seq in range(frames):
        frame = (args.warmup + seq) % total_frames
        depth, colour = corpus[frame]
        result = run(depth, colour, width, height)

        stage_ns = result.stage_ns
        row = stages[seq]
        row[0] = stage_ns['decode']
        row[1] = stage_ns['deproject']
        row[2] = stage_ns['transform_crop']
        row[3] = stage_ns['plane']
        row[4] = stage_ns['cluster']
        row[5] = stage_ns['grasp']
        row[6] = stage_ns['ik']
        row[7] = stage_ns['traj']
        row = scalars[seq]
        row[0] = seq
        row[1] = frame
        row[2] = result.total_ns
        row[3] = result.points
        row[4] = result.cluster_points
        row[5] = result.iterations
        row = flags[seq]
        row[0] = result.plane_found
        row[1] = result.graspable
        row[2] = result.converged
        row = collections[seq]
        row[0] = counts[0]
        row[1] = counts[1]
        row[2] = counts[2]

        if frame not in outputs:
            outputs[frame] = output_record(impl, frame, result)
    elapsed = time.time() - started
    counter.detach()

    args.out_timing.parent.mkdir(parents=True, exist_ok=True)
    with args.out_timing.open('w', encoding='utf-8') as handle:
        for seq in range(frames):
            row = (list(scalars[seq]) + list(flags[seq])
                   + list(collections[seq]) + list(stages[seq]))
            handle.write(timing_record(impl, store.name, row) + '\n')

    args.out_output.parent.mkdir(parents=True, exist_ok=True)
    with args.out_output.open('w', encoding='utf-8') as handle:
        for frame in sorted(outputs):
            handle.write(outputs[frame] + '\n')

    totals = scalars[:, 2]
    summary = {
        'impl': impl,
        'dataset': store.name,
        'resolution': [width, height],
        'store_frames': total_frames,
        'warmup_frames': args.warmup,
        'measured_frames': frames,
        'wall_clock_s': elapsed,
        'percentile_method': PERCENTILE_METHOD,
        'total_ns': percentiles(totals),
        'stage_ns': {key: percentiles(stages[:, i])
                     for i, key in enumerate(STAGE_KEYS)},
        'gc_collections_total': [int(counts[i] - baseline[i]) for i in range(3)],
        'gc_collections_per_100_frames': [
            (counts[i] - baseline[i]) * 100.0 / frames for i in range(3)],
        'bail_outs': {
            'plane_not_found': int((~flags[:, 0]).sum()),
            'not_graspable': int((~flags[:, 1]).sum()),
            'ik_not_converged': int((~flags[:, 2]).sum()),
        },
        'ik_iterations': {
            'median': float(np.median(scalars[:, 5])),
            'max': int(scalars[:, 5].max()),
        },
        'points': {'median': float(np.median(scalars[:, 3]))},
        'timer': timer,
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count(),
            'load_average': list(os.getloadavg()),
            'blas_threads': {name: os.environ.get(name)
                             for name in ('OPENBLAS_NUM_THREADS',
                                          'OMP_NUM_THREADS',
                                          'MKL_NUM_THREADS')},
        },
        'inputs': {
            'config': str(args.config),
            'chain': str(args.chain),
            'ransac_table': str(args.ransac_table),
        },
    }
    meta_path = args.out_meta
    if meta_path is None:
        name = args.out_timing.name
        stem = name[:-len('.timing.jsonl')] if name.endswith('.timing.jsonl') \
            else args.out_timing.stem
        meta_path = args.out_timing.with_name(stem + '.meta.json')
    meta_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    if not args.quiet:
        report(summary, stages, totals)
    return 0


def report(summary, stages, totals) -> None:
    print(f"{summary['impl']} on {summary['dataset']} "
          f"{summary['resolution'][0]}x{summary['resolution'][1]}, "
          f"{summary['measured_frames']} measured frames "
          f"in {summary['wall_clock_s']:.1f} s")
    print(f"  {'stage':<16}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}"
          f"{'share':>8}")
    median_total = summary['total_ns']['p50_ns']
    for i, key in enumerate(STAGE_KEYS):
        stage = summary['stage_ns'][key]
        share = 100.0 * stage['p50_ns'] / median_total if median_total else 0.0
        print(f"  {key:<16}{stage['p50_ns'] / 1e6:>10.3f}"
              f"{stage['p95_ns'] / 1e6:>10.3f}{stage['p99_ns'] / 1e6:>10.3f}"
              f"{share:>7.1f}%")
    print(f"  {'total':<16}{median_total / 1e6:>10.3f}"
          f"{summary['total_ns']['p95_ns'] / 1e6:>10.3f}"
          f"{summary['total_ns']['p99_ns'] / 1e6:>10.3f}")
    print(f"  timer read: median {summary['timer']['median_ns']:.0f} ns "
          f"over {summary['timer']['samples']} samples")
    print(f"  gc collections per 100 frames: "
          f"{summary['gc_collections_per_100_frames']}")
    print(f"  bail-outs: {summary['bail_outs']}, "
          f"ik iterations median {summary['ik_iterations']['median']:.0f} "
          f"max {summary['ik_iterations']['max']}")


if __name__ == '__main__':
    sys.exit(main())
