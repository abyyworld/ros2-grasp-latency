#!/usr/bin/env python3
"""Threat T1, measured: how the py:cpp ratio moves when the workload mix moves.

The headline ratio in `results/RESULTS.md` is a property of one pipeline at
one set of constants. At `deproject.stride` 1 and `plane.iterations` 128 the
RANSAC plane stage is the overwhelming majority of the frame in both
implementations, and that stage is one large array reduction: Python issues it
as a handful of NumPy calls per block of points and then waits inside compiled
loops. A pipeline shaped like that is close to the best case for Python.
`docs/METHOD.md` used to say so and then assert, without measuring, that
changing the constants changes the ratio.

This script turns the assertion into a curve. It sweeps the two constants that
set how much of the frame is bulk array work:

* `deproject.stride`, which divides the point count by its square, and
* `plane.iterations`, which divides the candidate count outright.

Both shrink the plane stage without touching the small-matrix stages (grasp,
IK, trajectory), so the interpreter's per-call overhead goes from rounding
error to a large share of the frame, and the ratio has to move. The x axis of
the result is the plane stage's share of the C++ frame, which is the shape
variable that transfers to somebody else's pipeline; the y axis is the ratio.

## What this does per configuration

1. Writes a derived config: the committed `assets/pipeline_config.json` with
   exactly two keys changed, plus `ransac_table.count` to match the new
   iteration count. Neither implementation is patched and neither is told which
   configuration it is running: both read every constant from the file they are
   handed, which is the invariant `tests/test_config_is_sole_source.py` exists
   to protect.
2. Writes a derived deviate table: the **first** `3 * iterations` doubles of
   the committed `assets/ransac_uniform.bin`. A prefix rather than a fresh
   draw, so a 32-iteration run scores the same first 32 candidate triples the
   128-iteration run scored, and the difference between two rows is the amount
   of work and nothing else.
3. Re-chooses `implementation.ransac_block_points` by measurement, the way the
   committed value was chosen. That constant is read only by the Python plane
   stage, it changes memory traffic and never output, and 512 was picked when
   the distance block was 512 x 128 float64. At 8 iterations the same 512
   leaves the Python loop running the same number of times over a sixteenth of
   the arithmetic, so keeping it would charge Python for a constant tuned to a
   configuration it is no longer in, and the fairness rule in ALGORITHM.md
   forbids exactly that. The candidate that wins and the whole calibration
   curve go into `workload_sweep.json`, so the cost of the stale value is
   visible rather than hidden.
4. Runs both benchmark binaries over the same corpus, in the same order the
   published run uses: C++ first, then Python, no BLAS thread pinning, so a row
   here is comparable with the headline row.
5. Runs `harness/compare_outputs.py` as a gate. A configuration where the two
   implementations disagree is not a data point, it is a bug, and the sweep
   stops rather than publishing the row.

## What it deliberately does not do

It does not lower `benchmark.measured_frames` silently. The default here is 500
measured frames per implementation per configuration rather than the 2000 of
the published run, because the sweep runs a benchmark pair and a block
calibration for every configuration rather than one pair in total, and the
count is written into `workload_sweep.json`, into the CSV and into the plot
caption. At 500 samples over a 100-frame store the p99 is a specific order
statistic of a specific size, so the metadata records which sample it is:
quoting a p99 without saying what it was counted from is how a tail number
becomes decoration.

It also does not touch either implementation, `assets/pipeline_config.json`, or
the committed deviate table. Everything it writes goes to a work directory that
is temporary by default, and the published outputs are the three
`results/workload_sweep.*` files plus the section it maintains inside
`results/RESULTS.md`.

Usage:
  workload_sweep.py --dataset data/table_640x480 --frames 500
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'assets' / 'pipeline_config.json'
DEFAULT_TABLE = REPO_ROOT / 'assets' / 'ransac_uniform.bin'
DEFAULT_CHAIN = REPO_ROOT / 'assets' / 'franka' / 'panda_chain.json'

STAGE_KEYS = ('decode', 'deproject', 'transform_crop', 'plane',
              'cluster', 'grasp', 'ik', 'traj')
NS_PER_MS = 1e6
NS_PER_US = 1e3

# ALGORITHM.md S3 draws one sample triple per RANSAC iteration, so the table a
# configuration needs is three deviates per iteration and no more. Both
# implementations check the length against the config rather than trusting it.
DEVIATES_PER_ITERATION = 3
DOUBLE_BYTES = 8

# Markers around the section this script maintains inside results/RESULTS.md.
# analyze.py rewrites that file from scratch, so the section has to be
# re-insertable rather than hand-typed: a number in prose that no run produced
# is exactly what this repository refuses to publish.
SECTION_BEGIN = '<!-- workload-sweep:begin -->'
SECTION_END = '<!-- workload-sweep:end -->'


def die(message: str) -> None:
    print(f'workload_sweep: {message}', file=sys.stderr)
    raise SystemExit(1)


def run(command: list[str], quiet: bool) -> None:
    """Run a child process and stop the sweep if it fails.

    Output is passed through rather than captured: a benchmark that dies
    halfway needs its own error message visible, not a traceback here.
    """
    if not quiet:
        print('    $ ' + ' '.join(command), file=sys.stderr)
    result = subprocess.run(command, cwd=REPO_ROOT,
                            stdout=subprocess.DEVNULL if quiet else None)
    if result.returncode != 0:
        die(f'{command[0]} exited {result.returncode}')


# ------------------------------------------------------- the configurations

class Variant:
    """One point on the curve: a stride, an iteration count, and its files."""

    def __init__(self, stride: int, iterations: int, baseline: bool):
        self.stride = stride
        self.iterations = iterations
        self.baseline = baseline
        self.key = f'stride{stride}_iter{iterations}'
        self.label = f'stride {stride}, {iterations} iters'
        self.config_path: Path | None = None
        self.table_path: Path | None = None

    def materialise(self, base_config: dict, table: bytes, work_dir: Path,
                    block_points: int | None = None,
                    suffix: str = '') -> Path:
        config = copy.deepcopy(base_config)
        config['deproject']['stride'] = self.stride
        config['plane']['iterations'] = self.iterations
        if block_points is not None:
            config['implementation']['ransac_block_points'] = block_points
        needed = self.iterations * DEVIATES_PER_ITERATION
        if needed * DOUBLE_BYTES > len(table):
            die(f'{self.label} needs {needed} deviates and '
                f'{DEFAULT_TABLE.name} holds {len(table) // DOUBLE_BYTES}')
        config['ransac_table']['count'] = needed
        # Written for the record: both runners are handed the table path on the
        # command line, so this field is documentation of what was used.
        config['ransac_table']['path'] = f'{self.key}.ransac.bin'
        config['ransac_table']['count_note'] = (
            'A prefix of the committed table, so this configuration scores the '
            'same candidate triples the full-length one scored first.')
        path = work_dir / f'{self.key}{suffix}.config.json'
        path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        self.table_path = work_dir / f'{self.key}.ransac.bin'
        self.table_path.write_bytes(table[:needed * DOUBLE_BYTES])
        if not suffix:
            self.config_path = path
        return path


def plan(base_config: dict, strides: list[int],
         iterations: list[int]) -> list[Variant]:
    """One axis at a time, crossing only at the published configuration.

    A full grid would be sixteen benchmark pairs and would answer a question
    nobody asked: the two knobs shrink the same stage, so their cross terms are
    not where the information is. Sweeping each against the published values
    keeps every row one edit away from the run in RESULTS.md.
    """
    base_stride = base_config['deproject']['stride']
    base_iterations = base_config['plane']['iterations']
    ordered = [Variant(base_stride, base_iterations, baseline=True)]
    seen = {(base_stride, base_iterations)}
    for stride in strides:
        if (stride, base_iterations) not in seen:
            seen.add((stride, base_iterations))
            ordered.append(Variant(stride, base_iterations, baseline=False))
    for count in iterations:
        if (base_stride, count) not in seen:
            seen.add((base_stride, count))
            ordered.append(Variant(base_stride, count, baseline=False))
    return ordered


# ------------------------------------------------------------- measurement

def load_timing(path: Path) -> dict:
    """Read one *.timing.jsonl into arrays, per docs/FORMATS.md."""
    records = []
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                die(f'{path}:{number}: {error}')
    if not records:
        die(f'{path} holds no records')
    records.sort(key=lambda r: r['seq'])
    out = {
        'total': np.array([r['total_ns'] for r in records], dtype=np.int64),
        'frame': np.array([r['frame'] for r in records], dtype=np.int64),
        'points': np.array([r['points'] for r in records], dtype=np.int64),
        'cluster_points': np.array([r['cluster_points'] for r in records],
                                   dtype=np.int64),
        'timer_overhead_ns': records[0].get('timer_overhead_ns'),
        'dataset': records[0]['dataset'],
        'impl': records[0]['impl'],
    }
    out['stages'] = {key: np.array([r['stage_ns'][key] for r in records],
                                   dtype=np.int64) for key in STAGE_KEYS}
    out['bail_outs'] = {
        'plane_not_found': int(sum(1 for r in records if not r['plane_found'])),
        'not_graspable': int(sum(1 for r in records if not r['graspable'])),
        'ik_not_converged': int(sum(1 for r in records
                                    if r['graspable'] and not r['converged'])),
    }
    return out


def stats(samples: np.ndarray, wanted, method: str) -> dict:
    out = {f'p{p:g}_ns': float(np.percentile(samples, p, method=method))
           for p in wanted}
    out['max_ns'] = float(samples.max())
    out['mean_ns'] = float(samples.mean())
    out['n'] = int(samples.size)
    return out


def rank_from_slowest(samples: np.ndarray, percentile: float,
                      method: str) -> int:
    """How many samples sit at or above the quoted percentile.

    A p99 taken from 500 samples with the `nearest` method is one specific
    observed sample, and the honest way to describe it is by its rank in the
    run rather than by the word "p99". This is that rank.
    """
    value = np.percentile(samples, percentile, method=method)
    return int(np.count_nonzero(samples >= value))


def summarise(timing: dict, percentiles, method: str) -> dict:
    entry = {
        'impl': timing['impl'],
        'total_ns': stats(timing['total'], percentiles, method),
        'stage_ns': {key: stats(values, percentiles, method)
                     for key, values in timing['stages'].items()},
        'distinct_frames': int(np.unique(timing['frame']).size),
        'median_points': int(np.median(timing['points'])),
        'median_cluster_points': int(np.median(timing['cluster_points'])),
        'bail_outs': timing['bail_outs'],
        'timer_overhead_ns': timing['timer_overhead_ns'],
    }
    entry['p99_rank_from_slowest'] = rank_from_slowest(timing['total'], 99.0,
                                                       method)
    return entry


def block_candidates(variant: Variant, pixels: int, committed: int,
                     cap_elements: int) -> list[int]:
    """Block sizes worth measuring for this configuration.

    Bounded above twice: by the sampled pixel count, since a block larger than
    the cloud is the same run as one block over the whole cloud, and by a cap
    on `block * iterations` so the distance matrix cannot grow into something
    no sensible implementation would allocate. The committed value is always in
    the list, because the interesting comparison is against it.
    """
    sampled = (pixels + variant.stride ** 2 - 1) // variant.stride ** 2
    limit = max(1, cap_elements // variant.iterations)
    values = {committed}
    for value in (512, 2048, 8192, 32768, 131072, 524288):
        if value <= sampled and value <= limit:
            values.add(value)
    if sampled <= limit:
        values.add(sampled)          # the whole cloud in one block
    return sorted(values)


def calibrate_block(variant: Variant, args, base_config: dict,
                    work_dir: Path, pixels: int) -> dict:
    """Pick the Python plane stage's block size the way the committed one was.

    Only the Python side reads this constant, and the config records that it
    changes memory traffic and never output, so choosing it per configuration
    is tuning and not cheating: the equivalence gate still runs afterwards on
    whatever was chosen. The alternative, holding 512 across the sweep, would
    hand Python a loop that runs 415 times over a sixteenth of the arithmetic
    at 8 iterations and then report the overhead as a language cost.
    """
    committed = base_config['implementation']['ransac_block_points']
    candidates = block_candidates(variant, pixels, committed,
                                  args.block_cap_elements)
    measured = []
    for block in candidates:
        config_path = variant.materialise(
            base_config, Path(args.ransac_table).read_bytes(), work_dir,
            block_points=block, suffix=f'.block{block}')
        timing = work_dir / f'{variant.key}.block{block}.timing.jsonl'
        output = work_dir / f'{variant.key}.block{block}.output.jsonl'
        run([sys.executable, 'python/bench/bench_pipeline.py',
             '--dataset', str(args.dataset),
             '--config', str(config_path),
             '--chain', str(args.chain),
             '--ransac-table', str(variant.table_path),
             '--out-timing', str(timing),
             '--out-output', str(output),
             '--warmup', str(args.calibration_warmup),
             '--frames', str(args.calibration_frames),
             '--quiet'], quiet=True)
        loaded = load_timing(timing)
        method = base_config['analysis']['percentile_method']
        measured.append({
            'block_points': block,
            'plane_p50_ns': float(np.percentile(loaded['stages']['plane'],
                                                50.0, method=method)),
            'total_p50_ns': float(np.percentile(loaded['total'], 50.0,
                                                method=method)),
        })
        print(f"    block {block:>7}: plane p50 "
              f"{measured[-1]['plane_p50_ns'] / NS_PER_MS:8.3f} ms",
              file=sys.stderr)
    best = min(measured, key=lambda entry: entry['plane_p50_ns'])
    at_committed = next(entry for entry in measured
                        if entry['block_points'] == committed)
    return {
        'committed_block_points': committed,
        'chosen_block_points': best['block_points'],
        'frames': args.calibration_frames,
        'warmup': args.calibration_warmup,
        'plane_p50_ns_at_committed': at_committed['plane_p50_ns'],
        'plane_p50_ns_at_chosen': best['plane_p50_ns'],
        'candidates': measured,
    }


def gate(cpp_output: Path, py_output: Path, report: Path,
         quiet: bool) -> dict:
    """The equivalence gate, run as the repository's own gate and not a copy.

    Calling `compare_outputs.py` as a subprocess rather than importing pieces
    of it keeps one implementation of the comparison: a second one that drifted
    would let a divergence through while reporting agreement, which is worse
    than having no gate at all.
    """
    command = [sys.executable, 'harness/compare_outputs.py',
               str(cpp_output), str(py_output), '--json', str(report)]
    if quiet:
        command.append('--quiet')
    result = subprocess.run(command, cwd=REPO_ROOT)
    payload = json.loads(report.read_text(encoding='utf-8')) \
        if report.is_file() else {}
    if result.returncode != 0 or not payload.get('agree', False):
        die('the two implementations disagree on this configuration; '
            f'see {report}. A configuration where they disagree is a bug, '
            'not a data point, so no row is published for it')
    worst = 0.0
    worst_field = None
    for name, field in payload['fields'].items():
        deviation = field.get('worst_abs_deviation')
        if deviation is not None and deviation > worst:
            worst = deviation
            worst_field = name
    return {
        'frames_compared': payload['frames_compared'],
        'tolerance': payload['tolerance'],
        'worst_abs_deviation': worst,
        'worst_field': worst_field,
        'bail_points': payload['bail_points'],
        'report': str(report.relative_to(REPO_ROOT))
        if report.is_relative_to(REPO_ROOT) else str(report),
    }


def measure(variant: Variant, args, base_config: dict, work_dir: Path,
            calibration: dict) -> dict:
    """Benchmark both implementations under one configuration and gate them."""
    percentiles = base_config['analysis']['percentiles']
    method = base_config['analysis']['percentile_method']

    outputs = {}
    for impl, command in (
        ('cpp', [str(Path(args.build_dir) / 'bench_pipeline')]),
        ('py', [sys.executable, 'python/bench/bench_pipeline.py']),
    ):
        timing = work_dir / f'{variant.key}.{impl}.timing.jsonl'
        output = work_dir / f'{variant.key}.{impl}.output.jsonl'
        run(command + [
            '--dataset', str(args.dataset),
            '--config', str(variant.config_path),
            '--chain', str(args.chain),
            '--ransac-table', str(variant.table_path),
            '--out-timing', str(timing),
            '--out-output', str(output),
            '--warmup', str(args.warmup),
            '--frames', str(args.frames),
        ] + (['--quiet'] if args.quiet and impl == 'py' else []), args.quiet)
        outputs[impl] = (timing, output)

    equivalence = gate(outputs['cpp'][1], outputs['py'][1],
                       work_dir / f'{variant.key}.equivalence.json', args.quiet)

    row = {
        'key': variant.key,
        'label': variant.label,
        'baseline': variant.baseline,
        'stride': variant.stride,
        'plane_iterations': variant.iterations,
        'ransac_deviates': variant.iterations * DEVIATES_PER_ITERATION,
        'block_calibration': calibration,
        'equivalence': equivalence,
    }
    for impl in ('cpp', 'py'):
        row[impl] = summarise(load_timing(outputs[impl][0]), percentiles,
                              method)

    cpp_total = row['cpp']['total_ns']
    py_total = row['py']['total_ns']
    row['ratio_p50'] = py_total['p50_ns'] / cpp_total['p50_ns']
    row['ratio_p99'] = py_total['p99_ns'] / cpp_total['p99_ns']
    # The share is taken at p50 for the same reason RESULTS.md takes it there:
    # per-stage p99s belong to different frames and do not sum to the frame's
    # own p99, so a share built from them would not be a share of anything.
    row['plane_share_cpp'] = (row['cpp']['stage_ns']['plane']['p50_ns']
                              / cpp_total['p50_ns'])
    row['plane_share_py'] = (row['py']['stage_ns']['plane']['p50_ns']
                             / py_total['p50_ns'])
    # The stages that run in the interpreter over 6x7 and 3x3 matrices, where
    # Python pays per call rather than per element. Their share is the other
    # half of the mechanism the curve is about.
    small = ('grasp', 'ik', 'traj')
    row['small_matrix_share_cpp'] = sum(
        row['cpp']['stage_ns'][key]['p50_ns'] for key in small) / cpp_total['p50_ns']
    row['small_matrix_share_py'] = sum(
        row['py']['stage_ns'][key]['p50_ns'] for key in small) / py_total['p50_ns']
    return row


# ----------------------------------------------------------------- outputs

def published_reference(out_dir: Path, dataset: str) -> dict | None:
    """The headline run's own numbers for this corpus, if they are on disk.

    The baseline configuration in this sweep is the published one at a quarter
    of the frame count, so it is a reproduction and should be reported as one:
    if 500 frames do not land near the 2000-frame run, the shorter runs in
    every other row are worth less. Missing or unreadable `summary.json` is not
    an error, it just means the check cannot be made this time.
    """
    path = out_dir / 'summary.json'
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding='utf-8'))
        picked = {}
        for entry in summary['groups']:
            if entry['dataset'] == dataset and entry['transport'] == 'inproc' \
                    and entry['impl'] in ('cpp', 'py'):
                picked[entry['impl']] = entry['total_ns']
        if {'cpp', 'py'} - set(picked):
            return None
        return {
            'source': str(path.relative_to(REPO_ROOT))
            if path.is_relative_to(REPO_ROOT) else str(path),
            'frames': picked['cpp']['n'],
            'cpp_p50_ns': picked['cpp']['p50'],
            'cpp_p99_ns': picked['cpp']['p99'],
            'py_p50_ns': picked['py']['p50'],
            'py_p99_ns': picked['py']['p99'],
            'ratio_p50': picked['py']['p50'] / picked['cpp']['p50'],
            'ratio_p99': picked['py']['p99'] / picked['cpp']['p99'],
        }
    except (json.JSONDecodeError, KeyError, ZeroDivisionError, OSError):
        return None


def git_revision() -> str | None:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()[:12] or None


def observed_host() -> dict:
    host = {
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'python': platform.python_version(),
        'numpy': np.__version__,
    }
    try:
        host['load_average'] = list(os.getloadavg())
    except OSError:
        host['load_average'] = None
    host['thread_env'] = {
        name: os.environ.get(name)
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                     'MKL_NUM_THREADS')}
    return host


CSV_COLUMNS = (
    'key', 'stride', 'plane_iterations', 'baseline', 'frames', 'distinct_frames',
    'median_points', 'cpp_p50_ms', 'cpp_p99_ms', 'py_p50_ms', 'py_p99_ms',
    'ratio_p50', 'ratio_p99', 'plane_share_cpp', 'plane_share_py',
    'small_matrix_share_cpp', 'small_matrix_share_py',
    'cpp_plane_p50_ms', 'py_plane_p50_ms', 'cpp_ik_p50_us', 'py_ik_p50_us',
    'py_block_points', 'py_block_points_committed',
    'py_plane_p50_ms_at_committed_block',
    'bail_plane', 'bail_grasp', 'bail_ik', 'gate_worst_abs_deviation',
    'gate_frames_compared',
)


def format_load(load) -> str:
    if not load:
        return 'unavailable'
    return ', '.join(f'{value:.2f}' for value in load)


def ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
    return f'{number}{suffix}'


def maybe_ms(value) -> str:
    return '' if value is None else f'{value / NS_PER_MS:.4f}'


def csv_row(row: dict) -> dict:
    return {
        'key': row['key'],
        'stride': row['stride'],
        'plane_iterations': row['plane_iterations'],
        'baseline': int(row['baseline']),
        'frames': row['cpp']['total_ns']['n'],
        'distinct_frames': row['cpp']['distinct_frames'],
        'median_points': row['cpp']['median_points'],
        'cpp_p50_ms': f"{row['cpp']['total_ns']['p50_ns'] / NS_PER_MS:.4f}",
        'cpp_p99_ms': f"{row['cpp']['total_ns']['p99_ns'] / NS_PER_MS:.4f}",
        'py_p50_ms': f"{row['py']['total_ns']['p50_ns'] / NS_PER_MS:.4f}",
        'py_p99_ms': f"{row['py']['total_ns']['p99_ns'] / NS_PER_MS:.4f}",
        'ratio_p50': f"{row['ratio_p50']:.4f}",
        'ratio_p99': f"{row['ratio_p99']:.4f}",
        'plane_share_cpp': f"{row['plane_share_cpp']:.6f}",
        'plane_share_py': f"{row['plane_share_py']:.6f}",
        'small_matrix_share_cpp': f"{row['small_matrix_share_cpp']:.6f}",
        'small_matrix_share_py': f"{row['small_matrix_share_py']:.6f}",
        'cpp_plane_p50_ms':
            f"{row['cpp']['stage_ns']['plane']['p50_ns'] / NS_PER_MS:.4f}",
        'py_plane_p50_ms':
            f"{row['py']['stage_ns']['plane']['p50_ns'] / NS_PER_MS:.4f}",
        'cpp_ik_p50_us':
            f"{row['cpp']['stage_ns']['ik']['p50_ns'] / NS_PER_US:.2f}",
        'py_ik_p50_us':
            f"{row['py']['stage_ns']['ik']['p50_ns'] / NS_PER_US:.2f}",
        'py_block_points': row['block_calibration']['chosen_block_points'],
        'py_block_points_committed':
            row['block_calibration']['committed_block_points'],
        'py_plane_p50_ms_at_committed_block':
            maybe_ms(row['block_calibration'].get('plane_p50_ns_at_committed')),
        'bail_plane': row['cpp']['bail_outs']['plane_not_found'],
        'bail_grasp': row['cpp']['bail_outs']['not_graspable'],
        'bail_ik': row['cpp']['bail_outs']['ik_not_converged'],
        'gate_worst_abs_deviation':
            f"{row['equivalence']['worst_abs_deviation']:.3e}",
        'gate_frames_compared': row['equivalence']['frames_compared'],
    }


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_row(row))


def plot(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = summary['configurations']
    order = sorted(rows, key=lambda r: r['plane_share_cpp'])
    baseline = next(r for r in rows if r['baseline'])

    # Two families, drawn as two lines through the shared published point,
    # because they move the shape differently: striding shrinks the whole
    # cloud, so the C++ frame keeps roughly its proportions while Python's
    # per-call overheads do not shrink with it, and cutting iterations attacks
    # the plane stage alone. One line through all six points would hide that.
    families = [
        ('deproject.stride', '#1f4e79',
         [r for r in rows if r['plane_iterations'] == baseline['plane_iterations']],
         lambda r: f"stride {r['stride']}"),
        ('plane.iterations', '#7a3d9c',
         [r for r in rows if r['stride'] == baseline['stride']],
         lambda r: f"{r['plane_iterations']} iters"),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    axis = axes[0]
    for name, colour, members, tag in families:
        members = sorted(members, key=lambda r: r['plane_share_cpp'])
        shares = [100.0 * r['plane_share_cpp'] for r in members]
        axis.plot(shares, [r['ratio_p50'] for r in members], '-o',
                  color=colour, linewidth=1.7, label=f'{name}, p50')
        axis.plot(shares, [r['ratio_p99'] for r in members], '--s',
                  color=colour, linewidth=1.1, alpha=0.55,
                  markerfacecolor='none', label=f'{name}, p99')
        # Families are annotated on opposite sides: the stride points crowd
        # together near the published share, and one label on top of another
        # is a plot that has to be read from the CSV instead.
        below = name.startswith('deproject')
        for row, x in zip(members, shares):
            axis.annotate(tag(row), (x, row['ratio_p50']),
                          textcoords='offset points',
                          xytext=(7, -13 if below else 7),
                          fontsize=8, color=colour)
    axis.plot([100.0 * baseline['plane_share_cpp']], [baseline['ratio_p50']],
              marker='*', markersize=15, linestyle='none', color='#c0504d',
              label='published configuration')
    axis.axhline(1.0, color='0.6', linewidth=0.8)
    axis.set_xlabel('plane stage as a share of the C++ p50 frame, per cent')
    axis.set_ylabel('Python latency divided by C++ latency')
    axis.set_title('The ratio against how much of the frame\n'
                   'is one large array reduction')
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc='upper right')

    # The second panel is the mechanism behind the first: what each frame is
    # made of, C++ beside Python at every configuration. The C++ bar barely
    # changes shape while the Python bar's plane block collapses and the
    # per-call stages take its place, which is the whole of the effect.
    axis = axes[1]
    centres = np.arange(len(order), dtype=np.float64)
    width = 0.36
    palette = plt.get_cmap('tab20')
    for offset, impl in ((-0.5 * width - 0.02, 'cpp'), (0.5 * width + 0.02, 'py')):
        bottom = np.zeros(len(order))
        for index, key in enumerate(STAGE_KEYS):
            values = np.array([100.0 * r[impl]['stage_ns'][key]['p50_ns']
                               / r[impl]['total_ns']['p50_ns'] for r in order])
            axis.bar(centres + offset, values, width=width, bottom=bottom,
                     label=key if impl == 'cpp' else None,
                     color=palette(index / len(STAGE_KEYS)),
                     edgecolor='white', linewidth=0.4)
            bottom += values
        for centre in centres:
            axis.text(centre + offset, 101.0, impl, ha='center', fontsize=7,
                      color='0.3')
    axis.set_xticks(centres)
    axis.set_xticklabels([r['label'].replace(', ', '\n') for r in order],
                         fontsize=8)
    axis.set_ylabel('share of that implementation\'s p50 frame, per cent')
    axis.set_ylim(0, 106)
    axis.set_title('What the frame is made of, C++ beside Python')
    axis.legend(fontsize=7, ncol=4, loc='upper center',
                bbox_to_anchor=(0.5, -0.09), frameon=False)
    axis.grid(axis='y', alpha=0.25)

    meta = summary['metadata']
    figure.suptitle(
        f"Workload mix against the language ratio, {meta['dataset']} "
        f"({meta['width']}x{meta['height']}), "
        f"{meta['measured_frames']} measured frames per implementation "
        f"per configuration", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ------------------------------------------------- the section in RESULTS.md

def markdown_section(summary: dict) -> str:
    meta = summary['metadata']
    rows = sorted(summary['configurations'],
                  key=lambda r: -r['plane_share_cpp'])
    baseline = next(r for r in summary['configurations'] if r['baseline'])
    extreme = min(summary['configurations'], key=lambda r: r['plane_share_cpp'])

    lines = [
        SECTION_BEGIN,
        '',
        '## The ratio against the workload mix',
        '',
        f"Produced by `harness/workload_sweep.py` on `{meta['dataset']}` "
        f"({meta['width']}x{meta['height']}), "
        f"{meta['measured_frames']} measured frames per implementation per "
        f"configuration after {meta['warmup_frames']} warm-up frames, against "
        f"the {meta['reference_frames']} of the headline run. Every "
        'configuration passed the equivalence gate before its row was written; '
        'the worst deviation column is what the gate observed.',
        '',
    ]

    reference = meta.get('published_reference')
    if reference:
        lines += [
            f"The first row is the published configuration re-run at "
            f"{meta['measured_frames']} frames, so it is also a check on the "
            f"shorter runs the other rows use: the "
            f"{reference['frames']}-frame run in the table above gives "
            f"{reference['cpp_p50_ns'] / NS_PER_MS:.3f} ms and "
            f"{reference['py_p50_ns'] / NS_PER_MS:.3f} ms at p50 for "
            f"{reference['ratio_p50']:.2f}x, and this sweep gives "
            f"{baseline['cpp']['total_ns']['p50_ns'] / NS_PER_MS:.3f} ms and "
            f"{baseline['py']['total_ns']['p50_ns'] / NS_PER_MS:.3f} ms for "
            f"{baseline['ratio_p50']:.2f}x.",
            '',
        ]

    lines += [
        'The headline 1.50x is one point on a curve, and this is the curve. '
        'Two constants set how much of the frame is bulk array work: '
        '`deproject.stride`, which divides the point count by its square, and '
        '`plane.iterations`, which divides the candidate count. Shrinking '
        'either shrinks the RANSAC plane stage, which is the stage where '
        'NumPy and Eigen reach comparable compiled loops, and leaves the '
        'small-matrix stages where Python pays per call. Nothing else about '
        'the pipeline changes: both implementations read the same derived '
        'config file, and the deviate table each configuration uses is a '
        'prefix of the committed one, so a 32-iteration run scores the first '
        '32 triples the 128-iteration run scored.',
        '',
    ]

    headings = ['stride', 'iters', 'median points', 'plane share of cpp frame',
                'cpp p50', 'py p50', 'ratio p50', 'cpp p99', 'py p99',
                'ratio p99', 'py block', 'bail plane/grasp/ik', 'gate worst']
    align = [':---'] + ['---:'] * (len(headings) - 1)
    lines.append('| ' + ' | '.join(headings) + ' |')
    lines.append('| ' + ' | '.join(align) + ' |')
    for row in rows:
        bail = row['cpp']['bail_outs']
        lines.append('| ' + ' | '.join([
            str(row['stride']) + (' (published)' if row['baseline'] else ''),
            str(row['plane_iterations']),
            f"{row['cpp']['median_points']:,}",
            f"{100.0 * row['plane_share_cpp']:.1f}%",
            f"{row['cpp']['total_ns']['p50_ns'] / NS_PER_MS:.3f}",
            f"{row['py']['total_ns']['p50_ns'] / NS_PER_MS:.3f}",
            f"{row['ratio_p50']:.2f}x",
            f"{row['cpp']['total_ns']['p99_ns'] / NS_PER_MS:.3f}",
            f"{row['py']['total_ns']['p99_ns'] / NS_PER_MS:.3f}",
            f"{row['ratio_p99']:.2f}x",
            f"{row['block_calibration']['chosen_block_points']:,}",
            f"{bail['plane_not_found']}/{bail['not_graspable']}/"
            f"{bail['ik_not_converged']}",
            f"{row['equivalence']['worst_abs_deviation']:.1e}",
        ]) + ' |')

    lines += [
        '',
        'All latencies in milliseconds. The ratio columns are Python divided '
        'by C++ at that percentile, and the plane share is taken at p50 for '
        'the reason the per-stage tables above take it there: the frame that '
        'produced one stage\'s p99 is rarely the frame that produced '
        'another\'s, so p99 shares are not shares of anything.',
        '',
    ]

    committed_block = None
    retuned = []
    for row in rows:
        calibration = row['block_calibration']
        committed_block = calibration['committed_block_points']
        if calibration.get('candidates') and \
                calibration['chosen_block_points'] != committed_block:
            retuned.append(row)

    def block_paragraph() -> list[str]:
        if not retuned:
            return []
        out = [
            '',
            f"**The `py block` column is a knob that had to move.** "
            f"`implementation.ransac_block_points` is read only by the Python "
            f"plane stage, it changes memory traffic and never output, and its "
            f"committed value of {committed_block:,} was chosen when the "
            f"distance block was {committed_block:,} x 128 float64. Shrink the "
            f"candidate count and that block stops filling the cache while the "
            f"Python loop still runs once per {committed_block:,} points, so "
            f"the constant becomes a tax on a configuration it was not chosen "
            f"for. Each row here re-chooses it by measurement over "
            f"{retuned[0]['block_calibration']['frames']} frames per "
            f"candidate, exactly the way the committed value was chosen, and "
            f"the full curve is in `workload_sweep.json`. What holding it "
            f"fixed would have cost:",
            '',
        ]
        for entry in retuned:
            calibration = entry['block_calibration']
            out.append(
                f"* {entry['label']}: the Python plane stage runs at p50 "
                f"{calibration['plane_p50_ns_at_committed'] / NS_PER_MS:.3f} ms "
                f"with the committed {committed_block:,}-point block and "
                f"{calibration['plane_p50_ns_at_chosen'] / NS_PER_MS:.3f} ms "
                f"with the {calibration['chosen_block_points']:,}-point block "
                f"this row uses. Reporting the first would have charged Python "
                f"for a stale constant and called it a language cost.")
        return out

    lines += [
        f"**The ratio moves from {baseline['ratio_p50']:.2f}x at p50 when the "
        f"plane stage is {100.0 * baseline['plane_share_cpp']:.1f}% of the C++ "
        f"frame to {extreme['ratio_p50']:.2f}x when it is "
        f"{100.0 * extreme['plane_share_cpp']:.1f}%.** "
        'The published pipeline sits at the flattering end of that range by '
        'construction, and a reader whose own pipeline subsamples its cloud '
        'or runs a shorter RANSAC should read the row that matches their '
        'shape rather than the headline.',
        '',
        'What is worth knowing about the individual rows:',
        '',
    ]

    for row in rows:
        bail = row['cpp']['bail_outs']
        note = (f"* **{row['label']}**: plane is "
                f"{100.0 * row['plane_share_cpp']:.1f}% of the C++ frame and "
                f"{100.0 * row['plane_share_py']:.1f}% of the Python one; the "
                f"grasp, IK and trajectory stages together are "
                f"{100.0 * row['small_matrix_share_py']:.1f}% of the Python "
                f"frame against "
                f"{100.0 * row['small_matrix_share_cpp']:.1f}% of the C++ one. "
                f"Ratio {row['ratio_p50']:.2f}x at p50, "
                f"{row['ratio_p99']:.2f}x at p99.")
        if bail['not_graspable'] or bail['plane_not_found']:
            note += (f" This row is not the same pipeline as the others: "
                     f"{bail['plane_not_found']} of "
                     f"{row['cpp']['total_ns']['n']} measured frames found no "
                     f"plane and {bail['not_graspable']} found no cluster "
                     f"holding `cluster.min_points`, so they stop at S4 and "
                     f"never reach grasp synthesis, IK or the trajectory. "
                     f"Subsampling that hard removes the object as well as the "
                     f"cost, which is itself the answer to whether a stride "
                     f"like this is free.")
        lines.append(note)

    lines += block_paragraph()

    lines += [
        '',
        f"The p99 columns come from {meta['measured_frames']} samples over "
        f"{baseline['cpp']['distinct_frames']} distinct store frames, so the "
        f"value quoted is the "
        f"{ordinal(baseline['cpp']['p99_rank_from_slowest'])} slowest sample "
        'of the run and carries the repeat structure described in threat T8. '
        'The p50 '
        'columns are the ones to lean on; the p99 columns are here because a '
        'ratio that moves at the median and not in the tail would be a '
        'different finding from one that moves in both.',
        '',
        f"The box is shared and unpinned, so the load average is recorded at "
        f"both ends of the sweep: "
        f"{format_load(meta.get('load_average_at_start'))} at the start and "
        f"{format_load(meta['host'].get('load_average'))} at the end. Threat "
        f"T2 is why that matters and what it does not excuse.",
        '',
        f"Files: `workload_sweep.json`, `workload_sweep.csv`, "
        f"`workload_sweep.png`. Method and the threat this measures: "
        f"[../docs/METHOD.md](../docs/METHOD.md) T1.",
        '',
        SECTION_END,
    ]
    return '\n'.join(lines) + '\n'


def insert_section(results_md: Path, section: str) -> str:
    """Put the section into RESULTS.md, replacing any earlier copy.

    `harness/analyze.py` rewrites RESULTS.md from scratch, so this section
    cannot be a one-time hand edit: rerunning the analysis would delete it and
    the file would then be missing a result that was measured. Replacing a
    marked block, or inserting before the `## Files` heading when the markers
    are gone, makes the section a function of the last sweep rather than of
    whoever last typed it.
    """
    text = results_md.read_text(encoding='utf-8')
    begin = text.find(SECTION_BEGIN)
    end = text.find(SECTION_END)
    if begin != -1 and end != -1:
        return text[:begin] + section + text[end + len(SECTION_END):].lstrip('\n')
    anchor = text.find('\n## Files\n')
    if anchor != -1:
        return text[:anchor + 1] + section + '\n' + text[anchor + 1:]
    return text.rstrip('\n') + '\n\n' + section


# -------------------------------------------------------------------- main

def main(argv=None) -> int:
    base_config = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    analysis = base_config['analysis']
    benchmark = base_config['benchmark']

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path,
                        default=REPO_ROOT / 'data' / analysis['reference_dataset'],
                        help='frame store directory, holding manifest.json')
    parser.add_argument('--build-dir', type=Path,
                        default=REPO_ROOT / 'cpp' / 'build')
    parser.add_argument('--out-dir', type=Path, default=REPO_ROOT / 'results')
    parser.add_argument('--work-dir', type=Path, default=None,
                        help='where the per-configuration configs, tables and '
                             'JSONL land; a temporary directory by default, '
                             'because they are inputs to this summary rather '
                             'than results in their own right')
    parser.add_argument('--chain', type=Path, default=DEFAULT_CHAIN)
    parser.add_argument('--ransac-table', type=Path, default=DEFAULT_TABLE)
    parser.add_argument('--frames', type=int, default=500,
                        help='measured frames per implementation per '
                             'configuration; below the '
                             f"{benchmark['measured_frames']} of the published "
                             'run because this runs many benchmarks, and '
                             'recorded in every output')
    parser.add_argument('--warmup', type=int,
                        default=benchmark['warmup_frames'])
    parser.add_argument('--strides', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--iterations', type=int, nargs='+',
                        default=[128, 32, 8])
    parser.add_argument('--block-points', default='auto',
                        help="Python plane-stage block size: 'auto' re-chooses "
                             'it per configuration by measurement, an integer '
                             'holds it fixed. Only the Python side reads it '
                             'and it never changes output.')
    parser.add_argument('--calibration-frames', type=int, default=60,
                        help='measured frames per block-size candidate')
    parser.add_argument('--calibration-warmup', type=int, default=40)
    parser.add_argument('--block-cap-elements', type=int, default=4_194_304,
                        help='upper bound on block_points * iterations, so a '
                             'candidate cannot allocate a distance matrix no '
                             'sensible implementation would')
    parser.add_argument('--results-md', type=Path,
                        default=REPO_ROOT / 'results' / 'RESULTS.md',
                        help='RESULTS.md to insert the section into; pass an '
                             'empty string to skip')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    bench = Path(args.build_dir) / 'bench_pipeline'
    if not bench.is_file():
        die(f'{bench} is not built; cmake --build {args.build_dir} first')
    manifest_path = Path(args.dataset) / 'manifest.json'
    if not manifest_path.is_file():
        die(f'{manifest_path} is missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.work_dir is None:
        import tempfile
        work_holder = tempfile.TemporaryDirectory(prefix='workload_sweep.')
        work_dir = Path(work_holder.name)
    else:
        work_holder = None
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    table = Path(args.ransac_table).read_bytes()
    variants = plan(base_config, args.strides, args.iterations)
    pixels = manifest['width'] * manifest['height']

    # Recorded at both ends because the box is shared and unpinned (threat T2):
    # a row taken while something else was running should be spottable rather
    # than trusted, and one number at the end cannot show that.
    try:
        load_at_start = list(os.getloadavg())
    except OSError:
        load_at_start = None

    started = time.time()
    rows = []
    for index, variant in enumerate(variants, 1):
        print(f'\n==> [{index}/{len(variants)}] {variant.label}',
              file=sys.stderr)
        if args.block_points == 'auto':
            calibration = calibrate_block(variant, args, base_config,
                                          work_dir, pixels)
            block = calibration['chosen_block_points']
            print(f"    block chosen by measurement: {block} "
                  f"(committed {calibration['committed_block_points']})",
                  file=sys.stderr)
        else:
            block = int(args.block_points)
            calibration = {'committed_block_points':
                           base_config['implementation']['ransac_block_points'],
                           'chosen_block_points': block,
                           'candidates': []}
        variant.materialise(base_config, table, work_dir, block_points=block)
        rows.append(measure(variant, args, base_config, work_dir, calibration))
        last = rows[-1]
        print(f"    cpp p50 {last['cpp']['total_ns']['p50_ns'] / NS_PER_MS:.3f} ms, "
              f"py p50 {last['py']['total_ns']['p50_ns'] / NS_PER_MS:.3f} ms, "
              f"ratio {last['ratio_p50']:.2f}x at p50 and "
              f"{last['ratio_p99']:.2f}x at p99, plane is "
              f"{100.0 * last['plane_share_cpp']:.1f}% of the C++ frame",
              file=sys.stderr)

    summary = {
        'metadata': {
            'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                              time.gmtime()),
            'revision': git_revision(),
            'dataset': manifest['name'],
            'width': manifest['width'],
            'height': manifest['height'],
            'measured_frames': args.frames,
            'warmup_frames': args.warmup,
            'reference_frames': benchmark['measured_frames'],
            'percentile_method': analysis['percentile_method'],
            'percentiles': analysis['percentiles'],
            'equivalence_tolerance': analysis['equivalence_tolerance'],
            'base_config': str(DEFAULT_CONFIG.relative_to(REPO_ROOT)),
            'ransac_table': str(Path(args.ransac_table).relative_to(REPO_ROOT))
            if Path(args.ransac_table).is_relative_to(REPO_ROOT)
            else str(args.ransac_table),
            'ransac_table_note':
                'Each configuration reads the first 3 * plane.iterations '
                'doubles of the committed table, so a shorter run scores the '
                'same candidate triples the full run scored first.',
            'elapsed_s': round(time.time() - started, 1),
            'load_average_at_start': load_at_start,
            'host': observed_host(),
            'published_reference': published_reference(out_dir,
                                                       manifest['name']),
        },
        'configurations': rows,
    }

    json_path = out_dir / 'workload_sweep.json'
    json_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    csv_path = out_dir / 'workload_sweep.csv'
    write_csv(rows, csv_path)
    png_path = out_dir / 'workload_sweep.png'
    plot(summary, png_path)
    written = [json_path, csv_path, png_path]

    if str(args.results_md):
        results_md = Path(args.results_md)
        if results_md.is_file():
            results_md.write_text(
                insert_section(results_md, markdown_section(summary)),
                encoding='utf-8')
            written.append(results_md)
        else:
            print(f'workload_sweep: {results_md} is missing, section not '
                  f'inserted', file=sys.stderr)

    for path in written:
        print(f'wrote {path}', file=sys.stderr)
    if work_holder is not None:
        work_holder.cleanup()
    return 0


if __name__ == '__main__':
    sys.exit(main())
