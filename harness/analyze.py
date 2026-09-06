#!/usr/bin/env python3
"""Turn timing JSONL into the numbers the repository exists to publish.

Four things this does that a `numpy.percentile` one-liner would not:

* It reports the mean beside the percentiles rather than instead of them, so a
  reader can see by how much the mean of a right-skewed latency distribution
  misleads. A control loop misses a deadline on the tail, never on the mean.
* It puts a confidence interval on p99. A p99 quoted from a few thousand
  samples without an interval invites being picked apart, and rightly. Two
  intervals are given: an ordinary bootstrap over samples, and a bootstrap that
  resamples *distinct store frames* and keeps every repeat of each. Samples of
  the same frame are not independent, so the second interval is the honest one
  and it is wider.
* It attributes the Python tail. Every Python record carries the cumulative
  collection counters, so a slow frame either coincides with a collection or it
  does not, and both answers are findings.
* It finds where the pipeline crosses the control budget as resolution grows,
  by interpolating p99 against pixel count rather than by asserting a limit.

Reads any number of `*.timing.jsonl` (docs/FORMATS.md) and writes
`RESULTS.md`, `summary.json`, `summary.csv` and the plots into the output
directory. A ROS run's records carry `"transport":"ros2"` and are grouped
separately without any further special casing.

Usage:
  analyze.py results/*.timing.jsonl --out-dir results
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'assets' / 'pipeline_config.json'
DEFAULT_TOOLCHAIN = REPO_ROOT / 'docs' / 'TOOLCHAIN.md'

STAGE_KEYS = ('decode', 'deproject', 'transform_crop', 'plane',
              'cluster', 'grasp', 'ik', 'traj')
NS_PER_MS = 1e6
NS_PER_S = 1_000_000_000


def die(message: str) -> None:
    print(f'analyze: {message}', file=sys.stderr)
    raise SystemExit(1)


# ---------------------------------------------------------------- loading

class Group:
    """Every measured frame from one (implementation, dataset, transport)."""

    def __init__(self, impl: str, dataset: str, transport: str):
        self.impl = impl
        self.dataset = dataset
        self.transport = transport
        self.total = None
        self.stages = None
        self.frame = None
        self.gc = None
        self.plane_found = None
        self.graspable = None
        self.converged = None
        self.ik_iterations = None
        self.points = None
        self.timer_overhead_ns = None
        self.width = None
        self.height = None
        self.sources: list[str] = []

    @property
    def key(self) -> tuple:
        return (self.impl, self.dataset, self.transport)

    @property
    def label(self) -> str:
        suffix = '' if self.transport == 'inproc' else f' [{self.transport}]'
        return f'{self.impl}{suffix}'

    @property
    def pixels(self) -> int:
        return (self.width or 0) * (self.height or 0)

    @property
    def n(self) -> int:
        return int(self.total.size)


def parse_resolution(name: str) -> tuple[int, int] | None:
    match = re.search(r'(\d+)x(\d+)', name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def resolution_for(dataset: str, data_root: Path) -> tuple[int, int]:
    """Prefer the manifest, fall back to the name.

    The manifest is authoritative and the name is a convention, but a corpus
    can be deleted after a run while its timing file lives on, and refusing to
    analyse a committed result because the 1.2 GB it came from is gone would be
    the wrong trade.
    """
    manifest = data_root / dataset / 'manifest.json'
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        return int(payload['width']), int(payload['height'])
    parsed = parse_resolution(dataset)
    if parsed is None:
        die(f'cannot determine the resolution of dataset {dataset!r}: '
            f'no {manifest} and no WxH in the name')
    return parsed


def load(paths: list[Path], data_root: Path) -> list[Group]:
    buckets: dict[tuple, list[dict]] = {}
    origin: dict[tuple, list[str]] = {}
    for path in paths:
        with path.open(encoding='utf-8') as handle:
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    die(f'{path}:{number}: {error}')
                key = (record['impl'], record['dataset'],
                       record.get('transport', 'inproc'))
                buckets.setdefault(key, []).append(record)
                origin.setdefault(key, [])
                if str(path) not in origin[key]:
                    origin[key].append(str(path))

    groups = []
    for key, records in sorted(buckets.items()):
        records.sort(key=lambda r: r['seq'])
        group = Group(*key)
        group.sources = origin[key]
        count = len(records)
        group.total = np.array([r['total_ns'] for r in records],
                               dtype=np.int64)
        group.stages = np.empty((count, len(STAGE_KEYS)), dtype=np.int64)
        for i, stage in enumerate(STAGE_KEYS):
            group.stages[:, i] = [r['stage_ns'][stage] for r in records]
        group.frame = np.array([r['frame'] for r in records], dtype=np.int64)
        group.plane_found = np.array([r['plane_found'] for r in records])
        group.graspable = np.array([r['graspable'] for r in records])
        group.converged = np.array([r['converged'] for r in records])
        group.ik_iterations = np.array([r['ik_iterations'] for r in records],
                                       dtype=np.int64)
        group.points = np.array([r['points'] for r in records], dtype=np.int64)
        group.timer_overhead_ns = records[0].get('timer_overhead_ns')
        if 'gc' in records[0]:
            group.gc = np.array([[r['gc']['gen0'], r['gc']['gen1'],
                                  r['gc']['gen2']] for r in records],
                                dtype=np.int64)
        group.width, group.height = resolution_for(group.dataset, data_root)
        groups.append(group)
    return groups


# ------------------------------------------------------------- statistics

def stats(samples: np.ndarray, wanted, method: str) -> dict:
    out = {f'p{p:g}': float(np.percentile(samples, p, method=method))
           for p in wanted}
    out['max'] = float(samples.max())
    out['mean'] = float(samples.mean())
    out['n'] = int(samples.size)
    return out


def bootstrap_ci(samples: np.ndarray, percentile: float, resamples: int,
                 confidence: float, seed: int, method: str,
                 blocks: np.ndarray | None = None) -> dict:
    """Percentile bootstrap on one order statistic.

    `blocks` resamples distinct block labels with replacement and keeps every
    sample belonging to a drawn block. That is the right unit here: a measured
    run cycles the frame store many times, so twenty samples of frame 7 tell
    you about frame 7 once, not twenty times, and treating them as twenty
    independent draws narrows the interval by roughly the square root of the
    repeat count.
    """
    rng = np.random.default_rng(seed)
    if blocks is None:
        n = samples.size
        draws = np.empty(resamples, dtype=np.float64)
        # Bounded so the index matrix stays inside a few tens of megabytes.
        chunk = max(1, int(4_000_000 // max(n, 1)))
        done = 0
        while done < resamples:
            take = min(chunk, resamples - done)
            index = rng.integers(0, n, size=(take, n))
            draws[done:done + take] = np.percentile(
                samples[index], percentile, axis=1, method=method)
            done += take
    else:
        labels, inverse = np.unique(blocks, return_inverse=True)
        order = np.argsort(inverse, kind='stable')
        sorted_samples = samples[order]
        counts = np.bincount(inverse, minlength=labels.size)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        draws = np.empty(resamples, dtype=np.float64)
        for i in range(resamples):
            picked = rng.integers(0, labels.size, size=labels.size)
            pieces = [sorted_samples[starts[b]:starts[b] + counts[b]]
                      for b in picked]
            draws[i] = np.percentile(np.concatenate(pieces), percentile,
                                     method=method)

    alpha = 0.5 * (1.0 - confidence)
    low, high = np.percentile(draws, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return {'low': float(low), 'high': float(high),
            'confidence': confidence, 'resamples': resamples,
            'standard_error': float(draws.std(ddof=1))}


def gc_attribution(group: Group, tail_percentile: float, method: str) -> dict:
    """Did the Python tail happen where the collector ran?

    The counters are cumulative and sampled at the end of a frame, so a
    collection during frame k shows up as an increase between k-1 and k. Frame
    0 has nothing to difference against and is excluded from both sides of the
    split rather than guessed at.
    """
    if group.gc is None:
        return {}
    cumulative = group.gc.sum(axis=1)
    total = group.total
    known = slice(1, group.n)
    fired = np.diff(cumulative) > 0
    latency = total[known]
    n_known = latency.size
    if n_known == 0:
        return {}

    threshold = float(np.percentile(total, tail_percentile, method=method))
    in_tail = latency >= threshold
    clean = latency[~fired]
    dirty = latency[fired]

    # Point-biserial correlation is just Pearson with a 0/1 regressor, and it
    # is the right summary of "does a collection predict a slow frame".
    if fired.any() and not fired.all():
        correlation = float(np.corrcoef(fired.astype(np.float64), latency)[0, 1])
    else:
        correlation = None

    per_generation = (group.gc[-1] - group.gc[0]).tolist()
    result = {
        'collections_over_run': {'gen0': per_generation[0],
                                 'gen1': per_generation[1],
                                 'gen2': per_generation[2]},
        'frames_analysed': int(n_known),
        'frames_with_collection': int(fired.sum()),
        'collection_frame_share': float(fired.mean()),
        'tail_percentile': tail_percentile,
        'tail_threshold_ns': threshold,
        'tail_frames': int(in_tail.sum()),
        'tail_frames_with_collection': int((in_tail & fired).sum()),
        'tail_collection_share': float(
            (in_tail & fired).sum() / in_tail.sum()) if in_tail.any() else None,
        'point_biserial_r': correlation,
        f'p{tail_percentile:g}_all_ns': float(
            np.percentile(latency, tail_percentile, method=method)),
        f'p{tail_percentile:g}_without_collection_ns': float(
            np.percentile(clean, tail_percentile, method=method))
        if clean.size else None,
        'median_with_collection_ns': float(np.median(dirty))
        if dirty.size else None,
        'median_without_collection_ns': float(np.median(clean))
        if clean.size else None,
    }

    all_p = result[f'p{tail_percentile:g}_all_ns']
    clean_p = result[f'p{tail_percentile:g}_without_collection_ns']
    if result['frames_with_collection'] == 0:
        result['verdict'] = (
            'the collector never ran during a measured frame, so it explains '
            'none of the tail')
    elif clean_p is None:
        result['verdict'] = 'every measured frame saw a collection'
    else:
        shift = (all_p - clean_p) / all_p if all_p else 0.0
        base = result['collection_frame_share']
        tail_share = result['tail_collection_share'] or 0.0
        enriched = tail_share > 2.0 * base and tail_share > 0.1
        if shift < 0.02 and not enriched:
            result['verdict'] = (
                f'removing collection frames moves p{tail_percentile:g} by '
                f'{100 * shift:.1f}%, and collections are no more common in '
                f'the tail ({100 * tail_share:.1f}%) than overall '
                f'({100 * base:.1f}%): the tail is not the collector')
        else:
            result['verdict'] = (
                f'removing collection frames moves p{tail_percentile:g} by '
                f'{100 * shift:.1f}%, and {100 * tail_share:.1f}% of tail '
                f'frames saw a collection against {100 * base:.1f}% overall')
    return result


def crossover(rows: list[dict], budget_ns: float, metric: str) -> dict:
    """Where p99 against pixel count crosses the control budget.

    Linear in pixels between the two measured resolutions that bracket the
    budget. If nothing brackets it the two nearest points are extended instead
    and the result says so, because an extrapolated crossing is a different
    kind of claim from an interpolated one.
    """
    points = sorted(rows, key=lambda r: r['pixels'])
    if len(points) < 2:
        return {'pixels': None, 'kind': 'not enough resolutions'}
    xs = [float(p['pixels']) for p in points]
    ys = [float(p[metric]) for p in points]

    for i in range(len(points) - 1):
        if (ys[i] - budget_ns) * (ys[i + 1] - budget_ns) <= 0.0:
            span = ys[i + 1] - ys[i]
            if span == 0.0:
                continue
            t = (budget_ns - ys[i]) / span
            return {'pixels': xs[i] + t * (xs[i + 1] - xs[i]),
                    'kind': 'interpolated',
                    'between': [points[i]['dataset'], points[i + 1]['dataset']]}

    if ys[0] > budget_ns:
        i, j, kind = 0, 1, 'extrapolated below the smallest measured resolution'
    else:
        i, j, kind = len(points) - 2, len(points) - 1, \
            'extrapolated above the largest measured resolution'
    span = ys[j] - ys[i]
    if span == 0.0:
        return {'pixels': None, 'kind': 'flat, no crossing'}
    t = (budget_ns - ys[i]) / span
    return {'pixels': xs[i] + t * (xs[j] - xs[i]), 'kind': kind,
            'between': [points[i]['dataset'], points[j]['dataset']]}


# ------------------------------------------------------------- provenance

def toolchain(path: Path) -> dict:
    """The declared machine, lifted out of the docs/TOOLCHAIN.md table."""
    if not path.is_file():
        return {}
    entries = {}
    for line in path.read_text().splitlines():
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) != 2 or cells[0] in ('Component', '---'):
            continue
        if set(cells[0]) <= set('-: '):
            continue
        entries[cells[0]] = cells[1].replace('`', '')
    return entries


def observed_host() -> dict:
    return {
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'cpu_count': os.cpu_count(),
        'load_average': list(os.getloadavg()),
        'blas_threads': {name: os.environ.get(name)
                         for name in ('OPENBLAS_NUM_THREADS',
                                      'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
    }


def git_revision() -> str | None:
    try:
        completed = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                   cwd=REPO_ROOT, capture_output=True,
                                   text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


# ------------------------------------------------------------------ table

def table(headings: list[str], rows: list[list[str]],
          align: str | None = None) -> list[str]:
    """A markdown table. The one table writer everything in here goes through.

    `align` is one character per column: 'l' or 'r'.
    """
    align = align or 'l' * len(headings)
    rule = {'l': ':---', 'r': '---:'}
    return (['| ' + ' | '.join(headings) + ' |',
             '| ' + ' | '.join(rule[a] for a in align) + ' |']
            + ['| ' + ' | '.join(row) + ' |' for row in rows])


def ms(value: float | None, places: int = 3) -> str:
    return '-' if value is None else f'{value / NS_PER_MS:.{places}f}'


def us(value: float | None, places: int = 2) -> str:
    """Stages span four orders of magnitude, and 0.000 ms hides three of them."""
    return '-' if value is None else f'{value / 1e3:.{places}f}'


# ------------------------------------------------------------- the report

def build(groups: list[Group], config: dict, args) -> dict:
    analysis = config['analysis']
    wanted = analysis['percentiles']
    method = analysis['percentile_method']
    budget_ns = NS_PER_S / config['benchmark']['deadline_hz']

    summary = {
        'metadata': {
            'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                              time.gmtime()),
            'git_revision': git_revision(),
            'toolchain_declared': toolchain(args.toolchain),
            'host_observed': observed_host(),
            'config': str(args.config),
            'percentiles': wanted,
            'percentile_method': method,
            'deadline_hz': config['benchmark']['deadline_hz'],
            'budget_ns': budget_ns,
            'bootstrap': {
                'resamples': analysis['bootstrap_resamples'],
                'confidence': analysis['bootstrap_confidence'],
                'seed': analysis['bootstrap_seed'],
            },
            'sources': sorted({s for g in groups for s in g.sources}),
        },
        'groups': [],
        'speedups': [],
        'gc_attribution': {},
        'crossover': {},
    }

    for group in groups:
        entry = {
            'impl': group.impl,
            'dataset': group.dataset,
            'transport': group.transport,
            'width': group.width,
            'height': group.height,
            'pixels': group.pixels,
            'samples': group.n,
            'distinct_frames': int(np.unique(group.frame).size),
            'timer_overhead_ns': group.timer_overhead_ns,
            'total_ns': stats(group.total, wanted, method),
            'stage_ns': {key: stats(group.stages[:, i], wanted, method)
                         for i, key in enumerate(STAGE_KEYS)},
            'over_budget_fraction': float((group.total > budget_ns).mean()),
            'bail_outs': {
                'plane_not_found': int((~group.plane_found).sum()),
                'not_graspable': int((~group.graspable).sum()),
                'ik_not_converged': int((~group.converged).sum()),
            },
            'ik_iterations': {
                'median': float(np.median(group.ik_iterations)),
                'max': int(group.ik_iterations.max()),
            },
            'points_median': float(np.median(group.points)),
        }
        entry['p99_ci_iid'] = bootstrap_ci(
            group.total, 99.0, analysis['bootstrap_resamples'],
            analysis['bootstrap_confidence'], analysis['bootstrap_seed'],
            method)
        entry['p99_ci_by_frame'] = bootstrap_ci(
            group.total, 99.0, analysis['bootstrap_resamples'],
            analysis['bootstrap_confidence'], analysis['bootstrap_seed'],
            method, blocks=group.frame)
        summary['groups'].append(entry)

        if group.gc is not None:
            summary['gc_attribution'][f'{group.impl}/{group.dataset}'] = \
                gc_attribution(group, analysis['tail_percentile'], method)

    by_key = {(g['impl'], g['dataset'], g['transport']): g
              for g in summary['groups']}
    for (impl, dataset, transport), entry in sorted(by_key.items()):
        if impl == 'cpp':
            continue
        reference = by_key.get(('cpp', dataset, transport))
        if reference is None:
            continue
        ratio = {
            'impl': impl,
            'dataset': dataset,
            'transport': transport,
            'pixels': entry['pixels'],
            'total': {p: entry['total_ns'][p] / reference['total_ns'][p]
                      for p in ('p50', 'p99', 'mean', 'max')},
            'stages': {key: (entry['stage_ns'][key]['p50']
                             / reference['stage_ns'][key]['p50']
                             if reference['stage_ns'][key]['p50'] else None)
                       for key in STAGE_KEYS},
        }
        summary['speedups'].append(ratio)

    for impl in sorted({g['impl'] for g in summary['groups']}):
        rows = [g for g in summary['groups']
                if g['impl'] == impl and g['transport'] == 'inproc']
        # One point per resolution. Two corpora at the same pixel count would
        # make the interpolation ambiguous, so the canonically named one wins
        # and the other is still reported in the tables above.
        best: dict[int, dict] = {}
        for row in rows:
            canonical = f"table_{row['width']}x{row['height']}"
            existing = best.get(row['pixels'])
            if existing is None or row['dataset'] == canonical:
                best[row['pixels']] = row
        points = [{'dataset': r['dataset'], 'pixels': r['pixels'],
                   'p50': r['total_ns']['p50'], 'p99': r['total_ns']['p99'],
                   'over_budget_fraction': r['over_budget_fraction']}
                  for r in best.values()]
        summary['crossover'][impl] = {
            'points': sorted(points, key=lambda p: p['pixels']),
            'p99': crossover(points, budget_ns, 'p99'),
            'p50': crossover(points, budget_ns, 'p50'),
        }

    return summary


def blas_note(meta: dict) -> str:
    """Say how many cores each side was allowed, because it is not the same.

    NumPy reaches OpenBLAS, which threads across every core it can find unless
    told not to; Eigen here is built without OpenMP and runs on one. A ratio
    taken with those defaults compares a multi-core program against a
    single-core one, which is a fact about the deployment rather than about
    the language, and it has to be said next to the number rather than in a
    footnote.
    """
    threads = meta['host_observed']['blas_threads']
    unset = [name for name, value in threads.items() if value is None]
    if len(unset) == len(threads):
        return (
            f"**Thread counts.** None of {', '.join(f'`{n}`' for n in unset)} "
            f"was set, so NumPy's BLAS was free to use all "
            f"{meta['host_observed']['cpu_count']} cores while the C++ build, "
            'which has no OpenMP, used one. Rows labelled `py1t` are the same '
            'Python pipeline with the BLAS pinned to a single thread. See '
            'threat T4 in [../docs/METHOD.md](../docs/METHOD.md).')
    return ('**Thread counts.** BLAS thread environment at run time: '
            + ', '.join(f'`{n}`={v!r}' for n, v in threads.items()) + '.')


def headline(summary: dict, reference: str) -> list[str]:
    """The one-paragraph answer, on the corpus the repository leans on."""
    budget = summary['metadata']['budget_ns']
    pair = {e['impl']: e for e in summary['groups']
            if e['dataset'] == reference and e['transport'] == 'inproc'}
    if 'cpp' not in pair or 'py' not in pair:
        return ['', f'No cpp/py pair on `{reference}`, so there is no headline '
                    'number to state.', '']
    cpp, py = pair['cpp']['total_ns'], pair['py']['total_ns']
    fits = {name: entry['total_ns']['p99'] <= budget
            for name, entry in pair.items()}
    if fits['cpp'] and fits['py']:
        verdict = ('both hold the budget at p99, so at this rate and this '
                   'resolution the language does not decide whether the loop '
                   'closes')
    elif fits['cpp']:
        verdict = ('cpp holds the budget at p99 and py does not, so here the '
                   'language is the difference between closing the loop and '
                   'not')
    elif fits['py']:
        verdict = 'py holds the budget at p99 and cpp does not'
    else:
        verdict = ('neither holds the budget at p99, so the pipeline is the '
                   'problem before the language is')
    lines = [
        '',
        f"**On `{reference}` "
        f"({pair['cpp']['width']}x{pair['cpp']['height']}, "
        f"n={pair['cpp']['samples']} per implementation): "
        f"C++ p50 {cpp['p50'] / NS_PER_MS:.1f} ms, p99 "
        f"{cpp['p99'] / NS_PER_MS:.1f} ms. Python p50 "
        f"{py['p50'] / NS_PER_MS:.1f} ms, p99 {py['p99'] / NS_PER_MS:.1f} ms. "
        f"Python costs {py['p50'] / cpp['p50']:.2f}x at p50 and "
        f"{py['p99'] / cpp['p99']:.2f}x at p99. Against a "
        f"{budget / NS_PER_MS:.1f} ms budget, {verdict}.**",
    ]
    pinned = pair.get('py1t')
    if pinned is not None:
        one = pinned['total_ns']
        lines.append('')
        lines.append(
            f"**And the Python figure above was taken with NumPy's BLAS free "
            f"to use every core. Pinned to one thread, the same pipeline on "
            f"the same corpus is p50 {one['p50'] / NS_PER_MS:.1f} ms and p99 "
            f"{one['p99'] / NS_PER_MS:.1f} ms, which is "
            f"{one['p50'] / cpp['p50']:.2f}x C++ at p50 and "
            f"{one['p99'] / cpp['p99']:.2f}x at p99. The headline ratio is a "
            f"four-core program against a one-core one; this is the per-core "
            f"one.**")
    lines.append('')
    return lines


def markdown(summary: dict, rate_sweep: dict | None,
             reference: str) -> list[str]:
    meta = summary['metadata']
    budget_ms = meta['budget_ns'] / NS_PER_MS
    lines = [
        '# Results',
        '',
        'Generated by `harness/analyze.py`. Every number here came out of a '
        'run on the machine below; nothing is projected or rounded up from a '
        'shorter run.',
        '',
        f"Generated {meta['generated_at_utc']}"
        + (f", revision `{meta['git_revision'][:12]}`"
           if meta['git_revision'] else '') + '.',
    ]
    lines += headline(summary, reference)
    lines += ['## The machine', '']
    declared = meta['toolchain_declared']
    if declared:
        lines += table(['Component', 'Version'],
                       [[k, v] for k, v in declared.items()])
    host = meta['host_observed']
    lines += [
        '',
        f"Observed at run time: {host['platform']}, {host['cpu_count']} CPUs, "
        f"load average {', '.join(f'{x:.2f}' for x in host['load_average'])}, "
        f"CPython {host['python']}, NumPy {host['numpy']}.",
        '',
        blas_note(meta),
        '',
        f"Percentiles use the `{meta['percentile_method']}` method: the value "
        'reported is one that was actually observed, never an interpolation '
        'between two samples that were not.',
        '',
        '## End to end, per implementation and corpus',
        '',
        'All figures in milliseconds. The mean is here to be compared against '
        'p50, not to be quoted: on a right-skewed latency distribution it sits '
        'above the median and below the tail, and describes neither. The '
        '`mean/p50` column is how far it misleads.',
        '',
    ]

    rows = []
    for entry in summary['groups']:
        total = entry['total_ns']
        label = entry['impl'] if entry['transport'] == 'inproc' \
            else f"{entry['impl']} [{entry['transport']}]"
        ci = entry['p99_ci_by_frame']
        rows.append([
            label, entry['dataset'], f"{entry['width']}x{entry['height']}",
            str(entry['samples']),
            ms(total['p50']), ms(total['p90']), ms(total['p95']),
            ms(total['p99']),
            f"{ms(ci['low'])} to {ms(ci['high'])}",
            ms(total['p99.9']), ms(total['max']), ms(total['mean']),
            f"{total['mean'] / total['p50']:.2f}",
        ])
    lines += table(
        ['impl', 'corpus', 'res', 'n', 'p50', 'p90', 'p95', 'p99',
         'p99 95% CI', 'p99.9', 'max', 'mean', 'mean/p50'],
        rows, align='llrrrrrrrrrrr')

    lines += ['', '### What each run was doing', '']
    rows = []
    for entry in summary['groups']:
        bail = entry['bail_outs']
        rows.append([
            entry['impl'] if entry['transport'] == 'inproc'
            else f"{entry['impl']} [{entry['transport']}]",
            entry['dataset'], str(entry['samples']),
            str(entry['distinct_frames']),
            f"{entry['points_median']:,.0f}",
            f"{entry['ik_iterations']['median']:.0f}",
            str(entry['ik_iterations']['max']),
            f"{bail['plane_not_found']}/{bail['not_graspable']}"
            f"/{bail['ik_not_converged']}",
            ('-' if entry['timer_overhead_ns'] is None
             else f"{entry['timer_overhead_ns']:.0f}"),
        ])
    lines += table(
        ['impl', 'corpus', 'samples', 'distinct frames', 'median points',
         'median IK iters', 'max IK iters',
         'bail-outs plane/grasp/ik', 'clock read ns'],
        rows, align='llrrrrrrr')
    lines += [
        '',
        'Bail-outs are counted because a frame that gives up early is a '
        'shorter pipeline and a cheaper sample. The corpus generator '
        'guarantees a graspable object in every frame, so a zero here is a '
        'property of the corpus, not a result: see docs/METHOD.md.',
        '',
        'The p99 interval is a percentile bootstrap that resamples distinct '
        'store frames and keeps every repeat of each, because a run cycles a '
        '100-frame corpus many times and twenty samples of one frame are not '
        'twenty independent draws. `summary.json` also carries the narrower '
        'interval that treats them as independent, for comparison.',
        '',
        '## What the language costs, per stage',
        '',
        'This is the analytically interesting table. A stage that is one large '
        'NumPy call over a big array should show almost no gap; a stage that '
        'iterates over small matrices in Python should show a large one. '
        'Ratios are py p50 divided by cpp p50.',
        '',
    ]

    for ratio in summary['speedups']:
        dataset = ratio['dataset']
        cpp = next(g for g in summary['groups']
                   if g['impl'] == 'cpp' and g['dataset'] == dataset
                   and g['transport'] == ratio['transport'])
        py = next(g for g in summary['groups']
                  if g['impl'] == ratio['impl'] and g['dataset'] == dataset
                  and g['transport'] == ratio['transport'])
        lines += [f"### {dataset} ({cpp['width']}x{cpp['height']}), "
                  f"cpp against {ratio['impl']}", '']
        stage_rows = []
        for key in STAGE_KEYS:
            c, p = cpp['stage_ns'][key], py['stage_ns'][key]
            factor = ratio['stages'][key]
            stage_rows.append([
                key, us(c['p50']), us(p['p50']),
                '-' if factor is None else f'{factor:.1f}x',
                us(c['p99']), us(p['p99']),
                f"{100 * c['p50'] / cpp['total_ns']['p50']:.1f}%",
                f"{100 * p['p50'] / py['total_ns']['p50']:.1f}%",
            ])
        stage_rows.append([
            '**total**', us(cpp['total_ns']['p50']), us(py['total_ns']['p50']),
            f"{ratio['total']['p50']:.1f}x",
            us(cpp['total_ns']['p99']), us(py['total_ns']['p99']),
            '100%', '100%'])
        lines += table(
            ['stage', 'cpp p50 us', f"{ratio['impl']} p50 us", 'p50 ratio',
             'cpp p99 us', f"{ratio['impl']} p99 us", 'cpp share',
             f"{ratio['impl']} share"],
            stage_rows, align='lrrrrrrr')
        lines += [
            '',
            f"Whole pipeline: {ratio['impl']} is "
            f"{ratio['total']['p50']:.2f}x cpp at p50 "
            f"and {ratio['total']['p99']:.2f}x at p99. The two differ because "
            'the tail is not a scaled copy of the median.',
            '',
        ]

    lines += [
        'Per-stage percentiles do not add up to the total percentile, and are '
        'not meant to: the frame that produced the p99 of one stage is rarely '
        'the frame that produced the p99 of another. The p50 column sums to '
        'about the p50 of the total, which is why the shares are taken there.',
        '',
        '## Where the Python tail comes from',
        '',
    ]

    if not summary['gc_attribution']:
        lines += ['No record carried garbage-collection counters, so no '
                  'attribution was possible.', '']
    else:
        rows = []
        for name, entry in sorted(summary['gc_attribution'].items()):
            if not entry:
                continue
            tail = entry['tail_percentile']
            clean = entry[f'p{tail:g}_without_collection_ns']
            rows.append([
                name,
                f"{entry['collections_over_run']['gen0']}/"
                f"{entry['collections_over_run']['gen1']}/"
                f"{entry['collections_over_run']['gen2']}",
                f"{entry['frames_with_collection']} "
                f"({100 * entry['collection_frame_share']:.1f}%)",
                ms(entry[f'p{tail:g}_all_ns']),
                ms(clean),
                ('-' if entry['tail_collection_share'] is None
                 else f"{100 * entry['tail_collection_share']:.1f}%"),
                ('-' if entry['point_biserial_r'] is None
                 else f"{entry['point_biserial_r']:+.3f}"),
            ])
        lines += table(
            ['run', 'collections gen0/1/2', 'frames with a collection',
             'p99 all', 'p99 excluding them', 'share of tail frames',
             'point-biserial r'],
            rows, align='lrrrrrr')
        lines += ['']
        for name, entry in sorted(summary['gc_attribution'].items()):
            if entry:
                lines.append(f"* `{name}`: {entry['verdict']}.")
        lines += ['']

    lines += [
        '## The resolution crossover',
        '',
        f'The control budget is {budget_ms:.1f} ms, one period at '
        f"{meta['deadline_hz']:g} Hz. The crossing is a linear interpolation "
        'of p99 against pixel count between the two measured resolutions that '
        'bracket the budget, which is a straight line through two points and '
        'nothing more: the pipeline is not exactly linear in pixels, so treat '
        'it as a locator, not a law.',
        '',
    ]
    rows = []
    for impl, entry in sorted(summary['crossover'].items()):
        for point in entry['points']:
            rows.append([impl, point['dataset'], f"{point['pixels']:,}",
                         ms(point['p50']), ms(point['p99']),
                         f"{100 * point['over_budget_fraction']:.1f}%"])
    lines += table(['impl', 'corpus', 'pixels', 'p50', 'p99',
                    f'frames over {budget_ms:.1f} ms'],
                   rows, align='llrrrr')
    lines += ['']
    for impl, entry in sorted(summary['crossover'].items()):
        crossing = entry['p99']
        if crossing['pixels'] is None:
            lines.append(f"* **{impl}**: {crossing['kind']}.")
            continue
        pixels = crossing['pixels']
        side = round((pixels * 3 / 4) ** 0.5)
        lines.append(
            f"* **{impl}**: p99 crosses {budget_ms:.1f} ms at about "
            f"{pixels:,.0f} pixels, roughly {side * 4 // 3:.0f}x{side:.0f} at "
            f"4:3, {crossing['kind']} between "
            f"{crossing.get('between', ['?', '?'])[0]} and "
            f"{crossing.get('between', ['?', '?'])[1]}.")
    lines += ['']

    if rate_sweep:
        lines += rate_sweep_section(rate_sweep)

    lines += [
        '## Files',
        '',
        '* `summary.json`: every aggregate above, plus the bootstrap draws\' '
        'parameters and the machine metadata.',
        '* `summary.csv`: the same aggregates in long form, one metric per row.',
        '* `rate_sweep.json`, `rate_sweep.csv`: the control-rate sweep.',
        '* `latency_cdf.png`, `stage_breakdown.png`, '
        '`resolution_scaling.png`, `rate_sweep.png`.',
        '',
        'Method and threats to validity: [../docs/METHOD.md](../docs/METHOD.md).',
    ]
    return lines


def rate_sweep_section(sweep: dict) -> list[str]:
    """The control-rate sweep, rendered through the same table writer."""
    meta = sweep['metadata']
    lines = [
        '## Control-rate sweep',
        '',
        f"Each implementation driven over the frame store at a fixed rate, "
        f"{meta['jobs_per_rate']} jobs per rate, releases and deadlines taken "
        'as absolute offsets from one monotonic origin. A job that overruns '
        'delays the next one rather than being queued, which is what a '
        'best-effort depth subscriber does. `worst finish` is signed: negative '
        'means the slowest job of the run still beat its deadline.',
        '',
    ]
    rows = []
    for run in sweep['runs']:
        compute, response = run['compute'], run['response']
        rows.append([
            run['impl'], run['dataset'], f"{run['rate_hz']:g}",
            f"{run['period_ms']:.1f}",
            ms(compute['p50']), ms(compute['p99']),
            ms(response['p50']), ms(response['p99']),
            f"{100 * run['deadline_miss_rate']:.1f}%",
            f"{run['worst_finish_vs_deadline_ms']:+.1f}",
            f"{run['worst_finish_vs_deadline_periods']:+.1f}",
            f"{run['utilisation']:.2f}",
        ])
    lines += table(
        ['impl', 'corpus', 'rate Hz', 'period ms', 'compute p50',
         'compute p99', 'response p50', 'response p99', 'misses',
         'worst finish ms', 'worst finish periods', 'utilisation'],
        rows, align='llrrrrrrrrrr')
    lines += [
        '',
        '`compute` is the pipeline. `response` is release to finish, so it '
        'carries the backlog a previous overrun left behind, and it is the '
        'number a downstream controller actually waits for.',
        '',
    ]
    rows = []
    for row in sweep['onset']:
        clean = row['highest_clean_rate_hz']
        missing = row['lowest_missing_rate_hz']
        rows.append([
            row['impl'], row['dataset'],
            'none swept' if clean is None else f'{clean:g} Hz',
            'none swept' if missing is None else f'{missing:g} Hz',
            f"{row['sustainable_hz_p50']:.1f} Hz",
            f"{row['sustainable_hz_p99']:.1f} Hz",
        ])
    lines += table(
        ['impl', 'corpus', 'highest clean rate', 'lowest rate that missed',
         'sustainable from p50', 'sustainable from p99'],
        rows, align='llrrrr')
    lines += [
        '',
        'The two sustainable-rate columns are `1 / latency` at that '
        'percentile: the rate a loop could hold if every frame cost the '
        'median, and the rate it could hold if every frame cost the p99. A '
        'real loop lives between them and misses deadlines from the moment the '
        'period drops below the second.',
        '',
    ]
    return lines


# ------------------------------------------------------------------ plots

def plots(summary: dict, groups: list[Group], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    budget_ms = summary['metadata']['budget_ns'] / NS_PER_MS
    colours = {'cpp': '#1f4e79', 'py': '#c0504d'}
    written = []

    inproc = [g for g in groups if g.transport == 'inproc']
    datasets = sorted({g.dataset for g in inproc},
                      key=lambda d: next(g.pixels for g in inproc
                                         if g.dataset == d))

    # 1. Latency CDF, one panel per corpus, deadline marked.
    columns = min(3, max(1, len(datasets)))
    rows = (len(datasets) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, squeeze=False,
                                figsize=(4.6 * columns, 3.6 * rows))
    for index, dataset in enumerate(datasets):
        axis = axes[index // columns][index % columns]
        shape = next(f'{g.width}x{g.height}' for g in inproc
                     if g.dataset == dataset)
        for group in (g for g in inproc if g.dataset == dataset):
            samples = np.sort(group.total) / NS_PER_MS
            quantiles = np.arange(1, samples.size + 1) / samples.size
            axis.plot(samples, quantiles, color=colours.get(group.impl, '0.4'),
                      linewidth=1.6, label=f'{group.label} (n={group.n})')
        axis.axvline(budget_ms, color='0.3', linestyle=(0, (5, 3)),
                     linewidth=1.0)
        axis.text(budget_ms, 0.04, f' {budget_ms:.1f} ms budget', fontsize=7,
                  color='0.3', rotation=90, va='bottom')
        axis.set_title(f'{dataset} ({shape})', fontsize=10)
        axis.set_xlabel('end-to-end latency, ms')
        axis.set_ylabel('fraction of frames at or below')
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc='lower right')
    for index in range(len(datasets), rows * columns):
        axes[index // columns][index % columns].axis('off')
    figure.suptitle('Latency distribution per corpus, with the 30 Hz budget',
                    fontsize=12)
    figure.tight_layout()
    path = out_dir / 'latency_cdf.png'
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    # 2. Where the time goes: p50 per stage, stacked.
    entries = [e for e in summary['groups'] if e['transport'] == 'inproc']
    entries.sort(key=lambda e: (e['pixels'], e['impl']))
    labels = [f"{e['impl']}\n{e['width']}x{e['height']}" for e in entries]
    figure, axis = plt.subplots(figsize=(1.15 * len(entries) + 3.0, 5.6))
    bottom = np.zeros(len(entries))
    palette = plt.get_cmap('tab20')
    for i, key in enumerate(STAGE_KEYS):
        values = np.array([e['stage_ns'][key]['p50'] / NS_PER_MS
                           for e in entries])
        axis.bar(labels, values, bottom=bottom, label=key,
                 color=palette(i / len(STAGE_KEYS)), edgecolor='white',
                 linewidth=0.5)
        bottom += values
    axis.axhline(budget_ms, color='0.2', linestyle=(0, (5, 3)), linewidth=1.0)
    axis.text(len(entries) - 0.4, budget_ms, f'{budget_ms:.1f} ms budget',
              fontsize=8, va='bottom', ha='right', color='0.2')
    axis.set_ylabel('p50 latency, ms')
    axis.set_title('Where the median frame spends its time')
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis='y', alpha=0.25)
    figure.tight_layout()
    path = out_dir / 'stage_breakdown.png'
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    # 3. Scaling with pixel count, and where p99 crosses the budget.
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    for impl, entry in sorted(summary['crossover'].items()):
        points = entry['points']
        if not points:
            continue
        xs = [p['pixels'] / 1e6 for p in points]
        axis.plot(xs, [p['p50'] / NS_PER_MS for p in points], '-o',
                  color=colours.get(impl, '0.4'), label=f'{impl} p50')
        axis.plot(xs, [p['p99'] / NS_PER_MS for p in points], '--s',
                  color=colours.get(impl, '0.4'), markerfacecolor='none',
                  label=f'{impl} p99')
        crossing = entry['p99']
        if crossing['pixels'] is not None and crossing['kind'] == 'interpolated':
            axis.plot([crossing['pixels'] / 1e6], [budget_ms], marker='*',
                      markersize=14, color=colours.get(impl, '0.4'),
                      linestyle='none',
                      label=f"{impl} p99 crossing "
                            f"{crossing['pixels'] / 1e6:.2f} MP")
    axis.axhline(budget_ms, color='0.3', linestyle=(0, (5, 3)), linewidth=1.0)
    axis.text(axis.get_xlim()[0], budget_ms, f' {budget_ms:.1f} ms budget',
              fontsize=8, va='bottom', color='0.3')
    axis.set_xlabel('pixels per frame, megapixels')
    axis.set_ylabel('latency, ms')
    axis.set_title('Latency against resolution, and where p99 leaves the budget')
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = out_dir / 'resolution_scaling.png'
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    return written


# ------------------------------------------------------------------- csv

CSV_COLUMNS = ('impl', 'dataset', 'transport', 'width', 'height', 'pixels',
               'scope', 'metric', 'value')


def csv_rows(summary: dict):
    for entry in summary['groups']:
        base = {k: entry[k] for k in ('impl', 'dataset', 'transport', 'width',
                                      'height', 'pixels')}
        for metric, value in entry['total_ns'].items():
            yield {**base, 'scope': 'total', 'metric': f'{metric}_ns'
                   if metric != 'n' else 'samples', 'value': value}
        for stage, values in entry['stage_ns'].items():
            for metric, value in values.items():
                if metric == 'n':
                    continue
                yield {**base, 'scope': f'stage:{stage}',
                       'metric': f'{metric}_ns', 'value': value}
        yield {**base, 'scope': 'total', 'metric': 'p99_ci_low_ns',
               'value': entry['p99_ci_by_frame']['low']}
        yield {**base, 'scope': 'total', 'metric': 'p99_ci_high_ns',
               'value': entry['p99_ci_by_frame']['high']}
        yield {**base, 'scope': 'run', 'metric': 'over_budget_fraction',
               'value': entry['over_budget_fraction']}
        yield {**base, 'scope': 'run', 'metric': 'distinct_frames',
               'value': entry['distinct_frames']}
        yield {**base, 'scope': 'run', 'metric': 'timer_overhead_ns',
               'value': entry['timer_overhead_ns']}
        yield {**base, 'scope': 'run', 'metric': 'ik_iterations_median',
               'value': entry['ik_iterations']['median']}
        for name, value in entry['bail_outs'].items():
            yield {**base, 'scope': 'run', 'metric': f'bail_{name}',
                   'value': value}


# ------------------------------------------------------------------- main

def main(argv=None) -> int:
    config_default = json.loads(DEFAULT_CONFIG.read_text())
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('timing', type=Path, nargs='*',
                        help='*.timing.jsonl; defaults to <out-dir>/*.timing.jsonl')
    parser.add_argument('--out-dir', type=Path, default=REPO_ROOT / 'results')
    parser.add_argument('--data-root', type=Path, default=REPO_ROOT / 'data')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--toolchain', type=Path, default=DEFAULT_TOOLCHAIN)
    parser.add_argument('--rate-sweep', type=Path, default=None,
                        help='rate_sweep.json; defaults to one beside the '
                             'output if it exists')
    parser.add_argument('--reference', default=None,
                        help='corpus the prose leans on; defaults to '
                             'analysis.reference_dataset')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text()) if args.config != DEFAULT_CONFIG \
        else config_default
    reference = args.reference or config['analysis']['reference_dataset']

    paths = args.timing or sorted(args.out_dir.glob('*.timing.jsonl'))
    if not paths:
        die(f'no timing files given and none in {args.out_dir}; '
            f'run harness/run_all.sh first')
    missing = [p for p in paths if not p.is_file()]
    if missing:
        die(f'these timing files do not exist: '
            f'{", ".join(str(p) for p in missing)}')

    groups = load(list(paths), args.data_root)
    print(f'analyze: {len(groups)} run(s) over '
          f'{sum(g.n for g in groups)} measured frames', flush=True)

    summary = build(groups, config, args)

    sweep_path = args.rate_sweep
    if sweep_path is None:
        candidate = args.out_dir / 'rate_sweep.json'
        sweep_path = candidate if candidate.is_file() else None
    sweep = json.loads(sweep_path.read_text()) if sweep_path else None
    if sweep:
        summary['rate_sweep'] = sweep

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = '\n'.join(markdown(summary, sweep, reference)) + '\n'
    (args.out_dir / 'RESULTS.md').write_text(report, encoding='utf-8')
    (args.out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    with (args.out_dir / 'summary.csv').open('w', newline='',
                                             encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows(summary))

    written = [] if args.no_plots else plots(summary, groups, args.out_dir)

    print(report)
    print(f'wrote {args.out_dir}/RESULTS.md, summary.json, summary.csv'
          + (''.join(f', {p.name}' for p in written)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
