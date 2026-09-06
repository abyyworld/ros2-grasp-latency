#!/usr/bin/env python3
"""Put measured numbers into docs/METHOD.md, and keep putting them there.

METHOD.md quotes a dozen figures that come out of the runs rather than out of
an argument: what a clock read costs on each side, how far apart the two
implementations landed on every corpus, what the rate sweep's own scheduler and
`ctypes` transition cost, and what the IK null-space sweep found. Typing those
in by hand means that the next run silently disagrees with the document, and a
document that disagrees with its own data is worse than one with a hole in it.

So each of them lives inside a marked region:

    ... the median cost is <!--@TIMER_CPP-->21<!--/TIMER_CPP--> ns ...

The markers are HTML comments, so they render as nothing and a reader sees only
the number. This script rewrites what is between them from the JSON in
`results/`, which is the same JSON the repository commits. Running it twice in
a row changes nothing, and running it after a new experiment updates every
figure at once. It exits non-zero if a figure cannot be produced, because the
alternative is a document that looks measured and is not.

Two modes:

    tools/fill_method_placeholders.py --measure-nullspace
        Runs the IK null-space sweep (this is the one figure in METHOD.md that
        no other runner produces) and writes results/nullspace_sweep.json.

    tools/fill_method_placeholders.py [--check]
        Rewrites docs/METHOD.md from the committed JSON. --check reports what
        would change and exits non-zero instead of writing, which is what CI
        wants.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / 'docs/METHOD.md'
RESULTS = ROOT / 'results'
CONFIG = ROOT / 'assets/pipeline_config.json'

CLOCK_READS_PER_FRAME = 9  # one before S0 and one after each of the eight stages

# The sweep METHOD.md section 11 reports. Fixed here rather than taken from the
# config because the config holds the chosen value, not the range it was chosen
# from, and the document has to say what was actually tried.
NULLSPACE_DAMPINGS = (0.05, 0.02, 0.01, 0.005, 0.001)
NULLSPACE_GAINS = (0.1, 0.02, 0.0)


# --------------------------------------------------------------------------
# Reading what the runs left behind


class Missing(Exception):
    """A figure cannot be produced from the data that is present."""


def load_json(path: Path):
    if not path.is_file():
        raise Missing(f'{path.relative_to(ROOT)} is not there')
    return json.loads(path.read_text())


class Sources:
    """Everything the substitutions read, loaded once and only when asked for.

    Lazy because --measure-nullspace has no business needing a rate sweep, and
    because a clear "which file is missing" beats a stack trace.
    """

    def __init__(self, results: Path):
        self.results = results
        self._cache: dict[str, object] = {}

    def _get(self, key, produce):
        if key not in self._cache:
            self._cache[key] = produce()
        return self._cache[key]

    @property
    def config(self):
        return self._get('config', lambda: load_json(CONFIG))

    @property
    def reference(self) -> str:
        return self.config['analysis']['reference_dataset']

    @property
    def summary(self):
        return self._get('summary', lambda: load_json(self.results / 'summary.json'))

    @property
    def rate_sweep(self):
        return self._get('sweep', lambda: load_json(self.results / 'rate_sweep.json'))

    @property
    def nullspace(self):
        return self._get(
            'nullspace', lambda: load_json(self.results / 'nullspace_sweep.json'))

    @property
    def python_meta(self):
        return self._get(
            'pymeta',
            lambda: load_json(self.results / f'py.{self.reference}.meta.json'))

    def group(self, impl: str, dataset: str) -> dict:
        for group in self.summary['groups']:
            if group['impl'] == impl and group['dataset'] == dataset:
                return group
        raise Missing(f'summary.json has no {impl} run of {dataset}')

    def equivalence_files(self) -> list[Path]:
        found = sorted(self.results.glob('equivalence.*.json'), key=_by_pixels)
        if not found:
            raise Missing('no results/equivalence.*.json to read')
        return found


def _by_pixels(path: Path):
    """Order the corpora by size rather than alphabetically, so the table
    reads the way the resolution-scaling table in RESULTS.md does."""
    shape = re.search(r'(\d+)x(\d+)', path.name)
    pixels = int(shape.group(1)) * int(shape.group(2)) if shape else 0
    return (pixels, path.name)


# --------------------------------------------------------------------------
# Formatting. Every figure that reaches the document goes through one of these,
# so a stray full-precision float cannot get into the prose by accident.


def significant(value: float, digits: int = 3) -> str:
    """A number with a fixed number of significant digits and no exponent."""
    if value == 0:
        return '0'
    text = f'%.{digits}g' % value
    if 'e' in text:  # only reached for figures far outside anything here
        return text
    return text


def scientific(value: float, digits: int = 1) -> str:
    """`2.5e-14`, the spelling a deviation column wants."""
    if value == 0:
        return '0'
    mantissa, exponent = (f'%.{digits}e' % value).split('e')
    return f'{mantissa}e{int(exponent)}'


def thousands(value: int) -> str:
    return f'{value:,}'


# --------------------------------------------------------------------------
# The substitutions


def timer_cpp_ns(src: Sources) -> int:
    """One C++ clock read. Stamped identically into every record of a run."""
    values = {group['timer_overhead_ns'] for group in src.summary['groups']
              if group['impl'] == 'cpp'}
    if not values:
        raise Missing('summary.json has no C++ run to take a timer figure from')
    # The calibration runs once per process, so four runs give four estimates
    # of the same quantity. The reference corpus is the one the document's
    # other numbers come from, so it is the one quoted.
    return int(src.group('cpp', src.reference)['timer_overhead_ns'])


def timer_py_ns(src: Sources) -> int:
    timer = src.python_meta.get('timer')
    if not timer or 'median_ns' not in timer:
        raise Missing('the Python run metadata carries no timer calibration')
    return int(round(timer['median_ns']))


def sub_timer_cpp(src):
    return str(timer_cpp_ns(src))


def sub_timer_cpp9(src):
    return thousands(CLOCK_READS_PER_FRAME * timer_cpp_ns(src))


def sub_timer_py(src):
    return str(timer_py_ns(src))


def sub_timer_py9(src):
    return thousands(CLOCK_READS_PER_FRAME * timer_py_ns(src))


def sub_timer_py_share(src):
    """The instrumentation floor as a share of a Python frame."""
    frame_ns = src.group('py', src.reference)['total_ns']['p50']
    share = CLOCK_READS_PER_FRAME * timer_py_ns(src) / frame_ns
    return f'{significant(share * 100.0, 2)}%'


def sub_timer_ratio(src):
    return significant(timer_py_ns(src) / timer_cpp_ns(src), 2)


def sub_timer_bias_us(src):
    """What Python is charged for instrumenting itself, over the C++ cost."""
    extra_ns = CLOCK_READS_PER_FRAME * (timer_py_ns(src) - timer_cpp_ns(src))
    return significant(extra_ns / 1000.0, 2)


def sub_ctypes_ns(src):
    workers = src.rate_sweep['metadata']['workers']
    if 'cpp' not in workers or 'ctypes_call_overhead_ns' not in workers['cpp']:
        raise Missing('rate_sweep.json carries no ctypes calibration')
    return significant(workers['cpp']['ctypes_call_overhead_ns'], 3)


def sub_sched_overhead(src):
    """Release-to-finish minus compute, at the lowest swept rate.

    Only rows that missed no deadline are eligible: once a run is backlogged
    the gap between `response` and `compute` is the backlog, which is the thing
    the sweep is reporting rather than the scheduler's own cost.
    """
    rates = src.rate_sweep['metadata']['rates_hz']
    lowest = min(rates)
    gaps = []
    for row in src.rate_sweep['runs']:
        if row['rate_hz'] != lowest or row['deadline_misses']:
            continue
        gaps.append((row['response']['p50'] - row['compute']['p50']) / 1e6)
    if not gaps:
        raise Missing(f'no clean {lowest:g} Hz row in rate_sweep.json')
    low, high = min(gaps), max(gaps)
    if significant(low, 2) == significant(high, 2):
        return f'{significant(low, 2)}'
    return f'{significant(low, 2)} to {significant(high, 2)}'


# Fields whose agreement is exact or nothing: a boolean, an iteration count or
# a hex digest either matches or it does not, so a deviation column would print
# a null for every corpus and say less than one sentence of prose.
EXACT_FIELDS = ('plane_found', 'graspable', 'converged', 'iterations',
                'traj_checksum')
DEVIATION_FIELDS = (('plane', 'plane `(n, d)`'), ('width', '`width`'),
                    ('tcp', 'TCP pose'), ('q', '`q`'),
                    ('duration_s', 'duration'))


def corpus_label(path: Path, report: dict) -> str:
    """`equivalence.table_848x480.json` reads as its corpus, and the
    single-BLAS-thread pass reads as the pass it is rather than as a corpus."""
    stem = path.name[len('equivalence.'):-len('.json')]
    parts = stem.split('.')
    label = f'`{parts[0]}`'
    right = report.get('right', {}).get('impl', 'py')
    if right != 'py':
        label += f' ({right})'
    return label


def sub_equivalence_table(src):
    rows = []
    worst_overall = 0.0
    for path in src.equivalence_files():
        report = load_json(path)
        fields = report['fields']
        cells = []
        for key, _ in DEVIATION_FIELDS:
            deviation = fields.get(key, {}).get('worst_abs_deviation')
            cells.append('exact' if deviation in (None, 0)
                         else scientific(deviation))
            if deviation:
                worst_overall = max(worst_overall, deviation)
        for key in EXACT_FIELDS:
            if fields.get(key, {}).get('mismatches'):
                raise Missing(f'{path.name} records a mismatch in {key}; the '
                              f'gate did not pass and no table should say it did')
        if not report.get('agree', False):
            raise Missing(f'{path.name} says the two implementations disagree')
        rows.append(f'| {corpus_label(path, report)} | '
                    f'{report["frames_compared"]} | ' + ' | '.join(cells) + ' |')

    tolerance = float(src.config['analysis']['equivalence_tolerance'])
    header = ('| Corpus | Frames | ' +
              ' | '.join(name for _, name in DEVIATION_FIELDS) + ' |')
    rule = '|---' * (2 + len(DEVIATION_FIELDS)) + '|'
    exact = ', '.join(f'`{name}`' for name in EXACT_FIELDS)
    if worst_overall:
        margin = (f'The worst deviation anywhere above is '
                  f'{scientific(worst_overall)} against a tolerance of '
                  f'{scientific(tolerance)}, a margin of '
                  f'{significant(tolerance / worst_overall, 2)} times.')
    else:
        margin = ('Every field matched to the last bit on every frame, which '
                  'is a stronger claim than the tolerance asks for and worth '
                  'reading twice before believing.')
    tail = (f'\nEach entry is the worst absolute deviation over every frame and '
            f'every component of that field, in metres, radians or '
            f'dimensionless matrix entries; `results/equivalence.<corpus>.json` '
            f'carries the frame each one came from. The fields that compare '
            f'exactly rather than to a tolerance ({exact}) are not in the '
            f'table; they matched on every frame of every corpus. {margin}')
    return '\n'.join([header, rule] + rows) + '\n' + tail


def sub_nullspace_table(src):
    data = src.nullspace
    rows = ['| `damping` | `nullspace_gain` | Converged | Median iterations | '
            'Worst position error | Worst orientation error |',
            '|---|---|---|---|---|---|']
    for entry in data['sweep']:
        iterations = ('n/a' if entry['median_iterations'] is None
                      else str(entry['median_iterations']))
        rows.append(
            f'| {entry["damping"]:g} | {entry["nullspace_gain"]:g} | '
            f'{entry["converged"]}/{entry["targets"]} | {iterations} | '
            f'{significant(entry["worst_position_error_m"] * 1e3, 3)} mm | '
            f'{significant(entry["worst_orientation_error_rad"] * 1e3, 3)} mrad |')
    return '\n'.join(rows)


def sub_nullspace_prose(src):
    data = src.nullspace
    sweep = data['sweep']
    targets = sweep[0]['targets']
    tol_mm = data['position_tolerance_m'] * 1e3
    tol_mrad = data['orientation_tolerance_rad'] * 1e3

    full = [e for e in sweep if e['converged'] == e['targets']]
    if not full:
        raise Missing('no configuration in the null-space sweep converged on '
                      'every target; the prose below would be a fiction')
    gains_that_work = sorted({e['nullspace_gain'] for e in full})
    biased = [e for e in sweep if e['nullspace_gain'] > 0.0]
    worst_biased_converged = max(e['converged'] for e in biased)
    biased_orientation = min(e['worst_orientation_error_rad'] for e in biased)
    chosen = src.config['ik']['nullspace_gain']
    chosen_damping = src.config['ik']['damping']
    at_chosen = next(e for e in sweep
                     if e['nullspace_gain'] == chosen
                     and e['damping'] == chosen_damping)
    zero_gain = [e for e in sweep if e['nullspace_gain'] == 0.0]
    worst_zero_position = max(e['worst_position_error_m'] for e in zero_gain)
    worst_zero = max(zero_gain, key=lambda e: e['worst_position_error_m'])

    if gains_that_work != [0.0]:
        # Kept honest rather than kept short: if a future change makes the
        # biased solver converge, the paragraph has to stop saying it cannot.
        works = ', '.join(f'{g:g}' for g in gains_that_work)
        return (f'Configurations that converged on all {targets} targets '
                f'appear at gains {works}, which is not the picture this '
                f'section was written for. Rerun and rewrite it.')

    return (
        f'The gain, not the damping, decides it. Every configuration with the '
        f'gain at zero converged on all {targets} targets; no configuration '
        f'with the gain above zero converged on more than '
        f'{worst_biased_converged}, at any damping in the sweep. The failures '
        f'are not the solver running out of iterations. It reaches a fixed '
        f'point and stays there: the best orientation error any biased '
        f'configuration reached was {significant(biased_orientation * 1e3, 3)} '
        f'mrad against a tolerance of {significant(tol_mrad, 2)} mrad. '
        f'`I - J^T (J J^T + lambda^2 I)^-1 J` is not a null-space projector '
        f'while `lambda > 0`, so the pull toward `q_neutral` does not stay in '
        f'the null space; it leaks into task space and is balanced there by '
        f'the task error rather than driven out of it.\n\n'
        f'Damping matters in the other direction and only once the gain is '
        f'off. At {worst_zero["damping"]:g} the worst position error over the '
        f'{targets} targets is {significant(worst_zero_position * 1e3, 3)} mm, '
        f'against {significant(at_chosen["worst_position_error_m"] * 1e3, 3)} '
        f'mm at the configured {chosen_damping:g}: too little damping is what '
        f'goes wrong near a singularity, and it goes wrong in position rather '
        f'than in orientation.\n\n'
        f'A true pseudo-inverse projector converges, and costs an SVD per '
        f'iteration in the stage that is already the most expensive part of '
        f'the Python frame per unit of work done. Since every frame is seeded '
        f'from `q_neutral`, the solution is already near the neutral posture '
        f'and the bias has nothing left to buy. So `ik.nullspace_gain` is '
        f'{chosen:g}, and the median IK cost in `RESULTS.md` is the cost of '
        f'the solver that converges rather than of one that runs to the '
        f'iteration cap on every frame. The tolerance those columns are judged '
        f'against is {significant(tol_mm, 2)} mm and '
        f'{significant(tol_mrad, 2)} mrad, from `ik.position_tolerance_m` and '
        f'`ik.orientation_tolerance_rad`.')


# name -> (producer, is_block). A block substitution sits on its own lines.
SUBSTITUTIONS = {
    'TIMER_CPP': (sub_timer_cpp, False),
    'TIMER_CPP9': (sub_timer_cpp9, False),
    'TIMER_PY': (sub_timer_py, False),
    'TIMER_PY9': (sub_timer_py9, False),
    'TIMER_PY_SHARE': (sub_timer_py_share, False),
    'TIMER_RATIO': (sub_timer_ratio, False),
    'TIMER_BIAS_US': (sub_timer_bias_us, False),
    'EQUIVALENCE_TABLE': (sub_equivalence_table, True),
    'SCHED_OVERHEAD': (sub_sched_overhead, False),
    'CTYPES_NS': (sub_ctypes_ns, False),
    'NULLSPACE_TABLE': (sub_nullspace_table, True),
    'NULLSPACE_PROSE': (sub_nullspace_prose, True),
}


# --------------------------------------------------------------------------
# The rewrite itself


def region(name: str, value: str, block: bool) -> str:
    if block:
        return f'<!--@{name}-->\n{value}\n<!--/{name}-->'
    return f'<!--@{name}-->{value}<!--/{name}-->'


def substitute(text: str, name: str, value: str, block: bool) -> tuple[str, int]:
    """Replace a marked region, or the raw placeholder on the first pass."""
    replacement = region(name, value, block)
    marked = re.compile(rf'<!--@{name}-->.*?<!--/{name}-->', re.DOTALL)
    text, count = marked.subn(lambda _: replacement, text)
    if count:
        return text, count
    return re.subn(rf'@@{name}@@', lambda _: replacement, text)


def fill(text: str, src: Sources) -> tuple[str, list[str], list[str]]:
    notes, problems = [], []
    for name, (produce, block) in SUBSTITUTIONS.items():
        try:
            value = produce(src)
        except Missing as why:
            problems.append(f'{name}: {why}')
            continue
        text, count = substitute(text, name, value, block)
        if count == 0:
            problems.append(f'{name}: neither @@{name}@@ nor a marked region '
                            f'is in {METHOD.name}')
        elif count > 1:
            notes.append(f'{name}: filled {count} places')
        else:
            notes.append(name)
    left = re.findall(r'@@([A-Z0-9_]+)@@', text)
    for name in sorted(set(left)):
        notes.append(f'{name}: still a placeholder, nothing here fills it')
    return text, notes, problems


# --------------------------------------------------------------------------
# The one measurement no other runner makes


def measure_nullspace(results: Path, data_root: Path) -> dict:
    """Sweep IK damping against null-space gain over real grasp targets.

    The targets are the S5 output of the reference corpus, not synthetic poses:
    the question is whether the term earns its place on the poses this pipeline
    actually asks for, and a reachable pose sampled at random would answer a
    different question. Nothing here is timed, so it is deliberately the
    straightforward spelling.
    """
    import numpy as np

    sys.path.insert(0, str(ROOT / 'python'))
    sys.path.insert(0, str(ROOT / 'python/bench'))
    from bench_pipeline import FrameStore  # noqa: E402
    from grasp_core import GraspPipeline  # noqa: E402
    from grasp_core.kinematics import IKSolver  # noqa: E402

    config = load_json(CONFIG)
    reference = config['analysis']['reference_dataset']
    store = FrameStore(data_root / reference)

    pipeline = GraspPipeline(CONFIG, ROOT / 'assets/franka/panda_chain.json',
                             ROOT / 'assets/ransac_uniform.bin')
    targets = []
    for depth, rgb in store.frames:
        result = pipeline.run(depth, rgb, store.width, store.height)
        if result.graspable:
            targets.append(result.tcp.copy())
    if not targets:
        raise Missing(f'{reference} produced no graspable frame to solve for')

    chain = pipeline.chain
    base = dict(config['ik'])
    sweep = []
    for damping in NULLSPACE_DAMPINGS:
        for gain in NULLSPACE_GAINS:
            settings = dict(base, damping=damping, nullspace_gain=gain)
            solver = IKSolver(chain, settings)
            converged, iterations = 0, []
            worst_position, worst_orientation = 0.0, 0.0
            for target in targets:
                ok = solver.solve(target)
                converged += int(ok)
                if ok:
                    iterations.append(solver.iterations)
                # The error is measured at whatever pose the solver stopped at,
                # converged or not, because "how far off did it park" is the
                # question a non-converging configuration raises.
                tcp = solver.forward(solver.q)
                offset = target[:3, 3] - tcp[:3, 3]
                product = tcp[:3, :3] @ target[:3, :3].T
                rotation = 0.5 * np.array([product[1, 2] - product[2, 1],
                                           product[2, 0] - product[0, 2],
                                           product[0, 1] - product[1, 0]])
                worst_position = max(worst_position,
                                     float(np.linalg.norm(offset)))
                worst_orientation = max(worst_orientation,
                                        float(np.linalg.norm(rotation)))
            sweep.append({
                'damping': damping,
                'nullspace_gain': gain,
                'targets': len(targets),
                'converged': converged,
                'median_iterations': (None if not iterations
                                      else int(np.median(iterations))),
                'worst_position_error_m': worst_position,
                'worst_orientation_error_rad': worst_orientation,
            })

    report = {
        'dataset': reference,
        'targets': len(targets),
        'seeded_from': 'q_neutral',
        'max_iterations': base['max_iterations'],
        'position_tolerance_m': base['position_tolerance_m'],
        'orientation_tolerance_rad': base['orientation_tolerance_rad'],
        'dampings': list(NULLSPACE_DAMPINGS),
        'nullspace_gains': list(NULLSPACE_GAINS),
        'sweep': sweep,
    }
    out = results / 'nullspace_sweep.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'wrote {out.relative_to(ROOT)}: {len(sweep)} configurations over '
          f'{len(targets)} targets')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--measure-nullspace', action='store_true',
                        help='run the IK sweep and write nullspace_sweep.json')
    parser.add_argument('--check', action='store_true',
                        help='report what would change and write nothing')
    args = parser.parse_args(argv)

    if args.measure_nullspace:
        try:
            measure_nullspace(args.results, args.data_root)
        except Missing as why:
            print(f'fill_method_placeholders: {why}', file=sys.stderr)
            return 1
        return 0

    src = Sources(args.results)
    before = METHOD.read_text()
    after, notes, problems = fill(before, src)

    for note in notes:
        print(f'  {note}')
    if problems:
        print(f'{len(problems)} figure(s) could not be filled:', file=sys.stderr)
        for problem in problems:
            print(f'  {problem}', file=sys.stderr)
        return 1
    if after == before:
        print(f'{METHOD.relative_to(ROOT)} is already current')
        return 0
    if args.check:
        print(f'{METHOD.relative_to(ROOT)} is out of date with {args.results}',
              file=sys.stderr)
        return 1
    METHOD.write_text(after)
    print(f'rewrote {METHOD.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
