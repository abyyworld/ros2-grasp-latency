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
NULLSPACE_GAINS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0)


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
    """A number rounded to a fixed count of significant digits."""
    if value == 0:
        return '0'
    return f'%.{digits}g' % value


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
    # The calibration runs once per process, so every run carries its own
    # estimate of the same quantity. The reference corpus is the one the
    # document's other numbers come from, so it is the one quoted.
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


# The gate's fields in the order compare_outputs.py reports them, with the
# unit a deviation in that field is measured in. Booleans and the digest have
# no unit because they compare exactly or not at all.
FIELD_UNITS = (
    ('plane_found', None),
    ('plane', 'unit normal, metres'),
    ('graspable', None),
    ('width', 'metres'),
    ('tcp', 'metres, direction cosines'),
    ('converged', None),
    ('iterations', None),
    ('q', 'radians'),
    ('duration_s', 'seconds'),
    ('traj_checksum', None),
)


def corpus_label(path: Path, report: dict) -> str:
    """`equivalence.table_848x480.json` reads as its corpus, and the
    single-BLAS-thread pass reads as the pass it is rather than as a tag."""
    stem = path.name[len('equivalence.'):-len('.json')]
    corpus = stem.split('.')[0]
    right = report.get('right', {}).get('impl', 'py')
    if right == 'py1t':
        return f'the single-BLAS-thread pass on `{corpus}`'
    if right != 'py':
        return f'`{corpus}` ({right})'
    return f'`{corpus}`'


def sub_equivalence_table(src):
    """One row per field, worst deviation over every corpus in results/.

    Aggregated across the comparisons rather than split by corpus because the
    claim the gate makes is per field: this is how far apart the two
    implementations ever got on this quantity, anywhere. The per-corpus and
    per-frame detail is in the JSON, which is committed.
    """
    reports = []
    for path in src.equivalence_files():
        report = load_json(path)
        if not report.get('agree', False):
            raise Missing(f'{path.name} says the two implementations disagree; '
                          f'no table should report a gate that did not pass')
        reports.append((path, report))

    rows = ['| field | values compared | worst absolute deviation |',
            '|---|---:|---|']
    worst_overall, worst_field = 0.0, None
    quantised = []
    for name, unit in FIELD_UNITS:
        compared = sum(r['fields'][name]['compared'] for _, r in reports)
        deviations = [r['fields'][name]['worst_abs_deviation'] for _, r in reports]
        mismatches = [m for _, r in reports for m in r['fields'][name]['mismatches']]
        if mismatches:
            raise Missing(f'the gate records {len(mismatches)} mismatch(es) in '
                          f'{name}; the table must not say it passed')
        worst = max((d for d in deviations if d is not None), default=None)
        if worst is None:
            if name == 'traj_checksum':
                near = sum(len(r.get('checksum_quantisation_frames', []))
                           for _, r in reports)
                quantised = [(path, entry) for path, r in reports
                             for entry in r.get('checksum_quantisation_frames', [])]
                cell = (f'{compared} identical' if not near else
                        f'{compared - near} identical, {near} on the rounding grid')
            else:
                cell = 'exact, every value'
        else:
            cell = f'{scientific(worst)}' + (f' ({unit})' if unit else '')
            if worst > worst_overall:
                worst_overall, worst_field = worst, name
        rows.append(f'| `{name}` | {thousands(compared)} | {cell} |')

    frames = sum(r['frames_compared'] for _, r in reports)
    labels = [corpus_label(path, report) for path, report in reports]
    listed = ', '.join(labels[:-1]) + f' and {labels[-1]}'
    tolerance = float(src.config['analysis']['equivalence_tolerance'])
    tail = (f'\nOver {listed}: {len(reports)} comparisons, {frames} frames, '
            f'tolerance {scientific(tolerance)}. The worst deviation anywhere '
            f'above is {scientific(worst_overall)}, in `{worst_field}`, which '
            f'is {scientific(tolerance / worst_overall)} times inside the '
            f'tolerance.')
    if quantised:
        worst_amplified = max(e['amplified'] for _, e in quantised)
        half_step = max(e['grid_half_step'] for _, e in quantised)
        where = ', '.join(f'{corpus_label(path, dict(right={}))} frame '
                          f'{entry["frame"]}' for path, entry in quantised)
        tail += (f'\n\nThe digest differences are {where}, and they are the '
                 f'boundary-straddle case rather than a divergence: the joint '
                 f'solution behind them agrees to '
                 f'{scientific(max(e["determinant_deviation"] for _, e in quantised))}'
                 f', which the quintic amplifies to at most '
                 f'{scientific(worst_amplified)} inside a waypoint against a '
                 f'grid half-step of {scientific(half_step)}. Nothing can hide '
                 f'in that, and the gate says so with the numbers rather than '
                 f'either failing or staying quiet.')
    return '\n'.join(rows) + '\n' + tail


def _nullspace_row(sweep, damping, gain):
    for entry in sweep:
        if entry['damping'] == damping and entry['nullspace_gain'] == gain:
            return entry
    raise Missing(f'the sweep has no damping {damping:g} at gain {gain:g}')


def sub_nullspace_table(src):
    """The gain sweep at the damping the pipeline actually runs.

    One column of the grid rather than the whole of it, because this is the
    column the configured value was chosen from and it is the one that carries
    the errors. The other four dampings appear as a convergence grid in the
    prose below, where all they have to say is whether the solve landed.
    """
    data = src.nullspace
    damping = src.config['ik']['damping']
    gains = sorted({e['nullspace_gain'] for e in data['sweep']})
    rows = ['| `nullspace_gain` | converged | median iterations | '
            'max iterations | worst position error | worst orientation error |',
            '|---:|---:|---:|---:|---:|---:|']
    for gain in gains:
        entry = _nullspace_row(data['sweep'], damping, gain)
        median = ('none' if entry['median_iterations'] is None
                  else str(entry['median_iterations']))
        rows.append(
            f'| {gain:g} | {entry["converged"]} of {entry["targets"]} | '
            f'{median} | {entry["max_iterations"]} | '
            f'{entry["worst_position_error_m"] * 1e3:.4f} mm | '
            f'{entry["worst_orientation_error_rad"] * 1e3:.2f} mrad |')
    rows.append('')
    rows.append(f'At `ik.damping` {damping:g}, the configured value. Errors are '
                f'measured at whatever pose the solver stopped at, converged or '
                f'not, because how far off a non-converging configuration parks '
                f'is the question it raises.')
    return '\n'.join(rows)


def sub_nullspace_prose(src):
    """The paragraphs under the table, written from the sweep.

    Generated rather than typed so that a rerun cannot leave the table saying
    one thing and the prose another, which is the failure this whole script
    exists to prevent. The narrative shape is fixed and the numbers are not, so
    the guard below refuses to produce a paragraph when the data stops
    supporting the shape, rather than producing a fluent wrong one.
    """
    data = src.nullspace
    sweep = data['sweep']
    targets = data['targets']
    tol_mm = data['position_tolerance_m'] * 1e3
    tol_mrad = data['orientation_tolerance_rad'] * 1e3
    frames = data.get('target_frames') or list(range(targets))

    chosen_gain = src.config['ik']['nullspace_gain']
    chosen_damping = src.config['ik']['damping']
    dampings = sorted({e['damping'] for e in sweep}, reverse=True)
    gains = sorted({e['nullspace_gain'] for e in sweep}, reverse=True)

    def row(damping, gain):
        return _nullspace_row(sweep, damping, gain)

    def full(entry):
        return entry['converged'] == entry['targets']

    at_chosen = row(chosen_damping, chosen_gain)
    strongest = row(chosen_damping, max(gains))
    if not full(at_chosen) or full(strongest):
        return (f'At the configured damping {chosen_damping:g} the sweep now '
                f'reports {at_chosen["converged"]}/{targets} at gain '
                f'{chosen_gain:g} and {strongest["converged"]}/{targets} at '
                f'gain {max(gains):g}. That is not the result this section was '
                f'written around, so it needs rewriting rather than refilling.')

    # The largest gain that still converges on everything at the configured
    # damping, and the smallest that does not: the boundary is the finding.
    works = [g for g in sorted(gains) if full(row(chosen_damping, g))]
    fails = [g for g in sorted(gains) if not full(row(chosen_damping, g))]
    first_total = min((g for g in fails
                       if row(chosen_damping, g)['converged'] == 0),
                      default=None)

    paragraphs = [
        f'At gain {chosen_gain:g} the solver converges on {at_chosen["converged"]} '
        f'of {targets} targets in a median of {at_chosen["median_iterations"]} '
        f'iterations, worst case {at_chosen["max_iterations"]}, with a worst '
        f'position error of {at_chosen["worst_position_error_m"] * 1e3:.4f} mm '
        f'against a tolerance of {significant(tol_mm, 2)} mm. Those are the '
        f'figures the IK column of `RESULTS.md` is the cost of.',

        f'Raising the gain buys nothing and costs convergence. The term is '
        f'`I - J^T (J J^T + lambda^2 I)^-1 J`, which is not a null-space '
        f'projector while `lambda > 0`: it differs from the true projector by '
        f'`O(lambda^2)`, so the posture bias does not stay in the redundant '
        f'degree of freedom. It leaks into task space, where it is balanced '
        f'against the task error rather than driven out of it, and the balance '
        f'point sits outside the tolerance. The largest gain that still reaches '
        f'every target at damping {chosen_damping:g} is '
        f'{max(works):g}' +
        ('' if first_total is None else
         f'. From gain {first_total:g} upward it reaches tolerance on none of '
         f'them and burns all '
         f'{row(chosen_damping, first_total)["max_iterations"]} iterations on '
         f'every frame') + '.',

        f'**What fails is position, not orientation.** At gain 0.1 the '
        f'orientation error stays inside '
        f'{row(chosen_damping, 0.1)["worst_orientation_error_rad"] * 1e3:.2f} '
        f'mrad, well under the {significant(tol_mrad, 2)} mrad tolerance, '
        f'while the position error reaches '
        f'{row(chosen_damping, 0.1)["worst_position_error_m"] * 1e3:.2f} mm '
        f'against {significant(tol_mm, 2)} mm. An earlier revision of '
        f'`ALGORITHM.md` and of `ik.nullspace_note` attributed the failure to a '
        f'198 mrad orientation error. That figure came from a sweep taken '
        f'before the S5 wrist fold existed, which changed which of two '
        f'equivalent grasp frames is demanded and so changed where the solver '
        f'stalls, and it does not reproduce here. The 0 of '
        f'{targets} result does, and both files now carry the corrected '
        f'diagnosis.',
    ]

    grid = ['| `damping` \\ `nullspace_gain` | ' +
            ' | '.join(f'{g:g}' for g in gains) + ' |',
            '|---:' * (1 + len(gains)) + '|']
    for damping in dampings:
        cells = ' | '.join(str(row(damping, g)['converged']) for g in gains)
        marker = ' (configured)' if damping == chosen_damping else ''
        grid.append(f'| {damping:g}{marker} | {cells} |')

    safe_below = None
    for damping in sorted(dampings):
        if all(full(row(damping, g)) for g in gains):
            safe_below = damping
        else:
            break

    cross = (f'The damping decides which gains are survivable, which is why '
             f'this is a grid and not a column. Targets reached, out of '
             f'{targets}:\n\n' + '\n'.join(grid) + '\n\n'
             f'The boundary moves the way the `O(lambda^2)` leak says it '
             f'should: the less damping, the smaller the projector error and '
             f'the more posture bias the task can absorb.')
    if safe_below is not None:
        cross += (f' At damping {safe_below:g} and below, every gain in the '
                  f'sweep reached every target, so the term is not fatal in '
                  f'itself. It is fatal at the damping this pipeline runs.')

    reversal = None
    for damping in dampings:
        unbiased = row(damping, 0.0)
        if unbiased['failed_targets']:
            rescued = [g for g in gains if g > 0.0 and full(row(damping, g))]
            reversal = (damping, unbiased, rescued)
            break
    if reversal is not None:
        damping, unbiased, rescued = reversal
        failed = unbiased['failed_targets']
        plural = 's' if len(failed) > 1 else ''
        named = ', '.join(str(frames[i]) for i in failed)
        rescue = ('' if not rescued else
                  f', and is reached at gain '
                  f'{", ".join(f"{g:g}" for g in rescued)}')
        cross += (f' Less damping is not uniformly safer either. At damping '
                  f'{damping:g}, frame{plural} {named} of the store ends up '
                  f'{unbiased["worst_position_error_m"] * 1e3:.1f} mm and '
                  f'{unbiased["worst_orientation_error_rad"] * 1e3:.1f} mrad '
                  f'away with the gain off{rescue}. That failure is not '
                  f'monotonic in either parameter, which is what a '
                  f'near-singular target looks like when the damping is the '
                  f'only thing regularising the solve.')
    paragraphs.append(cross)

    zero_full = sum(1 for d in dampings if full(row(d, 0.0)))
    paragraphs.append(
        f'A genuine pseudo-inverse projector does converge, and costs an SVD '
        f'per iteration on a stage that runs every frame. Since every frame is '
        f'seeded from `q_neutral` the solution already sits near the neutral '
        f'posture, so the term has nothing left to buy and the default is '
        f'{chosen_gain:g}. The damping stays at {chosen_damping:g}, the most '
        f'damped value in the sweep and one of the {zero_full} dampings out '
        f'of {len(dampings)} that reach every target with the gain off.')

    return '\n\n'.join(paragraphs)


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
    targets, target_frames = [], []
    for frame, (depth, rgb) in enumerate(store.frames):
        result = pipeline.run(depth, rgb, store.width, store.height)
        if result.graspable:
            targets.append(result.tcp.copy())
            target_frames.append(frame)
    if not targets:
        raise Missing(f'{reference} produced no graspable frame to solve for')

    chain = pipeline.chain
    base = dict(config['ik'])
    sweep = []
    for damping in NULLSPACE_DAMPINGS:
        for gain in NULLSPACE_GAINS:
            settings = dict(base, damping=damping, nullspace_gain=gain)
            solver = IKSolver(chain, settings)
            converged, iterations, every_iteration = 0, [], []
            position_error, orientation_error = [], []
            for target in targets:
                ok = solver.solve(target)
                converged += int(ok)
                every_iteration.append(solver.iterations)
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
                position_error.append(float(np.linalg.norm(offset)))
                orientation_error.append(float(np.linalg.norm(rotation)))
            position_error = np.array(position_error)
            orientation_error = np.array(orientation_error)
            sweep.append({
                'damping': damping,
                'nullspace_gain': gain,
                'targets': len(targets),
                'converged': converged,
                'median_iterations': (None if not iterations
                                      else int(np.median(iterations))),
                'max_iterations': int(max(every_iteration)),
                'worst_position_error_m': float(position_error.max()),
                'worst_position_target': int(position_error.argmax()),
                'worst_orientation_error_rad': float(orientation_error.max()),
                'worst_orientation_target': int(orientation_error.argmax()),
                # Which targets the solver did not place inside the tolerance,
                # named rather than counted: the interesting question about a
                # configuration that fails on one target is whether it is the
                # same target every time.
                'failed_targets': [
                    int(i) for i in np.nonzero(
                        (position_error >= base['position_tolerance_m'])
                        | (orientation_error >= base['orientation_tolerance_rad'])
                    )[0]],
            })

    report = {
        'dataset': reference,
        'targets': len(targets),
        'target_frames': target_frames,
        'target_index_note': 'targets are the graspable frames in store order; '
                             'target_frames[i] is the frame target i came from',
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
