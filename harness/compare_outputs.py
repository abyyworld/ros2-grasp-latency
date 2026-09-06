#!/usr/bin/env python3
"""The equivalence gate: do the two implementations compute the same thing?

A latency comparison between two programs that produce different answers is
worthless, so this runs before any number is quoted and CI runs it as a gate
rather than as a report.

It does not print "passed". It prints the worst deviation observed in every
field, because "the TCP pose agrees to 3e-13 over 100 frames" is evidence and
"passed" is an assertion. A run that squeaks under the tolerance at 9e-7 and a
run that sits at 1e-15 both exit zero, and only one of them should let you
sleep.

Bail-outs are compared to the point where they happened. A frame that found no
plane has no plane to compare, but the two implementations must still agree
that it found no plane, and agree at which stage they gave up. Disagreeing
about *where* a frame bailed is a mismatch even if every field that survives
the bail-out matches.

## The trajectory digest is a proxy, and proxies can lie

`traj_checksum` (docs/FORMATS.md) is a SHA-256 over every waypoint value
rounded to nine decimal places. Rounding is an *equality* test on a grid, and
the two implementations are only ever known to agree to a *tolerance*, so the
two can straddle a grid boundary and produce different digests while agreeing
to 1e-13. That is a false positive, and a gate with false positives is not a
gate. It was observed: `table_848x480` frame 95 carries a waypoint value 6.7e-15
from a boundary while the joint solutions that generate it agree to 5.0e-14.

The digest is not independent evidence. Every waypoint value is a closed-form
function of `q_neutral`, `q` and `duration_s` (ALGORITHM.md S7), and `q` and
`duration_s` are already compared numerically here. So when a digest differs,
this asks whether the values that determine it agree closely enough that no
real divergence could hide inside the grid. The amplification of `dq` into the
waypoint block is bounded by the quintic's own derivatives, computed from the
config rather than assumed; if `amplification * dq` is under half a grid step,
a digest difference can only be a boundary straddle and is reported as one
rather than failing the run. If it is not, the digest fails the run.

Usage:
  compare_outputs.py A.output.jsonl B.output.jsonl [--json report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'assets' / 'pipeline_config.json'

# Which gate has to be open before a field means anything. docs/ALGORITHM.md
# S3/S5/S6: the plane vector is only defined once a plane was found, the pose
# and width only once the cluster was judged graspable, and everything from IK
# onward rides on that same graspable gate because a non-graspable frame never
# calls the solver. `converged` itself is always defined: false is an answer.
ALWAYS = ()
FIELD_GATES = {
    'plane_found': ALWAYS,
    'plane': ('plane_found',),
    'graspable': ALWAYS,
    'width': ('graspable',),
    'tcp': ('graspable',),
    'converged': ALWAYS,
    'iterations': ('graspable',),
    'q': ('graspable',),
    'duration_s': ('graspable',),
    'traj_checksum': ('graspable',),
}

FLOAT_FIELDS = ('plane', 'width', 'tcp', 'q', 'duration_s')
EXACT_FIELDS = ('plane_found', 'graspable', 'converged', 'iterations',
                'traj_checksum')

# Reported in this order, which is the order the pipeline produces them.
FIELD_ORDER = ('plane_found', 'plane', 'graspable', 'width', 'tcp',
               'converged', 'iterations', 'q', 'duration_s', 'traj_checksum')

UNITS = {
    'plane': 'unit normal and metres',
    'width': 'metres',
    'tcp': 'metres and direction cosines',
    'q': 'radians',
    'duration_s': 'seconds',
}

# The fields that fully determine the waypoint block, per ALGORITHM.md S7.
CHECKSUM_DETERMINANTS = ('q', 'duration_s')

# docs/FORMATS.md rounds every waypoint value to this many decimals before
# hashing, so this is the digest's grid step.
CHECKSUM_DECIMALS = 9


def waypoint_amplification(config: dict) -> float:
    """How far a joint-angle disagreement can be magnified inside a waypoint.

    S7 writes position, velocity and acceleration as `h`, `hd` and `hdd` times
    `(q_goal - q_start)`, so a disagreement `dq` appears in the block scaled by
    at most `max(|h|, |hd|, |hdd|)`. Evaluated from the quintic itself over a
    dense grid of `s`, at the shortest duration the planner will emit, rather
    than quoting a constant somebody derived once by hand.
    """
    duration = config['trajectory']['min_duration_s']
    steps = 4096
    peak = 0.0
    for i in range(steps + 1):
        s = i / steps
        h = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        hd = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / duration
        hdd = (60 * s - 180 * s ** 2 + 120 * s ** 3) / duration ** 2
        peak = max(peak, abs(h), abs(hd), abs(hdd))
    # The time column is `s * duration`, so a duration disagreement enters it
    # unamplified and is already covered by the peak above, which exceeds 1.
    return peak


def bail_point(record: dict) -> str:
    """The first gate this frame failed, or 'none'.

    Ordered by stage, so the earliest failure is the one named. `converged` is
    last because a frame that ran IK and did not reach tolerance still produced
    a joint vector and a trajectory: it is a worse answer, not a shorter
    pipeline.
    """
    if not record['plane_found']:
        return 'plane'
    if not record['graspable']:
        return 'grasp'
    if not record['converged']:
        return 'ik'
    return 'none'


def load(path: Path) -> tuple[str, dict[int, dict]]:
    frames: dict[int, dict] = {}
    impl = None
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f'{path}:{number}: {error}') from error
            frame = record['frame']
            if frame in frames:
                raise SystemExit(f'{path}:{number}: frame {frame} appears twice')
            frames[frame] = record
            impl = record.get('impl', impl)
    if not frames:
        raise SystemExit(f'{path} holds no records')
    return impl or path.stem, frames


class Deviation:
    """Worst absolute disagreement seen in one field, and where."""

    __slots__ = ('field', 'compared', 'skipped', 'worst', 'worst_frame',
                 'worst_index', 'worst_pair', 'mismatches')

    def __init__(self, field: str):
        self.field = field
        self.compared = 0
        self.skipped = 0
        self.worst = 0.0
        self.worst_frame = -1
        self.worst_index = -1
        self.worst_pair = (0.0, 0.0)
        self.mismatches: list[str] = []

    def observe(self, frame: int, index: int, left, right) -> None:
        self.compared += 1
        delta = abs(float(left) - float(right))
        if not math.isfinite(delta):
            self.mismatches.append(
                f'frame {frame} index {index}: {left!r} against {right!r} '
                f'is not a finite difference')
            return
        if delta > self.worst:
            self.worst = delta
            self.worst_frame = frame
            self.worst_index = index
            self.worst_pair = (float(left), float(right))

    def observe_exact(self, frame: int, left, right) -> None:
        self.compared += 1
        if left != right:
            self.mismatches.append(
                f'frame {frame}: {left!r} against {right!r}')


def compare(left: dict[int, dict], right: dict[int, dict], tolerance: float,
            amplification: float):
    problems: list[str] = []

    left_frames, right_frames = set(left), set(right)
    if left_frames != right_frames:
        only_left = sorted(left_frames - right_frames)[:8]
        only_right = sorted(right_frames - left_frames)[:8]
        problems.append(
            f'frame sets differ: {len(left_frames)} against '
            f'{len(right_frames)} records, only in the first {only_left}, '
            f'only in the second {only_right}')

    frames = sorted(left_frames & right_frames)
    deviations = {name: Deviation(name) for name in FIELD_ORDER}
    bails: dict[str, int] = {}
    bail_disagreements: list[str] = []
    # Half a grid step: two values closer together than this cannot land more
    # than one step apart after rounding, so a digest difference between them
    # is a boundary straddle and nothing else.
    grid_half_step = 0.5 * 10.0 ** -CHECKSUM_DECIMALS
    quantisation: list[dict] = []

    for frame in frames:
        a, b = left[frame], right[frame]
        frame_worst = 0.0
        here, there = bail_point(a), bail_point(b)
        if here != there:
            bail_disagreements.append(
                f'frame {frame}: bailed at {here!r} against {there!r}')
        bails[here] = bails.get(here, 0) + 1

        for name in FIELD_ORDER:
            gates = FIELD_GATES[name]
            if any(not (a[gate] and b[gate]) for gate in gates):
                deviations[name].skipped += 1
                continue
            if name in EXACT_FIELDS:
                if name == 'traj_checksum' and a[name] != b[name]:
                    # Judged below, once every determinant has been measured.
                    continue
                deviations[name].observe_exact(frame, a[name], b[name])
                continue
            value_a, value_b = a[name], b[name]
            if isinstance(value_a, list):
                if len(value_a) != len(value_b):
                    problems.append(
                        f'frame {frame}: {name} has {len(value_a)} entries '
                        f'against {len(value_b)}')
                    continue
                for index, (x, y) in enumerate(zip(value_a, value_b)):
                    deviations[name].observe(frame, index, x, y)
                    if name in CHECKSUM_DETERMINANTS:
                        frame_worst = max(frame_worst, abs(float(x) - float(y)))
            else:
                deviations[name].observe(frame, -1, value_a, value_b)
                if name in CHECKSUM_DETERMINANTS:
                    frame_worst = max(frame_worst,
                                      abs(float(value_a) - float(value_b)))

        digest = deviations['traj_checksum']
        gates = FIELD_GATES['traj_checksum']
        if all(a[gate] and b[gate] for gate in gates) \
                and a['traj_checksum'] != b['traj_checksum']:
            digest.compared += 1
            reach = amplification * frame_worst
            if reach < grid_half_step:
                quantisation.append({
                    'frame': frame,
                    'determinant_deviation': frame_worst,
                    'amplified': reach,
                    'grid_half_step': grid_half_step,
                })
            else:
                digest.mismatches.append(
                    f'frame {frame}: {a["traj_checksum"][:16]}... against '
                    f'{b["traj_checksum"][:16]}..., and the values that '
                    f'determine it disagree by up to {frame_worst:.3e}, which '
                    f'reaches {reach:.3e} inside a waypoint against a grid '
                    f'half-step of {grid_half_step:.3e}')

    for name in FLOAT_FIELDS:
        deviation = deviations[name]
        if deviation.worst > tolerance:
            deviation.mismatches.append(
                f'frame {deviation.worst_frame} index {deviation.worst_index}: '
                f'{deviation.worst_pair[0]!r} against '
                f'{deviation.worst_pair[1]!r}, off by {deviation.worst:.3e}')

    return frames, deviations, bails, bail_disagreements, problems, quantisation


def render(name_a: str, name_b: str, frames, deviations, bails,
           bail_disagreements, problems, quantisation,
           tolerance: float) -> list[str]:
    lines = [
        f'{name_a} against {name_b}: {len(frames)} frames compared, '
        f'tolerance {tolerance:g}',
        '',
        f"  {'field':<14}{'compared':>10}{'skipped':>9}"
        f"{'worst deviation':>18}  where",
    ]
    for name in FIELD_ORDER:
        deviation = deviations[name]
        if name in EXACT_FIELDS:
            worst = 'exact' if not deviation.mismatches else 'DIFFERS'
            where = '' if not deviation.mismatches \
                else f'{len(deviation.mismatches)} frame(s)'
            if name == 'traj_checksum' and quantisation:
                note = f'{len(quantisation)} on the grid, see below'
                where = f'{where}, {note}' if where else note
                if not deviation.mismatches:
                    worst = 'on the grid'
        elif deviation.compared == 0:
            worst = 'not compared'
            where = ''
        else:
            worst = f'{deviation.worst:.3e}'
            where = f'frame {deviation.worst_frame}'
            if deviation.worst_index >= 0:
                where += f'[{deviation.worst_index}]'
            unit = UNITS.get(name)
            if unit:
                where += f', {unit}'
        lines.append(f'  {name:<14}{deviation.compared:>10}'
                     f'{deviation.skipped:>9}{worst:>18}  {where}')

    lines.append('')
    lines.append('  bail-out points: '
                 + ', '.join(f'{k}={v}' for k, v in sorted(bails.items())))

    if quantisation:
        worst = max(quantisation, key=lambda q: q['amplified'])
        lines += [
            '',
            f'  {len(quantisation)} frame(s) differ in traj_checksum while '
            f'agreeing numerically. docs/FORMATS.md rounds every waypoint to '
            f'{CHECKSUM_DECIMALS} decimals before hashing, which is an '
            f'equality test on a grid of '
            f'{10.0 ** -CHECKSUM_DECIMALS:g}, and two values either side of a '
            'grid boundary hash differently however close together they are.',
            f'  The joint solutions and durations that generate those '
            f'waypoints agree to at worst '
            f'{worst["determinant_deviation"]:.3e}, which reaches '
            f'{worst["amplified"]:.3e} inside a waypoint against a grid '
            f'half-step of {worst["grid_half_step"]:.3e}: '
            f'{worst["amplified"] / worst["grid_half_step"]:.1e} of one step. '
            'No divergence can hide in that, so these are reported and not '
            'failed.',
            '  Frames: ' + ', '.join(str(q['frame']) for q in quantisation[:16])
            + ('' if len(quantisation) <= 16 else ' ...'),
        ]

    failures = list(problems) + list(bail_disagreements)
    for name in FIELD_ORDER:
        failures.extend(f'{name}: {m}' for m in deviations[name].mismatches[:8])
    if failures:
        lines.append('')
        lines.append(f'  {len(failures)} disagreement(s):')
        lines.extend(f'    {failure}' for failure in failures)
    return lines


def main(argv=None) -> int:
    config = json.loads(DEFAULT_CONFIG.read_text())
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('left', type=Path, help='first *.output.jsonl')
    parser.add_argument('right', type=Path, help='second *.output.jsonl')
    parser.add_argument('--tolerance', type=float,
                        default=config['analysis']['equivalence_tolerance'],
                        help='absolute, in metres or radians')
    parser.add_argument('--json', type=Path, default=None,
                        help='also write the report as machine-readable JSON')
    parser.add_argument('--quiet', action='store_true',
                        help='print only on failure')
    args = parser.parse_args(argv)

    name_a, left = load(args.left)
    name_b, right = load(args.right)
    if name_a == name_b:
        print(f'compare_outputs: both files claim impl {name_a!r}; '
              f'comparing a run against itself proves nothing',
              file=sys.stderr)
        return 2

    amplification = waypoint_amplification(config)
    (frames, deviations, bails, bail_disagreements, problems,
     quantisation) = compare(left, right, args.tolerance, amplification)

    failed = bool(problems) or bool(bail_disagreements) or any(
        deviations[name].mismatches for name in FIELD_ORDER)

    lines = render(name_a, name_b, frames, deviations, bails,
                   bail_disagreements, problems, quantisation, args.tolerance)
    if failed or not args.quiet:
        stream = sys.stderr if failed else sys.stdout
        print('\n'.join(lines), file=stream)

    if args.json is not None:
        report = {
            'left': {'impl': name_a, 'path': str(args.left)},
            'right': {'impl': name_b, 'path': str(args.right)},
            'frames_compared': len(frames),
            'tolerance': args.tolerance,
            'agree': not failed,
            'bail_points': bails,
            'waypoint_amplification': amplification,
            'checksum_grid': 10.0 ** -CHECKSUM_DECIMALS,
            'checksum_quantisation_frames': quantisation,
            'fields': {
                name: {
                    'compared': deviations[name].compared,
                    'skipped': deviations[name].skipped,
                    'worst_abs_deviation':
                        None if name in EXACT_FIELDS else deviations[name].worst,
                    'worst_frame': deviations[name].worst_frame,
                    'mismatches': deviations[name].mismatches,
                }
                for name in FIELD_ORDER
            },
            'problems': problems + bail_disagreements,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n',
                             encoding='utf-8')

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
