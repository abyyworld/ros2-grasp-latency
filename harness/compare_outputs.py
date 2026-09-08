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

## What is compared, and where

`docs/ALGORITHM.md` decides, not either implementation. A field is compared on
every frame where the spec says what both implementations must hold, and
skipped only where the spec is silent and the record therefore carries whatever
its producer happened to leave in the slot. Two conditions do that gating:

* `plane_found`: S3 defines `(n, d)` only when it found a plane.
* a non-empty cluster: S4's cluster is what S5 needs, so with no cluster there
  is no pose and no width to compare.

`graspable` gates nothing. It is S5's verdict on the gripper width, and S6 and
S7 run whatever it says, so a frame too wide for the jaws still has a pose,
joints and a trajectory that both implementations must agree on. Gating them on
`graspable` would blind the gate on exactly the frames where two
implementations are most likely to have taken different paths: the earlier
version of this file did that, having read the gate off one implementation
rather than off the spec, and it could not have detected a divergence in S6 or
S7 on a non-graspable frame at all.

For the same reason the joints, the iteration count, the duration and the
trajectory are compared even on a frame with no cluster: the spec states what
they hold there (`q_neutral`, not converged, zero iterations, a zero trajectory
of zero duration), so a record that holds something else is a divergence and
not an undefined slot.

Bail-outs are compared to the point where they happened. A frame that found no
plane has no plane to compare, but the two implementations must still agree
that it found no plane, and agree at which stage they gave up. Disagreeing
about *where* a frame bailed is a mismatch even if every field that survives
the bail-out matches.

## The trajectory is compared, not hashed

`docs/FORMATS.md` used to carry a SHA-256 of the waypoint block rounded to nine
decimals. Rounding is an *equality* test on a grid, and the two implementations
are only ever known to agree to a *tolerance*, so two values 1e-13 apart either
side of a grid boundary hashed differently and failed the gate for no real
reason. It happened: `table_848x480` frame 95 carried a waypoint value 6.7e-15
from a boundary while the joint solutions generating it agreed to 5.0e-14.

Both runners now emit the whole block, so it is held to the same absolute
tolerance as the pose and the joints and reports a worst deviation like they
do. Because S7 scales a joint disagreement by the quintic's own derivatives,
this is the strictest row in the table rather than a proxy for the others: see
`waypoint_amplification`, which computes that factor from the config.

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

# The conditions under which docs/ALGORITHM.md defines a field's value. Named
# rather than inlined so the table below reads as the spec does.
CONDITIONS = {
    'plane_found': lambda record: bool(record['plane_found']),
    'cluster': lambda record: record['cluster_points'] > 0,
}

# Which conditions have to hold on *both* sides before a field means anything.
# An empty tuple is a field the spec defines on every frame.
FIELD_GATES = {
    'plane_found': (),
    'plane': ('plane_found',),
    'cluster_points': (),
    'graspable': (),
    'width': ('cluster',),
    'tcp': ('cluster',),
    'converged': (),
    'iterations': (),
    'q': (),
    'duration_s': (),
    'trajectory': (),
}

FLOAT_FIELDS = ('plane', 'width', 'tcp', 'q', 'duration_s', 'trajectory')
EXACT_FIELDS = ('plane_found', 'cluster_points', 'graspable', 'converged',
                'iterations')

# Reported in this order, which is the order the pipeline produces them.
FIELD_ORDER = ('plane_found', 'plane', 'cluster_points', 'graspable', 'width',
               'tcp', 'converged', 'iterations', 'q', 'duration_s',
               'trajectory')

UNITS = {
    'plane': 'unit normal and metres',
    'width': 'metres',
    'tcp': 'metres and direction cosines',
    'q': 'radians',
    'duration_s': 'seconds',
}

# Every stage that can come back negative, in the order the pipeline reaches
# them. Only 'cluster' shortens the pipeline; the other three are verdicts that
# the frame carries onward, and they are listed here because the two
# implementations must agree about them frame by frame.
VERDICTS = (
    ('plane', lambda record: bool(record['plane_found'])),
    ('cluster', lambda record: record['cluster_points'] > 0),
    ('grasp', lambda record: bool(record['graspable'])),
    ('ik', lambda record: bool(record['converged'])),
)


def waypoint_amplification(config: dict) -> float:
    """How far a joint-angle disagreement is magnified inside a waypoint.

    S7 writes position, velocity and acceleration as `h`, `hd` and `hdd` times
    `(q_goal - q_start)`, so a disagreement `dq` reaches the block scaled by at
    most `max(|h|, |hd|, |hdd|)`. Evaluated from the quintic itself over a dense
    grid of `s`, at the shortest duration the planner will emit, rather than
    quoting a constant somebody derived once by hand. Reported so a reader can
    see that the trajectory row fails at a `q` disagreement this many times
    smaller than the `q` row does.
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


def waypoint_label(index: int, dof: int) -> tuple[str, str]:
    """Name the flat trajectory index, so a deviation says where it landed.

    docs/FORMATS.md orders the block by waypoint: `dof` positions, then `dof`
    velocities, then `dof` accelerations, then `time_from_start`.
    """
    stride = 3 * dof + 1
    waypoint, offset = divmod(index, stride)
    if offset == stride - 1:
        return f'waypoint {waypoint} time_from_start', 'seconds'
    kind, joint = divmod(offset, dof)
    name = ('position', 'velocity', 'acceleration')[kind]
    unit = ('radians', 'radians per second',
            'radians per second squared')[kind]
    return f'waypoint {waypoint} {name} of joint {joint}', unit


def bail_point(record: dict) -> str:
    """The first stage whose verdict came back negative, or 'none'.

    Ordered by stage, so the earliest is the one named. `converged` is last
    because a frame that ran IK and did not reach tolerance still produced a
    joint vector and a trajectory: it is a worse answer, not a shorter
    pipeline.
    """
    for name, holds in VERDICTS:
        if not holds(record):
            return name
    return 'none'


def load(path: Path, block_length: int) -> tuple[str, dict[int, dict]]:
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
            # A file written by a runner that predates the current
            # docs/FORMATS.md would otherwise be compared on the fields it
            # happens to share, which is a gate that quietly stopped checking
            # the rest.
            missing = [name for name in FIELD_ORDER if name not in record]
            if missing:
                raise SystemExit(
                    f'{path}:{number}: no {", ".join(missing)} in this record. '
                    f'docs/FORMATS.md section 4 lists what a *.output.jsonl '
                    f'line carries; regenerate the file with the current '
                    f'runner')
            if len(record['trajectory']) != block_length:
                raise SystemExit(
                    f'{path}:{number}: trajectory holds '
                    f'{len(record["trajectory"])} values, and the config asks '
                    f'for {block_length}')
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


def compare(left: dict[int, dict], right: dict[int, dict], tolerance: float):
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

    for frame in frames:
        a, b = left[frame], right[frame]
        here, there = bail_point(a), bail_point(b)
        if here != there:
            bail_disagreements.append(
                f'frame {frame}: bailed at {here!r} against {there!r}')
        bails[here] = bails.get(here, 0) + 1

        for name in FIELD_ORDER:
            deviation = deviations[name]
            gates = FIELD_GATES[name]
            # Both sides have to have produced the field. One side alone would
            # be comparing a value against a slot nothing wrote.
            if any(not (CONDITIONS[gate](a) and CONDITIONS[gate](b))
                   for gate in gates):
                deviation.skipped += 1
                continue
            value_a, value_b = a[name], b[name]
            if name in EXACT_FIELDS:
                deviation.observe_exact(frame, value_a, value_b)
                continue
            if isinstance(value_a, list):
                if len(value_a) != len(value_b):
                    problems.append(
                        f'frame {frame}: {name} has {len(value_a)} entries '
                        f'against {len(value_b)}')
                    continue
                for index, (x, y) in enumerate(zip(value_a, value_b)):
                    deviation.observe(frame, index, x, y)
            else:
                deviation.observe(frame, -1, value_a, value_b)

    for name in FLOAT_FIELDS:
        deviation = deviations[name]
        if deviation.worst > tolerance:
            deviation.mismatches.append(
                f'frame {deviation.worst_frame} index {deviation.worst_index}: '
                f'{deviation.worst_pair[0]!r} against '
                f'{deviation.worst_pair[1]!r}, off by {deviation.worst:.3e}')

    return frames, deviations, bails, bail_disagreements, problems


def where(deviation: Deviation, dof: int) -> str:
    """The frame, index and unit a field's worst deviation landed on."""
    if deviation.worst_index < 0:
        text = f'frame {deviation.worst_frame}'
        unit = UNITS.get(deviation.field)
        return f'{text}, {unit}' if unit else text
    if deviation.field == 'trajectory':
        label, unit = waypoint_label(deviation.worst_index, dof)
        return f'frame {deviation.worst_frame} {label}, {unit}'
    text = f'frame {deviation.worst_frame}[{deviation.worst_index}]'
    unit = UNITS.get(deviation.field)
    return f'{text}, {unit}' if unit else text


def render(name_a: str, name_b: str, frames, deviations, bails,
           bail_disagreements, problems, tolerance: float,
           amplification: float, dof: int) -> list[str]:
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
            location = '' if not deviation.mismatches \
                else f'{len(deviation.mismatches)} frame(s)'
        elif deviation.compared == 0:
            worst = 'not compared'
            location = ''
        else:
            worst = f'{deviation.worst:.3e}'
            location = where(deviation, dof)
        lines.append(f'  {name:<14}{deviation.compared:>10}'
                     f'{deviation.skipped:>9}{worst:>18}  {location}')

    lines.append('')
    lines.append('  bail-out points: '
                 + ', '.join(f'{k}={v}' for k, v in sorted(bails.items())))
    lines.append(
        f'  S7 turns a joint disagreement dq into at most {amplification:.1f} '
        f'* dq inside the waypoint block, so the trajectory row above fails at '
        f'a q disagreement {amplification:.1f} times smaller than the q row '
        f'does.')

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

    trajectory = config['trajectory']
    dof = len(trajectory['joint_names'])
    block_length = trajectory['waypoints'] * (3 * dof + 1)

    name_a, left = load(args.left, block_length)
    name_b, right = load(args.right, block_length)
    if name_a == name_b:
        print(f'compare_outputs: both files claim impl {name_a!r}; '
              f'comparing a run against itself proves nothing',
              file=sys.stderr)
        return 2

    amplification = waypoint_amplification(config)
    (frames, deviations, bails, bail_disagreements,
     problems) = compare(left, right, args.tolerance)

    failed = bool(problems) or bool(bail_disagreements) or any(
        deviations[name].mismatches for name in FIELD_ORDER)

    lines = render(name_a, name_b, frames, deviations, bails,
                   bail_disagreements, problems, args.tolerance,
                   amplification, dof)
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
            'trajectory_block_length': block_length,
            'fields': {
                name: {
                    'compared': deviations[name].compared,
                    'skipped': deviations[name].skipped,
                    'worst_abs_deviation':
                        None if name in EXACT_FIELDS else deviations[name].worst,
                    'worst_frame': deviations[name].worst_frame,
                    'worst_index': deviations[name].worst_index,
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
