"""The two implementations on frames that do not produce a grasp.

Every committed corpus runs to completion on every frame, so until this file
nothing in the repository exercised what happens when a frame gives up early:
no plane, no cluster, or a cluster the gripper cannot span. That is where the
two implementations last disagreed. `cpp/src/pipeline.cpp` ran S6 and S7
whenever the cluster was non-empty, while the Python gated both on `graspable`,
which is the gripper-width verdict, so a frame holding a too-wide object timed
a full pipeline in C++ and a truncated one in Python. No frame of any committed
corpus is affected, and the equivalence gate could not have caught it either:
it stops comparing at the bail-out, which is exactly where the difference sat.

So this file compares the two implementations *past* the bail-out, field by
field with no gate, and asserts they gave up in the same place.

The scenes are rendered from the camera model in the config, as in
`test_pipeline.py`, rather than read from `data/`: a checkout that has never
generated a corpus still runs them. They are written out as a frame store
because the C++ side is reached through `cpp/build/bench_pipeline`, which reads
one. Those tests skip when the binary has not been built; the in-process tests
below do not need it.
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import CHAIN_PATH, CONFIG_PATH, RANSAC_PATH, ROOT

CPP_BENCH = ROOT / 'cpp' / 'build' / 'bench_pipeline'
PY_BENCH = ROOT / 'python' / 'bench' / 'bench_pipeline.py'

# Half-extents in metres. The narrow box is the one `test_pipeline.py` grasps;
# the wide one is 0.10 m across its short side, which no `max_width_m` in the
# config comes near, so it reaches S5 and fails there on any config.
NARROW_BOX = (0.02, 0.035)
WIDE_BOX = (0.05, 0.06)

# Jaws too small for either box, used to move the narrow box's verdict without
# touching the scene. The point of the run is that only the verdict moves.
NARROW_JAWS_MAX_WIDTH_M = 0.03

# The scenes, in store order, with the stage each is built to give up at.
# `plane` is a frame of zero depth, which is what a sensor returns when it sees
# nothing in range: every pixel falls outside [z_min_m, z_max_m], so S3 has
# fewer than 3 points and S4 has none.
SCENES = ('none', 'grasp', 'cluster', 'plane')


def write_config(path, overrides):
    """The shipped config with `grasp` entries replaced, written to `path`."""
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    config['grasp'].update(overrides)
    path.write_text(json.dumps(config), encoding='utf-8')
    return path


def build_store(directory, config):
    """A four-frame store on disk, one frame per bail-out point."""
    from conftest import _render_box_on_table

    directory.mkdir(parents=True, exist_ok=True)
    scenes = [
        _render_box_on_table(config, half=NARROW_BOX),
        _render_box_on_table(config, half=WIDE_BOX),
        # A table with nothing on it: S3 removes every point, so S4 has no
        # cluster to return.
        _render_box_on_table(config, height_m=0.0),
    ]
    width, height = scenes[0]['width'], scenes[0]['height']
    scenes.append({
        'depth': bytes(2 * width * height),
        'rgb': bytes(3 * width * height),
        'width': width,
        'height': height,
    })

    frames = []
    for index, scene in enumerate(scenes):
        names = (f'{index:06d}.depth.bin', f'{index:06d}.rgb.bin')
        (directory / names[0]).write_bytes(scene['depth'])
        (directory / names[1]).write_bytes(scene['rgb'])
        frames.append({'id': index, 'depth': names[0], 'rgb': names[1]})
    (directory / 'manifest.json').write_text(json.dumps({
        'name': 'bailout_scenes',
        'width': width,
        'height': height,
        'depth_scale_m': config['camera']['depth_scale_m'],
        'frame_count': len(frames),
        'frames': frames,
    }), encoding='utf-8')
    return directory


def run_bench(command, dataset, config_path, out_dir, impl):
    """One implementation over the store, through its own benchmark runner."""
    output = out_dir / f'{impl}.output.jsonl'
    subprocess.run(
        command + ['--dataset', str(dataset),
                   '--out-timing', str(out_dir / f'{impl}.timing.jsonl'),
                   '--out-output', str(output),
                   '--config', str(config_path),
                   '--chain', str(CHAIN_PATH),
                   '--ransac-table', str(RANSAC_PATH),
                   '--impl', impl,
                   '--warmup', '0', '--frames', str(len(SCENES))],
        check=True, capture_output=True, cwd=ROOT)
    records = [json.loads(line) for line in
               output.read_text(encoding='utf-8').splitlines() if line.strip()]
    return {record['frame']: record for record in records}


def bail_point(record):
    """The first stage this frame gave up at, or 'none'.

    Ordered by stage. `cluster` is read from `cluster_points` when the record
    carries it; otherwise from the all-zero pose ALGORITHM.md S5 specifies for
    a frame with no cluster, which is the only other thing that distinguishes
    it from a frame whose cluster was too wide.
    """
    if not record['plane_found']:
        return 'plane'
    if 'cluster_points' in record:
        empty = record['cluster_points'] == 0
    else:
        empty = not any(record['tcp'])
    if empty:
        return 'cluster'
    if not record['graspable']:
        return 'grasp'
    if not record['converged']:
        return 'ik'
    return 'none'


@pytest.fixture(scope='module')
def store(tmp_path_factory):
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    return build_store(tmp_path_factory.mktemp('bailout_store'), config)


@pytest.fixture(scope='module', params=['shipped', 'narrow_jaws'])
def paired_run(request, store, tmp_path_factory):
    """Both implementations over the same store under the same config.

    Parametrised over two gripper widths because the second is what the C++
    behaviour makes visible: with 3 cm jaws the narrow box stops being
    graspable while the frame stays exactly as long, and both implementations
    have to agree on everything downstream of a verdict neither of them acts
    on.
    """
    if not CPP_BENCH.exists():
        pytest.skip(f'{CPP_BENCH} is not built; nothing to compare against')
    out_dir = tmp_path_factory.mktemp(f'bailout_{request.param}')
    overrides = ({} if request.param == 'shipped'
                 else {'max_width_m': NARROW_JAWS_MAX_WIDTH_M})
    config_path = write_config(out_dir / 'pipeline_config.json', overrides)
    cpp = run_bench([str(CPP_BENCH)], store, config_path, out_dir, 'cpp')
    py = run_bench([sys.executable, str(PY_BENCH)], store, config_path,
                   out_dir, 'py')
    return request.param, cpp, py


def test_the_run_covers_every_bail_out_point(paired_run):
    """A parity check over frames that all bail the same way proves little."""
    name, cpp, _ = paired_run
    points = [bail_point(cpp[frame]) for frame in sorted(cpp)]
    expected = list(SCENES) if name == 'shipped' \
        else ['grasp', 'grasp', 'cluster', 'plane']
    assert points == expected


def test_both_implementations_bail_in_the_same_place(paired_run):
    _, cpp, py = paired_run
    assert sorted(cpp) == sorted(py)
    for frame in sorted(cpp):
        assert bail_point(cpp[frame]) == bail_point(py[frame]), \
            f'frame {frame}'


def test_every_field_agrees_past_the_bail_out(paired_run, config):
    """No gate: a bailed frame's fields are compared like any other frame's.

    `harness/compare_outputs.py` stops at the bail-out by design, so it is the
    wrong instrument for the question this file asks. Fields are compared by
    type, which keeps this independent of how the runners encode the waypoint
    block: a digest is a string and compares exactly, a block of doubles
    compares to the same tolerance as every other double.
    """
    tolerance = config['analysis']['equivalence_tolerance']
    _, cpp, py = paired_run
    for frame in sorted(cpp):
        left, right = dict(cpp[frame]), dict(py[frame])
        # The one field that is meant to differ: it names the implementation.
        left.pop('impl'), right.pop('impl')
        assert set(left) == set(right), f'frame {frame}: field sets differ'
        for field, value in left.items():
            other = right[field]
            if isinstance(value, (bool, int, str)):
                assert value == other, f'frame {frame}: {field}'
                continue
            worst = np.abs(np.subtract(value, other, dtype=np.float64)).max()
            assert worst <= tolerance, \
                f'frame {frame}: {field} off by {worst:.3e}'


def test_the_width_verdict_changes_the_verdict_and_nothing_else(tmp_path,
                                                                config):
    """S6 and S7 do not know what `graspable` says, and must not.

    This is the divergence itself, in one assertion: the same frame under two
    gripper widths. Everything the solver and the planner produce has to be
    identical, because the width test is a statement about the gripper and not
    a branch in the pipeline (ALGORITHM.md S5, S6).
    """
    from conftest import _render_box_on_table
    from grasp_core import GraspPipeline

    scene = _render_box_on_table(config, half=NARROW_BOX)
    answers = {}
    for name, overrides in (('shipped', {}),
                            ('narrow_jaws',
                             {'max_width_m': NARROW_JAWS_MAX_WIDTH_M})):
        path = write_config(tmp_path / f'{name}.json', overrides)
        pipeline = GraspPipeline(str(path), str(CHAIN_PATH), str(RANSAC_PATH))
        result = pipeline.run(scene['depth'], scene['rgb'],
                              scene['width'], scene['height'])
        answers[name] = (result.graspable, result.width, result.tcp.copy(),
                         result.q.copy(), result.iterations, result.converged,
                         result.duration_s, result.positions.copy())

    assert answers['shipped'][0] is True
    assert answers['narrow_jaws'][0] is False
    # Same width measured, same pose demanded, same joints reached, same plan.
    for index in range(1, len(answers['shipped'])):
        left, right = answers['shipped'][index], answers['narrow_jaws'][index]
        assert np.array_equal(left, right), f'field {index}'
    assert answers['narrow_jaws'][4] > 0, 'S6 did not run'
    assert answers['narrow_jaws'][6] > 0.0, 'S7 did not run'


def test_a_frame_the_gripper_cannot_span_is_still_timed_end_to_end(tmp_path,
                                                                   config):
    """The stage clocks are the reason the gate above matters."""
    from conftest import _render_box_on_table
    from grasp_core import GraspPipeline

    path = write_config(tmp_path / 'narrow_jaws.json',
                        {'max_width_m': NARROW_JAWS_MAX_WIDTH_M})
    pipeline = GraspPipeline(str(path), str(CHAIN_PATH), str(RANSAC_PATH))
    scene = _render_box_on_table(config, half=NARROW_BOX)
    result = pipeline.run(scene['depth'], scene['rgb'],
                          scene['width'], scene['height'])
    assert not result.graspable
    assert result.stage_ns['ik'] > 0
    assert result.stage_ns['traj'] > 0
    assert result.total_ns == sum(result.stage_ns.values())


def test_no_cluster_reports_the_empty_answer_and_not_the_last_one(pipeline,
                                                                  config,
                                                                  chain):
    """ALGORITHM.md S5 to S7 for a frame with nothing on the table.

    The graspable frame runs first on purpose: every buffer then holds a real
    pose, a real joint vector and a real plan, and the empty frame has to
    report none of them. A reused buffer that is never cleared is how a stale
    answer gets published as a current one.
    """
    from conftest import _render_box_on_table

    filled = _render_box_on_table(config, half=NARROW_BOX)
    before = pipeline.run(filled['depth'], filled['rgb'],
                          filled['width'], filled['height'])
    assert before.graspable and before.cluster_points > 0

    bare = _render_box_on_table(config, height_m=0.0)
    result = pipeline.run(bare['depth'], bare['rgb'],
                          bare['width'], bare['height'])

    assert result.plane_found
    assert result.cluster_points == 0
    assert result.graspable is False
    assert result.width == 0.0
    assert not result.tcp.any()
    assert np.array_equal(result.q, chain.q_neutral)
    assert result.iterations == 0
    assert result.converged is False
    assert result.duration_s == 0.0
    assert not result.positions.any()
    assert not result.velocities.any()
    assert not result.accelerations.any()
    assert not result.times.any()


def test_the_pose_buffer_is_live_again_on_the_next_frame(pipeline, config):
    """The zero pose above is a swap, so check the swap goes both ways."""
    from conftest import _render_box_on_table

    bare = _render_box_on_table(config, height_m=0.0)
    filled = _render_box_on_table(config, half=NARROW_BOX)
    pipeline.run(bare['depth'], bare['rgb'], bare['width'], bare['height'])
    result = pipeline.run(filled['depth'], filled['rgb'],
                          filled['width'], filled['height'])
    assert result.graspable
    assert result.tcp[3, 3] == 1.0
    assert np.array_equal(result.tcp[:3, 2], config['grasp']['approach_axis_base'])
