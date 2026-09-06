"""Put `python/` on the path and hand the tests the frozen assets.

CI collects `tests/` and `python/tests/` in one pytest invocation from the
repository root, so the import has to work without an installed package and
without a `PYTHONPATH` the caller has to remember.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

CONFIG_PATH = ROOT / 'assets' / 'pipeline_config.json'
CHAIN_PATH = ROOT / 'assets' / 'franka' / 'panda_chain.json'
RANSAC_PATH = ROOT / 'assets' / 'ransac_uniform.bin'


@pytest.fixture(scope='session')
def config():
    from grasp_core.config import load_json
    return load_json(CONFIG_PATH)


@pytest.fixture(scope='session')
def chain():
    from grasp_core.config import Chain, load_json
    return Chain(load_json(CHAIN_PATH))


@pytest.fixture(scope='session')
def deviates(config):
    from grasp_core.config import load_deviates
    return load_deviates(RANSAC_PATH, config['ransac_table']['count'])


@pytest.fixture(scope='session')
def pipeline():
    from grasp_core import GraspPipeline
    return GraspPipeline(str(CONFIG_PATH), str(CHAIN_PATH), str(RANSAC_PATH))


@pytest.fixture(scope='session')
def top_down_scene(config):
    """A synthetic top-down frame: one box on a level table.

    Built from the camera model rather than from `data/`, because the corpora
    are generated artefacts that a checkout does not have to carry, and a unit
    test should not depend on one having been generated.
    """
    return _render_box_on_table(config)


def _render_box_on_table(config, height_m=0.06, half=(0.02, 0.035),
                         centre=(0.45, -0.05)):
    camera = config['camera']
    width = camera['reference_width'] // 4
    height = camera['reference_height'] // 4
    scale = width / camera['reference_width']
    fx, fy = camera['fx'] * scale, camera['fy'] * scale
    cx, cy = camera['cx'] * scale, camera['cy'] * scale
    transform = np.array(camera['T_base_cam'], dtype=np.float64).reshape(4, 4)
    origin = transform[:3, 3]
    rotation = transform[:3, :3]

    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(u, v)
    rays = np.stack([(grid_u - cx) / fx, (grid_v - cy) / fy,
                     np.ones_like(grid_u)], axis=-1) @ rotation.T

    # The camera looks straight down, so z_cam is the ray parameter and the
    # table plane z = 0 is hit at t = origin_z.
    table_t = origin[2] / -rays[..., 2]
    top_t = (origin[2] - height_m) / -rays[..., 2]
    hit = origin + top_t[..., None] * rays
    inside = ((np.abs(hit[..., 0] - centre[0]) <= half[0])
              & (np.abs(hit[..., 1] - centre[1]) <= half[1]))
    depth_m = np.where(inside, top_t, table_t)
    depth = np.rint(depth_m / camera['depth_scale_m']).astype('<u2')
    colour = np.zeros((height, width, 3), dtype=np.uint8)
    return {
        'depth': depth.tobytes(),
        'rgb': colour.tobytes(),
        'width': width,
        'height': height,
        'centre': centre,
        'half': half,
        'height_m': height_m,
    }
