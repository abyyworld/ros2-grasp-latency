"""The blocking constant must not change what the pipeline computes.

`assets/pipeline_config.json` says of `implementation.ransac_block_points`
that "any value gives bit-identical output, only the memory traffic differs".
That was false: the plane stage accumulated each block's inlier count in an
int16, which wraps once a block holds more than 32767 points, and RANSAC then
selected a different candidate. The equivalence gate caught it at a
131072-point block, with plane normals 0.139 apart.

A claim a config file makes about itself is worth testing.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from grasp_core import GraspPipeline

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "assets/pipeline_config.json"
CHAIN = ROOT / "assets/franka/panda_chain.json"
TABLE = ROOT / "assets/ransac_uniform.bin"
CORPUS = ROOT / "data/table_640x480"

# Spans the int16 boundary at 32767 deliberately.
BLOCK_SIZES = [512, 4096, 32767, 32768, 131072]
FRAMES = 6


def load_frames(manifest):
    for entry in manifest["frames"][:FRAMES]:
        yield (np.fromfile(CORPUS / entry["depth"], dtype="<u2"),
               np.fromfile(CORPUS / entry["rgb"], dtype=np.uint8))


@pytest.fixture(scope="module")
def manifest():
    if not CORPUS.exists():
        pytest.skip("corpus not generated; run harness/make_dataset.py")
    return json.loads((CORPUS / "manifest.json").read_text())


def run_with_block(block_points: int, manifest, tmp_path: Path):
    cfg = json.loads(CONFIG.read_text())
    cfg["implementation"]["ransac_block_points"] = block_points
    path = tmp_path / f"cfg_{block_points}.json"
    path.write_text(json.dumps(cfg))

    pipeline = GraspPipeline(str(path), str(CHAIN), str(TABLE))
    out = []
    for depth, rgb in load_frames(manifest):
        r = pipeline.run(depth, rgb, manifest["width"], manifest["height"])
        out.append((np.array(r.plane, dtype=np.float64),
                    bool(r.graspable),
                    np.array(r.q, dtype=np.float64)))
    return out


def test_block_size_does_not_change_the_answer(manifest, tmp_path):
    reference = run_with_block(BLOCK_SIZES[0], manifest, tmp_path)

    for block in BLOCK_SIZES[1:]:
        got = run_with_block(block, manifest, tmp_path)
        for frame, (want, have) in enumerate(zip(reference, got)):
            assert want[1] == have[1], (
                f"block {block}, frame {frame}: graspable flipped")
            # Bit-identical is the claim, so assert it rather than a tolerance.
            assert np.array_equal(want[0], have[0]), (
                f"block {block}, frame {frame}: plane differs by "
                f"{np.abs(want[0] - have[0]).max():.3e}, so RANSAC chose a "
                f"different candidate")
            assert np.array_equal(want[2], have[2]), (
                f"block {block}, frame {frame}: joint solution differs by "
                f"{np.abs(want[2] - have[2]).max():.3e} rad")
