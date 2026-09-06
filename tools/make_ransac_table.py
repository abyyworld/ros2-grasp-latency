#!/usr/bin/env python3
"""Precompute the uniform deviates RANSAC uses to pick its sample triples.

RANSAC needs randomness, but a pseudo-random *generator* in the measured path
would make the C++ and Python pipelines diverge for a reason that has nothing
to do with the question being asked -- two language runtimes do not agree on
PRNG output, and hand-rolling a matching one in pure Python would penalise
Python for a cost no competent implementation would pay.

So the deviates are drawn once, here, and shipped as a flat binary table both
implementations mmap/read at construction time. At run time each iteration does
`index = min(n - 1, int(u * n))` -- one multiply, no generator state. This is
also what a real-time perception stack does, for the same reason.

Usage: make_ransac_table.py <config.json> <out.bin>
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np


def main(config_path: Path, out_path: Path) -> None:
    cfg = json.loads(config_path.read_text())["ransac_table"]
    values = np.random.default_rng(cfg["seed"]).random(cfg["count"], dtype=np.float64)
    assert values.min() >= 0.0 and values.max() < 1.0

    out_path.write_bytes(struct.pack(f"<{values.size}d", *values.tolist()))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, "
          f"{values.size} float64 in [{values.min():.6f}, {values.max():.6f}])")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
