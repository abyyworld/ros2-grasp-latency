#!/usr/bin/env python3
"""Which BLAS is NumPy actually reaching, and how fast is it on our shape?

The in-process benchmark found Python at parity with C++ because one stage,
RANSAC plane scoring, is a large matrix product that NumPy hands to BLAS. The
same stage measured 34 ms on an x86 host and 144 ms in an arm64 container. The
obvious explanation is that the two NumPy builds link different BLAS
implementations, but "obvious" is not "measured", so this measures it.

The product benchmarked here is the one the plane stage actually issues:
`(block x 4) @ (4 x candidates)` repeated over the cloud, which for the shipped
config is 512 x 4 x 128 per block over 212574 points. It is reported as
GFLOP/s so a run on one machine can be compared against a run on another
without either being the same speed.

Usage: blas_probe.py [--out results/blas_probe.json]
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def blas_identity() -> dict:
    """Whatever NumPy will admit about its own build."""
    out = {}
    try:
        cfg = np.show_config(mode="dicts")
        build = cfg.get("Build Dependencies", {})
        blas = build.get("blas", {})
        out["name"] = blas.get("name")
        out["version"] = blas.get("version")
        out["detection_method"] = blas.get("detection method")
        out["openblas_config"] = blas.get("openblas configuration")
    except Exception as exc:  # older NumPy has no dicts mode
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["numpy_version"] = np.__version__
    return out


def gemm_rate(block: int, candidates: int, repeats: int) -> float:
    """GFLOP/s on the plane stage's shape. Best of `repeats`, not the mean:
    the fastest run is the one least disturbed by whatever else the machine
    was doing."""
    a = np.ones((block, 4), dtype=np.float64)
    b = np.ones((4, candidates), dtype=np.float64)
    out = np.empty((block, candidates), dtype=np.float64)
    np.matmul(a, b, out=out)  # warm the path

    flops = 2.0 * block * 4 * candidates
    best = 0.0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(100):
            np.matmul(a, b, out=out)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        best = max(best, 100 * flops / elapsed / 1e9)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/blas_probe.json")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    cfg = json.loads((ROOT / "assets/pipeline_config.json").read_text())
    block = cfg["implementation"]["ransac_block_points"]
    candidates = cfg["plane"]["iterations"]

    report = {
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "blas": blas_identity(),
        "shape": {"block_points": block, "candidates": candidates, "k": 4},
        "gflops_plane_shape": gemm_rate(block, candidates, args.repeats),
        # A large square product, for a second point that does not depend on
        # this pipeline's unusually thin k=4.
        "gflops_1024_square": gemm_rate(1024, 1024, max(1, args.repeats // 2)),
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  numpy {report['blas']['numpy_version']}, blas {report['blas'].get('name')}")
    print(f"  plane shape ({block}x4 @ 4x{candidates}): {report['gflops_plane_shape']:.2f} GFLOP/s")
    print(f"  1024 square:                              {report['gflops_1024_square']:.2f} GFLOP/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
