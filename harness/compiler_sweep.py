#!/usr/bin/env python3
"""How much of the C++ side of the headline ratio is the compiler?

The published numbers are built with the flags cpp/CMakeLists.txt defaults to.
That is a choice, and a choice that flatters the Python side, so it is measured
rather than left implicit: a reader is entitled to know how much of "Python
costs 1.6x" survives building the C++ the way anyone shipping it would.

Each variant is also checked against the Python implementation, because
-march=native enables FMA contraction and therefore changes results. It changes
them about 1e-14, which is nine orders of magnitude inside the gate's
tolerance, but that is a claim worth checking rather than assuming.

Usage: compiler_sweep.py [--dataset data/table_640x480] [--frames 300]
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["-O2 -DNDEBUG", "-O3 -DNDEBUG", "-O3 -DNDEBUG -march=native"]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def build(flags: str, workdir: Path) -> Path:
    build_dir = workdir / ("build_" + "".join(c for c in flags if c.isalnum()))
    run(["cmake", "-S", str(ROOT / "cpp"), "-B", str(build_dir),
         "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CXX_FLAGS_RELEASE={flags}"])
    run(["cmake", "--build", str(build_dir), "-j2"])
    return build_dir / "bench_pipeline"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/table_640x480")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default="results/compiler_flags.json")
    ap.add_argument("--workdir", default="/tmp/compiler_sweep")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)

    load_before = Path("/proc/loadavg").read_text().split()[:3]
    variants = []
    for flags in VARIANTS:
        binary = build(flags, workdir)
        tag = "".join(c for c in flags if c.isalnum())
        timing = workdir / f"{tag}.timing.jsonl"
        output = workdir / f"{tag}.output.jsonl"
        run([str(binary), "--dataset", args.dataset, "--out-timing", str(timing),
             "--out-output", str(output), "--warmup", str(args.warmup),
             "--frames", str(args.frames)])
        rows = [json.loads(line) for line in timing.read_text().splitlines() if line.strip()]
        variants.append({
            "flags": flags,
            "binary_md5": run(["md5sum", str(binary)]).stdout.split()[0],
            "total_p50_ms": percentile([r["total_ns"] for r in rows], 50) / 1e6,
            "total_p99_ms": percentile([r["total_ns"] for r in rows], 99) / 1e6,
            "plane_p50_ms": percentile([r["stage_ns"]["plane"] for r in rows], 50) / 1e6,
            "ik_p50_us": percentile([r["stage_ns"]["ik"] for r in rows], 50) / 1e3,
            "output": str(output),
        })

    baseline = variants[0]["total_p50_ms"]
    for v in variants:
        v["speedup_vs_default"] = baseline / v["total_p50_ms"]
        del v["output"]

    report = {
        "note": ("Identical binaries across flags would mean CMakeLists is overriding the "
                 "cache; the md5 column is what catches that."),
        "dataset": args.dataset,
        "frames": args.frames,
        "warmup": args.warmup,
        "load_average_before": [float(x) for x in load_before],
        "load_average_after": [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]],
        "variants": variants,
    }
    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    for v in variants:
        print(f"  {v['flags']:<28} total p50 {v['total_p50_ms']:7.3f} ms  "
              f"plane {v['plane_p50_ms']:7.3f} ms  {v['speedup_vs_default']:.2f}x  "
              f"md5 {v['binary_md5'][:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
