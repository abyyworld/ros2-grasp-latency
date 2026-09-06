"""No tuning constant may be hard-coded in either implementation.

The whole comparison rests on the two pipelines being the same pipeline. The
easiest way for them to silently stop being the same is for one of them to grow
a literal -- an inlier threshold, a damping term, a link offset -- that no
longer tracks `assets/pipeline_config.json` or `assets/franka/panda_chain.json`
when those change. A grep-based guard is crude, but it catches exactly the
failure that would be hardest to notice by reading either file alone.

Only *distinctive* values are checked. `0.0`, `1.0`, `2` and friends appear
legitimately everywhere and carry no information, so requiring them to come
from a config file would be noise, not rigour.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "assets/pipeline_config.json"
CHAIN = ROOT / "assets/franka/panda_chain.json"

SOURCE_GLOBS = ("cpp/include/**/*.hpp", "cpp/src/**/*.cpp", "cpp/bench/**/*.cpp",
                "python/grasp_core/**/*.py", "python/bench/**/*.py")

# Values common enough that a literal says nothing about where it came from.
UNREMARKABLE = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0,
                12.0, 16.0, 100.0, 180.0, 255.0, 256.0, 1000.0, 1e3, 1e6, 1e9}
MIN_SIGNIFICANT_DIGITS = 3


def walk(node, path=""):
    """Yield (json_path, value) for every numeric leaf."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key.startswith("_") or key.endswith("_note") or key == "_comment":
                continue
            yield from walk(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from walk(value, f"{path}[{i}]")
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        yield path, float(node)


def significant_digits(value: float) -> int:
    text = repr(abs(value))
    if "e" in text or "E" in text:
        return MIN_SIGNIFICANT_DIGITS  # scientific notation is distinctive enough
    return len(text.replace(".", "").lstrip("0")) or 1


def distinctive_values() -> dict[float, list[str]]:
    """Config and chain values worth policing, mapped to where they are defined."""
    found: dict[float, list[str]] = {}
    for source, label in ((CONFIG, "config"), (CHAIN, "chain")):
        for path, value in walk(json.loads(source.read_text())):
            if abs(value) in UNREMARKABLE or value == 0.0:
                continue
            if significant_digits(value) < MIN_SIGNIFICANT_DIGITS:
                continue
            found.setdefault(value, []).append(f"{label}:{path}")
    return found


def source_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SOURCE_GLOBS:
        files.extend(p for p in ROOT.glob(pattern) if p.is_file())
    return files


def strip_noise(text: str) -> str:
    """Drop comments and strings so a value quoted in prose is not a violation."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    text = re.sub(r'"""(?:.|\n)*?"""', " ", text)
    text = re.sub(r"#[^\n]*", " ", text)
    text = re.sub(r'"[^"\n]*"', " ", text)
    return text


@pytest.fixture(scope="module")
def sources():
    files = source_files()
    if not files:
        pytest.skip("no implementation sources present yet")
    return [(p, strip_noise(p.read_text())) for p in files]


# Resolution defaults legitimately appear as CLI defaults in both runners.
EXEMPT_ORIGINS = {"config:camera.reference_width", "config:camera.reference_height"}


def literal_forms(value: float) -> list[str]:
    """How this value would plausibly be typed into source, and no more.

    Deliberately narrow: `repr(640.0)` is `'640.0'`, and naive trailing-zero
    stripping turns that into `'64'`, which then matches inside every `640` in
    the file. Integral values get exactly one integer spelling.
    """
    if value == int(value):
        return [str(int(value))]
    return [repr(value)]


def test_no_config_constant_is_hard_coded(sources):
    violations = []
    for value, origins in distinctive_values().items():
        if set(origins) <= EXEMPT_ORIGINS:
            continue
        for literal in literal_forms(value):
            pattern = re.compile(rf"(?<![\w.]){re.escape(literal)}(?![\w.])")
            for path, text in sources:
                for match in pattern.finditer(text):
                    line = text[:match.start()].count("\n") + 1
                    violations.append(
                        f"{path.relative_to(ROOT)}:{line} has literal {match.group(0)}, "
                        f"which is {' / '.join(origins)}")

    assert not violations, (
        "these constants must be read from the JSON, not typed into the source:\n  "
        + "\n  ".join(sorted(violations)))


def test_both_implementations_exist(sources):
    """Guard the guard: the check above passes trivially if it scans nothing."""
    scanned = {p.parts[len(ROOT.parts)] for p, _ in sources}
    assert "cpp" in scanned, "no C++ sources were scanned"
    assert "python" in scanned, "no Python sources were scanned"


def test_distinctive_value_set_is_not_empty():
    values = distinctive_values()
    assert len(values) > 20, f"only {len(values)} values policed; the filter is too aggressive"
