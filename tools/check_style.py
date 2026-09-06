#!/usr/bin/env python3
"""Enforce the repository's writing constraints mechanically.

These rules are easy to state and easy to forget, and a reviewer reading the
README is the audience that matters most, so they are checked rather than
trusted. Run by CI on every commit.

Usage: check_style.py [paths...]   (defaults to every tracked text file)
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# U+2014 EM DASH, U+2013 EN DASH used as punctuation, and the horizontal bar.
FORBIDDEN = {
    "—": "em dash: use a colon, a comma, or a full stop",
    "―": "horizontal bar: use a colon, a comma, or a full stop",
}

# Adjectives that assert quality instead of measuring it. Flagged in prose
# only; a variable named `fast_path` is not a marketing claim.
PUFFERY = ("robust", "seamless", "blazing", "blazingly", "effortless",
           "cutting-edge", "state-of-the-art", "world-class", "lightning-fast")

TEXT_SUFFIXES = {".md", ".py", ".cpp", ".hpp", ".h", ".txt", ".sh", ".yml",
                 ".yaml", ".json", ".xml", ".msg", ".cfg"}
SKIP_DIRS = {"assets/franka", "data", ".git", "__pycache__", "build", "install"}
# This file necessarily contains every character and word it rejects.
SKIP_FILES = {"tools/check_style.py"}


def tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    files = []
    for rel in out:
        if rel in SKIP_FILES or any(rel.startswith(d) for d in SKIP_DIRS):
            continue
        path = ROOT / rel
        if path.suffix in TEXT_SUFFIXES and path.is_file():
            files.append(path)
    return files


def check(path: Path) -> list[str]:
    problems = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []

    for n, line in enumerate(lines, 1):
        for char, why in FORBIDDEN.items():
            if char in line:
                col = line.index(char) + 1
                problems.append(f"{path.relative_to(ROOT)}:{n}:{col} {why}")
        for word in PUFFERY:
            if re.search(rf"\b{word}\b", line, re.IGNORECASE):
                problems.append(
                    f"{path.relative_to(ROOT)}:{n} '{word}' asserts quality; "
                    f"give the measurement instead")
    return problems


def main(argv: list[str]) -> int:
    targets = [Path(a).resolve() for a in argv] or tracked_text_files()
    problems = [p for target in targets for p in check(target)]
    if problems:
        print(f"{len(problems)} style violation(s):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"style clean across {len(targets)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
