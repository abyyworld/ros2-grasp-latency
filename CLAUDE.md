# Working rules for this repository

## What this repository is

It answers one question with a number: how much end-to-end latency does writing
a ROS 2 grasp node in Python cost against the same pipeline in C++, from RGB-D
frame to joint command, and at what control rate does it start to matter.

It is not a pick-and-place demo. A ROS 2 demo without a measurement is
indistinguishable from the thousands of them online and reads as coursework.
Every addition should be framed as a question with a measured answer.

## Style constraints, not negotiable

1. **No em dashes.** Anywhere. READMEs, docs, code comments, commit messages.
   Use a colon, a comma, or a full stop.
2. **Never state a result that was not produced.** If a run does not finish,
   the honest output is that it did not finish, not a number from an early
   epoch. If a number could not be measured on this machine, say which machine
   it needs.
3. **Prefer measured claims over adjectives.** "p99 of 4.1 ms" beats "fast".
   `tools/check_style.py` lists the quality-asserting words it rejects.
4. **Publish unflattering results.** If Python turns out to be sufficient at
   30 Hz, that is the finding. The honesty is the point and it is the most
   distinctive thing about this work.
5. **Do not overstate authorship or imply institutional backing.**

## Git

Commit and push as `abyyworld <annolieberto@gmail.com>`, and only ever to that
account. Do not add a `Co-Authored-By` trailer, a session link, or any other
assistant attribution to a commit message, a pull request, or anything else
pushed to a remote. This is portfolio work and it carries one author's name.

## Technical invariants

- Both implementations read every constant from `assets/pipeline_config.json`
  and the chain from `assets/franka/panda_chain.json`. Nothing is hard-coded;
  `tests/test_config_is_sole_source.py` enforces this.
- `docs/ALGORITHM.md` is the frozen spec. Changing pipeline behaviour means
  changing the spec first, then both implementations.
- The output-equivalence gate (`harness/compare_outputs.py`) must pass before
  any latency number is quoted. A latency comparison between two programs that
  compute different things is worthless.
- The Python implementation must be the best reasonable NumPy/SciPy code, not a
  transliteration of the C++. Forcing scalar loops would manufacture the result.
  No numba, no Cython, no C extensions: the question is what an ordinary rclpy
  node costs.
- Nothing enters a measured region that does not belong there: no allocation,
  file I/O, JSON parsing, string formatting or logging.
- Results are committed as data (JSON or CSV) as well as prose.

## Environment note

ROS 2 cannot be installed in the Claude Code container: the egress proxy
returns 403 for packages.ros.org. The pure C++ and Python cores, the dataset
generator, the benchmark and the analysis all run without it. The ROS 2
packages build and run through `docker/`, on a host that has Docker.
