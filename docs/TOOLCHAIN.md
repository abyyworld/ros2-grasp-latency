# The machine the numbers came from

A latency figure without a machine attached to it is not a measurement. Every
number in [RESULTS.md](../results/RESULTS.md) was produced on this stack, and
`harness/analyze.py` stamps it into the results metadata at run time so a
committed result can always be traced back.

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Kernel | Linux 6.18.44 |
| CPU | 4 vCPU, shared, no pinning and no isolation |
| GPU | none |
| Compiler | g++ 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| CMake | 3.28.3 |
| C++ flags | `-O2 -DNDEBUG` |
| Eigen | 3.4 (`libeigen3-dev`, Ubuntu archive) |
| Python | CPython 3.11.15 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |

The four shared vCPUs matter and are not a detail to skip past. Tail latency on
a machine with no CPU pinning includes scheduler noise from whatever else the
host is doing, which inflates p99 and p99.9 for both implementations. It does
not obviously favour either one, but it means the tail figures are an upper
bound for this stack rather than a clean measurement of the pipeline alone.
[METHOD.md](METHOD.md) covers this and the rest of the threats to validity.
