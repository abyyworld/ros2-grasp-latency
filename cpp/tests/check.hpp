#pragma once

// assert() is compiled out by -DNDEBUG, which is exactly the configuration CI
// builds and therefore the one the tests run in, so the checks here are
// ordinary code that reports and exits non-zero.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace grasp_test
{

inline int failures = 0;

inline void check(bool condition, const char * what, const char * file, int line)
{
  if (!condition) {
    std::fprintf(stderr, "%s:%d: FAILED %s\n", file, line, what);
    ++failures;
  }
}

inline void check_near(
  double a, double b, double tolerance, const char * what, const char * file, int line)
{
  const double delta = std::abs(a - b);
  if (!(delta <= tolerance)) {
    std::fprintf(
      stderr, "%s:%d: FAILED %s: %.17g vs %.17g, delta %.3g > %.3g\n",
      file, line, what, a, b, delta, tolerance);
    ++failures;
  }
}

inline int report(const char * name)
{
  if (failures != 0) {
    std::fprintf(stderr, "%s: %d check(s) failed\n", name, failures);
    return 1;
  }
  std::fprintf(stderr, "%s: ok\n", name);
  return 0;
}

inline std::string asset(const char * relative)
{
  return std::string(GRASP_REPO_ROOT) + "/" + relative;
}

}  // namespace grasp_test

#define CHECK(cond) ::grasp_test::check((cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) ::grasp_test::check_near((a), (b), (tol), #a " ~ " #b, \
    __FILE__, __LINE__)
