#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <fmt/core.h>

#include <catch2/catch.hpp>
#include <skimpy/detail/eval.hpp>
#include <unordered_map>
#include <vector>

using namespace skimpy::detail::step;

TEST_CASE("Benchmark static lower bound", "[step_static_lower_bound]") {
  constexpr auto d = 256;
  constexpr auto n = d * d * d;

  std::vector<int> v(n);
  for (int i = 0; i < n; i += 1) {
    v[i] = i % 10;
  }

  auto x0 = 2, x1 = 254, xs = 1;
  auto y0 = 2, y1 = 254, ys = 1;
  auto z0 = 2, z1 = 254, zs = 1;

  BENCHMARK("v[2:254, 2:254, 2:254].sum(); d=256") {
    int64_t cnt = 0;
    int64_t sum = 0;
    for (int z = z0; z < z1; z += zs) {
      for (int y = y0; y < y1; y += ys) {
        for (int x = x0; x < x1; x += xs) {
          sum += v[cnt++];
        }
      }
    }
    REQUIRE(cnt == (x1 - x0) * (y1 - y0) * (z1 - z0));
    REQUIRE(sum == 72013528);
  };
}

TEST_CASE("Benchmark cycle fn", "[step_cycle_fn]") {
  constexpr auto d = 256;
  constexpr auto n = d * d * d;

  auto x0 = 2, x1 = 254;
  auto y0 = 2, y1 = 254;
  auto z0 = 2, z1 = 254;

  auto x_s = x1 - x0;
  auto y_s = y1 - y0;
  auto z_s = z1 - z0;

  auto i0 = x0 + y0 * d + z0 * d * d;
  auto i1 = i0 + x_s + d * (y_s - 1) + d * d * (z_s - 1);

  // Create the cycle step functino.
  auto fn = [&] {
    using namespace cyclic;
    auto steps = stack(
        z_s,
        stack(y_s, scaled<1>(x_s), fixed<0>(d - x_s)),
        fixed<0>(d * d - d * y_s));
    return build(stack(fixed<0>(i0), steps));
  }();

  // Initialize some indexable data to access by the step function.
  std::vector<int> v(n);
  for (int i = 0; i < n; i += 1) {
    v[i] = i % 10;
  }

  // Run the benchmark.
  BENCHMARK("v[2:254, 2:254, 2:254].sum(); d=256") {
    int64_t cnt = 0;
    int64_t sum = 0;
    auto prev_end = 0;
    for (int i = i0 + 1; i <= i1; i += 1) {
      auto end = fn(i);
      if (prev_end != end) {
        sum += v[end - 1];
        prev_end = end;
        cnt++;
      }
    }
    REQUIRE(cnt == x_s * y_s * z_s);
    REQUIRE(sum == 72013528);
  };
}

TEST_CASE("Benchmark simple cycle fn", "[step_simple_cycle_fn]") {
  constexpr auto n = 1024 * 1024;

  // Run the simple benchmark.
  {
    auto fn = identity();
    BENCHMARK("n=1024 * 1024; simple") {
      volatile int64_t sum = 0;
      for (int i = 1; i <= n; i += 1) {
        sum += fn(i);
      }
    };
  }

  // Run the cyclic benchmark.
  {
    auto fn = cyclic::build(cyclic::scaled<1>(n));
    BENCHMARK("n=1024 * 1024; cyclic") {
      volatile int64_t sum = 0;
      for (int i = 1; i <= n; i += 1) {
        sum += fn(i);
      }
    };
  }
}
