#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>
#include <thread>
#include <vector>

#include "skimpy/skimpy.hpp"

TEST_CASE("Benchmark skimpy arrays", "[bench_arrays]") {
  // Assign to a random entry in a univariate array.
  BENCHMARK("univariate_assign_1") {
    skimpy::Array<int> x(1000 * 1000, 0);
    x.set(892, 1);
  };

  // Assign 10 random entries in a univariate array.
  BENCHMARK("univariate_assign_10") {
    skimpy::Array<int> x(1000 * 1000, 0);
    for (int i = 0; i < 10; i += 1) {
      x.set(892 + i, i + 1);
    }
  };
}
