#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <array>
#include <catch2/catch.hpp>
#include <thread>
#include <vector>

#include "skimpy/skimpy.hpp"

/*
TEST_CASE("Benchmark skimpy arrays", "[bench_arrays]") {
  // Assign to a random entry in a univariate array.
  BENCHMARK("univariate_assign_1") {
    skimpy::Array<int> x(1000 * 1000, 0);
    x.set(892, 1);
    x.eval();
  };

  // Assign 10 random entries in a univariate array.
  BENCHMARK("univariate_assign_10") {
    skimpy::Array<int> x(1000 * 1000, 0);
    x.set(skimpy::Slice(872, 882), 1);
    x.eval();
  };

  // Assign 10 random entries in a univariate array.
  BENCHMARK("univariate_assign_100") {
    skimpy::Array<int> x(1000 * 1000, 0);
    x.set(skimpy::Slice(872, 972), 1);
    x.eval();
  };

  // Populate a dense store for testing large batch operations.
  auto store = std::make_shared<skimpy::Store<int>>(1000 * 1000);
  for (int i = 0; i < 1000 * 1000; i += 1) {
    store->ends[i] = i + 1;
    store->vals[i] = i;
  }

  // Merge two arrays with a batch operation.
  BENCHMARK("multiply_2") {
    auto a = skimpy::Array(store);
    auto b = skimpy::Array(store);
    auto z = a * b;
    z.eval();
  };

  // Merge four arrays with a batch operation.
  BENCHMARK("multiply_4") {
    auto a = skimpy::Array(store);
    auto b = skimpy::Array(store);
    auto c = skimpy::Array(store);
    auto d = skimpy::Array(store);
    auto z = a * b * c * d;
    z.eval();
  };

  // Merge four arrays with a batch operation.
  BENCHMARK("multiply_8") {
    auto a = skimpy::Array(store);
    auto b = skimpy::Array(store);
    auto c = skimpy::Array(store);
    auto d = skimpy::Array(store);
    auto e = skimpy::Array(store);
    auto f = skimpy::Array(store);
    auto g = skimpy::Array(store);
    auto h = skimpy::Array(store);
    auto z = a * b * c * d * e * f * g * h;
    z.eval();
  };

  BENCHMARK("multiply_16") {
    std::array<skimpy::Array<int>, 16> a = {
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
        skimpy::Array(store),
    };
    auto z = a[0];
    for (int i = 1; i < 16; i += 1) {
      z = z * a[i];
    }
    z.eval();
  };
}
*/

TEST_CASE("Benchmark skimpy array builders", "[bench_builders]") {
  BENCHMARK("build_10k") {
    skimpy::ArrayBuilder<int> b(10 * 1000, 0);
    for (int i = 0; i < 10 * 1000; i += 1) {
      b.set(i, i);
    }
    b.build();
  };

  BENCHMARK("build_100k") {
    skimpy::ArrayBuilder<int> b(100 * 1000, 0);
    for (int i = 0; i < 100 * 1000; i += 1) {
      b.set(i, i);
    }
    b.build();
  };

  BENCHMARK("build_1m") {
    skimpy::ArrayBuilder<int> b(1000 * 1000, 0);
    for (int i = 0; i < 1000 * 1000; i += 1) {
      b.set(i, i);
    }
    b.build();
  };
}
