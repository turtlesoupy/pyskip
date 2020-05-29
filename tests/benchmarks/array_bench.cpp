#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <array>
#include <catch2/catch.hpp>
#include <thread>
#include <vector>

#include "skimpy/detail/box.hpp"
#include "skimpy/detail/util.hpp"
#include "skimpy/skimpy.hpp"

namespace box = skimpy::detail::box;
namespace util = skimpy::detail::util;

TEST_CASE("Benchmark simple assignments", "[arrays]") {
  // Assign to a random entry in a univariate array.
  BENCHMARK("univariate_assign_1") {
    auto x = skimpy::make_array(1000 * 1000, 0);
    x.set(892, 1);
    x.eval();
  };

  // Assign 10 random entries in a univariate array.
  BENCHMARK("univariate_assign_10") {
    auto x = skimpy::make_array(1000 * 1000, 0);
    x.set(skimpy::Slice(872, 882), 1);
    x.eval();
  };

  // Assign 10 random entries in a univariate array.
  BENCHMARK("univariate_assign_100") {
    auto x = skimpy::make_array(1000 * 1000, 0);
    x.set(skimpy::Slice(872, 972), 1);
    x.eval();
  };
}

TEST_CASE("Benchmark merge operations", "[arrays]") {
  // Populate a dense store for testing large batch operations.
  auto store = std::make_shared<box::BoxStore>(1000 * 1000);
  for (int i = 0; i < 1000 * 1000; i += 1) {
    store->ends[i] = i + 1;
    store->vals[i] = i;
  }

  // Merge two arrays with a batch operation.
  BENCHMARK("multiply_2") {
    auto a = skimpy::Array<int>(store);
    auto b = skimpy::Array<int>(store);
    auto z = a * b;
    z.eval();
  };

  // Merge four arrays with a batch operation.
  BENCHMARK("multiply_4") {
    auto a = skimpy::Array<int>(store);
    auto b = skimpy::Array<int>(store);
    auto c = skimpy::Array<int>(store);
    auto d = skimpy::Array<int>(store);
    auto z = a * b * c * d;
    z.eval();
  };

  // Merge four arrays with a batch operation.
  BENCHMARK("multiply_8") {
    auto a = skimpy::Array<int>(store);
    auto b = skimpy::Array<int>(store);
    auto c = skimpy::Array<int>(store);
    auto d = skimpy::Array<int>(store);
    auto e = skimpy::Array<int>(store);
    auto f = skimpy::Array<int>(store);
    auto g = skimpy::Array<int>(store);
    auto h = skimpy::Array<int>(store);
    auto z = a * b * c * d * e * f * g * h;
    z.eval();
  };

  BENCHMARK("multiply_16") {
    std::array<skimpy::Array<int>, 16> a = {
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
        skimpy::Array<int>(store),
    };
    auto z = a[0];
    for (int i = 1; i < 16; i += 1) {
      z = z * a[i];
    }
    z.eval();
  };
}

TEST_CASE("Benchmark skimpy array builders", "[builders]") {
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

TEST_CASE("Benchmark reductions", "[arrays][reduce]") {
  // Populate a dense store for testing large batch operations.
  auto store = std::make_shared<box::BoxStore>(1000 * 1000);
  for (int i = 0; i < 1000 * 1000; i += 1) {
    store->ends[i] = i + 1;
    store->vals[i] = i;
  }

  util::Fix sum([](auto& sum, auto& t) {
    if (t.len() == 0) {
      return skimpy::make_array(1, 0);
    } else if (t.len() == 1) {
      return t;
    } else if (t.len() % 2 != 0) {
      auto last = t.get({t.len() - 1, t.len(), 1});
      return sum(t.get({0, t.len() - 1, 1})) + last;
    } else {
      auto lo = t.get({0, t.len(), 2});
      auto hi = t.get({1, t.len(), 2});
      return sum(lo + hi);
    }
  });

  BENCHMARK("sum") {
    skimpy::Array<int> a(store);
    volatile auto ret = sum(a).eval();
  };

  util::Fix prod([](auto& prod, auto& t) {
    if (t.len() == 0) {
      return skimpy::make_array(1, 1);
    } else if (t.len() == 1) {
      return t;
    } else if (t.len() % 2 != 0) {
      auto last = t.get({t.len() - 1, t.len(), 1});
      return prod(t.get({0, t.len() - 1, 1})) * last;
    } else {
      auto lo = t.get({0, t.len(), 2});
      auto hi = t.get({1, t.len(), 2});
      return prod(lo * hi);
    }
  });

  BENCHMARK("prod") {
    skimpy::Array<int> a(store);
    volatile auto ret = prod(a).eval();
  };
}