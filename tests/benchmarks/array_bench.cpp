#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <array>
#include <catch2/catch.hpp>
#include <thread>
#include <vector>

#include "pyskip/detail/box.hpp"
#include "pyskip/detail/util.hpp"
#include "pyskip/pyskip.hpp"

namespace box = pyskip::detail::box;
namespace conf = pyskip::detail::config;
namespace util = pyskip::detail::util;

TEST_CASE("Benchmark simple assignments", "[arrays]") {
  // Assign to a random entry in a univariate array.
  BENCHMARK("univariate_assign_1") {
    auto x = pyskip::make_array(1000 * 1000, 0);
    x.set(892, 1);
    x.eval();
  };

  // Assign 10 random entries in a univariate array.
  BENCHMARK("univariate_assign_10") {
    auto x = pyskip::make_array(1000 * 1000, 0);
    x.set(pyskip::Slice(872, 882), 1);
    x.eval();
  };

  // Assign 10 random entries in a univariate array.
  BENCHMARK("univariate_assign_100") {
    auto x = pyskip::make_array(1000 * 1000, 0);
    x.set(pyskip::Slice(872, 972), 1);
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
    auto a = pyskip::Array<int>(store);
    auto b = pyskip::Array<int>(store);
    auto z = a * b;
    z.eval();
  };

  // Merge four arrays with a batch operation.
  BENCHMARK("multiply_4") {
    auto a = pyskip::Array<int>(store);
    auto b = pyskip::Array<int>(store);
    auto c = pyskip::Array<int>(store);
    auto d = pyskip::Array<int>(store);
    auto z = a * b * c * d;
    z.eval();
  };

  // Merge four arrays with a batch operation.
  BENCHMARK("multiply_8") {
    auto a = pyskip::Array<int>(store);
    auto b = pyskip::Array<int>(store);
    auto c = pyskip::Array<int>(store);
    auto d = pyskip::Array<int>(store);
    auto e = pyskip::Array<int>(store);
    auto f = pyskip::Array<int>(store);
    auto g = pyskip::Array<int>(store);
    auto h = pyskip::Array<int>(store);
    auto z = a * b * c * d * e * f * g * h;
    z.eval();
  };

  BENCHMARK("multiply_16") {
    std::array<pyskip::Array<int>, 16> a = {
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
        pyskip::Array<int>(store),
    };
    auto z = a[0];
    for (int i = 1; i < 16; i += 1) {
      z = z * a[i];
    }
    z.eval();
  };
}

TEST_CASE("Benchmark pyskip array builders", "[builders]") {
  BENCHMARK("build_10k") {
    pyskip::ArrayBuilder<int> b(10 * 1000, 0);
    for (int i = 0; i < 10 * 1000; i += 1) {
      b.set(i, i);
    }
    b.build();
  };

  BENCHMARK("build_100k") {
    pyskip::ArrayBuilder<int> b(100 * 1000, 0);
    for (int i = 0; i < 100 * 1000; i += 1) {
      b.set(i, i);
    }
    b.build();
  };

  BENCHMARK("build_1m") {
    pyskip::ArrayBuilder<int> b(1000 * 1000, 0);
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

  util::Fix sum([](auto& sum, auto t) {
    if (t.len() == 0) {
      return pyskip::make_array(1, 0);
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
    pyskip::Array<int> a(store);
    volatile auto ret = sum(a).eval();
  };

  util::Fix prod([](auto& prod, auto t) {
    if (t.len() == 0) {
      return pyskip::make_array(1, 1);
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
    pyskip::Array<int> a(store);
    volatile auto ret = prod(a).eval();
  };
}

TEST_CASE("Benchmark evaluation options", "[arrays][config]") {
  static constexpr auto kArrayNonZeroCount = 1000000;

  auto make_arrays = [](auto k) {
    std::vector<pyskip::Array<int>> arrays;
    for (int i = 0; i < k; i += 1) {
      auto a = pyskip::make_array(k * kArrayNonZeroCount, 0);
      a.set({i, k * kArrayNonZeroCount, k}, 1);
      arrays.push_back(std::move(a.eval()));
    }
    return arrays;
  };

  // Pre-compute a bunch of misaligned arrays.
  auto arrays = make_arrays(16);

  // Disable auto-materialization.
  conf::set("flush_tree_size_threshold", std::numeric_limits<int64_t>::max());

  BENCHMARK("mul(k=16); MT; accelerated_eval=false") {
    [&](...) {
      conf::set("accelerated_eval", false);
      auto ret = [&arrays] {
        auto ret = arrays[0];
        for (int i = 1; i < 16; i += 1) {
          ret = ret + arrays[i];
        }
        return ret.eval();
      }();
      REQUIRE(ret.str() == "16000000=>1");
    }();
  };

  BENCHMARK("mul(k=16); MT; accelerated_eval=true") {
    [&](...) {
      conf::set("accelerated_eval", true);
      auto ret = [&arrays] {
        auto ret = arrays[0];
        for (int i = 1; i < 16; i += 1) {
          ret = ret + arrays[i];
        }
        return ret.eval();
      }();
      REQUIRE(ret.str() == "16000000=>1");
    }();
  };

  {
    // Disable eval parallelism.
    struct Scope {
      Scope() {
        conf::set("parallelize_threshold", std::numeric_limits<int64_t>::max());
      }
      ~Scope() {
        conf::clear("parallelize_threshold");
      }
    } scope_guard;

    BENCHMARK("mul(k=16); ST; accelerated_eval=false") {
      [&](...) {
        conf::set("accelerated_eval", false);
        auto ret = [&arrays] {
          auto ret = arrays[0];
          for (int i = 1; i < 16; i += 1) {
            ret = ret + arrays[i];
          }
          return ret.eval();
        }();
        REQUIRE(ret.str() == "16000000=>1");
      }();
    };

    BENCHMARK("mul(k=16); ST; accelerated_eval=true") {
      [&](...) {
        conf::set("accelerated_eval", true);
        auto ret = [&arrays] {
          auto ret = arrays[0];
          for (int i = 1; i < 16; i += 1) {
            ret = ret + arrays[i];
          }
          return ret.eval();
        }();
        REQUIRE(ret.str() == "16000000=>1");
      }();
    };
  }
}
