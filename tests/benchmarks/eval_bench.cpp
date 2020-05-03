#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <array>
#include <catch2/catch.hpp>
#include <thread>
#include <vector>

#include "skimpy/detail/conv.hpp"
#include "skimpy/detail/core.hpp"
#include "skimpy/detail/eval.hpp"
#include "skimpy/detail/step.hpp"
#include "skimpy/detail/threads.hpp"

using namespace skimpy::detail;

template <typename FnRange>
void run_in_parallel(const FnRange& fns) {
  static auto executor = [] {
    auto n = std::thread::hardware_concurrency();
    return std::make_unique<threads::QueueExecutor>(n);
  }();

  std::vector<std::future<void>> futures;
  for (const auto& fn : fns) {
    futures.push_back(executor->schedule(fn));
  }
  for (auto& future : futures) {
    future.get();
  }
}

template <typename Fn>
void partition(int t, int n, Fn&& fn) {
  std::vector<std::function<void()>> tasks;
  for (int i = 0; i < t; i += 1) {
    tasks.emplace_back([i, t, n, fn = std::forward<Fn>(fn)] {
      fn(i * n / t, (i + 1) * n / t, i);
    });
  }
  run_in_parallel(tasks);
}

auto make_store(int n, int seed) {
  auto store = std::make_shared<core::Store<int>>(n);
  for (int i = 0; i < n; i += 1) {
    store->ends[i] = i + 1;
    store->vals[i] = (7909 * seed * (i + 7703)) & 4095;
  }
  return store;
}

TEST_CASE("Test post-parallelism fusing", "[eval_test_fuse]") {
  static constexpr auto n = 1024 * 1024;  // size of input

  auto sources = eval::make_pool(
      eval::SimpleSource(make_store(n, 0)),
      eval::SimpleSource(make_store(n, 0)),
      eval::SimpleSource(make_store(n, 0)),
      eval::SimpleSource(make_store(n, 0)));

  auto x = eval::eval_simple<int, int>(
      [](const int* v) { return v[0] * v[1] * v[2] * v[3]; }, sources);

  REQUIRE(x->size == 1);
  REQUIRE(x->ends[0] == n);
  REQUIRE(x->vals[0] == 0);
}

TEST_CASE("Benchmark 1-source evaluation", "[eval_1]") {
  static constexpr auto n = 1024 * 1024;  // size of input

  auto sources = eval::make_pool(eval::SimpleSource<int>(make_store(n, 1)));

  BENCHMARK("eval") {
    volatile auto x = eval::eval_simple<int, int>(
        [](const int* v) { return 2 * v[0]; }, sources);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<core::Store<int>>(n);
    partition(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        x->ends[i] = sources[0].store()->ends[i];
        x->vals[i] = 2 * sources[0].store()->vals[i];
      }
    });
  };
}

TEST_CASE("Benchmark 2-source plan evaluation", "[eval_2]") {
  static constexpr auto n = 1024 * 1024;  // size of input

  auto sources = eval::make_pool(
      eval::SimpleSource<int>(make_store(n, 1)),
      eval::SimpleSource<int>(make_store(n, 2)));

  BENCHMARK("eval") {
    volatile auto x = eval::eval_simple<int, int>(
        [](const int* v) { return v[0] * v[1]; }, sources);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<core::Store<int>>(n);
    partition(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        x->ends[i] = sources[0].store()->ends[i];
        x->vals[i] = sources[0].store()->vals[i];
        for (int j = 1; j < 2; j += 1) {
          x->vals[i] *= sources[j].store()->vals[i];
        }
      }
    });
  };
}

TEST_CASE("Benchmark 4-source plan evaluation", "[eval_4]") {
  static constexpr auto n = 1024 * 1024;  // size of input

  auto sources = eval::make_pool(
      eval::SimpleSource(make_store(n, 1)),
      eval::SimpleSource(make_store(n, 2)),
      eval::SimpleSource(make_store(n, 3)),
      eval::SimpleSource(make_store(n, 4)));

  BENCHMARK("eval") {
    volatile auto x = eval::eval_simple<int, int>(
        [](const int* v) { return v[0] * v[1] * v[2] * v[3]; }, sources);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<core::Store<int>>(n);
    partition(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        x->ends[i] = sources[0].store()->ends[i];
        x->vals[i] = sources[0].store()->vals[i];
        for (int j = 1; j < 4; j += 1) {
          x->vals[i] *= sources[j].store()->vals[i];
        }
      }
    });
  };
}

TEST_CASE("Benchmark 8-source plan evaluation", "[eval_8]") {
  static constexpr auto n = 1024 * 1024;  // size of input

  auto sources = eval::make_pool(
      eval::SimpleSource(make_store(n, 1)),
      eval::SimpleSource(make_store(n, 2)),
      eval::SimpleSource(make_store(n, 3)),
      eval::SimpleSource(make_store(n, 4)),
      eval::SimpleSource(make_store(n, 5)),
      eval::SimpleSource(make_store(n, 6)),
      eval::SimpleSource(make_store(n, 7)),
      eval::SimpleSource(make_store(n, 8)));
  static constexpr auto sources_size = decltype(sources)::size;

  BENCHMARK("eval") {
    volatile auto x = eval::eval_simple<int, int>(
        [](const int* v) {
          auto ret = v[0];
          for (int i = 1; i < sources_size; i += 1) {
            ret *= v[i];
          }
          return ret;
        },
        sources);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<core::Store<int>>(n);
    partition(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        x->ends[i] = sources[0].store()->ends[i];
        x->vals[i] = sources[0].store()->vals[i];
        for (int j = 1; j < sources_size; j += 1) {
          x->vals[i] *= sources[j].store()->vals[i];
        }
      }
    });
  };
}

TEST_CASE("Benchmark 16-source plan evaluation", "[eval_16]") {
  static constexpr auto n = 1024 * 1024;  // size of input

  auto sources = eval::make_pool(
      eval::SimpleSource(make_store(n, 1)),
      eval::SimpleSource(make_store(n, 2)),
      eval::SimpleSource(make_store(n, 3)),
      eval::SimpleSource(make_store(n, 4)),
      eval::SimpleSource(make_store(n, 5)),
      eval::SimpleSource(make_store(n, 6)),
      eval::SimpleSource(make_store(n, 7)),
      eval::SimpleSource(make_store(n, 8)),
      eval::SimpleSource(make_store(n, 9)),
      eval::SimpleSource(make_store(n, 10)),
      eval::SimpleSource(make_store(n, 11)),
      eval::SimpleSource(make_store(n, 12)),
      eval::SimpleSource(make_store(n, 13)),
      eval::SimpleSource(make_store(n, 14)),
      eval::SimpleSource(make_store(n, 15)),
      eval::SimpleSource(make_store(n, 16)));
  static constexpr auto sources_size = decltype(sources)::size;

  BENCHMARK("eval") {
    volatile auto x = eval::eval_simple<int, int>(
        [](const int* v) {
          auto ret = v[0];
          for (int i = 1; i < sources_size; i += 1) {
            ret *= v[i];
          }
          return ret;
        },
        sources);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<core::Store<int>>(n);
    partition(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        x->ends[i] = sources[0].store()->ends[i];
        x->vals[i] = sources[0].store()->vals[i];
        for (int j = 1; j < sources_size; j += 1) {
          x->vals[i] *= sources[j].store()->vals[i];
        }
      }
    });
  };
}

TEST_CASE("Benchmark 32-source plan evaluation", "[eval_32]") {
  static constexpr auto n = 1024 * 1024;  // size of input

  auto sources = eval::make_pool(
      eval::SimpleSource(make_store(n, 1)),
      eval::SimpleSource(make_store(n, 2)),
      eval::SimpleSource(make_store(n, 3)),
      eval::SimpleSource(make_store(n, 4)),
      eval::SimpleSource(make_store(n, 5)),
      eval::SimpleSource(make_store(n, 6)),
      eval::SimpleSource(make_store(n, 7)),
      eval::SimpleSource(make_store(n, 8)),
      eval::SimpleSource(make_store(n, 9)),
      eval::SimpleSource(make_store(n, 10)),
      eval::SimpleSource(make_store(n, 11)),
      eval::SimpleSource(make_store(n, 12)),
      eval::SimpleSource(make_store(n, 13)),
      eval::SimpleSource(make_store(n, 14)),
      eval::SimpleSource(make_store(n, 15)),
      eval::SimpleSource(make_store(n, 16)),
      eval::SimpleSource(make_store(n, 17)),
      eval::SimpleSource(make_store(n, 18)),
      eval::SimpleSource(make_store(n, 19)),
      eval::SimpleSource(make_store(n, 20)),
      eval::SimpleSource(make_store(n, 21)),
      eval::SimpleSource(make_store(n, 22)),
      eval::SimpleSource(make_store(n, 23)),
      eval::SimpleSource(make_store(n, 24)),
      eval::SimpleSource(make_store(n, 25)),
      eval::SimpleSource(make_store(n, 26)),
      eval::SimpleSource(make_store(n, 27)),
      eval::SimpleSource(make_store(n, 28)),
      eval::SimpleSource(make_store(n, 29)),
      eval::SimpleSource(make_store(n, 30)),
      eval::SimpleSource(make_store(n, 31)),
      eval::SimpleSource(make_store(n, 32)));
  static constexpr auto sources_size = decltype(sources)::size;

  BENCHMARK("eval") {
    volatile auto x = eval::eval_simple<int, int>(
        [](const int* v) {
          auto ret = v[0];
          for (int i = 1; i < sources_size; i += 1) {
            ret *= v[i];
          }
          return ret;
        },
        sources);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<core::Store<int>>(n);
    partition(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        x->ends[i] = sources[0].store()->ends[i];
        x->vals[i] = sources[0].store()->vals[i];
        for (int j = 1; j < sources_size; j += 1) {
          x->vals[i] *= sources[j].store()->vals[i];
        }
      }
    });
  };
}

TEST_CASE("Benchmark mixed 1-source evaluation", "[eval_mixed_1]") {
  static constexpr auto n = 1024 * 1024;  // size of input

  auto sources = eval::make_pool<eval::MixSourceBase>(
      eval::MixSource<int>(make_store(n, 1)));

  BENCHMARK("eval") {
    volatile auto x = eval::eval_mixed<int>(
        [](const eval::Mix* v) {
          auto v_0 = std::get<int>(v[0]);
          return 2 * v_0;
        },
        sources);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<core::Store<int>>(n);
    partition(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        x->ends[i] = sources[0].end(i);
        x->vals[i] = 2 * std::get<int>(sources[0].val(i));
      }
    });
  };
}

TEST_CASE("Benchmark mixed 2-source evaluation", "[eval_mixed_2]") {
  static constexpr auto n = 1024 * 1024;  // size of input

  auto sources = eval::make_pool<eval::MixSourceBase>(
      eval::MixSource<int>(make_store(n, 1)),
      eval::MixSource<int>(make_store(n, 2)));
  static constexpr auto sources_size = decltype(sources)::size;

  BENCHMARK("eval") {
    volatile auto x = eval::eval_mixed<int>(
        [](const eval::Mix* v) {
          auto v_0 = std::get<int>(v[0]);
          auto v_1 = std::get<int>(v[1]);
          return v_0 * v_1;
        },
        sources);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<core::Store<int>>(n);
    partition(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        x->ends[i] = sources[0].end(i);
        x->vals[i] = std::get<int>(sources[0].val(i));
        for (int j = 1; j < sources_size; j += 1) {
          x->vals[i] *= std::get<int>(sources[j].val(i));
        }
      }
    });
  };
}

TEST_CASE("Benchmark mixed 4-source evaluation", "[eval_mixed_4]") {
  static constexpr auto n = 1024 * 1024;  // size of input

  auto sources = eval::make_pool<eval::MixSourceBase>(
      eval::MixSource<int>(make_store(n, 1)),
      eval::MixSource<int>(make_store(n, 2)),
      eval::MixSource<int>(make_store(n, 3)),
      eval::MixSource<int>(make_store(n, 4)));
  static constexpr auto sources_size = decltype(sources)::size;

  BENCHMARK("eval") {
    volatile auto x = eval::eval_mixed<int>(
        [](const eval::Mix* v) {
          auto v_0 = std::get<int>(v[0]);
          auto v_1 = std::get<int>(v[1]);
          auto v_2 = std::get<int>(v[2]);
          auto v_3 = std::get<int>(v[3]);
          return v_0 * v_1 * v_2 * v_3;
        },
        sources);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<core::Store<int>>(n);
    partition(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        x->ends[i] = sources[0].end(i);
        x->vals[i] = std::get<int>(sources[0].val(i));
        for (int j = 1; j < sources_size; j += 1) {
          x->vals[i] *= std::get<int>(sources[j].val(i));
        }
      }
    });
  };
}

TEST_CASE("Benchmark mixed 8-source evaluation", "[eval_mixed_8]") {
  static constexpr auto n = 1024 * 1024;  // size of input

  auto sources = eval::make_pool<eval::MixSourceBase>(
      eval::MixSource<int>(make_store(n, 1)),
      eval::MixSource<int>(make_store(n, 2)),
      eval::MixSource<int>(make_store(n, 3)),
      eval::MixSource<int>(make_store(n, 4)),
      eval::MixSource<int>(make_store(n, 5)),
      eval::MixSource<int>(make_store(n, 6)),
      eval::MixSource<int>(make_store(n, 7)),
      eval::MixSource<int>(make_store(n, 8)));
  static constexpr auto sources_size = decltype(sources)::size;

  BENCHMARK("eval") {
    volatile auto x = eval::eval_mixed<int>(
        [](const eval::Mix* v) {
          auto v_0 = std::get<int>(v[0]);
          auto v_1 = std::get<int>(v[1]);
          auto v_2 = std::get<int>(v[2]);
          auto v_3 = std::get<int>(v[3]);
          auto v_4 = std::get<int>(v[4]);
          auto v_5 = std::get<int>(v[5]);
          auto v_6 = std::get<int>(v[6]);
          auto v_7 = std::get<int>(v[7]);
          return v_0 * v_1 * v_2 * v_3 * v_4 * v_5 * v_6 * v_7;
        },
        sources);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<core::Store<int>>(n);
    partition(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        x->ends[i] = sources[0].end(i);
        x->vals[i] = std::get<int>(sources[0].val(i));
        for (int j = 1; j < sources_size; j += 1) {
          x->vals[i] *= std::get<int>(sources[j].val(i));
        }
      }
    });
  };
}
