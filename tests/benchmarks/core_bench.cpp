
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <fmt/format.h>

#include <algorithm>
#include <catch2/catch.hpp>
#include <thread>

#include "skimpy/detail/core.hpp"

using namespace skimpy::detail::core;

template <typename Fn>
void parallel(int t, int n, Fn&& fn) {
  std::vector<std::thread> threads;
  for (int i = 0; i < t; i += 1) {
    threads.emplace_back([i, t, n, fn = std::forward<Fn>(fn)] {
      fn(i * n / t, (i + 1) * n / t, i);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_CASE("Benchmark 1-source plan evaluation", "[plan_eval_1_sources]") {
  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 1;            // number of input stores

  // Allocate the input stores.
  std::shared_ptr<EvalStore> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<EvalStore>(n);
  }

  // Initialize the input stores.
  const auto max_end = n;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      std::srand(j);
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = i + 1;
        stores[j]->vals[i] = j > 0 ? 1 : 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    return [&](...) {
      int ret = 0;
      for (int j = 0; j < q; j += 2) {
        ret += inputs[j] * inputs[j + 1];
      }
      return ret;
    }();
  };

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan;
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    EvalStep step(start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan.
  BENCHMARK("eval_plan") {
    volatile auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<EvalStore>(n);
    parallel(8, n - 1, [&](int start, int end, ...) {
      int ends[q];
      int vals[q];
      auto ends_ptr = &x->ends[start];
      auto vals_ptr = &x->vals[start];
      auto prev_end = 0;
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          ends[j] = stores[j]->ends[i];
          vals[j] = stores[j]->vals[i];
          if (ends[j] != prev_end) {
            *ends_ptr++ = ends[j];
            *vals_ptr++ = eval_fn(vals);
            prev_end = ends[j];
          }
        }
      }
    });
  };
}

TEST_CASE("Benchmark 2-source plan evaluation", "[plan_eval_2_sources]") {
  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 2;            // number of input stores

  // Allocate the input stores.
  std::shared_ptr<EvalStore> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<EvalStore>(n);
  }

  // Initialize the input stores.
  const auto max_end = n;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      std::srand(j);
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = i + 1;
        stores[j]->vals[i] = j > 0 ? 1 : 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    return [&](...) {
      int ret = 0;
      for (int j = 0; j < q; j += 2) {
        ret += inputs[j] * inputs[j + 1];
      }
      return ret;
    }();
  };

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan;
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    EvalStep step(start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan.
  BENCHMARK("eval_plan") {
    volatile auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<EvalStore>(n);
    parallel(8, n - 1, [&](int start, int end, ...) {
      int ends[q];
      int vals[q];
      auto ends_ptr = &x->ends[start];
      auto vals_ptr = &x->vals[start];
      auto prev_end = 0;
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          ends[j] = stores[j]->ends[i];
          vals[j] = stores[j]->vals[i];
          if (ends[j] != prev_end) {
            *ends_ptr++ = ends[j];
            *vals_ptr++ = eval_fn(vals);
            prev_end = ends[j];
          }
        }
      }
    });
  };
}

TEST_CASE("Benchmark 4-source plan evaluation", "[plan_eval_4_sources]") {
  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 4;            // number of input stores

  // Allocate the input stores.
  std::shared_ptr<EvalStore> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<EvalStore>(n);
  }

  // Initialize the input stores.
  const auto max_end = n;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      std::srand(j);
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = i + 1;
        stores[j]->vals[i] = j > 0 ? 1 : 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    return [&](...) {
      int ret = 0;
      for (int j = 0; j < q; j += 2) {
        ret += inputs[j] * inputs[j + 1];
      }
      return ret;
    }();
  };

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan;
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    EvalStep step(start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan.
  BENCHMARK("eval_plan") {
    volatile auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<EvalStore>(n);
    parallel(8, n - 1, [&](int start, int end, ...) {
      int ends[q];
      int vals[q];
      auto ends_ptr = &x->ends[start];
      auto vals_ptr = &x->vals[start];
      auto prev_end = 0;
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          ends[j] = stores[j]->ends[i];
          vals[j] = stores[j]->vals[i];
          if (ends[j] != prev_end) {
            *ends_ptr++ = ends[j];
            *vals_ptr++ = eval_fn(vals);
            prev_end = ends[j];
          }
        }
      }
    });
  };
}

TEST_CASE("Benchmark 8-source plan evaluation", "[plan_eval_8_sources]") {
  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 8;            // number of input stores

  // Allocate the input stores.
  std::shared_ptr<EvalStore> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<EvalStore>(n);
  }

  // Initialize the input stores.
  const auto max_end = n;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      std::srand(j);
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = i + 1;
        stores[j]->vals[i] = j > 0 ? 1 : 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    return [&](...) {
      int ret = 0;
      for (int j = 0; j < q; j += 2) {
        ret += inputs[j] * inputs[j + 1];
      }
      return ret;
    }();
  };

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan;
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    EvalStep step(start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan.
  BENCHMARK("eval_plan") {
    volatile auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<EvalStore>(n);
    parallel(8, n - 1, [&](int start, int end, ...) {
      int ends[q];
      int vals[q];
      auto ends_ptr = &x->ends[start];
      auto vals_ptr = &x->vals[start];
      auto prev_end = 0;
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          ends[j] = stores[j]->ends[i];
          vals[j] = stores[j]->vals[i];
          if (ends[j] != prev_end) {
            *ends_ptr++ = ends[j];
            *vals_ptr++ = eval_fn(vals);
            prev_end = ends[j];
          }
        }
      }
    });
  };
}

TEST_CASE("Benchmark 10-source plan evaluation", "[plan_eval_10_sources]") {
  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 10;           // number of input stores

  // Allocate the input stores.
  std::shared_ptr<EvalStore> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<EvalStore>(n);
  }

  // Initialize the input stores.
  const auto max_end = n;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      std::srand(j);
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = i + 1;
        stores[j]->vals[i] = j > 0 ? 1 : 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    return [&](...) {
      int ret = 0;
      for (int j = 0; j < q; j += 2) {
        ret += inputs[j] * inputs[j + 1];
      }
      return ret;
    }();
  };

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan;
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    EvalStep step(start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan.
  BENCHMARK("eval_plan") {
    volatile auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<EvalStore>(n);
    parallel(8, n - 1, [&](int start, int end, ...) {
      int ends[q];
      int vals[q];
      auto ends_ptr = &x->ends[start];
      auto vals_ptr = &x->vals[start];
      auto prev_end = 0;
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          ends[j] = stores[j]->ends[i];
          vals[j] = stores[j]->vals[i];
          if (ends[j] != prev_end) {
            *ends_ptr++ = ends[j];
            *vals_ptr++ = eval_fn(vals);
            prev_end = ends[j];
          }
        }
      }
    });
  };
}

TEST_CASE("Benchmark 16-source plan evaluation", "[plan_eval_16_sources]") {
  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 16;           // number of input stores

  // Allocate the input stores.
  std::shared_ptr<EvalStore> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<EvalStore>(n);
  }

  // Initialize the input stores.
  const auto max_end = n;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      std::srand(j);
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = i + 1;
        stores[j]->vals[i] = j > 0 ? 1 : 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    return [&](...) {
      int ret = 0;
      for (int j = 0; j < q; j += 2) {
        ret += inputs[j] * inputs[j + 1];
      }
      return ret;
    }();
  };

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan;
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    EvalStep step(start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan.
  BENCHMARK("eval_plan") {
    volatile auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<EvalStore>(n);
    parallel(8, n - 1, [&](int start, int end, ...) {
      int ends[q];
      int vals[q];
      auto ends_ptr = &x->ends[start];
      auto vals_ptr = &x->vals[start];
      auto prev_end = 0;
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          ends[j] = stores[j]->ends[i];
          vals[j] = stores[j]->vals[i];
          if (ends[j] != prev_end) {
            *ends_ptr++ = ends[j];
            *vals_ptr++ = eval_fn(vals);
            prev_end = ends[j];
          }
        }
      }
    });
  };
}

TEST_CASE("Benchmark 20-source plan evaluation", "[plan_eval_20_sources]") {
  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 20;           // number of input stores

  // Allocate the input stores.
  std::shared_ptr<EvalStore> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<EvalStore>(n);
  }

  // Initialize the input stores.
  const auto max_end = n;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      std::srand(j);
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = i + 1;
        stores[j]->vals[i] = j > 0 ? 1 : 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    return [&](...) {
      int ret = 0;
      for (int j = 0; j < q; j += 2) {
        ret += inputs[j] * inputs[j + 1];
      }
      return ret;
    }();
  };

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan;
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    EvalStep step(start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan.
  BENCHMARK("eval_plan") {
    volatile auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<EvalStore>(n);
    parallel(8, n - 1, [&](int start, int end, ...) {
      int ends[q];
      int vals[q];
      auto ends_ptr = &x->ends[start];
      auto vals_ptr = &x->vals[start];
      auto prev_end = 0;
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          ends[j] = stores[j]->ends[i];
          vals[j] = stores[j]->vals[i];
          if (ends[j] != prev_end) {
            *ends_ptr++ = ends[j];
            *vals_ptr++ = eval_fn(vals);
            prev_end = ends[j];
          }
        }
      }
    });
  };
}

TEST_CASE("Benchmark 32-source plan evaluation", "[plan_eval_32_sources]") {
  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 32;           // number of input stores

  // Allocate the input stores.
  std::shared_ptr<EvalStore> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<EvalStore>(n);
  }

  // Initialize the input stores.
  const auto max_end = n;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      std::srand(j);
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = i + 1;
        stores[j]->vals[i] = j > 0 ? 1 : 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    return [&](...) {
      int ret = 0;
      for (int j = 0; j < q; j += 2) {
        ret += inputs[j] * inputs[j + 1];
      }
      return ret;
    }();
  };

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan;
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    EvalStep step(start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan.
  BENCHMARK("eval_plan") {
    volatile auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<EvalStore>(n);
    parallel(8, n - 1, [&](int start, int end, ...) {
      int ends[q];
      int vals[q];
      auto ends_ptr = &x->ends[start];
      auto vals_ptr = &x->vals[start];
      auto prev_end = 0;
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          ends[j] = stores[j]->ends[i];
          vals[j] = stores[j]->vals[i];
          if (ends[j] != prev_end) {
            *ends_ptr++ = ends[j];
            *vals_ptr++ = eval_fn(vals);
            prev_end = ends[j];
          }
        }
      }
    });
  };
}

TEST_CASE("Benchmark 64-source plan evaluation", "[plan_eval_64_sources]") {
  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 64;           // number of input stores

  // Allocate the input stores.
  std::shared_ptr<EvalStore> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<EvalStore>(n);
  }

  // Initialize the input stores.
  const auto max_end = n;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      std::srand(j);
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = i + 1;
        stores[j]->vals[i] = j > 0 ? 1 : 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    return [&](...) {
      int ret = 0;
      for (int j = 0; j < q; j += 2) {
        ret += inputs[j] * inputs[j + 1];
      }
      return ret;
    }();
  };

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan;
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    EvalStep step(start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan.
  BENCHMARK("eval_plan") {
    volatile auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<EvalStore>(n);
    parallel(8, n - 1, [&](int start, int end, ...) {
      int ends[q];
      int vals[q];
      auto ends_ptr = &x->ends[start];
      auto vals_ptr = &x->vals[start];
      auto prev_end = 0;
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          ends[j] = stores[j]->ends[i];
          vals[j] = stores[j]->vals[i];
          if (ends[j] != prev_end) {
            *ends_ptr++ = ends[j];
            *vals_ptr++ = eval_fn(vals);
            prev_end = ends[j];
          }
        }
      }
    });
  };
}

TEST_CASE("Benchmark 128-source plan evaluation", "[plan_eval_128_sources]") {
  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 128;          // number of input stores

  // Allocate the input stores.
  std::shared_ptr<EvalStore> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<EvalStore>(n);
  }

  // Initialize the input stores.
  const auto max_end = n;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      std::srand(j);
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = i + 1;
        stores[j]->vals[i] = j > 0 ? 1 : 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    return [&](...) {
      int ret = 0;
      for (int j = 0; j < q; j += 2) {
        ret += inputs[j] * inputs[j + 1];
      }
      return ret;
    }();
  };

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan;
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    EvalStep step(start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan.
  BENCHMARK("eval_plan") {
    volatile auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<EvalStore>(n);
    parallel(8, n - 1, [&](int start, int end, ...) {
      int ends[q];
      int vals[q];
      auto ends_ptr = &x->ends[start];
      auto vals_ptr = &x->vals[start];
      auto prev_end = 0;
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          ends[j] = stores[j]->ends[i];
          vals[j] = stores[j]->vals[i];
          if (ends[j] != prev_end) {
            *ends_ptr++ = ends[j];
            *vals_ptr++ = eval_fn(vals);
            prev_end = ends[j];
          }
        }
      }
    });
  };
}
