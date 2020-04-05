
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <asmjit/asmjit.h>
#include <fmt/format.h>

#include <algorithm>
#include <catch2/catch.hpp>
#include <new>
#include <thread>

#include "skimpy/detail/errors.hpp"

using Pos = int32_t;
using Val = int;

struct Store {
  int size;
  std::unique_ptr<Pos[]> ends;
  std::unique_ptr<Val[]> vals;

  Store(Pos n) : size(n), ends(new Pos[n]), vals(new Val[n]) {}
  Store(Pos n, std::unique_ptr<Pos[]> ends, std::unique_ptr<Val[]> vals)
      : size(n), ends(std::move(ends)), vals(std::move(vals)) {}

  int index(Pos pos) {
    return std::upper_bound(&ends[0], &ends[size], pos) - &ends[0];
  }
};

struct EvalSource {
  std::shared_ptr<Store> store;
  Pos start;
  Pos stop;
  Pos stride;

  EvalSource(std::shared_ptr<Store> store, Pos start, Pos stop, Pos stride)
      : store(std::move(store)), start(start), stop(stop), stride(stride) {}
};

typedef int (*EvalFn)(int*);

struct EvalStep {
  int size;
  Pos start;
  Pos stop;
  std::vector<EvalSource> sources;
  EvalFn eval_fn;

  EvalStep(int size, Pos start, Pos stop, EvalFn eval_fn)
      : size(size), start(start), stop(stop), eval_fn(eval_fn) {}
};

struct EvalPlan {
  std::shared_ptr<Store> destination;
  std::vector<EvalStep> steps;

  EvalPlan(std::shared_ptr<Store> destination)
      : destination(std::move(destination)) {}
};

auto start_index(const EvalSource& source) {
  auto ends_b = source.store->ends.get();
  auto ends_e = ends_b + source.store->size;
  auto index = std::upper_bound(ends_b, ends_e, source.start) - ends_b;
  CHECK_ARGUMENT(0 <= index && index < source.store->size);
  return index;
}

template <size_t k>
void eval_step_fixed(Pos* ends, Val* vals, const EvalStep& step) {
  CHECK_ARGUMENT(step.sources.size() == k);

  // Initialize slices.
  Pos starts[k];
  Pos strides[k];
  for (auto i = 0; i < k; i += 1) {
    starts[i] = step.sources[i].start;
    strides[i] = step.sources[i].stride;
  }

  // Initialize iterators.
  Pos* iter_ends[k];
  Val* iter_vals[k];
  for (auto i = 0; i < k; i += 1) {
    auto index = start_index(step.sources[i]);
    iter_ends[i] = &step.sources[i].store->ends[index];
    iter_vals[i] = &step.sources[i].store->vals[index];
  }

  // Initialize frontier.
  Pos curr_ends[k];
  Pos curr_vals[k];
  for (auto i = 0; i < k; i += 1) {
    curr_ends[i] = 1 + (*iter_ends[i]++ - starts[i] - 1) / strides[i];
    curr_vals[i] = *iter_vals[i]++;
  }

  // Initialize source heap.
  int heap[k];
  std::iota(&heap[0], &heap[k], 0);
  std::sort(&heap[0], &heap[k], [&](int i, int j) {
    return curr_ends[i] < curr_ends[j];
  });

  auto eval_fn = step.eval_fn;
  auto sentinel = ends + step.size - 1;
  while (ends != sentinel) {
    auto src = heap[0];
    auto end = step.start + curr_ends[src];
    auto val = eval_fn(curr_vals);

    *vals++ = val;
    *ends++ = end;

    auto new_end = 1 + (*iter_ends[src]++ - starts[src] - 1) / strides[src];
    auto new_val = *iter_vals[src]++;
    curr_ends[src] = new_end;
    curr_vals[src] = new_val;

    // Update the heap.
    heap[0] = src;
    for (int i = 0; i < k - 1; i += 1) {
      auto next_src = heap[i + 1];
      if (new_end > curr_ends[next_src]) {
        heap[i + 1] = heap[i];
        heap[i] = next_src;
      } else {
        break;
      }
    }
  }

  // Emit the final end marker at the stop position.
  *vals = eval_fn(curr_vals);
  *ends = step.stop;
}

void eval_step(Pos* ends, Val* vals, const EvalStep& step) {
  auto k = step.sources.size();
  auto b = std::hardware_destructive_interference_size;

  struct SourceState {
    Pos start;
    Pos stride;
    Pos* iter_end;
    Val* iter_val;
    Pos curr_end;
    Pos curr_val;
  };

  // Initialize source state.
  std::unique_ptr<SourceState[]> state(new SourceState[k, b]);
  for (auto i = 0; i < k; i += 1) {
    auto& s = state[i];
    const auto& source = step.sources[i];
    auto index = start_index(source);
    s.iter_end = &source.store->ends[index];
    s.iter_val = &source.store->vals[index];
    s.start = source.start;
    s.stride = source.stride;
    s.curr_end = 1 + (*s.iter_end++ - s.start - 1) / s.stride;
    s.curr_val = *s.iter_val++;
  }

  // Initialize argument array to pass to the eval function.
  std::unique_ptr<Val[]> args(new Val[k, b]);
  for (auto i = 0; i < k; i += 1) {
    args[i] = state[i].curr_val;
  }

  // Initialize source heap.
  std::unique_ptr<Pos[]> heap(new Pos[k, b]);
  std::iota(&heap[0], &heap[k], 0);
  std::sort(&heap[0], &heap[k], [&](int si_1, int si_2) {
    return state[si_1].curr_end < state[si_2].curr_end;
  });

  auto eval_fn = step.eval_fn;
  auto sentinel = ends + step.size - 1;
  while (ends != sentinel) {
    auto si = heap[0];
    auto ss = state[si];
    auto out_end = step.start + ss.curr_end;
    auto out_val = eval_fn(args.get());

    *ends++ = out_end;
    *vals++ = out_val;

    // Update the frontier.
    auto new_end = 1 + (*ss.iter_end++ - ss.start - 1) / ss.stride;
    auto new_val = *ss.iter_val++;
    ss.curr_end = new_end;
    ss.curr_val = new_val;
    args[si] = new_val;

    // Update the heap.
    // TODO: Find k for when an actual heap is better and use that instead.
    heap[0] = si;
    for (int i = 0; i < k - 1; i += 1) {
      auto next_src = heap[i + 1];
      if (new_end > state[next_src].curr_end) {
        heap[i + 1] = heap[i];
        heap[i] = next_src;
      } else {
        break;
      }
    }
  }

  // Emit the final end marker at the stop position.
  *vals = eval_fn(args.get());
  *ends = step.stop;
}

auto check_step(const EvalStep& step) {
  CHECK_ARGUMENT(step.sources.size() > 0);

  // Compute and validate the span of each source.
  std::vector<int> spans;
  for (const auto& source : step.sources) {
    const auto& store = *source.store;
    CHECK_ARGUMENT(source.start >= 0);
    CHECK_ARGUMENT(source.start < store.ends[store.size - 1]);
    CHECK_ARGUMENT(source.start < source.stop);
    CHECK_ARGUMENT(source.stop <= store.ends[store.size - 1]);
    CHECK_ARGUMENT(source.stride > 0);
    spans.push_back(1 + (source.stop - source.start - 1) / source.stride);
  }

  // Check that all spans are aligned.
  for (int i = 1; i < spans.size(); i += 1) {
    CHECK_ARGUMENT(spans[i - 1] == spans[i]);
  }

  // Make sure that the spans match the output span.
  CHECK_ARGUMENT(spans[0] == (step.stop - step.start));
}

void eval_plan(const EvalPlan& plan) {
  CHECK_ARGUMENT(plan.steps.size() <= std::numeric_limits<int32_t>::max());
  for (const auto& step : plan.steps) {
    check_step(step);
  }

  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> exceptions;

  // Create a thread for each step in the plan.
  auto end_offset = 0;
  for (int i = 0; i < plan.steps.size(); i += 1) {
    exceptions.push_back(nullptr);
    threads.emplace_back([&plan, &exceptions, i, end_offset] {
      EvalStep step = plan.steps[i];
      Pos* ends = plan.destination->ends.get() + end_offset;
      Val* vals = plan.destination->vals.get() + end_offset;
      try {
        if (step.sources.size() == 1) {
          eval_step_fixed<1>(ends, vals, step);
        } else if (step.sources.size() == 2) {
          eval_step_fixed<2>(ends, vals, step);
        } else if (step.sources.size() == 3) {
          eval_step_fixed<3>(ends, vals, step);
        } else if (step.sources.size() == 4) {
          eval_step_fixed<4>(ends, vals, step);
        } else if (step.sources.size() == 5) {
          eval_step_fixed<5>(ends, vals, step);
        } else if (step.sources.size() == 6) {
          eval_step_fixed<6>(ends, vals, step);
        } else if (step.sources.size() == 7) {
          eval_step_fixed<7>(ends, vals, step);
        } else if (step.sources.size() == 8) {
          eval_step_fixed<8>(ends, vals, step);
        } else {
          eval_step(ends, vals, step);
        }
      } catch (...) {
        exceptions[i] = std::current_exception();
      }
    });
    end_offset += plan.steps[i].size;
  }

  // Wait for all threads to finish before returning.
  for (auto& thread : threads) {
    thread.join();
  }

  // Throw any exceptions that were raised.
  for (auto& exception : exceptions) {
    if (exception) {
      std::rethrow_exception(exception);
    }
  }

  // Compress the output store before returning.
  // TODO: Consider doing this in parallel by pre-compressing the data at merge
  // time, emitting the resulting output indices, and doing a parallel memcopy.
  /*
  auto ends_iter = plan.destination->ends.get();
  auto vals_iter = plan.destination->vals.get();
  for (int i = 1; i < end_offset; i += 1) {
    auto end = plan.destination->ends[i];
    auto val = plan.destination->vals[i];
    if (*ends_iter == end) {
      continue;
    } else if (*vals_iter == val) {
      *ends_iter = end;
    } else {
      ++ends_iter;
      ++vals_iter;
      *ends_iter = end;
      *vals_iter = val;
    }
  }
  plan.destination->size = 1 + ends_iter - plan.destination->ends.get();
  */
  plan.destination->size = end_offset;
}

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
  static constexpr auto n = 8 * 1024 * 1024;  // size of input
  static constexpr auto s = 8;                // number of steps
  static constexpr auto q = 1;                // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
  }

  // Initialize the input stores.
  const auto max_end = q * n - 1;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = q * i + j + 1;
        stores[j]->vals[i] = 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) { return 2 * inputs[0]; };

  // Initialize the output store.
  auto x = std::make_shared<Store>(q * n + s);

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan(x);
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    auto size = 1;
    for (int j = 0; j < q; j += 1) {
      size += stores[j]->index(stop - 1) - stores[j]->index(start);
    }
    EvalStep step(size, start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan, after which x will be populated.
  BENCHMARK("eval_plan") {
    eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    parallel(8, n - 1, [&](int start, int end, ...) {
      int vals[q];
      for (int j = 0; j < q; j += 1) {
        vals[j] = stores[j]->vals[start];
      }
      auto ends_ptr = &x->ends[q * start];
      auto vals_ptr = &x->vals[q * start];
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          *ends_ptr++ = stores[j]->ends[i];
          *vals_ptr++ = eval_fn(vals);
          vals[j] = stores[j]->vals[i + 1];
        }
      }
    });
  };
}

TEST_CASE("Benchmark 2-source plan evaluation", "[plan_eval_2_sources]") {
  static constexpr auto n = 8 * 1024 * 1024;  // size of input
  static constexpr auto s = 8;                // number of steps
  static constexpr auto q = 2;                // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
  }

  // Initialize the input stores.
  const auto max_end = q * n - 1;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = q * i + j + 1;
        stores[j]->vals[i] = 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    int a = inputs[0];
    int b = inputs[1];
    return a * b;
  };

  // Initialize the output store.
  auto x = std::make_shared<Store>(q * n + s);

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan(x);
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    auto size = 1;
    for (int j = 0; j < q; j += 1) {
      size += stores[j]->index(stop - 1) - stores[j]->index(start);
    }
    EvalStep step(size, start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan, after which x will be populated.
  BENCHMARK("eval_plan") {
    eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    parallel(8, n - 1, [&](int start, int end, ...) {
      int vals[q];
      for (int j = 0; j < q; j += 1) {
        vals[j] = stores[j]->vals[start];
      }
      auto ends_ptr = &x->ends[q * start];
      auto vals_ptr = &x->vals[q * start];
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          *ends_ptr++ = stores[j]->ends[i];
          *vals_ptr++ = eval_fn(vals);
          vals[j] = stores[j]->vals[i + 1];
        }
      }
    });
  };
}

TEST_CASE("Benchmark 4-source plan evaluation", "[plan_eval_4_sources]") {
  static constexpr auto n = 8 * 1024 * 1024;  // size of input
  static constexpr auto s = 8;                // number of steps
  static constexpr auto q = 4;                // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
  }

  // Initialize the input stores.
  const auto max_end = q * n - 1;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = q * i + j + 1;
        stores[j]->vals[i] = 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    int a = inputs[0];
    int b = inputs[1];
    int c = inputs[2];
    int d = inputs[3];
    return a * b + c * d;
  };

  // Initialize the output store.
  auto x = std::make_shared<Store>(q * n + s);

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan(x);
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    auto size = 1;
    for (int j = 0; j < q; j += 1) {
      size += stores[j]->index(stop - 1) - stores[j]->index(start);
    }
    EvalStep step(size, start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan, after which x will be populated.
  BENCHMARK("eval_plan") {
    eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    parallel(8, n - 1, [&](int start, int end, ...) {
      int vals[q];
      for (int j = 0; j < q; j += 1) {
        vals[j] = stores[j]->vals[start];
      }
      auto ends_ptr = &x->ends[q * start];
      auto vals_ptr = &x->vals[q * start];
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          *ends_ptr++ = stores[j]->ends[i];
          *vals_ptr++ = eval_fn(vals);
          vals[j] = stores[j]->vals[i + 1];
        }
      }
    });
  };
}

TEST_CASE("Benchmark 8-source plan evaluation", "[plan_eval_8_sources]") {
  static constexpr auto n = 8 * 1024 * 1024;  // size of input
  static constexpr auto s = 8;                // number of steps
  static constexpr auto q = 8;                // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
  }

  // Initialize the input stores.
  const auto max_end = q * n - 1;
  BENCHMARK("init_input") {
    for (int j = 0; j < q; j += 1) {
      for (int i = 0; i < n; i += 1) {
        stores[j]->ends[i] = q * i + j + 1;
        stores[j]->vals[i] = 1 + (std::rand() % 100);
      }
      stores[j]->ends[n - 1] = max_end;
    }
  };

  // Initialize the evaluation function.
  auto eval_fn = [](int* inputs) {
    int a = inputs[0];
    int b = inputs[1];
    int c = inputs[2];
    int d = inputs[3];
    int e = inputs[4];
    int f = inputs[5];
    int g = inputs[6];
    int h = inputs[7];
    return a * b + c * d + e * f + g * h;
  };

  // Initialize the output store.
  auto x = std::make_shared<Store>(q * n + s);

  // Initialize evaluation plan chunked into some number of steps.
  EvalPlan plan(x);
  for (int64_t i = 0; i < s; i += 1) {
    auto start = static_cast<int>(i * max_end / s);
    auto stop = static_cast<int>((i + 1) * max_end / s);
    auto size = 1;
    for (int j = 0; j < q; j += 1) {
      size += stores[j]->index(stop - 1) - stores[j]->index(start);
    }
    EvalStep step(size, start, stop, eval_fn);
    for (int j = 0; j < q; j += 1) {
      step.sources.emplace_back(stores[j], start, stop, 1);
    }
    plan.steps.emplace_back(std::move(step));
  }

  // Evaluate the plan, after which x will be populated.
  BENCHMARK("eval_plan") {
    eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    parallel(8, n - 1, [&](int start, int end, ...) {
      int vals[q];
      for (int j = 0; j < q; j += 1) {
        vals[j] = stores[j]->vals[start];
      }
      auto ends_ptr = &x->ends[q * start];
      auto vals_ptr = &x->vals[q * start];
      for (int i = start; i < end; i += 1) {
        for (int j = 0; j < q; j += 1) {
          *ends_ptr++ = stores[j]->ends[i];
          *vals_ptr++ = eval_fn(vals);
          vals[j] = stores[j]->vals[i + 1];
        }
      }
    });
  };
}
