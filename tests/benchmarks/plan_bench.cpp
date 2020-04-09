
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <asmjit/asmjit.h>
#include <fmt/format.h>

#include <algorithm>
#include <catch2/catch.hpp>
#include <new>
#include <thread>

#include "skimpy/detail/errors.hpp"
#include "skimpy/detail/utils.hpp"

namespace skimpy::detail {

using Pos = int32_t;
using Val = int;

struct Store {
  int size;
  std::unique_ptr<Pos[]> ends;
  std::unique_ptr<Val[]> vals;

  Store(int n) : size(n), ends(new Pos[n]), vals(new Val[n]) {}

  void reset(int n) {
    size = n;
    ends.reset(new Pos[n]);
    vals.reset(new Val[n]);
  }

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
  EvalFn eval_fn;
  std::vector<EvalSource> sources;

  EvalStep(int size, Pos start, Pos stop, EvalFn eval_fn)
      : size(size), start(start), stop(stop), eval_fn(eval_fn) {}
};

struct EvalPlan {
  std::vector<EvalStep> steps;
};

template <int k>
auto eval_step_fixed(Pos* const ends, Val* const vals, const EvalStep& step) {
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
    const auto& source = step.sources[i];
    const auto index = source.store->index(source.start);
    iter_ends[i] = &source.store->ends[index];
    iter_vals[i] = &source.store->vals[index];
  }

  // Initialize frontier.
  Pos curr_ends[k];
  Val curr_vals[k];
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

  auto ends_out = ends;
  auto vals_out = vals;
  auto eval_fn = step.eval_fn;
  Pos prev_end = 0;
  Val prev_val;

  for (int count = 0; count < step.size - 1; count += 1) {
    auto src = heap[0];
    auto end = step.start + curr_ends[src];

    // Emit the new marker but handle compression cases.
    if (!prev_end || prev_end != end) {
      auto val = eval_fn(curr_vals);
      if (prev_end && prev_val == val) {
        --vals_out;
        --ends_out;
      }
      *vals_out++ = val;
      *ends_out++ = end;
      prev_end = end;
      prev_val = val;
    }

    // auto new_end = 1 + (*iter_ends[src]++ - starts[src] - 1) / strides[src];
    auto new_end = 1 + (*iter_ends[src]++ - starts[src] - 1);
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
  auto val = eval_fn(curr_vals);
  if (prev_end && prev_val == val) {
    --vals_out;
    --ends_out;
  }
  *vals_out++ = val;
  *ends_out++ = step.stop;

  return ends_out - ends;
}

template <int buffer_size>
auto eval_step_buffered(
    Pos* const ends, Val* const vals, const EvalStep& step) {
  const int k = step.sources.size();
  CHECK_ARGUMENT(k <= buffer_size);

  // Initialize slices.
  Pos starts[buffer_size];
  Pos strides[buffer_size];
  for (auto i = 0; i < k; i += 1) {
    starts[i] = step.sources[i].start;
    strides[i] = step.sources[i].stride;
  }

  // Initialize iterators.
  Pos* iter_ends[buffer_size];
  Val* iter_vals[buffer_size];
  for (auto i = 0; i < k; i += 1) {
    const auto& source = step.sources[i];
    const auto index = source.store->index(source.start);
    iter_ends[i] = &source.store->ends[index];
    iter_vals[i] = &source.store->vals[index];
  }

  // Initialize frontier.
  Pos curr_ends[buffer_size];
  Val curr_vals[buffer_size];
  for (auto i = 0; i < k; i += 1) {
    curr_ends[i] = 1 + (*iter_ends[i]++ - starts[i] - 1) / strides[i];
    curr_vals[i] = *iter_vals[i]++;
  }

  // Initialize tournament tree.
  uint64_t tree[4 * buffer_size];
  const auto u = round_up_to_power_of_two(k);
  for (int src = 0; src < u; src += 1) {
    if (src < k) {
      tree[u + src - 1] = (static_cast<uint64_t>(curr_ends[src]) << 32) | src;
    } else {
      tree[u + src - 1] = std::numeric_limits<int64_t>::max();
    }
  }
  for (int h = u >> 1; h > 0; h >>= 1) {
    for (int i = h; i < h << 1; i += 1) {
      auto l = (i << 1) - 1;
      auto r = (i << 1);
      tree[i - 1] = tree[l] <= tree[r] ? tree[l] : tree[r];
    }
  }

  auto ends_out = ends;
  auto vals_out = vals;
  auto eval_fn = step.eval_fn;
  Pos prev_end = 0;
  Val prev_val;

  for (int count = 0; count < step.size - 1; count += 1) {
    auto src = tree[0] & 0xFFFFFFFF;
    auto end = step.start + curr_ends[src];

    // Emit the new marker but handle compression cases.
    if (prev_end != end) {
      auto val = eval_fn(curr_vals);
      if (prev_end && prev_val == val) {
        --ends_out;
        --vals_out;
      }
      *ends_out++ = end;
      *vals_out++ = val;
      prev_end = end;
      prev_val = val;
    }

    // auto new_end = 1 + (*iter_ends[src]++ - starts[src] - 1) / strides[src];
    auto new_end = 1 + (*iter_ends[src]++ - starts[src] - 1);
    auto new_val = *iter_vals[src]++;
    curr_ends[src] = new_end;
    curr_vals[src] = new_val;

    // Update the tournament tree.
    tree[u + src - 1] = (static_cast<uint64_t>(curr_ends[src]) << 32) | src;
    for (int i = (u + src) >> 1; i > 0; i >>= 1) {
      auto l = (i << 1) - 1;
      auto r = (i << 1);
      if (tree[l] < tree[r]) {
        tree[i - 1] = tree[l];
      } else {
        tree[i - 1] = tree[r];
      }
    }
  }

  // Emit the final end marker at the stop position.
  auto val = eval_fn(curr_vals);
  if (prev_end && prev_val == val) {
    --vals_out;
    --ends_out;
  }
  *vals_out++ = val;
  *ends_out++ = step.stop;

  return ends_out - ends;
}

auto eval_step(Pos* const ends, Val* const vals, const EvalStep& step) {
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
    const auto index = source.store->index(source.start);
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

  auto ends_out = ends;
  auto vals_out = vals;
  auto eval_fn = step.eval_fn;
  Pos prev_end = 0;
  Val prev_val;
  for (int count = 0; count < step.size - 1; count += 1) {
    auto si = heap[0];
    auto ss = state[si];
    auto end = step.start + ss.curr_end;

    if (!prev_end || prev_end != end) {
      auto val = eval_fn(args.get());
      if (prev_end && prev_val == val) {
        --vals_out;
        --ends_out;
      }
      *vals_out++ = val;
      *ends_out++ = end;
      prev_end = end;
      prev_val = val;
    }

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
  auto val = eval_fn(args.get());
  if (prev_end && prev_val == val) {
    --vals_out;
    --ends_out;
  }
  *vals_out++ = val;
  *ends_out++ = step.stop;

  return ends_out - ends;
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

template <typename FnRange>
void run_in_parallel(const FnRange& fns) {
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> exceptions;

  // Create a thread for each task function.
  int i = 0;
  for (const auto& fn : fns) {
    exceptions.push_back(nullptr);
    threads.emplace_back([&exceptions, &fn, i] {
      try {
        fn();
      } catch (...) {
        exceptions[i] = std::current_exception();
      }
    });
    ++i;
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
}

auto eval_plan(const EvalPlan& plan) {
  auto s = plan.steps.size();
  CHECK_ARGUMENT(s <= std::numeric_limits<int32_t>::max());

  // Validate that each step is properly configured.
  for (const auto& step : plan.steps) {
    check_step(step);
  }

  // We need to keep track of where we write each step's output into the
  // temporary store and the size of each output after compression.
  std::vector<int> dest_sizes(s);
  std::vector<int> step_offsets(s + 1);
  step_offsets[0] = 0;
  for (int i = 0; i < plan.steps.size(); i += 1) {
    CHECK_ARGUMENT(plan.steps[i].size > 0);
    step_offsets[i + 1] = step_offsets[i] + plan.steps[i].size;
  }

  // Allocate the destination store to fit all step outputs.
  auto store = std::make_shared<Store>(step_offsets[s]);

  // Create a task for each eval step in the plan.
  std::vector<std::function<void()>> eval_fns;
  for (int i = 0; i < plan.steps.size(); i += 1) {
    eval_fns.emplace_back([&, i] {
      EvalStep step = plan.steps[i];
      Pos* ends = store->ends.get() + step_offsets[i];
      Val* vals = store->vals.get() + step_offsets[i];

      // Emit the merged ranges into the destination.
      CHECK_ARGUMENT(step.sources.size());
      if (step.sources.size() == 1) {
        dest_sizes[i] = eval_step_fixed<1>(ends, vals, step);
      } else if (step.sources.size() == 2) {
        dest_sizes[i] = eval_step_fixed<2>(ends, vals, step);
      } else if (step.sources.size() == 3) {
        dest_sizes[i] = eval_step_fixed<3>(ends, vals, step);
      } else if (step.sources.size() == 4) {
        dest_sizes[i] = eval_step_fixed<4>(ends, vals, step);
      } else if (step.sources.size() == 5) {
        dest_sizes[i] = eval_step_fixed<5>(ends, vals, step);
      } else if (step.sources.size() == 6) {
        dest_sizes[i] = eval_step_fixed<6>(ends, vals, step);
      } else if (step.sources.size() == 7) {
        dest_sizes[i] = eval_step_fixed<7>(ends, vals, step);
      } else if (step.sources.size() == 8) {
        dest_sizes[i] = eval_step_fixed<8>(ends, vals, step);
      } else if (step.sources.size() == 9) {
        dest_sizes[i] = eval_step_fixed<9>(ends, vals, step);
      } else if (step.sources.size() == 10) {
        dest_sizes[i] = eval_step_fixed<10>(ends, vals, step);
      } else if (step.sources.size() == 11) {
        dest_sizes[i] = eval_step_fixed<11>(ends, vals, step);
      } else if (step.sources.size() == 12) {
        dest_sizes[i] = eval_step_fixed<12>(ends, vals, step);
      } else if (step.sources.size() == 13) {
        dest_sizes[i] = eval_step_fixed<13>(ends, vals, step);
      } else if (step.sources.size() == 14) {
        dest_sizes[i] = eval_step_fixed<14>(ends, vals, step);
      } else if (step.sources.size() == 15) {
        dest_sizes[i] = eval_step_fixed<15>(ends, vals, step);
      } else if (step.sources.size() == 16) {
        dest_sizes[i] = eval_step_fixed<16>(ends, vals, step);
      } else if (step.sources.size() <= 128) {
        dest_sizes[i] = eval_step_buffered<128>(ends, vals, step);
      } else if (step.sources.size() <= 1024) {
        dest_sizes[i] = eval_step_buffered<1024>(ends, vals, step);
      } else {
        dest_sizes[i] = eval_step(ends, vals, step);
      }
    });
  }

  // Run all of the eval tasks in parallel.
  run_in_parallel(eval_fns);

  // Update the output offsets and sizes to reflect comrpession at the edges.
  for (int i = 0; i < s - 1; i += 1) {
    auto l_head = step_offsets[i];
    auto l_size = dest_sizes[i];
    auto l_back = l_head + l_size - 1;
    auto r_head = step_offsets[i + 1];
    auto r_size = dest_sizes[i + 1];

    CHECK_ARGUMENT(l_size > 0);
    CHECK_ARGUMENT(r_size >= 0);
    if (r_size == 0) {
      store->ends[r_head - 1] = store->ends[l_back];
      store->vals[r_head - 1] = store->vals[l_back];
      --l_size;
      --r_head;
      ++r_size;
    } else if (r_size == 1) {
      if (store->ends[l_back] == store->ends[r_head]) {
        store->vals[r_head] = store->vals[l_back];
        --l_size;
      } else if (store->vals[l_back] == store->vals[r_head]) {
        --l_size;
      }
    } else {
      if (store->ends[l_back] != store->ends[r_head]) {
        if (store->vals[l_back] == store->vals[r_head]) {
          --l_size;
        }
      } else {
        if (store->vals[l_back] != store->vals[r_head + 1]) {
          ++r_head;
          --r_size;
        } else {
          --l_size;
          ++r_head;
          --r_size;
        }
      }
    }

    step_offsets[i] = l_head;
    dest_sizes[i] = l_size;
    step_offsets[i + 1] = r_head;
    dest_sizes[i + 1] = r_size;
  }

  // Calculate the final output offsets.
  std::vector<int> dest_offsets;
  dest_offsets.push_back(0);
  for (int i = 0; i < s; i += 1) {
    dest_offsets.push_back(dest_offsets.back() + dest_sizes[i]);
  }

  // Allocate the compressed destination store.
  auto destination = std::make_shared<Store>(dest_offsets[s]);

  // Create a task for to move the output of each step in the plan.
  std::vector<std::function<void()>> move_fns;
  for (int i = 0; i < plan.steps.size(); i += 1) {
    move_fns.emplace_back([&, i] {
      auto step_b = step_offsets[i];
      auto step_e = step_offsets[i] + dest_sizes[i];
      auto dest_b = dest_offsets[i];
      std::copy(
          &store->ends[step_b],
          &store->ends[step_e],
          &destination->ends[dest_b]);
      std::copy(
          &store->vals[step_b],
          &store->vals[step_e],
          &destination->vals[dest_b]);
    });
  }

  // Run all of the move tasks in parallel.
  run_in_parallel(move_fns);

  return destination;
}

}  // namespace skimpy::detail

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
  using namespace skimpy::detail;

  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 1;            // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
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
    auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<Store>(n);
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
  using namespace skimpy::detail;

  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 2;            // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
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
    auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<Store>(n);
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
  using namespace skimpy::detail;

  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 4;            // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
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
    auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<Store>(n);
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
  using namespace skimpy::detail;

  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 8;            // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
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
    auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<Store>(n);
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
  using namespace skimpy::detail;

  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 10;           // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
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
    auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<Store>(n);
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
  using namespace skimpy::detail;

  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 16;           // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
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
    auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<Store>(n);
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
  using namespace skimpy::detail;

  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 20;           // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
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
    auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<Store>(n);
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
  using namespace skimpy::detail;

  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 32;           // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
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
    auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<Store>(n);
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
  using namespace skimpy::detail;

  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 64;           // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
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
    auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<Store>(n);
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
  using namespace skimpy::detail;

  static constexpr auto n = 1024 * 1024;  // size of input
  static constexpr auto s = 8;            // number of steps
  static constexpr auto q = 128;          // number of input stores

  // Allocate the input stores.
  std::shared_ptr<Store> stores[q];
  for (int j = 0; j < q; j += 1) {
    stores[j] = std::make_shared<Store>(n);
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
    auto dst = eval_plan(plan);
  };

  BENCHMARK("lower_bound") {
    auto x = std::make_shared<Store>(n);
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
