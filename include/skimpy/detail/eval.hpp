#pragma once

#include <algorithm>
#include <memory>
#include <new>
#include <numeric>
#include <thread>

#include "skimpy/detail/errors.hpp"
#include "skimpy/detail/threads.hpp"
#include "skimpy/detail/util.hpp"

namespace skimpy::detail::eval {

using Pos = core::Pos;

// template <typename Val>
// using EvalFn = Val (*)(Val*);

template <typename Val>
using EvalFn = std::function<Val(Val*)>;

template <typename Val>
struct EvalSource {
  std::shared_ptr<core::Store<Val>> store;
  Pos start;
  Pos stop;
  Pos stride;

  EvalSource(
      std::shared_ptr<core::Store<Val>> store, Pos start, Pos stop, Pos stride)
      : store(std::move(store)), start(start), stop(stop), stride(stride) {}

  int span() const {
    return 1 + (stop - start - 1) / stride;
  }
};

template <typename Val>
struct EvalStep {
  Pos start;
  Pos stop;
  EvalFn<Val> eval_fn;
  std::vector<EvalSource<Val>> sources;

  EvalStep(Pos start, Pos stop, EvalFn<Val> eval_fn)
      : start(start), stop(stop), eval_fn(eval_fn) {}

  int span() const {
    return stop - start;
  }

  int size() const {
    int size = 1;
    for (const auto& source : sources) {
      auto start = source.start;
      auto stop = source.stop;
      size += source.store->index(stop - 1) - source.store->index(start);
    }
    return size;
  }
};

template <typename Val>
struct EvalPlan {
  std::vector<EvalStep<Val>> steps;
};

template <int k, typename Val>
auto eval_step_fixed(
    Pos* const ends, Val* const vals, const EvalStep<Val>& step) {
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

  // Compute the size of the iteration.
  auto step_size = step.size() - 1;

  // Iterate over all input arrays combining their ranges together.
  auto ends_out = ends;
  auto vals_out = vals;
  auto eval_fn = step.eval_fn;
  Pos prev_end = 0;
  Val prev_val;
  for (int count = 0; count < step_size; count += 1) {
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

    // Compute the new end coordinate, relative to the slice parameters.
    auto new_end = *iter_ends[src]++ - starts[src] - 1;
    if (is_power_of_two(strides[src])) {
      new_end >>= lg2(strides[src]);
    } else {
      new_end /= strides[src];
    }
    new_end += 1;

    // Update the frontier.
    curr_ends[src] = new_end;
    curr_vals[src] = *iter_vals[src]++;

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

template <int buffer_size, typename Val>
auto eval_step_stack(
    Pos* const ends, Val* const vals, const EvalStep<Val>& step) {
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

  // Initialize the hash table.
  struct HashNode {
    int key;
    int size;
    int vals[buffer_size];
  };
  constexpr int hash_size = round_up_to_power_of_two(16 * buffer_size);
  CHECK_STATE(is_power_of_two(hash_size));
  std::unique_ptr<HashNode[]> hash(new HashNode[hash_size]);
  for (int i = 0; i < hash_size; i += 1) {
    hash[i].key = 0;
  }

  // Compute the size of the iteration.
  auto step_size = step.size() - 1;

  // Iterate over all input arrays combining their ranges together.
  auto ends_out = ends;
  auto vals_out = vals;
  auto eval_fn = step.eval_fn;
  Pos prev_end = 0;
  Val prev_val;
  for (int count = 0; count < step_size; count += 1) {
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

    // Update all duplicate ends in the hash table as well.
    auto out_end = curr_ends[src];
    if (auto& slot = hash[out_end & (hash_size - 1)]; slot.key == out_end) {
      for (int i = 0; i < slot.size; i += 1) {
        auto hash_src = slot.vals[i];
        curr_vals[hash_src] = *iter_vals[hash_src]++;
        count += 1;
      }
      slot.key = 0;
    }

    // Compute the new end coordinate, relative to the slice parameters.
    auto new_end = *iter_ends[src]++ - starts[src] - 1;
    if (is_power_of_two(strides[src])) {
      new_end >>= lg2(strides[src]);
    } else {
      new_end /= strides[src];
    }
    new_end += 1;

    // Store duplicate ends in the hash table instead of the tournament tree.
    while (step.start + new_end < step.stop) {
      auto& slot = hash[new_end & (hash_size - 1)];
      if (slot.key != new_end) {
        if (slot.key == 0) {
          slot.key = new_end;
          slot.size = 0;
        }
        break;
      }
      slot.vals[slot.size++] = src;
      new_end = 1 + (*iter_ends[src]++ - starts[src] - 1);
    };

    // Update the frontier.
    curr_ends[src] = new_end;
    curr_vals[src] = *iter_vals[src]++;

    // Update the tournament tree.
    tree[u + src - 1] = (static_cast<uint64_t>(new_end) << 32) | src;
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

template <typename Val>
auto eval_step_heap(
    Pos* const ends, Val* const vals, const EvalStep<Val>& step) {
  const int k = step.sources.size();
  constexpr auto kCacheLineSize = std::hardware_destructive_interference_size;

  // Initialize slices.
  std::unique_ptr<Pos[]> starts(new Pos[k, kCacheLineSize]);
  std::unique_ptr<Pos[]> strides(new Pos[k, kCacheLineSize]);
  for (auto i = 0; i < k; i += 1) {
    starts[i] = step.sources[i].start;
    strides[i] = step.sources[i].stride;
  }

  // Initialize iterators.
  std::unique_ptr<Pos*[]> iter_ends(new Pos*[k, kCacheLineSize]);
  std::unique_ptr<Val*[]> iter_vals(new Val*[k, kCacheLineSize]);
  for (auto i = 0; i < k; i += 1) {
    const auto& source = step.sources[i];
    const auto index = source.store->index(source.start);
    iter_ends[i] = &source.store->ends[index];
    iter_vals[i] = &source.store->vals[index];
  }

  // Initialize frontier.
  std::unique_ptr<Pos[]> curr_ends(new Pos[k, kCacheLineSize]);
  std::unique_ptr<Val[]> curr_vals(new Val[k, kCacheLineSize]);
  for (auto i = 0; i < k; i += 1) {
    curr_ends[i] = 1 + (*iter_ends[i]++ - starts[i] - 1) / strides[i];
    curr_vals[i] = *iter_vals[i]++;
  }

  // Initialize tournament tree.
  std::unique_ptr<uint64_t[]> tree(new uint64_t[4 * k, kCacheLineSize]);
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

  // Initialize the hash table.
  struct HashNode {
    int key;
    int size;
    std::unique_ptr<int[]> vals;
  };
  const int hash_size = round_up_to_power_of_two(4 * k);
  CHECK_STATE(is_power_of_two(hash_size));
  std::unique_ptr<HashNode[]> hash(new HashNode[hash_size]);
  for (int i = 0; i < hash_size; i += 1) {
    hash[i].key = 0;
    hash[i].vals.reset(new int[k]);
  }

  // Compute the size of the iteration.
  auto step_size = step.size() - 1;

  // Iterate over all input arrays combining their ranges together.
  auto ends_out = ends;
  auto vals_out = vals;
  auto eval_fn = step.eval_fn;
  Pos prev_end = 0;
  Val prev_val;
  for (int count = 0; count < step_size; count += 1) {
    auto src = tree[0] & 0xFFFFFFFF;
    auto end = step.start + curr_ends[src];

    // Emit the new marker but handle compression cases.
    if (prev_end != end) {
      auto val = eval_fn(&curr_vals[0]);
      if (prev_end && prev_val == val) {
        --ends_out;
        --vals_out;
      }
      *ends_out++ = end;
      *vals_out++ = val;
      prev_end = end;
      prev_val = val;
    }

    // Update all duplicate ends in the hash table as well.
    auto out_end = curr_ends[src];
    if (auto& slot = hash[out_end & (hash_size - 1)]; slot.key == out_end) {
      for (int i = 0; i < slot.size; i += 1) {
        auto hash_src = slot.vals[i];
        curr_vals[hash_src] = *iter_vals[hash_src]++;
        count += 1;
      }
      slot.key = 0;
    }

    // Compute the new end coordinate, relative to the slice parameters.
    auto new_end = *iter_ends[src]++ - starts[src] - 1;
    if (is_power_of_two(strides[src])) {
      new_end >>= lg2(strides[src]);
    } else {
      new_end /= strides[src];
    }
    new_end += 1;

    // Store duplicate ends in the hash table instead of the tournament tree.
    while (step.start + new_end < step.stop) {
      auto& slot = hash[new_end & (hash_size - 1)];
      if (slot.key != new_end) {
        if (slot.key == 0) {
          slot.key = new_end;
          slot.size = 0;
        }
        break;
      }
      slot.vals[slot.size++] = src;
      new_end = 1 + (*iter_ends[src]++ - starts[src] - 1);
    };

    // Update the frontier.
    curr_ends[src] = new_end;
    curr_vals[src] = *iter_vals[src]++;

    // Update the tournament tree.
    tree[u + src - 1] = (static_cast<uint64_t>(new_end) << 32) | src;
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
  auto val = eval_fn(&curr_vals[0]);
  if (prev_end && prev_val == val) {
    --vals_out;
    --ends_out;
  }
  *vals_out++ = val;
  *ends_out++ = step.stop;

  return ends_out - ends;
}

template <typename Val>
void check_step(const EvalStep<Val>& step) {
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

template <typename Val>
auto parallelize_plan(const EvalPlan<Val>& plan) {
  // STRATEGY: Partition each step into a fixed number of equal-sized chunks
  // if their total size exceeds some threshold. This approach should have a
  // minimal overhead while enabling balanced parallelism.
  static constexpr int kThreshold = 32 * 1024;
  static int kNumChunks = [] { return std::thread::hardware_concurrency(); }();

  EvalPlan<Val> ret;
  for (const auto& step : plan.steps) {
    uint64_t span = step.span();
    if (span > kThreshold) {
      for (auto i = 0; i < kNumChunks; i += 1) {
        ret.steps.emplace_back(
            static_cast<Pos>(step.start + (span * i) / kNumChunks),
            static_cast<Pos>(step.start + (span * (i + 1)) / kNumChunks),
            step.eval_fn);
        for (auto& src : step.sources) {
          uint64_t src_span = src.stride * src.span();
          ret.steps.back().sources.emplace_back(
              src.store,
              static_cast<Pos>(src.start + (src_span * i) / kNumChunks),
              static_cast<Pos>(src.start + (src_span * (i + 1)) / kNumChunks),
              src.stride);
        }
      }
    } else {
      ret.steps.push_back(step);
    }
  }

  return ret;
}

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

template <typename FnRange>
void run_inline(const FnRange& fns) {
  for (const auto& fn : fns) {
    fn();
  }
}

template <typename Val>
auto eval_plan(const EvalPlan<Val>& input_plan) {
  // Validate that each step is properly configured.
  for (const auto& step : input_plan.steps) {
    check_step(step);
  }

  // Augment the plan to add better parallelization.
  auto plan = parallelize_plan(input_plan);
  auto s = plan.steps.size();
  CHECK_ARGUMENT(s <= std::numeric_limits<int32_t>::max());

  // We need to keep track of where we write each step's output into the
  // temporary store and the size of each output after compression.
  std::vector<int> dest_sizes(s);
  std::vector<int> step_offsets(s + 1);
  step_offsets[0] = 0;
  for (int i = 0; i < plan.steps.size(); i += 1) {
    auto step_size = plan.steps[i].size();
    CHECK_ARGUMENT(step_size > 0);
    step_offsets[i + 1] = step_offsets[i] + step_size;
  }

  // Allocate the destination store to fit all step outputs.
  auto store = std::make_shared<core::Store<Val>>(step_offsets[s]);

  // Create a task for each eval step in the plan.
  std::vector<std::function<void()>> eval_fns;
  for (int i = 0; i < plan.steps.size(); i += 1) {
    eval_fns.emplace_back([&, i] {
      EvalStep step = plan.steps[i];
      Pos* ends = store->ends.get() + step_offsets[i];
      Val* vals = store->vals.get() + step_offsets[i];

      // Emit the merged ranges into the destination.
      // TODO: Futher specialize cases (e.g. when size == 1).
      CHECK_ARGUMENT(step.sources.size());
      if (step.sources.size() == 1) {
        dest_sizes[i] = eval_step_fixed<1>(ends, vals, step);
      } else if (step.sources.size() == 2) {
        dest_sizes[i] = eval_step_fixed<2>(ends, vals, step);
      } else if (step.sources.size() == 3) {
        dest_sizes[i] = eval_step_fixed<3>(ends, vals, step);
      } else if (step.sources.size() == 4) {
        dest_sizes[i] = eval_step_fixed<4>(ends, vals, step);
      } else if (step.sources.size() <= 16) {
        dest_sizes[i] = eval_step_stack<16>(ends, vals, step);
      } else if (step.sources.size() <= 32) {
        dest_sizes[i] = eval_step_stack<32>(ends, vals, step);
      } else if (step.sources.size() <= 64) {
        dest_sizes[i] = eval_step_stack<64>(ends, vals, step);
      } else if (step.sources.size() <= 128) {
        dest_sizes[i] = eval_step_stack<128>(ends, vals, step);
      } else if (step.sources.size() <= 256) {
        dest_sizes[i] = eval_step_stack<256>(ends, vals, step);
      } else if (step.sources.size() <= 512) {
        dest_sizes[i] = eval_step_stack<512>(ends, vals, step);
      } else if (step.sources.size() <= 1024) {
        dest_sizes[i] = eval_step_stack<1024>(ends, vals, step);
      } else {
        dest_sizes[i] = eval_step_heap(ends, vals, step);
      }
    });
  }

  // Run all of the eval tasks in parallel.
  constexpr auto kParallelizeEvalThreshold = 16 * 1024;
  if (step_offsets[s] > kParallelizeEvalThreshold) {
    run_in_parallel(eval_fns);
  } else {
    run_inline(eval_fns);
  }

  // Update the output offsets and sizes to reflect comrpession at the edges.
  for (int i = 0; i < s - 1; i += 1) {
    auto l_head = step_offsets[i];
    auto l_size = dest_sizes[i];
    auto l_back = l_head + l_size - 1;
    auto r_head = step_offsets[i + 1];
    auto r_size = dest_sizes[i + 1];

    CHECK_STATE(l_size > 0);
    CHECK_STATE(r_size >= 0);
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
  auto destination = std::make_shared<core::Store<Val>>(dest_offsets[s]);

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
  constexpr auto kParallelizeMoveThreshold = 16 * 1024;
  if (dest_offsets[s] > kParallelizeMoveThreshold) {
    run_in_parallel(move_fns);
  } else {
    run_inline(move_fns);
  }

  return destination;
}

}  // namespace skimpy::detail::eval
