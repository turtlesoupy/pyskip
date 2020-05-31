#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <new>
#include <variant>

#include "config.hpp"
#include "core.hpp"
#include "step.hpp"
#include "threads.hpp"
#include "util.hpp"

namespace skimpy::detail::eval {

using Pos = core::Pos;

template <typename Val>
using StorePtr = std::shared_ptr<core::Store<Val>>;

// Convenience wrapper for the pointers to the output store of an evaluation.
template <typename Val>
struct EvalOutput {
  Pos* ends;
  Val* vals;

  EvalOutput(Pos* ends, Val* vals) : ends(ends), vals(vals) {}

  inline void rewind() {
    --ends;
    --vals;
  }

  inline void emit(Pos end, Val val) {
    *ends++ = end;
    *vals++ = val;
  }
};

// Provides a priority queue to select the source with the lowest end position.
template <int sources>
struct TournamentTree {
  static constexpr int base = util::round_up_to_power_of_two(sources);
  uint64_t keys[2 * base];

  template <typename Evaluator>
  TournamentTree(Evaluator& evaluator) {
    for (int src = 0; src < base; src += 1) {
      if (src < sources) {
        auto end = evaluator.next_end(src);
        keys[base + src - 1] = (static_cast<uint64_t>(end) << 32) | src;
      } else {
        keys[base + src - 1] = std::numeric_limits<int64_t>::max();
      }
    }
    for (int h = base >> 1; h > 0; h >>= 1) {
      for (int i = h; i < h << 1; i += 1) {
        auto l = (i << 1) - 1;
        auto r = (i << 1);
        keys[i - 1] = keys[l] <= keys[r] ? keys[l] : keys[r];
      }
    }
  }

  inline Pos end() const {
    return keys[0] >> 32;
  }

  inline int src() const {
    return keys[0] & 0xFFFFFFFF;
  }

  inline void push(int src, Pos end) {
    keys[base + src - 1] = (static_cast<uint64_t>(end) << 32) | src;
    for (int i = (base + src) >> 1; i > 0; i >>= 1) {
      auto l = (i << 1) - 1;
      auto r = (i << 1);
      if (keys[l] < keys[r]) {
        keys[i - 1] = keys[l];
      } else {
        keys[i - 1] = keys[r];
      }
    }
  }
};

// Provides a light-weight hash table to test for redundant insertions into the
// tournament tree. Insertion is best-effort as we care only about performance.
template <int sources>
struct HashTable {
  struct Node {
    int key;
    int size;
    int vals[sources];
  };
  static constexpr auto buckets = 2 * util::round_up_to_power_of_two(sources);
  Node nodes[buckets];

  HashTable() {
    for (int i = 0; i < buckets; i += 1) {
      nodes[i].key = 0;
    }
  }

  inline Node& lookup(Pos end) {
    return nodes[end & (buckets - 1)];
  }

  inline const Node& lookup(Pos end) const {
    return nodes[end & (buckets - 1)];
  }

  inline bool insert(int src, Pos end) {
    auto& node = lookup(end);
    if (node.key != end) {
      if (node.key == 0) {
        node.key = end;
        node.size = 0;
      }
      return true;
    } else if (node.size == sources) {
      return true;
    }
    node.vals[node.size++] = src;
    return false;
  }
};

// Simplified version of eval routine (no tournament tree, no hash) for
// comparison
template <int sources, typename Evaluator, typename Val>
void unaccelerated_eval(Evaluator evaluator, EvalOutput<Val>& output) {
  Pos prev_end = 0;
  Val prev_val;
  while (true) {
    int min_src = 0;
    Pos min_end = evaluator.peek_end(0);
    for (int src = 1; src < sources; src++) {
      if (auto ep = evaluator.peek_end(src); ep < min_end) {
        min_end = ep;
        min_src = src;
      }
    }

    if (prev_end != min_end) {
      auto val = evaluator.eval();
      if (prev_end && prev_val == val) {
        output.rewind();
      }

      if (min_end < evaluator.stop()) {
        output.emit(min_end, val);
      } else {
        output.emit(evaluator.stop(), val);
        break;
      }

      prev_end = min_end;
      prev_val = val;
    }

    for (int src = 0; src < sources; src++) {
      if (auto src_end = evaluator.peek_end(src); src_end == prev_end) {
        evaluator.next_val(src);
        evaluator.next_end(src);
      }
    }
  }
}

// Core evaluation routine. The evaluator is used to maintain the frontier of
// values, iterate them forward on a per-source basis, and evaluate them into
// an output value emitted into the output store. The implementation uses a
// tournament tree to speed up min selection as well as a hasing trick to avoid
// repeating the min-selection for repeated end position.
template <int sources, typename Evaluator, typename Val>
void eval(Evaluator evaluator, EvalOutput<Val>& output) {
  HashTable<sources> hash;
  TournamentTree<sources> tree(evaluator);

  Pos prev_end = 0;
  Val prev_val;
  for (;;) {
    auto src = tree.src();
    auto end = tree.end();

    // Emit the current range with compression.
    if (prev_end != end) {
      auto val = evaluator.eval();
      if (prev_end && prev_val == val) {
        output.rewind();
      }
      if (end < evaluator.stop()) {
        output.emit(end, val);
      } else {
        output.emit(evaluator.stop(), val);
        break;
      }
      prev_end = end;
      prev_val = val;
    }

    // Advance the value pointers of sources with this shared end position.
    if (auto& node = hash.lookup(end); node.key == end) {
      for (int i = 0; i < node.size; i += 1) {
        evaluator.next_val(node.vals[i]);
      }
      node.key = 0;
    }

    // Loop until we have a new end position for this source.
    end = evaluator.next_end(src);
    while (prev_end == end) {
      evaluator.next_val(src);
      end = evaluator.next_end(src);
    }

    // Also loop while the position is already in the hash table.
    while (end < evaluator.stop()) {
      if (hash.insert(src, end)) {
        break;
      }
      end = evaluator.next_end(src);
    }
    evaluator.next_val(src);

    // Update the tournament tree with the new end position.
    tree.push(src, end);
  }
}

// A source of input to an evaluation. A source provides iteration over a
// store along with a mapping of end positions to output coordinates.
template <typename Val, typename StepFn = step::IdentityStepFn>
class SimpleSource {
 public:
  SimpleSource() = delete;

  SimpleSource(StorePtr<Val> store) : SimpleSource(store, 0, store->span()) {}

  SimpleSource(StorePtr<Val> store, Pos start, Pos stop)
      : SimpleSource(std::move(store), start, stop, StepFn()) {}

  SimpleSource(StorePtr<Val> store, Pos start, Pos stop, StepFn step_fn)
      : store_(std::move(store)),
        start_(start),
        stop_(stop),
        step_fn_(std::move(step_fn)) {
    CHECK_ARGUMENT(start_ >= 0);
    CHECK_ARGUMENT(start_ <= stop_);
    CHECK_ARGUMENT(start_ < store_->span());
    CHECK_ARGUMENT(stop_ <= store_->span());
  }

  inline int span() const {
    return step::span(start_, stop_, step_fn_);
  }

  inline Pos stop() const {
    return step_fn_(stop_);
  }

  inline int capacity() const {
    return 1 + store_->index(stop_ - 1) - store_->index(start_);
  }

  int iter() const {
    return store_->index(start_);
  }

  Pos end(int index) const {
    return step_fn_(store_->ends[index]);
  }

  Val val(int index) const {
    return store_->vals[index];
  }

  inline auto split(Pos start, Pos stop) const {
    return std::make_shared<SimpleSource<Val, StepFn>>(
        store_,
        step::invert(start + 1, start_, stop_, step_fn_) - 1,
        step::invert(stop, start_, stop_, step_fn_),
        step_fn_);
  }

  inline StepFn step_fn() const {
    return step_fn_;
  }

  inline const StorePtr<Val>& store() const {
    return store_;
  }

  inline Pos* iter_ends() const {
    return &store_->ends[store_->index(start_)];
  }

  inline Val* iter_vals() const {
    return &store_->vals[store_->index(start_)];
  }

 private:
  StorePtr<Val> store_;
  Pos start_;
  Pos stop_;
  StepFn step_fn_;
};

// Mix sources provide multi-typed input to an evaluation. Iteration is
// exposed via an indexing method. End positions are mapped internally.
using Mix = std::variant<bool, char, int, float>;

struct MixSourceBase {
  ~MixSourceBase() = default;
  virtual Pos span() const = 0;
  virtual Pos stop() const = 0;
  virtual int capacity() const = 0;
  virtual int iter() const = 0;
  virtual Pos end(int index) const = 0;
  virtual Mix val(int index) const = 0;
  virtual std::shared_ptr<MixSourceBase> split(Pos start, Pos stop) const = 0;
};

template <typename Val, typename StepFn = step::IdentityStepFn>
class MixSource : public MixSourceBase {
 public:
  MixSource(StorePtr<Val> store) : MixSource(store, 0, store->span()) {}

  MixSource(StorePtr<Val> store, Pos start, Pos stop)
      : MixSource(std::move(store), start, stop, StepFn()) {}

  MixSource(StorePtr<Val> store, Pos start, Pos stop, StepFn step_fn)
      : store_(std::move(store)),
        start_(start),
        stop_(stop),
        step_fn_(std::move(step_fn)) {
    CHECK_ARGUMENT(start_ >= 0);
    CHECK_ARGUMENT(start_ <= stop_);
    CHECK_ARGUMENT(start_ < store_->span());
    CHECK_ARGUMENT(stop_ <= store_->span());
  }

  int span() const override {
    return step::span(start_, stop_, step_fn_);
  }

  int stop() const override {
    return step_fn_(stop_);
  }

  int capacity() const override {
    return 1 + store_->index(stop_ - 1) - store_->index(start_);
  }

  int iter() const override {
    return store_->index(start_);
  }

  Pos end(int index) const override {
    return step_fn_(store_->ends[index]);
  }

  Mix val(int index) const override {
    return store_->vals[index];
  }

  std::shared_ptr<MixSourceBase> split(Pos start, Pos stop) const override {
    return std::make_shared<MixSource<Val, StepFn>>(
        store_,
        step::invert(start + 1, start_, stop_, step_fn_) - 1,
        step::invert(stop, start_, stop_, step_fn_),
        step_fn_);
  }

 private:
  StorePtr<Val> store_;
  Pos start_;
  Pos stop_;
  StepFn step_fn_;
};

// Encapsulates a collection of input sources. The set of input sources
// compose an eval step and must have the same span. The pool is also designed
// to allow for paritioning so that it can be processed in parallel on
// multiple threads.
template <typename Source, int pool_size>
struct Pool {
  static_assert(pool_size > 0);
  static constexpr auto size = pool_size;
  using Val = decltype(std::declval<Source>().val(0));

  std::array<std::shared_ptr<Source>, pool_size> sources;

  Pool(std::array<std::shared_ptr<Source>, size> sources)
      : sources(std::move(sources)) {
    for (int i = 1; i < pool_size; i += 1) {
      CHECK_ARGUMENT(this->sources[i - 1]->span() == this->sources[i]->span());
      CHECK_ARGUMENT(this->sources[i - 1]->stop() == this->sources[i]->stop());
    }
  }

  inline Source& operator[](int index) {
    return *sources[index];
  }

  inline const Source& operator[](int index) const {
    return *sources[index];
  }

  inline int span() const {
    return sources[0]->span();
  }

  inline int stop() const {
    return sources[0]->stop();
  }

  inline int capacity() const {
    int ret = 1;
    for (const auto& source : sources) {
      ret += source->capacity() - 1;
    }
    return ret;
  }
};

template <typename Val, typename StepFn, int size>
using SimplePool = Pool<SimpleSource<Val, StepFn>, size>;

template <typename Source, typename Head, typename... Tail>
auto make_pool(Head&& head, Tail&&... tail) {
  constexpr auto size = 1 + sizeof...(tail);
  return Pool<Source, size>(std::array<std::shared_ptr<Source>, size>{
      std::make_shared<Head>(std::forward<Head>(head)),
      std::make_shared<Tail>(std::forward<Tail>(tail))...});
}

template <typename Head, typename... Tail>
auto make_pool(Head&& head, Tail&&... tail) {
  return make_pool<std::decay_t<Head>, Head, Tail...>(
      std::forward<Head>(head), std::forward<Tail>(tail)...);
}

template <typename Source, int size>
auto partition_pool(const Pool<Source, size>& pool, int parts) {
  // TODO: Instead of using uniform ranges, bisect for ranges with uniform
  // capacity so that parallel work is distributed evenly across threads.
  uint64_t span = pool.span();
  std::vector<Pool<Source, size>> ret(parts, pool);
  for (int i = 0; i < parts; i += 1) {
    auto l = static_cast<Pos>(i * span / parts);
    auto r = static_cast<Pos>((i + 1) * span / parts);
    for (int j = 0; j < pool.size; j += 1) {
      ret[i].sources[j] = ret[i][j].split(l, r);
    }
  }
  return ret;
}

// Maps a value frontier to an output value for simple evaluation.
template <typename Arg, typename Ret>
using EvalFn = std::function<Ret(const Arg*)>;

// Provides evaluation over a pool of simple sources via an eval function.
template <typename Ret, typename Arg, typename StepFn, int size>
class SimpleEvaluator {
 public:
  SimpleEvaluator(SimplePool<Arg, StepFn, size> pool, EvalFn<Arg, Ret> eval_fn)
      : stop_(pool.stop()),
        pool_(std::move(pool)),
        eval_fn_(std::move(eval_fn)) {
    for (int src = 0; src < size; src += 1) {
      step_fns_[src] = pool_[src].step_fn();
      iter_ends_[src] = pool_[src].iter_ends();
      iter_vals_[src] = pool_[src].iter_vals();
      next_val(src);
    }
  }

  inline auto stop() const {
    return stop_;
  }

  inline const auto& pool() const {
    return pool_;
  }

  inline auto next_val(int src) {
    return curr_vals_[src] = *iter_vals_[src]++;
  }

  inline auto next_end(int src) {
    return step_fns_[src](*iter_ends_[src]++);
  }

  inline auto peek_end(int src) {
    return step_fns_[src](*iter_ends_[src]);
  }

  inline auto eval() const {
    return eval_fn_(&curr_vals_[0]);
  }

 private:
  Pos stop_;
  SimplePool<Arg, StepFn, size> pool_;
  EvalFn<Arg, Ret> eval_fn_;
  StepFn step_fns_[size];
  Pos* iter_ends_[size];
  Arg* iter_vals_[size];
  Arg curr_vals_[size];
};

// Provides evaluation over a pool of input sources via an eval function.
template <typename Ret, typename Source, int size>
class SourceEvaluator {
 public:
  using Arg = typename Pool<Source, size>::Val;

  SourceEvaluator(Pool<Source, size> pool, EvalFn<Arg, Ret> eval_fn)
      : stop_(pool.stop()),
        pool_(std::move(pool)),
        eval_fn_(std::move(eval_fn)) {
    for (int src = 0; src < size; src += 1) {
      iter_ends_[src] = pool_[src].iter();
      iter_vals_[src] = pool_[src].iter();
      next_val(src);
    }
  }

  inline auto stop() const {
    return stop_;
  }

  inline const auto& pool() const {
    return pool_;
  }

  inline auto next_val(int src) {
    return curr_vals_[src] = pool_[src].val(iter_vals_[src]++);
  }

  inline auto next_end(int src) {
    return pool_[src].end(iter_ends_[src]++);
  }

  inline auto peek_end(int src) {
    return pool_[src].end(iter_ends_[src]);
  }

  inline auto eval() const {
    return eval_fn_(&curr_vals_[0]);
  }

 private:
  Pos stop_;
  Pool<Source, size> pool_;
  EvalFn<Arg, Ret> eval_fn_;
  int iter_ends_[size];
  int iter_vals_[size];
  Arg curr_vals_[size];
};

template <typename Val>
auto fuse_stores(const std::vector<StorePtr<Val>>& stores) {
  std::vector<std::tuple<int, int, int>> offsets(stores.size());

  // Compute the source and destination positives to copy the inputs stores
  // into the fused output store, handling compression at the edges.
  auto dst_b = 0;
  for (int i = 0; i < stores.size(); i += 1) {
    auto src_b = 0;
    auto src_e = stores[i]->size;
    CHECK_STATE(src_b < src_e);

    // Adjust the offsets based on compression.
    if (i > 0) {
      auto prev = stores[i - 1];
      if (prev->vals[prev->size - 1] == stores[i]->vals[0]) {
        auto [b, e, d] = offsets[i - 1];
        offsets[i - 1] = std::make_tuple(b, e - 1, d);
        dst_b -= 1;
      }
    }

    offsets[i] = std::make_tuple(src_b, src_e, dst_b);
    dst_b += src_e - src_b;
  }

  // Evaluate each part in parallel.
  auto ret = std::make_shared<core::Store<Val>>(dst_b);
  std::vector<std::function<void()>> tasks;
  for (int i = 0; i < stores.size(); i += 1) {
    tasks.emplace_back([&, i] {
      auto [src_b, src_e, dst_b] = offsets[i];
      auto& store = stores[i];
      std::move(&store->ends[src_b], &store->ends[src_e], &ret->ends[dst_b]);
      std::move(&store->vals[src_b], &store->vals[src_e], &ret->vals[dst_b]);
    });
  }
  threads::run_in_parallel(tasks);
  return ret;
}

template <typename Evaluator>
auto eval_generic(Evaluator evaluator) {
  CHECK_ARGUMENT(evaluator.pool().span() > 0);

  const bool useAcceleratedEval =
      GlobalConfig::get().getConfigVal<bool>("accelerated_eval", true);

  static constexpr auto size = std::decay_t<decltype(evaluator.pool())>::size;
  using Val = decltype(evaluator.eval());

  // Allocate the output store.
  auto store = std::make_shared<core::Store<Val>>(evaluator.pool().capacity());

  // Evaluate the output store.
  EvalOutput<Val> output(&store->ends[0], &store->vals[0]);
  if (useAcceleratedEval) {
    eval<size>(std::move(evaluator), output);
  } else {
    unaccelerated_eval<size>(std::move(evaluator), output);
  }

  // Resize the output (due to compression).
  store->size = output.ends - &store->ends[0];
  return store;
}

template <typename Ret, typename Arg, typename StepFn, int size>
auto eval_simple(EvalFn<Arg, Ret> eval_fn, SimplePool<Arg, StepFn, size> pool) {
  auto par_threshold = GlobalConfig::get().getConfigVal<int64_t>(
      "parallelize_threshold", 8 * 1024);
  auto par_parts = GlobalConfig::get().getConfigVal<int64_t>(
      "parallelize_parts", std::thread::hardware_concurrency());

  // Evaluate inline if the pool size is below the threshold.
  if (pool.capacity() < par_threshold || par_parts <= 1) {
    return eval_generic(SimpleEvaluator(std::move(pool), std::move(eval_fn)));
  }

  auto partition = partition_pool(pool, par_parts);

  // Evaluate each part in parallel.
  std::vector<std::function<void()>> tasks;
  std::vector<StorePtr<Ret>> stores(partition.size());
  for (int i = 0; i < partition.size(); i += 1) {
    tasks.emplace_back([&, i] {
      SimpleEvaluator evaluator(std::move(partition[i]), eval_fn);
      stores[i] = eval_generic(std::move(evaluator));
    });
  }
  threads::run_in_parallel(tasks);

  // Assemble each parts store back together.
  return fuse_stores(stores);
}

template <typename Ret, typename Arg, typename StepFn, typename... Sources>
auto eval_simple(
    EvalFn<Arg, Ret> eval_fn,
    SimpleSource<Arg, StepFn> head,
    Sources&&... tail) {
  constexpr auto size = 1 + sizeof...(tail);
  return eval_simple<Ret, Arg, StepFn, size>(
      std::move(eval_fn),
      make_pool(std::move(head), std::forward<Sources>(tail)...));
}

template <typename Ret, int size>
auto eval_mixed(EvalFn<Mix, Ret> eval_fn, Pool<MixSourceBase, size> pool) {
  static constexpr auto kParallelizeThreshold = 8 * 1024;
  static auto kParallelizeParts = std::thread::hardware_concurrency();

  // Evaluate inline if the pool size is below the threshold.
  if (pool.capacity() < kParallelizeThreshold) {
    return eval_generic(SourceEvaluator(std::move(pool), std::move(eval_fn)));
  }

  auto partition = partition_pool(pool, kParallelizeParts);

  // Evaluate each part in parallel.
  std::vector<std::function<void()>> tasks;
  std::vector<StorePtr<Ret>> stores(partition.size());
  for (int i = 0; i < partition.size(); i += 1) {
    tasks.emplace_back([&, i] {
      SourceEvaluator evaluator(std::move(partition[i]), eval_fn);
      stores[i] = eval_generic(std::move(evaluator));
    });
  }
  threads::run_in_parallel(tasks);

  // Assemble each parts store back together.
  return fuse_stores(stores);
}

template <typename Ret, typename Head, typename... Tail>
auto eval_mixed(EvalFn<Mix, Ret> eval_fn, Head head, Tail&&... tail) {
  constexpr auto size = 1 + sizeof...(tail);
  return eval_mixed<Ret, size>(
      std::move(eval_fn),
      make_pool<MixSourceBase>(std::move(head), std::forward<Tail>(tail)...));
}

}  // namespace skimpy::detail::eval
