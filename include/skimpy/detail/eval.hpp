#pragma once

#include <array>
#include <functional>
#include <memory>
#include <variant>

#include "core.hpp"
#include "step.hpp"
#include "util.hpp"

namespace skimpy::detail::eval {

template <typename Val>
using Store = core::Store<Val>;
using Pos = core::Pos;
using StepFn = step::StepFn;

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
  static constexpr int base = round_up_to_power_of_two(sources);
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
  static constexpr int size = round_up_to_power_of_two(sources);
  std::unique_ptr<Node[]> nodes;

  HashTable() : nodes(new Node[size]) {
    for (int i = 0; i < size; i += 1) {
      nodes[i].key = 0;
    }
  }

  inline Node& lookup(Pos end) {
    return nodes[end & (size - 1)];
  }

  inline const Node& lookup(Pos end) const {
    return nodes[end & (size - 1)];
  }

  inline bool insert(int src, Pos end) {
    auto& node = lookup(end);
    if (node.key != end) {
      if (node.key == 0) {
        node.key = end;
        node.size = 0;
      }
      return true;
    }
    node.vals[node.size++] = src;
    return false;
  }
};

// Core evaluation routine. The evaluator is used to maintain the frontier of
// values, iterate them forward on a per-source basis, and evaluate them into
// an output value emitted into the output store. The implementation uses a
// tournament tree to speed up min selection as well as a hasing trick to avoid
// repeating the min-selection for repeated end position.
template <int sources, typename Evaluator, typename Val>
void eval(Evaluator&& evaluator, EvalOutput<Val>& output) {
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
    if (auto& node = hash.lookup(prev_end); node.key == prev_end) {
      for (int i = 0; i < node.size; i += 1) {
        evaluator.next_val(node.vals[i]);
      }
      node.key = 0;
    }

    // Loop until we have a new distinct end position for this source.
    auto new_end = evaluator.next_end(src);
    while (new_end < evaluator.stop()) {
      if (hash.insert(src, new_end)) {
        break;
      }
      new_end = evaluator.next_end(src);
    }
    evaluator.next_val(src);

    // Update the tournament tree with the new end position.
    tree.push(src, new_end);
  }
}

// A source of input to an evaluation. A source provides iteration over a store
// along with a mapping of end positions to output coordinates.
template <typename Val>
class SimpleSource {
 public:
  SimpleSource() = delete;

  SimpleSource(std::shared_ptr<Store<Val>> store)
      : SimpleSource(store, 0, store->span()) {}

  SimpleSource(std::shared_ptr<Store<Val>> store, Pos start, Pos stop)
      : SimpleSource(std::move(store), start, stop, step::step_fn()) {}

  SimpleSource(
      std::shared_ptr<Store<Val>> store, Pos start, Pos stop, StepFn step_fn)
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

  inline Pos start() const {
    return start_;
  }

  inline Pos stop() const {
    return stop_;
  }

  inline int index(Pos pos) const {
    return store_->index(pos);
  }

  inline Pos end(int index) const {
    return step_fn_(store->ends[index]);
  }

  inline Val val(int index) const {
    return store_->vals[index];
  }

  inline auto split(Pos start, Pos stop) const {
    return std::make_unique<SimpleSource<Val>>(store_, start_, stop_, step_fn_);
  }

  inline StepFn step_fn() const {
    return step_fn_;
  }

  inline const std::shared_ptr<Store<Val>>& store() const {
    return store_;
  }

  inline Pos* iter_ends() const {
    return &store_->ends[index(start_)];
  }

  inline Val* iter_vals() const {
    return &store_->vals[index(start_)];
  }

 private:
  std::shared_ptr<Store<Val>> store_;
  Pos start_;
  Pos stop_;
  StepFn step_fn_;
};

// Mix sources provide multi-typed input to an evaluation. Iteration is exposed
// via an indexing method. End positions are mapped internally.
using Mix = std::variant<bool, char, int, float>;

struct MixSourceBase {
  ~MixSourceBase() = default;
  virtual int span() const = 0;
  virtual Pos start() const = 0;
  virtual Pos stop() const = 0;
  virtual int index(Pos pos) const = 0;
  virtual Pos end(int index) const = 0;
  virtual Mix val(int index) const = 0;
  virtual std::shared_ptr<MixSourceBase> split(Pos start, Pos stop) const = 0;
};

template <typename Val>
class MixSource : public MixSourceBase {
 public:
  MixSource(std::shared_ptr<Store<Val>> store)
      : MixSource(store, 0, store->span()) {}

  MixSource(std::shared_ptr<Store<Val>> store, Pos start, Pos stop)
      : MixSource(std::move(store), start, stop, step::step_fn()) {}

  MixSource(
      std::shared_ptr<Store<Val>> store, Pos start, Pos stop, StepFn step_fn)
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

  int start() const override {
    return start_;
  }

  int stop() const override {
    return stop_;
  }

  int index(Pos pos) const override {
    return store_->index(pos);
  }

  Pos end(int index) const override {
    return step_fn_(store_->ends[index]);
  }

  Mix val(int index) const override {
    return store_->vals[index];
  }

  std::shared_ptr<MixSourceBase> split(Pos start, Pos stop) const override {
    return std::make_shared<MixSource<Val>>(store_, start_, stop_, step_fn_);
  }

 private:
  std::shared_ptr<Store<Val>> store_;
  Pos start_;
  Pos stop_;
  StepFn step_fn_;
};

// Encapsulates a collection of input sources. The set of input sources compose
// an eval step and must have the same span. The pool is also designed to allow
// for paritioning so that it can be processed in parallel on multiple threads.
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

  inline int capacity() const {
    int ret = 1;
    for (const auto& source : sources) {
      ret += source->index(source->stop() - 1) - source->index(source->start());
    }
    return ret;
  }
};

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

// Maps a value frontier to an output value for simple evaluation.
template <typename Arg, typename Ret>
using EvalFn = std::function<Ret(const Arg*)>;

// Provides evaluation over a pool of simple sources via an eval function.
template <typename Ret, typename Arg, int size>
class SimpleEvaluator {
 public:
  SimpleEvaluator(
      Pos stop, Pool<SimpleSource<Arg>, size> pool, EvalFn<Arg, Ret> eval_fn)
      : stop_(stop), pool_(std::move(pool)), eval_fn_(std::move(eval_fn)) {
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

  inline auto eval() const {
    return eval_fn_(&curr_vals_[0]);
  }

 private:
  Pos stop_;
  Pool<SimpleSource<Arg>, size> pool_;
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

  SourceEvaluator(Pos stop, Pool<Source, size> pool, EvalFn<Arg, Ret> eval_fn)
      : stop_(stop), pool_(std::move(pool)), eval_fn_(std::move(eval_fn)) {
    for (int src = 0; src < size; src += 1) {
      iter_ends_[src] = pool_[src].index(pool_[src].start());
      iter_vals_[src] = pool_[src].index(pool_[src].start());
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

template <typename Evaluator>
auto eval_generic(Evaluator evaluator) {
  static constexpr auto size = std::decay_t<decltype(evaluator.pool())>::size;
  using Val = decltype(evaluator.eval());

  // Allocate the output store.
  auto store = std::make_shared<Store<Val>>(evaluator.pool().capacity());

  // Evaluate the output store.
  EvalOutput<Val> output(&store->ends[0], &store->vals[0]);
  eval<size>(evaluator, output);

  // Resize the output (due to compression).
  store->size = output.ends - &store->ends[0];
  return store;
}

template <typename Arg, typename Ret, int size>
auto eval_simple(EvalFn<Arg, Ret> eval_fn, Pool<SimpleSource<Arg>, size> pool) {
  auto s = pool.span();
  return eval_generic(SimpleEvaluator(s, std::move(pool), std::move(eval_fn)));
}

template <typename Arg, typename Ret, typename... Sources>
auto eval_simple(
    EvalFn<Arg, Ret> eval_fn, SimpleSource<Arg> head, Sources&&... tail) {
  constexpr auto size = 1 + sizeof...(tail);
  return eval_simple<Arg, Ret, size>(
      std::move(eval_fn),
      make_pool(std::move(head), std::forward<Sources>(tail)...));
}

template <typename Ret, int size>
auto eval_mixed(EvalFn<Mix, Ret> eval_fn, Pool<MixSourceBase, size> pool) {
  auto s = pool.span();
  return eval_generic(SourceEvaluator(s, std::move(pool), std::move(eval_fn)));
}

template <typename Ret, typename Head, typename... Tail>
auto eval_mixed(EvalFn<Mix, Ret> eval_fn, Head head, Tail&&... tail) {
  constexpr auto size = 1 + sizeof...(tail);
  return eval_mixed<Ret, size>(
      std::move(eval_fn),
      make_pool<MixSourceBase>(std::move(head), std::forward<Tail>(tail)...));
}

}  // namespace skimpy::detail::eval
