
#pragma once

#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core.hpp"
#include "dags.hpp"
#include "util.hpp"

namespace skimpy::detail::step {

using Pos = core::Pos;

using SimpleStepFn = Pos (*)(Pos);

struct IdentityStepFn {
  Pos operator()(Pos pos) const {
    return pos;
  }
};

namespace cyclic {

template <Pos pos>
static Pos kFixedPos = pos;

static constexpr auto kMaxExecDeps = 2;
static constexpr auto kMaxLutSize = 1 << 8;
static constexpr auto kMaxSpan = 1 << 30;

struct ExecNode {
  int span;
  int step;
  ExecNode* deps[kMaxExecDeps];
  enum { EMPTY, TABLE, STACK } kind;
  union {
    struct StackArgs {
      int reps;
      Pos loop_span;
      Pos loop_step;
      int bit_shift;
    } stack;
    struct TableArgs {
      Pos* lut;
      int mask;
    } table;
  };

  ExecNode() : span(0), step(0), kind(EMPTY) {
    deps[0] = nullptr;
    deps[1] = nullptr;
  }
};

struct ExecGraph {
  int size;
  ExecNode* root;
  std::shared_ptr<ExecNode[]> nodes;
  std::shared_ptr<Pos[]> table;

  ExecGraph() : size(0), root(nullptr), nodes(nullptr), table(nullptr) {}

  ExecGraph(
      int size,
      ExecNode* root,
      std::shared_ptr<ExecNode[]> nodes,
      std::shared_ptr<Pos[]> table)
      : size(size), root(root), nodes(std::move(nodes)), table(table) {}
};

inline auto empty_exec_graph() {
  std::shared_ptr<ExecNode[]> nodes(new ExecNode[1]);
  auto root = &nodes[0];
  root->span = kMaxSpan;
  root->step = kMaxSpan;
  root->kind = ExecNode::TABLE;
  root->table.lut = &kFixedPos<0>;
  root->table.mask = 0;
  return ExecGraph(1, root, std::move(nodes), nullptr);
}

class StepFn {
 public:
  StepFn(Pos start, Pos span, ExecGraph graph)
      : start_(start), span_(span), graph_(std::move(graph)) {
    CHECK_ARGUMENT(start >= 0);
    CHECK_ARGUMENT(span >= 0);
    CHECK_ARGUMENT(span_ <= graph.root->span);
  }

  StepFn(Pos start, Pos span, const StepFn& other)
      : StepFn(
            other.start_ + start,
            std::min<Pos>(span, other.span_ - start),
            other.graph_) {
    CHECK_ARGUMENT(start >= 0);
    CHECK_ARGUMENT(span >= 0);
  }

  StepFn() : StepFn(0, kMaxSpan, empty_exec_graph()) {}

  Pos operator()(Pos pos) const {
    pos -= start_ + 1;
    if (cache_.base > pos || pos >= cache_.stop) {
      if (pos < 0) {
        return 0;
      }
      if (pos >= span_) {
        pos = span_ - 1;
      }
      search(pos);
    }
    return cache_.step + cache_.lut[cache_.mask & (pos - cache_.base)];
  }

 private:
  void search(Pos pos) const {
    // TODO: Consider adding a stack to speed up the search for nearby nodes.
    auto node = graph_.root;
    auto base = 0, step = 0;
    while (node->kind != ExecNode::TABLE) {
      auto& args = node->stack;

      // Identify the repetition counter for the current position.
      auto quo = 0, rem = 0;
      if (args.bit_shift) {
        quo = (pos - base) >> args.bit_shift;
        rem = (pos - base) & (args.loop_span - 1);
      } else {
        quo = (pos - base) / args.loop_span;
        rem = (pos - base) % args.loop_span;
      }

      // Update the position offsets for the respective repetition.
      base += quo * args.loop_span;
      step += quo * args.loop_step;

      // Recurse on the apropriate node.
      if (auto l = node->deps[0]; rem < l->span) {
        node = l;
      } else {
        base += l->span;
        step += l->step;
        node = node->deps[1];
      }
    }

    cache_.base = base;
    cache_.step = step;
    cache_.stop = std::min<Pos>(base + node->span, span_);
    cache_.lut = node->table.lut;
    cache_.mask = node->table.mask;
  }

  Pos start_;
  Pos span_;
  ExecGraph graph_;
  mutable struct {
    Pos base = 0;
    Pos stop = 0;
    Pos step = 0;
    Pos* lut = nullptr;
    int mask = 0xFFFFFFFF;
  } cache_;
};

struct ExprData {
  enum { TABLE, STACK, FIXED, SCALED, STRIDED } kind;
  int span;
  int step;
  union {
    struct {
      int reps;
      Pos loop_span;
      Pos loop_step;
    } stack;
    struct {
      Pos* lut;
      Pos mask;
    } table;
    struct {
      Pos scale;
    } scaled;
    struct {
      Pos stride;
    } strided;
  };
};

constexpr auto kMaxExprDeps = 2;

using ExprNode = dags::SharedNode<kMaxExprDeps, ExprData>;

template <typename Fn>
inline void expr_dfs(ExprNode::Ptr expr, Fn&& fn) {
  dags::dfs(std::move(expr), [fn = std::forward<Fn>(fn)](auto e, auto& q) {
    if (fn(e)) {
      for (int i = 0; i < kMaxExprDeps; i += 1) {
        if (auto dep = e->deps[i]) {
          q.push_back(dep);
        } else {
          break;
        }
      }
      q.push_back(std::move(e));
    }
  });
};

inline auto stack(int reps, ExprNode::Ptr l, ExprNode::Ptr r = nullptr) {
  CHECK_ARGUMENT(l);
  auto ret = ExprNode::make_ptr();
  ret->data.stack.reps = reps;
  ret->data.stack.loop_span = l->data.span;
  ret->data.stack.loop_step = l->data.step;
  if (r) {
    ret->data.stack.loop_span += r->data.span;
    ret->data.stack.loop_step += r->data.step;
  }
  ret->data.span = reps * ret->data.stack.loop_span;
  ret->data.step = reps * ret->data.stack.loop_step;
  ret->data.kind = ExprData::STACK;
  ret->deps[0] = l;
  ret->deps[1] = r;
  return ret;
}

inline auto stack(ExprNode::Ptr l, ExprNode::Ptr r) {
  return stack(1, l, r);
}

inline auto table(Pos span, Pos* lut, Pos mask = 0xFFFFFFFF) {
  CHECK_ARGUMENT((span & mask) <= kMaxLutSize);
  auto ret = ExprNode::make_ptr();
  ret->data.table.lut = lut;
  ret->data.table.mask = mask;
  ret->data.span = span;
  ret->data.step = lut[mask & (span - 1)];
  ret->data.kind = ExprData::TABLE;
  return ret;
}

template <Pos k>
inline auto fixed(int span = 1) {
  CHECK_ARGUMENT(span > 0);
  return table(span, &kFixedPos<k>, 0);
}

template <Pos k>
inline auto scaled(int span) {
  CHECK_ARGUMENT(span > 0);
  if (span <= kMaxLutSize) {
    return table(span, scaled_lut<k>());
  } else {
    auto loop_span = kMaxLutSize;
    auto quo = span / loop_span, rem = span % loop_span;
    if (rem == 0) {
      return stack(quo, table(loop_span, scaled_lut<k>()));
    } else {
      return stack(
          stack(quo, table(loop_span, scaled_lut<k>())),
          table(rem, scaled_lut<k>()));
    }
  }
}

template <Pos k>
inline auto strided(int span) {
  static_assert(k < kMaxLutSize, "Static strides cannot exceed max LUT size.");
  CHECK_ARGUMENT(span > 0);
  if (span <= kMaxLutSize) {
    return table(span, strided_lut<k>());
  } else {
    auto loop_span = kMaxLutSize - (kMaxLutSize % k);
    auto quo = span / loop_span, rem = span % loop_span;
    if (rem == 0) {
      return stack(quo, table(loop_span, strided_lut<k>()));
    } else {
      return stack(
          stack(quo, table(loop_span, strided_lut<k>())),
          table(rem, strided_lut<k>()));
    }
  }
}

inline auto fixed(Pos span, Pos step) {
  CHECK_ARGUMENT(span > 0);
  CHECK_ARGUMENT(step >= 0);
  auto ret = ExprNode::make_ptr();
  ret->data.span = span;
  ret->data.step = step;
  ret->data.kind = ExprData::FIXED;
  return ret;
}

inline auto scaled(int span, Pos scale) -> ExprNode::Ptr {
  CHECK_ARGUMENT(span > 0);
  CHECK_ARGUMENT(scale > 0);
  if (span > kMaxLutSize) {
    auto loop_span = kMaxLutSize;
    auto quo = span / loop_span, rem = span % loop_span;
    if (rem == 0) {
      return stack(quo, scaled(loop_span, scale));
    } else {
      return stack(stack(quo, scaled(loop_span, scale)), scaled(rem, scale));
    }
  } else {
    auto ret = ExprNode::make_ptr();
    ret->data.scaled.scale = scale;
    ret->data.span = span;
    ret->data.step = scale * span;
    ret->data.kind = ExprData::SCALED;
    return ret;
  }
}

inline auto strided(int span, Pos stride) -> ExprNode::Ptr {
  CHECK_ARGUMENT(span > 0);
  CHECK_ARGUMENT(stride > 0);
  if (stride > kMaxLutSize) {
    auto quo = span / stride, rem = span % stride;
    if (rem == 0) {
      return stack(quo, fixed<0>(stride - 1), fixed<1>(1));
    } else {
      return stack(stack(quo, fixed<0>(stride - 1), fixed<1>()), fixed<0>(rem));
    }
  } else if (span > kMaxLutSize) {
    auto loop_span = kMaxLutSize - (kMaxLutSize % stride);
    auto quo = span / loop_span, rem = span % loop_span;
    if (rem == 0) {
      return stack(
          stack(quo, strided(loop_span, stride)), strided(rem, stride));
    } else {
      return stack(quo, strided(loop_span, stride));
    }
  } else {
    auto ret = ExprNode::make_ptr();
    ret->data.strided.stride = stride;
    ret->data.span = span;
    ret->data.step = 1 + (span - 1) / stride;
    ret->data.kind = ExprData::STRIDED;
    return ret;
  }
}

inline auto build(int start, int stop, ExprNode::Ptr in) {
  CHECK_ARGUMENT(stop - start <= kMaxSpan);

  // Figure out how much space is required in the exec graph.
  int nodes_size = 0;
  int table_size = 0;
  std::unordered_set<ExprNode::Ptr> visited;
  expr_dfs(in, [&](auto e) {
    if (visited.count(e)) {
      return false;
    }

    // Recurse if any child node is not yet mapped.
    for (int i = 0; i < kMaxExprDeps; i += 1) {
      if (e->deps[i] && !visited.count(e->deps[i])) {
        return true;
      }
    }

    // Count up any LUT space required for this node.
    if (e->data.kind == ExprData::FIXED) {
      table_size += 1;
    } else if (e->data.kind == ExprData::SCALED) {
      table_size += std::min(e->data.span, kMaxLutSize);
    } else if (e->data.kind == ExprData::STRIDED) {
      CHECK_STATE(e->data.span <= kMaxLutSize);
      table_size += e->data.span;
    }

    nodes_size += 1;
    visited.insert(std::move(e));
    return false;
  });

  // Initialize the execution graph storage.
  std::shared_ptr<ExecNode[]> nodes(new ExecNode[nodes_size]);
  std::shared_ptr<Pos[]> table(table_size > 0 ? new Pos[table_size] : nullptr);

  // Populate the execution graph nodes.
  auto nodes_ptr = &nodes[0];
  auto table_ptr = table ? &table[0] : nullptr;
  std::unordered_map<ExprNode::Ptr, ExecNode*> node_map;
  expr_dfs(in, [&](auto e) {
    if (node_map.count(e)) {
      return false;
    }

    // Recurse if any child node is not yet mapped.
    for (int i = 0; i < kMaxExprDeps; i += 1) {
      if (e->deps[i] && !node_map.count(e->deps[i])) {
        return true;
      }
    }

    // Map this node based on its type.
    auto node = node_map[e] = nodes_ptr++;
    node->span = e->data.span;
    node->step = e->data.step;
    if (e->data.kind == ExprData::STACK) {
      auto& args = e->data.stack;
      node->kind = ExecNode::STACK;
      node->stack.reps = args.reps;
      node->stack.loop_span = args.loop_span;
      node->stack.loop_step = args.loop_step;
      if (is_power_of_two(args.loop_span)) {
        node->stack.bit_shift = lg2(args.loop_span);
      } else {
        node->stack.bit_shift = 0;
      }
      if (auto dep = e->deps[0]) {
        node->deps[0] = node_map.at(dep);
      }
      if (auto dep = e->deps[1]) {
        node->deps[1] = node_map.at(dep);
      }
    } else if (e->data.kind == ExprData::TABLE) {
      CHECK_STATE(e->data.span > 0);
      CHECK_STATE((e->data.span & e->data.table.mask) <= kMaxLutSize);
      auto& args = e->data.table;
      node->kind = ExecNode::TABLE;
      node->table.lut = args.lut;
      node->table.mask = args.mask;
    } else if (e->data.kind == ExprData::FIXED) {
      CHECK_STATE(e->data.span > 0);
      node->kind = ExecNode::TABLE;
      node->table.lut = table_ptr;
      node->table.mask = 0;
      *table_ptr++ = e->data.step;
    } else if (e->data.kind == ExprData::SCALED) {
      CHECK_STATE(e->data.span > 0);
      CHECK_STATE(e->data.span <= kMaxLutSize);
      auto& args = e->data.scaled;
      node->kind = ExecNode::TABLE;
      node->table.lut = table_ptr;
      node->table.mask = 0xFFFFFFFF;
      for (int i = 0; i < e->data.span; i += 1) {
        *table_ptr++ = (i + 1) * args.scale;
      }
    } else if (e->data.kind == ExprData::STRIDED) {
      CHECK_STATE(e->data.span > 0);
      CHECK_STATE(e->data.span <= kMaxLutSize);
      auto& args = e->data.strided;
      node->kind = ExecNode::TABLE;
      node->table.lut = table_ptr;
      node->table.mask = 0xFFFFFFFF;
      if (is_power_of_two(args.stride)) {
        auto s = lg2(args.stride);
        for (int i = 0; i < e->data.span; i += 1) {
          *table_ptr++ = 1 + (i >> s);
        }
      } else {
        for (int i = 0; i < e->data.span; i += 1) {
          *table_ptr++ = 1 + i / args.stride;
        }
      }
    }
    return false;
  });

  // Assign the root handle to the exec state before returning.
  CHECK_STATE(nodes_size == nodes_ptr - nodes.get());
  CHECK_STATE(table_size == table_ptr - table.get());
  return StepFn(
      start,
      std::min<Pos>(in->data.span, stop - start),
      ExecGraph(
          nodes_size, node_map.at(in), std::move(nodes), std::move(table)));
}

inline auto build(int stop, ExprNode::Ptr in) {
  return build(0, stop, std::move(in));
}

inline auto build(ExprNode::Ptr in) {
  return build(kMaxSpan, std::move(in));
}

template <Pos k>
inline auto scaled_lut() {
  static const std::unique_ptr<Pos[]> lut = [] {
    std::unique_ptr<Pos[]> lut(new Pos[kMaxLutSize]);
    for (int i = 0; i < kMaxLutSize; i += 1) {
      lut[i] = (i + 1) * k;
    }
    return lut;
  }();
  return &lut[0];
}

template <Pos k>
inline auto strided_lut() {
  static const std::unique_ptr<Pos[]> lut = [] {
    std::unique_ptr<Pos[]> lut(new Pos[kMaxLutSize]);
    for (int i = 0; i < kMaxLutSize; i += 1) {
      lut[i] = 1 + i / k;
    }
    return lut;
  }();
  return &lut[0];
}

inline auto identity() {
  return cyclic::build(cyclic::scaled<1>(cyclic::kMaxSpan));
}

template <int k>
inline auto constant() {
  return cyclic::build(cyclic::fixed<k>(cyclic::kMaxSpan));
}

inline auto zero() {
  return constant<0>();
}

inline auto slice(Pos start, Pos stop, const cyclic::StepFn& fn) {
  return cyclic::StepFn(start, stop - start, fn);
}

inline auto slice(Pos stop, const cyclic::StepFn& fn) {
  return slice(0, stop, fn);
}

inline auto slice(Pos start, Pos stop) {
  return slice(start, stop, identity());
}

inline auto slice(Pos stop) {
  return slice(stop, identity());
}

template <int k>
inline auto scale_fn() {
  return cyclic::build(cyclic::scaled<k>(cyclic::kMaxSpan));
}

template <int k>
inline auto stride_fn() {
  return cyclic::build(cyclic::strided<k>(cyclic::kMaxSpan));
}

}  // namespace cyclic

namespace composite {

class StepFn {
 public:
  StepFn() = default;
  StepFn(std::vector<cyclic::StepFn> fns) : fns_(std::move(fns)) {}
  explicit StepFn(cyclic::StepFn fn) : fns_({std::move(fn)}) {}

  operator bool() const {
    return !fns_.empty();
  }

  auto compose(const StepFn& fn) {
    std::vector<cyclic::StepFn> fns(fn.fns_);
    for (auto& fn : fns_) {
      fns.push_back(std::move(fn));
    }
    fns_.swap(fns);
  }

  auto compose(StepFn&& fn) {
    std::vector<cyclic::StepFn> fns(std::move(fn.fns_));
    for (auto& fn : fns_) {
      fns.push_back(std::move(fn));
    }
    fns_.swap(fns);
  }

  Pos operator()(Pos pos) const {
    Pos ret = pos;
    for (const auto& fn : fns_) {
      ret = fn(ret);
    }
    return ret;
  }

 private:
  std::vector<cyclic::StepFn> fns_;
};

inline auto compose(StepFn f, const StepFn& g) {
  f.compose(g);
  return f;
}

inline auto compose(StepFn f, StepFn&& g) {
  f.compose(std::move(g));
  return f;
}

inline auto compose(cyclic::StepFn f, const StepFn& g) {
  return compose(StepFn(std::move(f)), g);
}

inline auto compose(cyclic::StepFn f, StepFn&& g) {
  return compose(StepFn(std::move(f)), std::move(g));
}

}  // namespace composite

template <typename Fn>
inline Pos span(Pos start, Pos stop, Fn&& fn) {
  return fn(stop) - fn(start);
}

inline Pos span(Pos start, Pos stop) {
  return span(start, stop, IdentityStepFn());
}

template <typename Fn>
inline Pos invert(Pos pos, Pos start, Pos stop, Fn&& step) {
  CHECK_ARGUMENT(start <= stop);
  auto l = start;
  auto h = stop;
  while (l < h) {
    auto m = (l + h) >> 1;
    if (step(m) < pos) {
      l = m + 1;
    } else {
      h = m;
    }
  }
  return l;
}

inline Pos invert(Pos pos, Pos start, Pos stop) {
  return invert(pos, start, stop, IdentityStepFn());
}

}  // namespace skimpy::detail::step
