#pragma once

#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "box.hpp"
#include "core.hpp"
#include "dags.hpp"
#include "eval.hpp"
#include "step.hpp"

namespace skimpy::detail::lang {

using box::BoxStore;

constexpr int kMaxExprDeps = 3;

// Argument to a store expression
struct StoreArgs {
  std::shared_ptr<BoxStore> store;

  StoreArgs(std::shared_ptr<BoxStore> store) : store(std::move(store)) {}
};

// Argument to a slice expression
struct SliceArgs {
  std::vector<step::cyclic::StepFn> step_fns;

  SliceArgs() = default;
  SliceArgs(std::vector<step::cyclic::StepFn> fns) : step_fns(std::move(fns)) {}
};

inline auto slice_eval(const SliceArgs& slice, core::Pos pos) {
  for (auto it = slice.step_fns.rbegin(); it != slice.step_fns.rend(); ++it) {
    pos = (*it)(pos);
  }
  return pos;
}

inline auto slice_invert(
    const SliceArgs& slice, core::Pos pos, core::Pos start, core::Pos stop) {
  auto step_fn = [&](core::Pos p) { return slice_eval(slice, p); };
  return step::invert(pos, start, stop, step_fn);
}

inline auto slice_compose(SliceArgs parent, const SliceArgs& child) {
  parent.step_fns.insert(
      parent.step_fns.end(), child.step_fns.begin(), child.step_fns.end());
  return parent;
}

using MergeFn = box::Box (*)(const box::Box*);

// Argument to a merge expression
struct MergeArgs {
  MergeFn merge_fn;

  MergeArgs(MergeFn merge_fn) : merge_fn(merge_fn) {}
};

template <typename Out, typename In, Out (*fn)(In)>
inline constexpr auto merge_fn() {
  return [](const box::Box* b) {
    auto in = b[0].get<In>();
    return box::Box(fn(in));
  };
}

template <typename Out, typename In1, typename In2, Out (*fn)(In1, In2)>
inline constexpr auto merge_fn() {
  return [](const box::Box* b) {
    auto in_1 = b[0].get<In1>();
    auto in_2 = b[1].get<In2>();
    return box::Box(fn(in_1, in_2));
  };
}

template <
    typename Out,
    typename In1,
    typename In2,
    typename In3,
    Out (*fn)(In1, In2, In3)>
inline constexpr auto merge_fn() {
  return [](const box::Box* b) {
    auto in_1 = b[0].get<In1>();
    auto in_2 = b[1].get<In2>();
    auto in_3 = b[2].get<In3>();
    return box::Box(fn(in_1, in_2, in_3));
  };
}

// Union of arguments composing an expression.
struct ExprData {
  int size;
  int span;
  enum { STORE, SLICE, MERGE_1, MERGE_2, MERGE_3 } kind;
  std::variant<std::monostate, StoreArgs, SliceArgs, MergeArgs> args;
};

using Expr = dags::SharedNode<kMaxExprDeps, ExprData>;

// Typed wrapper of an expression for generating typed output.
template <typename Val>
struct TypedExpr {
  Expr::Ptr expr;

  operator bool() const {
    return !!expr;
  }

  Expr::Ptr operator->() {
    return expr;
  };
};

template <typename Out, typename Val>
inline auto cast(TypedExpr<Val> in) {
  return TypedExpr<Out>{in.expr};
}

template <typename Val>
inline auto store(std::shared_ptr<BoxStore> store) {
  auto ret = Expr::make_ptr();
  ret->data.size = 1;
  ret->data.span = store->span();
  ret->data.kind = ExprData::STORE;
  ret->data.args.emplace<StoreArgs>(std::move(store));
  return TypedExpr<Val>{ret};
}

template <typename Val>
inline auto store(const core::Store<Val>& val_store) {
  auto box_store = std::make_shared<BoxStore>(val_store.size);
  for (int i = 0; i < val_store.size; i += 1) {
    box_store->ends[i] = val_store.ends[i];
    box_store->vals[i] = val_store.vals[i];
  }
  return store<Val>(std::move(box_store));
}

template <
    typename Val,
    typename = std::enable_if_t<!std::is_same_v<box::Box, Val>>>
inline auto store(const std::shared_ptr<core::Store<Val>>& val_store) {
  return store(*val_store);
}

template <typename Val>
inline auto store(core::Pos span, Val fill) {
  return store<Val>(core::make_shared_store(span, box::Box(fill)));
}

template <typename Val>
inline auto slice(TypedExpr<Val> in, step::cyclic::StepFn step_fn) {
  auto ret = Expr::make_ptr();
  ret->data.size = 1 + in->data.size;
  ret->data.span = step_fn(in->data.span);
  ret->data.kind = ExprData::SLICE;
  ret->data.args.emplace<SliceArgs>({std::move(step_fn)});
  ret->deps[0] = std::move(in.expr);
  return TypedExpr<Val>{ret};
}

template <typename Val>
inline auto slice(TypedExpr<Val> in, core::Pos start, core::Pos stop) {
  return slice(std::move(in), step::cyclic::slice(start, stop));
}

template <typename Val>
inline auto slice(TypedExpr<Val> in, core::Pos stop) {
  return slice(std::move(in), step::cyclic::slice(stop));
}

template <typename Out, typename In, Out (*fn)(In)>
inline auto merge(TypedExpr<In> in) {
  auto ret = Expr::make_ptr();
  ret->data.size = 1 + in->data.size;
  ret->data.span = in->data.span;
  ret->data.kind = ExprData::MERGE_1;
  ret->data.args.emplace<MergeArgs>(merge_fn<Out, In, fn>());
  ret->deps[0] = std::move(in.expr);
  return TypedExpr<Out>{ret};
}

template <typename In, typename Fn>
inline auto merge(TypedExpr<In> in, Fn fn) {
  using Out = decltype(fn(std::declval<In>()));
  return merge<Out, In, fn>(in);
}

template <typename Out, typename In1, typename In2, Out (*fn)(In1, In2)>
inline auto merge(TypedExpr<In1> in_1, TypedExpr<In2> in_2) {
  CHECK_ARGUMENT(in_1->data.span == in_2->data.span);
  auto ret = Expr::make_ptr();
  ret->data.size = 1 + in_1->data.size + in_2->data.size;
  ret->data.span = in_1->data.span;
  ret->data.kind = ExprData::MERGE_2;
  ret->data.args.emplace<MergeArgs>(merge_fn<Out, In1, In2, fn>());
  ret->deps[0] = std::move(in_1.expr);
  ret->deps[1] = std::move(in_2.expr);
  return TypedExpr<Out>{ret};
}

template <typename In1, typename In2, typename Fn>
inline auto merge(TypedExpr<In1> in_1, TypedExpr<In2> in_2, Fn fn) {
  using Out = decltype(fn(std::declval<In1>(), std::declval<In2>()));
  return merge<Out, In1, In2, fn>(in_1, in_2);
}

template <
    typename Out,
    typename In1,
    typename In2,
    typename In3,
    Out (*fn)(In1, In2, In3)>
inline auto merge(
    TypedExpr<In1> in_1, TypedExpr<In2> in_2, TypedExpr<In3> in_3) {
  CHECK_ARGUMENT(in_1->data.span == in_2->data.span);
  CHECK_ARGUMENT(in_1->data.span == in_3->data.span);
  auto ret = Expr::make_ptr();
  ret->data.size = 1 + in_1->data.size + in_2->data.size + in_3->data.size;
  ret->data.span = in_1->data.span;
  ret->data.kind = ExprData::MERGE_3;
  ret->data.args.emplace<MergeArgs>(merge_fn<Out, In1, In2, In3, fn>());
  ret->deps[0] = std::move(in_1.expr);
  ret->deps[1] = std::move(in_2.expr);
  ret->deps[2] = std::move(in_3.expr);
  return TypedExpr<Out>{ret};
}

template <typename In1, typename In2, typename In3, typename Fn>
inline auto merge(
    TypedExpr<In1> in_1, TypedExpr<In2> in_2, TypedExpr<In3> in_3, Fn fn) {
  using Out = decltype(
      fn(std::declval<In1>(), std::declval<In2>(), std::declval<In3>()));
  return merge<Out, In1, In2, In3, fn>(in_1, in_2, in_3);
}

template <typename Fn>
inline void expr_dfs(Expr::Ptr expr, Fn&& fn) {
  dags::dfs(std::move(expr), [fn = std::forward<Fn>(fn)](auto e, auto& q) {
    if (fn(e)) {
      for (int i = 0; i < kMaxExprDeps; i += 1) {
        if (e->deps[i]) {
          q.push_back(e->deps[i]);
        } else {
          break;
        }
      }
      q.push_back(std::move(e));
    }
  });
}

template <typename Val>
inline auto debug_str(TypedExpr<Val> in) {
  std::vector<std::string> stmts;
  std::unordered_map<Expr::Ptr, int> id_map;

  // Helper routine to push back a new expression statement.
  auto add_stmt = [&](Expr::Ptr expr, std::string expr_str) {
    auto id = id_map[expr] = id_map.size();
    stmts.push_back(fmt::format("x{} = {}", id, std::move(expr_str)));
  };

  // DFS to add a statement for each expression in reverse dependency order.
  expr_dfs(in.expr, [&](auto e) {
    if (id_map.count(e)) {
      return false;
    } else if (e->data.kind == ExprData::STORE) {
      add_stmt(e, fmt::format("store(span={})", e->data.span));
      return false;
    } else if (e->data.kind == ExprData::SLICE) {
      if (auto it = id_map.find(e->deps[0]); it != id_map.end()) {
        add_stmt(e, fmt::format("slice(x{})", it->second));
        return false;
      }
    } else if (e->data.kind == ExprData::MERGE_1) {
      if (auto it = id_map.find(e->deps[0]); it != id_map.end()) {
        add_stmt(e, fmt::format("merge(x{})", it->second));
        return false;
      }
    } else if (e->data.kind == ExprData::MERGE_2) {
      auto it_1 = id_map.find(e->deps[0]);
      auto it_2 = id_map.find(e->deps[1]);
      if (it_1 != id_map.end() && it_2 != id_map.end()) {
        auto id_1 = it_1->second;
        auto id_2 = it_2->second;
        add_stmt(e, fmt::format("merge(x{}, x{})", id_1, id_2));
        return false;
      }
    } else if (e->data.kind == ExprData::MERGE_3) {
      auto it_1 = id_map.find(e->deps[0]);
      auto it_2 = id_map.find(e->deps[1]);
      auto it_3 = id_map.find(e->deps[2]);
      auto it_end = id_map.end();
      if (it_1 != it_end && it_2 != it_end && it_3 != it_end) {
        auto id_1 = it_1->second;
        auto id_2 = it_2->second;
        auto id_3 = it_3->second;
        add_stmt(e, fmt::format("merge(x{}, x{}, x{})", id_1, id_2, id_3));
        return false;
      }
    }
    return true;
  });

  auto type_str = typeid(Val).name();
  stmts.push_back(fmt::format("x{} : {}", id_map.at(in.expr), type_str));
  return fmt::format("{}", fmt::join(stmts, ";\n"));
}

using ExprGraph = dags::Graph<kMaxExprDeps, ExprData>;

template <typename Fn>
inline void graph_dfs(ExprGraph::Handle handle, Fn&& fn) {
  dags::dfs(std::move(handle), [fn = std::forward<Fn>(fn)](auto h, auto& q) {
    if (fn(h)) {
      for (int i = 0; i < kMaxExprDeps; i += 1) {
        if (h->deps[i]) {
          q.push_back(h->deps[i]);
        } else {
          break;
        }
      }
      q.push_back(std::move(h));
    }
  });
}

template <typename Val>
inline auto span(TypedExpr<Val> expr) {
  return expr->data.span;
}

// Builds an expression recursively for a given DAG.
template <typename Val>
inline auto expressify(ExprGraph::Handle root) {
  auto bucket_hint = 2 * root->data.size;
  std::unordered_map<ExprGraph::Handle, Expr::Ptr> node_map(bucket_hint);
  graph_dfs(root, [&](auto h) {
    if (node_map.count(h)) {
      return false;
    }

    // Recurse if not all children have been mapped already.
    for (int i = 0; i < kMaxExprDeps; i += 1) {
      if (h->deps[i] && !node_map.count(h->deps[i])) {
        return true;
      }
    }

    // Map this node since it's children are all mapped.
    auto expr = node_map[h] = Expr::make_ptr(h->data);
    for (int i = 0; i < kMaxExprDeps; i += 1) {
      if (h->deps[i]) {
        expr->deps[i] = node_map.at(h->deps[i]);
      } else {
        break;
      }
    }
    return false;
  });

  return TypedExpr<Val>{node_map.at(root)};
}

// Builds a DAG recursively from an expression.
inline auto dagify(ExprGraph& graph, Expr::Ptr root) {
  auto bucket_hint = 2 * root->data.size;
  std::unordered_map<Expr::Ptr, ExprGraph::Handle> node_map(bucket_hint);
  expr_dfs(root, [&](auto e) {
    if (node_map.count(e)) {
      return false;
    } else {
      // Recurse if not all children have been mapped already.
      for (int i = 0; i < kMaxExprDeps; i += 1) {
        if (e->deps[i] && !node_map.count(e->deps[i])) {
          return true;
        }
      }

      // Map this node since it's children are all mapped.
      auto handle = node_map[e] = graph.emplace(e->data);
      for (int i = 0; i < kMaxExprDeps; i += 1) {
        if (e->deps[i]) {
          handle->deps[i] = node_map.at(e->deps[i]);
        } else {
          break;
        }
      }
      return false;
    }
  });

  return node_map.at(root);
}

inline auto debug_str(ExprGraph::Handle in) {
  std::vector<std::string> stmts;
  std::unordered_map<ExprGraph::Handle, int> id_map;

  // Helper routine to push back a new expression statement.
  auto add_stmt = [&](ExprGraph::Handle h, std::string expr_str) {
    auto id = id_map[h] = id_map.size();
    stmts.push_back(fmt::format("x{} = {}", id, std::move(expr_str)));
  };

  // DFS to add a statement for each expression in reverse dependency order.
  graph_dfs(in, [&](auto h) {
    if (id_map.count(h)) {
      return false;
    } else if (h->data.kind == ExprData::STORE) {
      add_stmt(h, fmt::format("store(span={})", h->data.span));
      return false;
    } else if (h->data.kind == ExprData::SLICE) {
      if (auto it = id_map.find(h->deps[0]); it != id_map.end()) {
        add_stmt(h, fmt::format("slice(x{})", it->second));
        return false;
      }
    } else if (h->data.kind == ExprData::MERGE_1) {
      if (auto it = id_map.find(h->deps[0]); it != id_map.end()) {
        add_stmt(h, fmt::format("merge(x{})", it->second));
        return false;
      }
    } else if (h->data.kind == ExprData::MERGE_2) {
      auto it_1 = id_map.find(h->deps[0]);
      auto it_2 = id_map.find(h->deps[1]);
      if (it_1 != id_map.end() && it_2 != id_map.end()) {
        auto id_1 = it_1->second;
        auto id_2 = it_2->second;
        add_stmt(h, fmt::format("merge(x{}, x{})", id_1, id_2));
        return false;
      }
    } else if (h->data.kind == ExprData::MERGE_3) {
      auto it_1 = id_map.find(h->deps[0]);
      auto it_2 = id_map.find(h->deps[1]);
      auto it_3 = id_map.find(h->deps[2]);
      auto it_end = id_map.end();
      if (it_1 != it_end && it_2 != it_end && it_3 != it_end) {
        auto id_1 = it_1->second;
        auto id_2 = it_2->second;
        auto id_3 = it_3->second;
        add_stmt(h, fmt::format("merge(x{}, x{}, x{})", id_1, id_2, id_3));
        return false;
      }
    }
    return true;
  });

  stmts.push_back(fmt::format("x{}", id_map.at(in)));
  return fmt::format("{}", fmt::join(stmts, ";\n"));
}

// Augments the given expression before scheduling to improve its efficiency.
inline auto optimize(ExprGraph& graph, ExprGraph::Handle root) {
  // TODO(taylorgrodon): Push down slice operations when doing so reduces cost.
}

// Schedules a sequence of nodes to materialize from the input graph.
inline auto schedule(ExprGraph::Handle root) {
  std::vector<ExprGraph::Handle> steps;

  // Start by scheduling every node for materialization.
  {
    std::unordered_set<ExprGraph::Handle> scheduled;
    graph_dfs(root, [&](auto h) {
      if (scheduled.count(h)) {
        return false;
      } else {
        for (int i = 0; i < kMaxExprDeps; i += 1) {
          if (h->deps[i] && !scheduled.count(h->deps[i])) {
            return true;
          }
        }
        scheduled.insert(h);
        steps.push_back(h);
        return false;
      }
    });
  }

  // Via a greedy bottom-up optimization, decide to skip evaluation for some
  // nodes. The process conserves "feasibility" (e.g. max step sources), while
  // choosing a materialization schedule with lower cost.
  CHECK_STATE(steps.size() > 0);
  std::unordered_map<ExprGraph::Handle, int> depth_map;
  std::unordered_map<ExprGraph::Handle, int> width_map;
  steps = [&] {
    std::vector<ExprGraph::Handle> filtered;
    for (int i = 0; i < steps.size() - 1; i += 1) {
      auto& step = steps[i];
      if (step->data.kind == ExprData::STORE) {
        depth_map[step] = 1;
        width_map[step] = 1;
        continue;
      }

      auto width = 0;
      auto depth = 1;
      for (int j = 0; j < kMaxExprDeps; j += 1) {
        if (auto dep = step->deps[j]) {
          width += width_map.at(dep);
          depth = std::max(depth, 1 + depth_map.at(dep));
        }
      }

      if (width <= 128 && depth < 128) {
        width_map[step] = width;
        depth_map[step] = depth;
      } else {
        width_map[step] = 1;
        depth_map[step] = 1;
        filtered.push_back(step);
      }
    }
    filtered.push_back(steps.back());
    return filtered;
  }();

  return steps;
}

// Expands an expression into the store [=> slice] [=> merge ...] normal form.
inline auto normalize(ExprGraph& graph, ExprGraph::Handle root) {
  // If the root is a store, introduce a root slice and return.
  if (root->data.kind == ExprData::STORE) {
    auto c = graph.emplace(ExprData(root->data));
    root->data.size = 2;
    root->data.span = c->data.span;
    root->data.kind = ExprData::SLICE;
    root->data.args.template emplace<SliceArgs>();
    root->deps[0] = c;
    return;
  }

  // The algorithm is as follows:
  // 1. If the node is a slice node, then:
  //   a. If the child is a slice node, coalesce the two slice nodes.
  //   b. If the child is a store node, stop recursion.
  //   c. If the child is a merge node, swap it with the slice node. This swap
  //      requires creating a new slice node for each merge input.
  // 2. If the node is a merge node, recurse without mutation. Before recursing
  //    on a store child, introduce a slice node.
  dfs(root, [&](auto h, auto& q) {
    CHECK_STATE(h->data.kind != ExprData::STORE);
    if (h->data.kind == ExprData::SLICE) {
      auto c = h->deps[0];
      if (c->data.kind == ExprData::STORE) {
        return;
      } else if (c->data.kind == ExprData::SLICE) {
        auto&& h_args = std::move(std::get<SliceArgs>(h->data.args));
        auto&& c_args = std::move(std::get<SliceArgs>(c->data.args));
        h->data.size -= 1;
        h->data.span = slice_eval(h_args, c->data.span);
        h->data.args.template emplace<SliceArgs>(slice_compose(h_args, c_args));
        h->deps[0] = c->deps[0];
        q.push_back(h);
      } else {
        auto h_size = c->data.size;
        for (int i = 0; i < kMaxExprDeps; i += 1) {
          if (auto dep = c->deps[i]) {
            h->deps[i] = graph.emplace(ExprData(h->data));
            h->deps[i]->data.size = 1 + dep->data.size;
            h->deps[i]->deps[0] = dep;
            h_size += 1;
          } else {
            break;
          }
        }
        h->data.size = h_size;
        h->data.kind = c->data.kind;
        h->data.args = std::move(c->data.args);
        q.push_back(h);
      }
    } else {
      for (int i = 0; i < kMaxExprDeps; i += 1) {
        if (auto dep = h->deps[i]) {
          if (dep->data.kind == ExprData::STORE) {
            h->deps[i] = graph.emplace();
            h->deps[i]->data.size = 2;
            h->deps[i]->data.span = dep->data.span;
            h->deps[i]->data.kind = ExprData::SLICE;
            h->deps[i]->data.args.template emplace<SliceArgs>();
            h->deps[i]->deps[0] = dep;
          } else {
            q.push_back(dep);
          }
        } else {
          break;
        }
      }
    }
  });
}

struct EvalPlan {
  struct Node {
    enum { SOURCE, MERGE_1, MERGE_2, MERGE_3 } kind;
    union {
      int index;
      MergeFn fn;
    };
  };

  int depth;
  std::vector<ExprGraph::Handle> sources;
  std::vector<Node> nodes;
};

inline auto build_plan(ExprGraph::Handle root) {
  EvalPlan plan;
  plan.depth = 0;
  plan.nodes.reserve(root->data.size);
  plan.sources.reserve(root->data.size);

  // Expand the DAG into a tree and linearize the nodes.
  int id_counter = 0;
  std::unordered_set<int> visited;
  dfs(std::tuple(root, id_counter++, 1), [&](auto t, auto& q) {
    auto [h, id, depth] = std::move(t);
    if (auto status = visited.insert(id); status.second) {
      for (int i = 0; i < kMaxExprDeps; i += 1) {
        if (h->deps[i]) {
          q.emplace_back(h->deps[i], id_counter++, depth + 1);
        }
      }
      q.emplace_back(h, id, depth);
    } else {
      if (h->data.kind == ExprData::SLICE) {
        CHECK_STATE(h->deps[0]);
        CHECK_STATE(h->deps[0]->data.kind == ExprData::STORE);
        auto& node = plan.nodes.emplace_back();
        node.kind = EvalPlan::Node::SOURCE;
        node.index = plan.sources.size();
        plan.sources.push_back(h);
        plan.depth = std::max(plan.depth, depth);
      } else if (h->data.kind == ExprData::MERGE_1) {
        auto& node = plan.nodes.emplace_back();
        node.kind = EvalPlan::Node::MERGE_1;
        node.fn = std::get<MergeArgs>(h->data.args).merge_fn;
      } else if (h->data.kind == ExprData::MERGE_2) {
        auto& node = plan.nodes.emplace_back();
        node.kind = EvalPlan::Node::MERGE_2;
        node.fn = std::get<MergeArgs>(h->data.args).merge_fn;
      } else if (h->data.kind == ExprData::MERGE_3) {
        auto& node = plan.nodes.emplace_back();
        node.kind = EvalPlan::Node::MERGE_3;
        node.fn = std::get<MergeArgs>(h->data.args).merge_fn;
      }
    }
  });

  return plan;
}

template <int size>
using EvalPool = std::variant<
    eval::SimplePool<box::Box, step::IdentityStepFn, size>,
    eval::SimplePool<box::Box, step::cyclic::StepFn, size>,
    eval::SimplePool<box::Box, step::composite::StepFn, size>>;

template <int size>
inline auto make_pool(const EvalPlan& plan) -> EvalPool<size> {
  auto b_iter = plan.sources.begin();
  auto e_iter = plan.sources.end();

  // If all slices are empty, emit an IdentityStepFn pool.
  auto all_empty = std::all_of(b_iter, e_iter, [&](auto h) {
    return std::get<SliceArgs>(h->data.args).step_fns.empty();
  });
  if (all_empty) {
    using P = eval::SimplePool<box::Box, step::IdentityStepFn, size>;
    using S = eval::SimpleSource<box::Box, step::IdentityStepFn>;
    std::array<std::shared_ptr<S>, size> sources;
    for (int i = 0; i < size; i += 1) {
      auto& h = plan.sources[i];
      auto& store_args = std::get<StoreArgs>(h->deps[0]->data.args);
      sources[i] = std::make_shared<S>(store_args.store);
    }
    return P(std::move(sources));
  }

  // If all slices have at most one step fn, emit an cyclic::StepFn pool.
  auto all_cyclic = std::all_of(b_iter, e_iter, [&](auto h) {
    return std::get<SliceArgs>(h->data.args).step_fns.size() <= 1;
  });
  if (all_cyclic) {
    using P = eval::SimplePool<box::Box, step::cyclic::StepFn, size>;
    using S = eval::SimpleSource<box::Box, step::cyclic::StepFn>;
    std::array<std::shared_ptr<S>, size> sources;
    for (int i = 0; i < size; i += 1) {
      auto& h = plan.sources[i];
      auto& slice_args = std::get<SliceArgs>(h->data.args);
      auto& store_args = std::get<StoreArgs>(h->deps[0]->data.args);
      if (slice_args.step_fns.empty()) {
        sources[i] = std::make_shared<S>(
            store_args.store, 0, h->data.span, step::cyclic::identity());
      } else {
        auto& step_fn = slice_args.step_fns.at(0);
        sources[i] = std::make_shared<S>(
            store_args.store,
            slice_invert(slice_args, 1, 0, store_args.store->span()) - 1,
            slice_invert(slice_args, h->data.span, 0, store_args.store->span()),
            step::cyclic::StepFn(step_fn));
      }
    }
    return P(std::move(sources));
  }

  // Fallback to emitting a composite::StepFn pool.
  {
    using P = eval::SimplePool<box::Box, step::composite::StepFn, size>;
    using S = eval::SimpleSource<box::Box, step::composite::StepFn>;
    std::array<std::shared_ptr<S>, size> sources;
    for (int i = 0; i < size; i += 1) {
      auto& h = plan.sources[i];
      auto& slice_args = std::get<SliceArgs>(h->data.args);
      auto& store_args = std::get<StoreArgs>(h->deps[0]->data.args);
      if (slice_args.step_fns.empty()) {
        sources[i] = std::make_shared<S>(
            store_args.store, 0, h->data.span, step::composite::StepFn());
      } else if (slice_args.step_fns.size() == 1) {
        auto& step_fn = slice_args.step_fns.at(0);
        sources[i] = std::make_shared<S>(
            store_args.store,
            slice_invert(slice_args, 1, 0, store_args.store->span()) - 1,
            slice_invert(slice_args, h->data.span, 0, store_args.store->span()),
            step::composite::StepFn(step_fn));
      } else {
        auto step_fns = slice_args.step_fns;
        std::reverse(step_fns.begin(), step_fns.end());
        sources[i] = std::make_shared<S>(
            store_args.store,
            slice_invert(slice_args, 1, 0, store_args.store->span()) - 1,
            slice_invert(slice_args, h->data.span, 0, store_args.store->span()),
            step::composite::StepFn(std::move(step_fns)));
      }
    }
    return P(std::move(sources));
  }
}

template <typename Val, int size>
inline auto execute_plan_fixed(EvalPlan plan) {
  static constexpr auto stack_capacity = 128;
  static constexpr auto max_branch_factor = 3;
  CHECK_ARGUMENT(max_branch_factor * plan.depth <= stack_capacity);
  auto eval_fn = [&](const box::Box* b) {
    thread_local box::Box stack[stack_capacity];
    auto sp = &stack[0];
    for (auto& node : plan.nodes) {
      switch (node.kind) {
        case EvalPlan::Node::SOURCE:
          *sp++ = b[node.index];
          break;
        case EvalPlan::Node::MERGE_1:
          *(sp - 1) = node.fn(sp - 1);
          break;
        case EvalPlan::Node::MERGE_2:
          *(sp - 2) = node.fn(sp - 2);
          sp -= 1;
          break;
        case EvalPlan::Node::MERGE_3:
          *(sp - 3) = node.fn(sp - 3);
          sp -= 2;
          break;
      }
    }

    return (sp - 1)->template get<Val>();
  };

  return std::visit(
      [&](auto&& pool) {
        return eval::eval_simple<Val, box::Box>(std::move(eval_fn), pool);
      },
      make_pool<size>(plan));
}

template <typename Val>
inline auto execute_plan(EvalPlan plan) {
  // Build a pool out of plan's sources.
  CHECK_STATE(plan.sources.size());
  switch (plan.sources.size()) {
    case 1:
      return execute_plan_fixed<Val, 1>(std::move(plan));
    case 2:
      return execute_plan_fixed<Val, 2>(std::move(plan));
    case 3:
      return execute_plan_fixed<Val, 3>(std::move(plan));
    case 4:
      return execute_plan_fixed<Val, 4>(std::move(plan));
    case 5:
      return execute_plan_fixed<Val, 5>(std::move(plan));
    case 6:
      return execute_plan_fixed<Val, 6>(std::move(plan));
    case 7:
      return execute_plan_fixed<Val, 7>(std::move(plan));
    case 8:
      return execute_plan_fixed<Val, 8>(std::move(plan));
    case 9:
      return execute_plan_fixed<Val, 9>(std::move(plan));
    case 10:
      return execute_plan_fixed<Val, 10>(std::move(plan));
    case 11:
      return execute_plan_fixed<Val, 11>(std::move(plan));
    case 12:
      return execute_plan_fixed<Val, 12>(std::move(plan));
    case 13:
      return execute_plan_fixed<Val, 13>(std::move(plan));
    case 14:
      return execute_plan_fixed<Val, 14>(std::move(plan));
    case 15:
      return execute_plan_fixed<Val, 15>(std::move(plan));
    case 16:
      return execute_plan_fixed<Val, 16>(std::move(plan));
    case 17:
      return execute_plan_fixed<Val, 17>(std::move(plan));
    case 18:
      return execute_plan_fixed<Val, 18>(std::move(plan));
    case 19:
      return execute_plan_fixed<Val, 19>(std::move(plan));
    case 20:
      return execute_plan_fixed<Val, 20>(std::move(plan));
    case 21:
      return execute_plan_fixed<Val, 21>(std::move(plan));
    case 22:
      return execute_plan_fixed<Val, 22>(std::move(plan));
    case 23:
      return execute_plan_fixed<Val, 23>(std::move(plan));
    case 24:
      return execute_plan_fixed<Val, 24>(std::move(plan));
    case 25:
      return execute_plan_fixed<Val, 25>(std::move(plan));
    case 26:
      return execute_plan_fixed<Val, 26>(std::move(plan));
    case 27:
      return execute_plan_fixed<Val, 27>(std::move(plan));
    case 28:
      return execute_plan_fixed<Val, 28>(std::move(plan));
    case 29:
      return execute_plan_fixed<Val, 29>(std::move(plan));
    case 30:
      return execute_plan_fixed<Val, 30>(std::move(plan));
    case 31:
      return execute_plan_fixed<Val, 31>(std::move(plan));
    case 32:
      return execute_plan_fixed<Val, 32>(std::move(plan));
    case 33:
      return execute_plan_fixed<Val, 33>(std::move(plan));
    case 34:
      return execute_plan_fixed<Val, 34>(std::move(plan));
    case 35:
      return execute_plan_fixed<Val, 35>(std::move(plan));
    case 36:
      return execute_plan_fixed<Val, 36>(std::move(plan));
    case 37:
      return execute_plan_fixed<Val, 37>(std::move(plan));
    case 38:
      return execute_plan_fixed<Val, 38>(std::move(plan));
    case 39:
      return execute_plan_fixed<Val, 39>(std::move(plan));
    case 40:
      return execute_plan_fixed<Val, 40>(std::move(plan));
    case 41:
      return execute_plan_fixed<Val, 41>(std::move(plan));
    case 42:
      return execute_plan_fixed<Val, 42>(std::move(plan));
    case 43:
      return execute_plan_fixed<Val, 43>(std::move(plan));
    case 44:
      return execute_plan_fixed<Val, 44>(std::move(plan));
    case 45:
      return execute_plan_fixed<Val, 45>(std::move(plan));
    case 46:
      return execute_plan_fixed<Val, 46>(std::move(plan));
    case 47:
      return execute_plan_fixed<Val, 47>(std::move(plan));
    case 48:
      return execute_plan_fixed<Val, 48>(std::move(plan));
    case 49:
      return execute_plan_fixed<Val, 49>(std::move(plan));
    case 50:
      return execute_plan_fixed<Val, 50>(std::move(plan));
    case 51:
      return execute_plan_fixed<Val, 51>(std::move(plan));
    case 52:
      return execute_plan_fixed<Val, 52>(std::move(plan));
    case 53:
      return execute_plan_fixed<Val, 53>(std::move(plan));
    case 54:
      return execute_plan_fixed<Val, 54>(std::move(plan));
    case 55:
      return execute_plan_fixed<Val, 55>(std::move(plan));
    case 56:
      return execute_plan_fixed<Val, 56>(std::move(plan));
    case 57:
      return execute_plan_fixed<Val, 57>(std::move(plan));
    case 58:
      return execute_plan_fixed<Val, 58>(std::move(plan));
    case 59:
      return execute_plan_fixed<Val, 59>(std::move(plan));
    case 60:
      return execute_plan_fixed<Val, 60>(std::move(plan));
    case 61:
      return execute_plan_fixed<Val, 61>(std::move(plan));
    case 62:
      return execute_plan_fixed<Val, 62>(std::move(plan));
    case 63:
      return execute_plan_fixed<Val, 63>(std::move(plan));
    case 64:
      return execute_plan_fixed<Val, 64>(std::move(plan));
    case 65:
      return execute_plan_fixed<Val, 65>(std::move(plan));
    case 66:
      return execute_plan_fixed<Val, 66>(std::move(plan));
    case 67:
      return execute_plan_fixed<Val, 67>(std::move(plan));
    case 68:
      return execute_plan_fixed<Val, 68>(std::move(plan));
    case 69:
      return execute_plan_fixed<Val, 69>(std::move(plan));
    case 70:
      return execute_plan_fixed<Val, 70>(std::move(plan));
    case 71:
      return execute_plan_fixed<Val, 71>(std::move(plan));
    case 72:
      return execute_plan_fixed<Val, 72>(std::move(plan));
    case 73:
      return execute_plan_fixed<Val, 73>(std::move(plan));
    case 74:
      return execute_plan_fixed<Val, 74>(std::move(plan));
    case 75:
      return execute_plan_fixed<Val, 75>(std::move(plan));
    case 76:
      return execute_plan_fixed<Val, 76>(std::move(plan));
    case 77:
      return execute_plan_fixed<Val, 77>(std::move(plan));
    case 78:
      return execute_plan_fixed<Val, 78>(std::move(plan));
    case 79:
      return execute_plan_fixed<Val, 79>(std::move(plan));
    case 80:
      return execute_plan_fixed<Val, 80>(std::move(plan));
    case 81:
      return execute_plan_fixed<Val, 81>(std::move(plan));
    case 82:
      return execute_plan_fixed<Val, 82>(std::move(plan));
    case 83:
      return execute_plan_fixed<Val, 83>(std::move(plan));
    case 84:
      return execute_plan_fixed<Val, 84>(std::move(plan));
    case 85:
      return execute_plan_fixed<Val, 85>(std::move(plan));
    case 86:
      return execute_plan_fixed<Val, 86>(std::move(plan));
    case 87:
      return execute_plan_fixed<Val, 87>(std::move(plan));
    case 88:
      return execute_plan_fixed<Val, 88>(std::move(plan));
    case 89:
      return execute_plan_fixed<Val, 89>(std::move(plan));
    case 90:
      return execute_plan_fixed<Val, 90>(std::move(plan));
    case 91:
      return execute_plan_fixed<Val, 91>(std::move(plan));
    case 92:
      return execute_plan_fixed<Val, 92>(std::move(plan));
    case 93:
      return execute_plan_fixed<Val, 93>(std::move(plan));
    case 94:
      return execute_plan_fixed<Val, 94>(std::move(plan));
    case 95:
      return execute_plan_fixed<Val, 95>(std::move(plan));
    case 96:
      return execute_plan_fixed<Val, 96>(std::move(plan));
    case 97:
      return execute_plan_fixed<Val, 97>(std::move(plan));
    case 98:
      return execute_plan_fixed<Val, 98>(std::move(plan));
    case 99:
      return execute_plan_fixed<Val, 99>(std::move(plan));
    case 100:
      return execute_plan_fixed<Val, 100>(std::move(plan));
    case 101:
      return execute_plan_fixed<Val, 101>(std::move(plan));
    case 102:
      return execute_plan_fixed<Val, 102>(std::move(plan));
    case 103:
      return execute_plan_fixed<Val, 103>(std::move(plan));
    case 104:
      return execute_plan_fixed<Val, 104>(std::move(plan));
    case 105:
      return execute_plan_fixed<Val, 105>(std::move(plan));
    case 106:
      return execute_plan_fixed<Val, 106>(std::move(plan));
    case 107:
      return execute_plan_fixed<Val, 107>(std::move(plan));
    case 108:
      return execute_plan_fixed<Val, 108>(std::move(plan));
    case 109:
      return execute_plan_fixed<Val, 109>(std::move(plan));
    case 110:
      return execute_plan_fixed<Val, 110>(std::move(plan));
    case 111:
      return execute_plan_fixed<Val, 111>(std::move(plan));
    case 112:
      return execute_plan_fixed<Val, 112>(std::move(plan));
    case 113:
      return execute_plan_fixed<Val, 113>(std::move(plan));
    case 114:
      return execute_plan_fixed<Val, 114>(std::move(plan));
    case 115:
      return execute_plan_fixed<Val, 115>(std::move(plan));
    case 116:
      return execute_plan_fixed<Val, 116>(std::move(plan));
    case 117:
      return execute_plan_fixed<Val, 117>(std::move(plan));
    case 118:
      return execute_plan_fixed<Val, 118>(std::move(plan));
    case 119:
      return execute_plan_fixed<Val, 119>(std::move(plan));
    case 120:
      return execute_plan_fixed<Val, 120>(std::move(plan));
    case 121:
      return execute_plan_fixed<Val, 121>(std::move(plan));
    case 122:
      return execute_plan_fixed<Val, 122>(std::move(plan));
    case 123:
      return execute_plan_fixed<Val, 123>(std::move(plan));
    case 124:
      return execute_plan_fixed<Val, 124>(std::move(plan));
    case 125:
      return execute_plan_fixed<Val, 125>(std::move(plan));
    case 126:
      return execute_plan_fixed<Val, 126>(std::move(plan));
    case 127:
      return execute_plan_fixed<Val, 127>(std::move(plan));
    case 128:
      return execute_plan_fixed<Val, 128>(std::move(plan));
    case 129:
      return execute_plan_fixed<Val, 129>(std::move(plan));
    case 130:
      return execute_plan_fixed<Val, 130>(std::move(plan));
    case 131:
      return execute_plan_fixed<Val, 131>(std::move(plan));
    case 132:
      return execute_plan_fixed<Val, 132>(std::move(plan));
    case 133:
      return execute_plan_fixed<Val, 133>(std::move(plan));
    case 134:
      return execute_plan_fixed<Val, 134>(std::move(plan));
    case 135:
      return execute_plan_fixed<Val, 135>(std::move(plan));
    case 136:
      return execute_plan_fixed<Val, 136>(std::move(plan));
    case 137:
      return execute_plan_fixed<Val, 137>(std::move(plan));
    case 138:
      return execute_plan_fixed<Val, 138>(std::move(plan));
    case 139:
      return execute_plan_fixed<Val, 139>(std::move(plan));
    case 140:
      return execute_plan_fixed<Val, 140>(std::move(plan));
    case 141:
      return execute_plan_fixed<Val, 141>(std::move(plan));
    case 142:
      return execute_plan_fixed<Val, 142>(std::move(plan));
    case 143:
      return execute_plan_fixed<Val, 143>(std::move(plan));
    case 144:
      return execute_plan_fixed<Val, 144>(std::move(plan));
    case 145:
      return execute_plan_fixed<Val, 145>(std::move(plan));
    case 146:
      return execute_plan_fixed<Val, 146>(std::move(plan));
    case 147:
      return execute_plan_fixed<Val, 147>(std::move(plan));
    case 148:
      return execute_plan_fixed<Val, 148>(std::move(plan));
    case 149:
      return execute_plan_fixed<Val, 149>(std::move(plan));
    case 150:
      return execute_plan_fixed<Val, 150>(std::move(plan));
    case 151:
      return execute_plan_fixed<Val, 151>(std::move(plan));
    case 152:
      return execute_plan_fixed<Val, 152>(std::move(plan));
    case 153:
      return execute_plan_fixed<Val, 153>(std::move(plan));
    case 154:
      return execute_plan_fixed<Val, 154>(std::move(plan));
    case 155:
      return execute_plan_fixed<Val, 155>(std::move(plan));
    case 156:
      return execute_plan_fixed<Val, 156>(std::move(plan));
    case 157:
      return execute_plan_fixed<Val, 157>(std::move(plan));
    case 158:
      return execute_plan_fixed<Val, 158>(std::move(plan));
    case 159:
      return execute_plan_fixed<Val, 159>(std::move(plan));
    case 160:
      return execute_plan_fixed<Val, 160>(std::move(plan));
    case 161:
      return execute_plan_fixed<Val, 161>(std::move(plan));
    case 162:
      return execute_plan_fixed<Val, 162>(std::move(plan));
    case 163:
      return execute_plan_fixed<Val, 163>(std::move(plan));
    case 164:
      return execute_plan_fixed<Val, 164>(std::move(plan));
    case 165:
      return execute_plan_fixed<Val, 165>(std::move(plan));
    case 166:
      return execute_plan_fixed<Val, 166>(std::move(plan));
    case 167:
      return execute_plan_fixed<Val, 167>(std::move(plan));
    case 168:
      return execute_plan_fixed<Val, 168>(std::move(plan));
    case 169:
      return execute_plan_fixed<Val, 169>(std::move(plan));
    case 170:
      return execute_plan_fixed<Val, 170>(std::move(plan));
    case 171:
      return execute_plan_fixed<Val, 171>(std::move(plan));
    case 172:
      return execute_plan_fixed<Val, 172>(std::move(plan));
    case 173:
      return execute_plan_fixed<Val, 173>(std::move(plan));
    case 174:
      return execute_plan_fixed<Val, 174>(std::move(plan));
    case 175:
      return execute_plan_fixed<Val, 175>(std::move(plan));
    case 176:
      return execute_plan_fixed<Val, 176>(std::move(plan));
    case 177:
      return execute_plan_fixed<Val, 177>(std::move(plan));
    case 178:
      return execute_plan_fixed<Val, 178>(std::move(plan));
    case 179:
      return execute_plan_fixed<Val, 179>(std::move(plan));
    case 180:
      return execute_plan_fixed<Val, 180>(std::move(plan));
    case 181:
      return execute_plan_fixed<Val, 181>(std::move(plan));
    case 182:
      return execute_plan_fixed<Val, 182>(std::move(plan));
    case 183:
      return execute_plan_fixed<Val, 183>(std::move(plan));
    case 184:
      return execute_plan_fixed<Val, 184>(std::move(plan));
    case 185:
      return execute_plan_fixed<Val, 185>(std::move(plan));
    case 186:
      return execute_plan_fixed<Val, 186>(std::move(plan));
    case 187:
      return execute_plan_fixed<Val, 187>(std::move(plan));
    case 188:
      return execute_plan_fixed<Val, 188>(std::move(plan));
    case 189:
      return execute_plan_fixed<Val, 189>(std::move(plan));
    case 190:
      return execute_plan_fixed<Val, 190>(std::move(plan));
    case 191:
      return execute_plan_fixed<Val, 191>(std::move(plan));
    case 192:
      return execute_plan_fixed<Val, 192>(std::move(plan));
    case 193:
      return execute_plan_fixed<Val, 193>(std::move(plan));
    case 194:
      return execute_plan_fixed<Val, 194>(std::move(plan));
    case 195:
      return execute_plan_fixed<Val, 195>(std::move(plan));
    case 196:
      return execute_plan_fixed<Val, 196>(std::move(plan));
    case 197:
      return execute_plan_fixed<Val, 197>(std::move(plan));
    case 198:
      return execute_plan_fixed<Val, 198>(std::move(plan));
    case 199:
      return execute_plan_fixed<Val, 199>(std::move(plan));
    case 200:
      return execute_plan_fixed<Val, 200>(std::move(plan));
    case 201:
      return execute_plan_fixed<Val, 201>(std::move(plan));
    case 202:
      return execute_plan_fixed<Val, 202>(std::move(plan));
    case 203:
      return execute_plan_fixed<Val, 203>(std::move(plan));
    case 204:
      return execute_plan_fixed<Val, 204>(std::move(plan));
    case 205:
      return execute_plan_fixed<Val, 205>(std::move(plan));
    case 206:
      return execute_plan_fixed<Val, 206>(std::move(plan));
    case 207:
      return execute_plan_fixed<Val, 207>(std::move(plan));
    case 208:
      return execute_plan_fixed<Val, 208>(std::move(plan));
    case 209:
      return execute_plan_fixed<Val, 209>(std::move(plan));
    case 210:
      return execute_plan_fixed<Val, 210>(std::move(plan));
    case 211:
      return execute_plan_fixed<Val, 211>(std::move(plan));
    case 212:
      return execute_plan_fixed<Val, 212>(std::move(plan));
    case 213:
      return execute_plan_fixed<Val, 213>(std::move(plan));
    case 214:
      return execute_plan_fixed<Val, 214>(std::move(plan));
    case 215:
      return execute_plan_fixed<Val, 215>(std::move(plan));
    case 216:
      return execute_plan_fixed<Val, 216>(std::move(plan));
    case 217:
      return execute_plan_fixed<Val, 217>(std::move(plan));
    case 218:
      return execute_plan_fixed<Val, 218>(std::move(plan));
    case 219:
      return execute_plan_fixed<Val, 219>(std::move(plan));
    case 220:
      return execute_plan_fixed<Val, 220>(std::move(plan));
    case 221:
      return execute_plan_fixed<Val, 221>(std::move(plan));
    case 222:
      return execute_plan_fixed<Val, 222>(std::move(plan));
    case 223:
      return execute_plan_fixed<Val, 223>(std::move(plan));
    case 224:
      return execute_plan_fixed<Val, 224>(std::move(plan));
    case 225:
      return execute_plan_fixed<Val, 225>(std::move(plan));
    case 226:
      return execute_plan_fixed<Val, 226>(std::move(plan));
    case 227:
      return execute_plan_fixed<Val, 227>(std::move(plan));
    case 228:
      return execute_plan_fixed<Val, 228>(std::move(plan));
    case 229:
      return execute_plan_fixed<Val, 229>(std::move(plan));
    case 230:
      return execute_plan_fixed<Val, 230>(std::move(plan));
    case 231:
      return execute_plan_fixed<Val, 231>(std::move(plan));
    case 232:
      return execute_plan_fixed<Val, 232>(std::move(plan));
    case 233:
      return execute_plan_fixed<Val, 233>(std::move(plan));
    case 234:
      return execute_plan_fixed<Val, 234>(std::move(plan));
    case 235:
      return execute_plan_fixed<Val, 235>(std::move(plan));
    case 236:
      return execute_plan_fixed<Val, 236>(std::move(plan));
    case 237:
      return execute_plan_fixed<Val, 237>(std::move(plan));
    case 238:
      return execute_plan_fixed<Val, 238>(std::move(plan));
    case 239:
      return execute_plan_fixed<Val, 239>(std::move(plan));
    case 240:
      return execute_plan_fixed<Val, 240>(std::move(plan));
    case 241:
      return execute_plan_fixed<Val, 241>(std::move(plan));
    case 242:
      return execute_plan_fixed<Val, 242>(std::move(plan));
    case 243:
      return execute_plan_fixed<Val, 243>(std::move(plan));
    case 244:
      return execute_plan_fixed<Val, 244>(std::move(plan));
    case 245:
      return execute_plan_fixed<Val, 245>(std::move(plan));
    case 246:
      return execute_plan_fixed<Val, 246>(std::move(plan));
    case 247:
      return execute_plan_fixed<Val, 247>(std::move(plan));
    case 248:
      return execute_plan_fixed<Val, 248>(std::move(plan));
    case 249:
      return execute_plan_fixed<Val, 249>(std::move(plan));
    case 250:
      return execute_plan_fixed<Val, 250>(std::move(plan));
    case 251:
      return execute_plan_fixed<Val, 251>(std::move(plan));
    case 252:
      return execute_plan_fixed<Val, 252>(std::move(plan));
    case 253:
      return execute_plan_fixed<Val, 253>(std::move(plan));
    case 254:
      return execute_plan_fixed<Val, 254>(std::move(plan));
    case 255:
      return execute_plan_fixed<Val, 255>(std::move(plan));
    case 256:
      return execute_plan_fixed<Val, 256>(std::move(plan));
    default:
      CHECK_UNREACHABLE("Invalid number of sources.");
  }
}

template <typename Val>
inline auto materialize(TypedExpr<Val> in) {
  // Build a operable graph out of the expression DAG.
  ExprGraph graph;
  auto root = dagify(graph, in.expr);

  // Optimize the expression DAG.
  optimize(graph, root);

  // Schedule the materialization steps for this DAG.
  auto steps = schedule(root);
  fmt::print("Number of steps in materialization: {}\n", steps.size());

  // Helper to build an eval plan for an expression to be materialized.
  auto plan_step = [&](ExprGraph::Handle step) {
    // Update size properties due to child mutations.
    step->data.size = 1;
    for (auto i = 0; i < kMaxExprDeps; i += 1) {
      if (step->deps[i]) {
        step->data.size += step->deps[i]->data.size;
      }
    }

    // Build and return a plan for the step.
    normalize(graph, step);
    return build_plan(step);
  };

  // Materialize each intermediate step in order.
  for (int i = 0; i < steps.size() - 1; i += 1) {
    auto& step = steps.at(i);
    auto store = execute_plan<box::Box>(plan_step(step));
    step->clear();
    step->data.size = 1;
    step->data.span = store->span();
    step->data.kind = ExprData::STORE;
    step->data.args.template emplace<StoreArgs>(std::move(store));
  }

  // Materialize the final step.
  return execute_plan<Val>(plan_step(steps.back()));
}

template <typename Val>
inline auto evaluate(TypedExpr<Val> in) {
  return store<Val>(materialize(TypedExpr<box::Box>{in.expr}));
}

}  // namespace skimpy::detail::lang
