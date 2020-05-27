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
  // TODO: Push down slice operations when doing so reduces "cost".
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

      if (width <= 16 && depth < 128) {
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
  static constexpr auto kStackCapacity = 128;
  static constexpr auto kMaxBranchFactor = 3;
  CHECK_ARGUMENT(kMaxBranchFactor * plan.depth <= kStackCapacity);
  auto eval_fn = [&](const box::Box* b) {
    thread_local box::Box stack[kStackCapacity];
    auto sp = &stack[0];
    for (auto node : plan.nodes) {
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
