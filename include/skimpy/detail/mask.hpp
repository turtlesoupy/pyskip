#pragma once

#include <memory>

#include "core.hpp"
#include "dags.hpp"
#include "step.hpp"

namespace skimpy::detail::mask {

// TODO(taylorgordon): Add a cache to store frequently used masks.

using Pos = core::Pos;

template <typename Val>
struct ExprData {
  enum Kind { RANGE, STACK } kind;
  int size;
  Pos span;
  union {
    struct {
      Val fill;
    } range;
    struct {
      int reps;
    } stack;
  };

  ExprData(Kind kind, int size, Pos span)
      : kind(kind), size(size), span(span) {}
};

template <typename Val>
using ExprNode = dags::SharedNode<2, ExprData<Val>>;

template <typename Val>
struct Expr {
  typename ExprNode<Val>::Ptr node;

  explicit operator bool() const {
    return !!node;
  }

  auto operator-> () {
    return node;
  }
};

template <typename Val>
inline auto range(int span, Val fill) {
  auto ret = ExprNode<Val>::make_ptr(ExprData<Val>::RANGE, 1, span);
  ret->data.range.fill = fill;
  return Expr<Val>{ret};
}

template <typename Val>
inline auto stack(int reps, Expr<Val> l, Expr<Val> r = Expr<Val>{nullptr}) {
  CHECK_ARGUMENT(l);
  auto size = reps * (l->data.size + (r ? r->data.size : 0));
  auto span = reps * (l->data.span + (r ? r->data.span : 0));
  auto ret = ExprNode<Val>::make_ptr(ExprData<Val>::STACK, size, span);
  ret->data.stack.reps = reps;
  ret->deps[0] = l.node;
  ret->deps[1] = r.node;
  return Expr<Val>{ret};
}

template <typename Val>
inline auto stack(Expr<Val> l, Expr<Val> r) {
  return stack(1, l, r);
}

template <typename Val>
inline auto clamp(int span, Expr<Val> in) {
  if (in->data.span <= span) {
    return in;
  }
  if (in->data.kind == ExprData<Val>::RANGE) {
    return range(span, in->data.range.fill);
  }
  auto loop_span = in->data.span / in->data.stack.reps;
  auto quo = span / loop_span, rem = span % loop_span;
  auto l = Expr<Val>{in->deps[0]}, r = Expr<Val>{in->deps[1]};
  auto body = stack(quo, l, r);
  if (rem == 0) {
    return body;
  } else if (rem <= l->data.span) {
    return stack(body, clamp(rem, l));
  } else {
    CHECK_STATE(r);
    return stack(body, stack(clamp(rem, l), clamp(rem - l->data.span, r)));
  }
}

template <typename Val = bool>
inline auto strided(int span, int stride, Val exclude = 0, Val include = 1) {
  CHECK_ARGUMENT(stride > 0);
  if (stride == 1) {
    return range(span, include);
  } else {
    auto reps = 1 + (span - 1) / stride;
    auto body = stack(reps, range(1, include), range(stride - 1, exclude));
    return clamp(span, body);
  }
}

template <typename Val>
inline auto debug_str(Expr<Val> expr) {
  std::unordered_map<typename ExprNode<Val>::Ptr, std::string> str;
  dags::dfs(expr.node, [&](auto e, auto& q) {
    if (e->data.kind == ExprData<Val>::RANGE) {
      str[e] = fmt::format("range({}, {})", e->data.span, e->data.range.fill);
    } else {
      if (auto dep = e->deps[0]; dep && !str.count(dep)) {
        q.push_back(dep);
      }
      if (auto dep = e->deps[1]; dep && !str.count(dep)) {
        q.push_back(dep);
      }
      if (q.empty()) {
        auto s1 = e->deps[0] ? fmt::format(", {}", str[e->deps[0]]) : "";
        auto s2 = e->deps[1] ? fmt::format(", {}", str[e->deps[1]]) : "";
        str[e] = fmt::format("stack({}{}{})", e->data.stack.reps, s1, s2);
      } else {
        q.push_back(e);
      }
    }
  });
  return fmt::format("{}: {}", str.at(expr.node), typeid(Val).name());
}

template <typename Val>
inline auto build(Expr<Val> expr) {
  // Initialize the output mask.
  auto mask = std::make_shared<core::Store<Val>>(expr->data.size);
  auto size = 0;

  // Helper function to initialzie the rep count of a node.
  auto init_node = [](auto expr) {
    if (expr->data.kind == ExprData<Val>::STACK) {
      return std::tuple(std::move(expr), expr->data.stack.reps);
    } else {
      return std::tuple(std::move(expr), 1);
    }
  };

  // DFS to populate the mask.
  dags::dfs(init_node(expr.node), [&](auto t, auto& q) {
    auto [e, i] = t;
    if (i == 0) {
      return;
    } else if (e->data.kind == ExprData<Val>::RANGE) {
      auto span = e->data.span;
      auto fill = e->data.range.fill;
      if (span == 0) {
        return;
      } else if (size == 0) {
        mask->ends[0] = span;
        mask->vals[0] = fill;
        ++size;
      } else if (mask->vals[size - 1] != fill) {
        mask->ends[size] = mask->ends[size - 1] + span;
        mask->vals[size] = fill;
        ++size;
      } else {
        mask->ends[size - 1] += span;
      }
    } else if (e->data.kind == ExprData<Val>::STACK) {
      if (e->deps[0]) {
        q.push_back(init_node(e->deps[0]));
      }
      if (e->deps[1]) {
        q.push_back(init_node(e->deps[1]));
      }
      q.push_back(std::tuple(e, i - 1));
    }
  });

  // Adjust the final mask's size due to compression and return.
  mask->size = size;
  return mask;
}

template <typename Val = bool, typename Bit = Val>
inline auto stride_mask(
    core::Pos span,
    core::Pos start,
    core::Pos stop,
    core::Pos stride = 1,
    Bit exclude = 0,
    Bit include = 1) {
  CHECK_ARGUMENT(0 <= start);
  CHECK_ARGUMENT(start <= stop);
  CHECK_ARGUMENT(stop <= span);
  CHECK_ARGUMENT(stride > 0);
  auto head = range<Val>(start, exclude);
  auto body = strided<Val>(stop - start, stride, exclude, include);
  auto tail = range<Val>(span - stop, exclude);
  return build<Val>(stack(head, stack(body, tail)));
}

}  // namespace skimpy::detail::mask
