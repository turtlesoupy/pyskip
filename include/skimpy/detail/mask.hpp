#pragma once

#include <memory>

#include "core.hpp"
#include "dags.hpp"
#include "step.hpp"

namespace skimpy::detail::mask {

// TODO: Add a cache to store frequently used masks.

using Pos = core::Pos;

template <typename Val>
struct MaskArgs {
  enum Kind { RANGE, STACK } kind;
  int size;
  union {
    struct {
      Pos span;
      Val fill;
    } range;
    struct {
      int reps;
    } stack;
  };

  MaskArgs(Kind kind, int size) : kind(kind), size(size) {}
};

template <typename Val>
using MaskExprNode = dags::SharedNode<2, MaskArgs<Val>>;

template <typename Val>
struct MaskExpr {
  typename MaskExprNode<Val>::Ptr node;

  operator bool() const {
    return !!node;
  }

  auto operator-> () {
    return node;
  }
};

template <typename Val>
inline auto range(int span, Val fill) {
  auto ret = MaskExprNode<Val>::make_ptr(MaskArgs<Val>::RANGE, 1);
  ret->data.range.span = span;
  ret->data.range.fill = fill;
  return MaskExpr<Val>{ret};
}

template <typename Val>
inline auto stack(int reps, MaskExpr<Val> l, MaskExpr<Val> r = nullptr) {
  CHECK_ARGUMENT(l);
  auto size = reps * (l->data.size + (r ? r->data.size : 0));
  auto ret = MaskExprNode<Val>::make_ptr(MaskArgs<Val>::STACK, size);
  ret->data.stack.reps = reps;
  ret->deps[0] = l.node;
  ret->deps[1] = r.node;
  return MaskExpr<Val>{ret};
}

template <typename Val>
inline auto build(MaskExpr<Val> expr) {
  // Initialize the output mask.
  auto mask = std::make_shared<core::Store<Val>>(expr->data.size);
  auto size = 0;

  // Helper function to initialzie the rep count of a node.
  auto init_node = [](auto expr) {
    if (expr->data.kind == MaskArgs<Val>::STACK) {
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
    } else if (e->data.kind == MaskArgs<Val>::RANGE) {
      auto span = e->data.range.span;
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
    } else if (e->data.kind == MaskArgs<Val>::STACK) {
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

  // Lead with zeros up to the starting position.
  auto head = range(start, exclude);

  // Define the stride by cycling back-and-forth from one to zero.
  auto reps = (stop - start) / stride;
  auto body = stack(reps, range(1, include), range(stride - 1, exclude));

  // End with possibly a one and trailing zeros.
  auto lead = (stop - start) % stride == 0 ? 0 : 1;
  auto last = span - start - stride * reps - lead;
  auto tail = stack(1, range(lead, include), range(last, exclude));

  return build(stack(1, head, stack(1, body, tail)));
}

}  // namespace skimpy::detail::mask
