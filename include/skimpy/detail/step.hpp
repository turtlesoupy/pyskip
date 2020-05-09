#pragma once

#include <array>
#include <limits>
#include <memory>
#include <vector>

#include "core.hpp"
#include "util.hpp"

namespace skimpy::detail::step {

using Pos = core::Pos;

using SimpleStepFn = Pos (*)(Pos);

// The underlying tree representing a CycleStepFn.
class CyclicTree {
 public:
  struct Node;

  struct StackNode {
    int loops;
    Node* l_child;
    Node* r_child;
    Pos loop_span;
    Pos loop_step;
    int bit_shift;

    StackNode(int loops, Node* l_child, Node* r_child = nullptr)
        : loops(loops), l_child(l_child), r_child(r_child) {
      CHECK_ARGUMENT(l_child != nullptr);
      loop_span = l_child->span;
      loop_step = l_child->step;
      if (r_child) {
        loop_span += r_child->span;
        loop_step += r_child->step;
      }
      bit_shift = is_power_of_two(loop_span) ? lg2(loop_span) : 0;
    }
  };

  struct RangeNode {
    SimpleStepFn fn;

    RangeNode(SimpleStepFn fn) : fn(std::move(fn)) {}
  };

  struct Node {
    enum { NONE, STACK, RANGE } kind;
    Pos span;
    Pos step;
    union {
      StackNode stack;
      RangeNode range;
    };

    Node() : kind(NONE), span(0), step(0) {}

    explicit Node(StackNode stack)
        : kind(Node::STACK),
          span(stack.loops * stack.loop_span),
          step(stack.loops * stack.loop_step),
          stack(std::move(stack)) {}

    Node(Pos span, RangeNode range)
        : kind(Node::RANGE),
          span(span),
          step(range.fn(span)),
          range(std::move(range)) {}
  };

  template <typename... Args>
  static auto stack_node(Args&&... args) {
    return Node(StackNode(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static auto range_node(int span, Args&&... args) {
    return Node(span, RangeNode(std::forward<Args>(args)...));
  }

  struct NodeStore {
    int size;
    std::shared_ptr<Node[]> nodes;

    explicit NodeStore(int size)
        : size(size),
          nodes(new Node[size, std::hardware_destructive_interference_size]) {}

    NodeStore(int size, std::shared_ptr<Node[]> nodes)
        : size(size), nodes(std::move(nodes)) {}

    Node& operator[](const int i) const {
      return nodes[i];
    };
  };

  template <typename... Args>
  static auto one_node_store(Args&&... args) {
    NodeStore store(1);
    store[0] = range_node(std::forward<Args>(args)...);
    return store;
  }

  CyclicTree() : CyclicTree(one_node_store(1, [](Pos p) { return p; }), 0) {}

  CyclicTree(NodeStore nodes, int root)
      : nodes_(std::move(nodes)),
        root_(root),
        depth_(node_depth(nodes_[root_])) {
    CHECK_ARGUMENT(0 <= root_);
    CHECK_ARGUMENT(root_ < nodes_.size);
  }

  auto root() const {
    return &nodes_[root_];
  }

  auto depth() const {
    return depth_;
  }

 private:
  static int node_depth(const Node& node) {
    if (node.kind == Node::RANGE) {
      return 1;
    }
    auto l_child = node.stack.l_child;
    auto r_child = node.stack.r_child;
    if (!r_child) {
      return 1 + node_depth(*l_child);
    } else {
      return 1 + std::max(node_depth(*l_child), node_depth(*r_child));
    }
  }

  NodeStore nodes_;
  int root_;
  int depth_;
};

class CyclicStepFn {
 public:
  CyclicStepFn() : CyclicStepFn(CyclicTree()) {}

  explicit CyclicStepFn(CyclicTree tree)
      : CyclicStepFn(0, std::numeric_limits<Pos>::max(), std::move(tree)) {}

  CyclicStepFn(Pos start, Pos stop, const CyclicStepFn& other)
      : CyclicStepFn(
            other.start_ + start,
            other.start_ + std::min(other.span_, stop),
            other.tree_) {}

  CyclicStepFn(Pos start, Pos stop, CyclicTree tree)
      : start_(start), span_(stop - start), tree_(std::move(tree)) {
    reset();
  }

  // Copy and move constructors
  CyclicStepFn(const CyclicStepFn& other) {
    *this = other;
  }
  CyclicStepFn(CyclicStepFn&& other) {
    *this = std::move(other);
  }

  // Copy and move assignment
  CyclicStepFn& operator=(const CyclicStepFn& other) {
    start_ = other.start_;
    span_ = other.span_;
    tree_ = other.tree_;
    reset();
    return *this;
  }
  CyclicStepFn& operator=(CyclicStepFn&& other) {
    start_ = other.start_;
    span_ = other.span_;
    tree_ = std::move(other.tree_);
    reset();
    return *this;
  }

  Pos operator()(Pos pos) const {
    pos -= start_;
    if (curr_->base >= pos || pos > curr_->stop) {
      if (pos <= 0) {
        return 0;
      }
      if (pos >= span_) {
        pos = span_;
      }
      if (curr_->base >= pos || pos > curr_->stop) {
        search(pos - 1);
      }
    }
    return curr_->step + curr_fn_(pos - curr_->base);
  }

 private:
  void reset() {
    stack_.resize(tree_.depth());
    curr_ = &stack_[0];
    curr_->base = 0;
    curr_->stop = span_;
    curr_->step = 0;
    curr_->node = tree_.root();
    if (span_ > 0) {
      search(0);
    }
  }

  inline void search(int pos) const {
    // Pop the stack until we find a node that includes pos.
    while (curr_->base > pos || pos >= curr_->stop) {
      --curr_;
    };

    // Advance down the cycle tree from here until we're at a leaf.
    auto base = curr_->base;
    auto step = curr_->step;
    while (curr_->node->kind != CyclicTree::Node::RANGE) {
      auto q = 0, r = 0;
      if (curr_->node->stack.bit_shift) {
        q = (pos - base) >> curr_->node->stack.bit_shift;
        r = (pos - base) & (curr_->node->stack.loop_span - 1);
      } else {
        q = (pos - base) / curr_->node->stack.loop_span;
        r = (pos - base) % curr_->node->stack.loop_span;
      }
      base += q * curr_->node->stack.loop_span;
      step += q * curr_->node->stack.loop_step;

      // Figure out which of the two children to recurse over.
      const CyclicTree::Node* child = nullptr;
      if (r < curr_->node->stack.l_child->span) {
        child = curr_->node->stack.l_child;
      } else {
        base += curr_->node->stack.l_child->span;
        step += curr_->node->stack.l_child->step;
        child = curr_->node->stack.r_child;
      }

      // Push this child onto the stack.
      push_stack(base, base + child->span, step, child);
    }

    curr_fn_ = curr_->node->range.fn;
    curr_base_ = curr_->base;
    curr_stop_ = curr_->stop;
    curr_step_ = curr_->step;
  }

  void push_stack(
      int base, int stop, int step, const CyclicTree::Node* node) const {
    ++curr_;
    curr_->base = base;
    curr_->stop = stop;
    curr_->step = step;
    curr_->node = node;
  }

  struct StackNode {
    Pos base = 0;
    Pos stop = 0;
    Pos step = 0;
    const CyclicTree::Node* node = nullptr;
  };

  Pos start_;
  Pos span_;
  CyclicTree tree_;
  mutable std::vector<StackNode> stack_;
  mutable StackNode* curr_;
  mutable SimpleStepFn curr_fn_;
  mutable Pos curr_base_;
  mutable Pos curr_stop_;
  mutable Pos curr_step_;
};

namespace cyclic {

struct CyclicExpr {
  using Dep = std::shared_ptr<CyclicExpr>;
  enum Kind { RANGE, STACK } kind;
  struct {
    int span;
    SimpleStepFn fn;
  } range;
  struct {
    int loops;
    Dep l_dep;
    Dep r_dep;
  } stack;

  CyclicExpr(Kind kind) : kind(kind) {}
};

inline auto range(int span, SimpleStepFn fn) {
  auto ret = std::make_shared<CyclicExpr>(CyclicExpr::RANGE);
  ret->range.span = span;
  ret->range.fn = fn;
  return ret;
}

inline auto stack(
    int loops, CyclicExpr::Dep l_dep, CyclicExpr::Dep r_dep = nullptr) {
  CHECK_ARGUMENT(l_dep);
  auto ret = std::make_shared<CyclicExpr>(CyclicExpr::STACK);
  ret->stack.loops = loops;
  ret->stack.l_dep = l_dep;
  ret->stack.r_dep = r_dep;
  return ret;
}

inline auto build(int start, int stop, CyclicExpr::Dep expr) {
  // Count the number of nodes we need to allocate.
  auto size = Fix([](auto f, const CyclicExpr::Dep& expr) -> int {
    auto ret = 1;
    if (expr->kind == CyclicExpr::STACK) {
      ret += f(expr->stack.l_dep);
      if (expr->stack.r_dep) {
        ret += f(expr->stack.r_dep);
      }
    }
    return ret;
  })(expr);

  // Generate the nodes composing the resulting tree.
  int index = 0;
  CyclicTree::NodeStore nodes(size);
  Fix([&](const auto& f, const CyclicExpr::Dep& expr) -> CyclicTree::Node* {
    auto& node = nodes[index++];
    if (expr->kind == CyclicExpr::RANGE) {
      node = CyclicTree::range_node(expr->range.span, expr->range.fn);
    } else {
      auto l_child = f(expr->stack.l_dep);
      auto r_child = expr->stack.r_dep ? f(expr->stack.r_dep) : nullptr;
      node = CyclicTree::stack_node(expr->stack.loops, l_child, r_child);
    }
    return &node;
  })(expr);

  return CyclicStepFn(start, stop, CyclicTree(std::move(nodes), 0));
}

inline auto build(int stop, CyclicExpr::Dep expr) {
  return build(0, stop, std::move(expr));
}

inline auto build(CyclicExpr::Dep expr) {
  return build(0, std::numeric_limits<Pos>::max(), std::move(expr));
}

}  // namespace cyclic

template <int k, typename F>
class CompositeStepFn {
 public:
  CompositeStepFn(std::array<F, k> fns) : fns_(std::move(fns)) {}

  template <typename Fn>
  explicit CompositeStepFn(Fn&& fn)
      : fns_(std::array<F, 1>{std::forward<Fn>(fn)}) {}

  template <typename Fn>
  auto compose(Fn&& fn) const {
    std::array<F, k + 1> fns;
    fns[0] = std::forward<Fn>(fn);
    for (int i = 0; i < k; i += 1) {
      fns[i + 1] = fns_[i];
    }
    return CompositeStepFn<k + 1, Fn>(std::move(fns));
  }

  Pos operator()(Pos pos) {
    Pos ret = pos;
    for (int i = 0; i < k; i += 1) {
      ret = fns_[i](ret);
    }
    return ret;
  }

 private:
  std::array<F, k> fns_;
};

inline constexpr SimpleStepFn identity() {
  return [](Pos p) { return p; };
}

inline constexpr SimpleStepFn zero() {
  return [](Pos p) { return 0; };
}

template <Pos value>
inline constexpr SimpleStepFn constant() {
  static_assert(value >= 0);
  return [](Pos p) {
    if (p <= 0) {
      return 0;
    }
    return value;
  };
}

template <int stride>
inline constexpr SimpleStepFn stride_fn() {
  static_assert(stride > 0);
  if constexpr (is_power_of_two(stride)) {
    static constexpr uint32_t shift = lg2(stride);
    return [](Pos p) {
      if (p <= 0) {
        return 0;
      }
      return 1 + (p - 1 >> shift);
    };
  } else {
    return [](Pos p) {
      if (p <= 0) {
        return 0;
      }
      return 1 + (p - 1) / stride;
    };
  }
}

inline auto slice(Pos start, Pos stop, CyclicStepFn fn) {
  return CyclicStepFn(start, stop, std::move(fn));
}

inline auto slice(Pos start, Pos stop, SimpleStepFn fn) {
  CyclicTree::NodeStore nodes(1);
  nodes[0] = CyclicTree::range_node(stop - start, std::move(fn));
  return CyclicStepFn(start, stop, CyclicTree(std::move(nodes), 0));
}

template <typename Fn>
inline auto slice(Pos stop, Fn&& fn) {
  return slice(0, stop, std::forward<Fn>(fn));
}

template <typename Fn>
inline auto slice(Fn&& fn) {
  return slice(0, std::numeric_limits<Pos>::max(), std::forward<Fn>(fn));
}

inline auto stride_fn(Pos start, Pos stop, Pos stride) {
  CHECK_ARGUMENT(stride > 0);
  if (stride < 32) {
    static constexpr SimpleStepFn fixed_strides[] = {
        identity(),      stride_fn<2>(),  stride_fn<3>(),  stride_fn<4>(),
        stride_fn<5>(),  stride_fn<6>(),  stride_fn<7>(),  stride_fn<8>(),
        stride_fn<9>(),  stride_fn<10>(), stride_fn<11>(), stride_fn<12>(),
        stride_fn<13>(), stride_fn<14>(), stride_fn<15>(), stride_fn<16>(),
        stride_fn<17>(), stride_fn<18>(), stride_fn<19>(), stride_fn<20>(),
        stride_fn<21>(), stride_fn<22>(), stride_fn<23>(), stride_fn<24>(),
        stride_fn<25>(), stride_fn<26>(), stride_fn<27>(), stride_fn<28>(),
        stride_fn<29>(), stride_fn<30>(), stride_fn<31>(),
    };
    return slice(start, stop, fixed_strides[stride]);
  } else if (is_power_of_two(stride) && stride <= 4096) {
    static constexpr SimpleStepFn pow_2_strides[] = {
        stride_fn<32>(),
        stride_fn<64>(),
        stride_fn<128>(),
        stride_fn<256>(),
        stride_fn<512>(),
        stride_fn<1024>(),
        stride_fn<2048>(),
        stride_fn<4096>(),
    };
    return slice(start, stop, pow_2_strides[lg2(stride) - 5]);
  } else {
    CyclicTree::NodeStore nodes(3);
    nodes[0] = CyclicTree::range_node(1, identity());
    nodes[1] = CyclicTree::range_node(stride - 1, constant<0>());
    nodes[2] = CyclicTree::stack_node(1, &nodes[0], &nodes[1]);
    return CyclicStepFn(start, stop, CyclicTree(std::move(nodes), 2));
  }
}

inline auto stride_fn(Pos stop, Pos stride) {
  return stride_fn(0, stop, stride);
}

inline auto stride_fn(Pos stride) {
  return stride_fn(0, std::numeric_limits<Pos>::max(), stride);
}

template <int k, typename F, typename Fn>
inline auto compose(const CompositeStepFn<k, F>& f, Fn&& g) {
  return f.compose(std::move(g));
}

template <typename F, typename G>
inline auto compose(F&& f, G&& g) {
  return CompositeStepFn(std::forward<F>(f)).compose(slice(std::forward<G>(g)));
}

template <typename Fn>
inline Pos span(Pos start, Pos stop, Fn&& fn) {
  return fn(stop) - fn(start);
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

}  // namespace skimpy::detail::step
