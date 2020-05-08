#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <fmt/core.h>

#include <catch2/catch.hpp>
#include <skimpy/detail/util.hpp>
#include <unordered_map>
#include <vector>

using namespace skimpy::detail;

// TODO: Add hashing / equality testing to enable caching of CycleTrees.
class CycleTree {
 public:
  struct Node;

  struct StackNode {
    int loops;
    Node* l_child;
    Node* r_child;
    int loop_span;
    int loop_step;
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
    int (*fn)(int);

    RangeNode(int (*fn)(int)) : fn(fn) {}
  };

  struct Node {
    enum { NONE, STACK, RANGE } kind;
    int span;
    int step;
    union {
      StackNode stack;
      RangeNode range;
    };

    Node() : kind(NONE), span(0), step(0) {}

    Node(StackNode stack)
        : kind(Node::STACK),
          span(stack.loops * stack.loop_span),
          step(stack.loops * stack.loop_step),
          stack(std::move(stack)) {}

    Node(int span, RangeNode range)
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
    std::unique_ptr<Node[]> nodes;

    NodeStore(int size) : size(size), nodes(new Node[size]) {}

    NodeStore(int size, std::unique_ptr<Node[]> nodes)
        : size(size), nodes(std::move(nodes)) {}

    Node& operator[](const int i) const {
      return nodes[i];
    };
  };

  CycleTree(NodeStore nodes, int root)
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

class CycleFn {
 public:
  CycleFn(int start, int stop, CycleTree tree)
      : start_(start),
        span_(stop - start),
        tree_(std::move(tree)),
        stack_(tree_.depth()),
        curr_(&stack_[0]) {
    curr_->base = 0;
    curr_->stop = span_;
    curr_->step = 0;
    curr_->node = tree_.root();
    if (span_ > 0) {
      search(0);
    }
  }

  int operator()(int pos) {
    pos -= start_;
    if (pos <= 0) {
      return 0;
    }
    if (pos >= span_) {
      pos = span_;
    }
    if (curr_->base >= pos || pos > curr_->stop) {
      search(pos - 1);
    }
    return curr_->step + curr_->node->range.fn(pos - curr_->base);
  }

 private:
  inline void search(int pos) {
    // Pop the stack until we find a node that includes pos.
    while (curr_->base > pos || pos >= curr_->stop) {
      --curr_;
    };

    // Advance down the cycle tree from here until we're at a leaf.
    auto base = curr_->base;
    auto step = curr_->step;
    while (curr_->node->kind != CycleTree::Node::RANGE) {
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
      const CycleTree::Node* child = nullptr;
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
  }

  void push_stack(int base, int stop, int step, const CycleTree::Node* node) {
    ++curr_;
    curr_->base = base;
    curr_->stop = stop;
    curr_->step = step;
    curr_->node = node;
  }

  struct StackNode {
    int base = 0;
    int stop = 0;
    int step = 0;
    const CycleTree::Node* node = nullptr;
  };

  int start_;
  int span_;
  CycleTree tree_;
  std::vector<StackNode> stack_;
  StackNode* curr_;
};

TEST_CASE("Basic cycle fn tests", "[cycle_fn_basic]") {
  CycleTree::NodeStore nodes(3);
  nodes[0] = CycleTree::range_node(8, [](int i) { return i; });
  nodes[1] = CycleTree::range_node(2, [](int i) { return 0; });
  nodes[2] = CycleTree::stack_node(4, &nodes[0], &nodes[1]);

  CycleTree tree(std::move(nodes), 2);

  // Validate the tree.
  REQUIRE(tree.depth() == 2);
  REQUIRE(tree.root()->span == 40);
  REQUIRE(tree.root()->kind == CycleTree::Node::STACK);
  REQUIRE(tree.root()->stack.loop_span == 10);
  REQUIRE(tree.root()->stack.loop_step == 8);
  REQUIRE(tree.root()->stack.l_child->kind == CycleTree::Node::RANGE);
  REQUIRE(tree.root()->stack.l_child->span == 8);
  REQUIRE(tree.root()->stack.l_child->step == 8);
  REQUIRE(tree.root()->stack.r_child->kind == CycleTree::Node::RANGE);
  REQUIRE(tree.root()->stack.r_child->span == 2);
  REQUIRE(tree.root()->stack.r_child->step == 0);

  // Create a cycle fn for this tree.
  CycleFn fn(0, 40, std::move(tree));
  for (int i = 0; i < 40; i += 1) {
    auto q = i / 10;
    auto r = i % 10;
    REQUIRE(fn(i) == 8 * q + std::min(r, 8));
  }
  REQUIRE(fn(-3) == 0);
  REQUIRE(fn(-2) == 0);
  REQUIRE(fn(-1) == 0);
  REQUIRE(fn(40) == 32);
  REQUIRE(fn(41) == 32);
  REQUIRE(fn(42) == 32);
}

TEST_CASE("Test cycle fn multi-dimensional slicing", "[cycle_fn_multi_dim]") {
  constexpr auto d = 4;
  constexpr auto n = d * d * d;

  std::vector<int> v(n);
  for (int i = 0; i < n; i += 1) {
    v[i] = i % 10;
  }

  auto x0 = 1, x1 = 3;
  auto y0 = 1, y1 = 3;
  auto z0 = 1, z1 = 3;

  auto x_s = x1 - x0;
  auto y_s = y1 - y0;
  auto z_s = z1 - z0;

  // Create the nodes defining the cycle tree.
  CycleTree::NodeStore nodes(5);
  nodes[0] = CycleTree::range_node(x_s, [](int i) { return i; });
  nodes[1] = CycleTree::range_node(d - x_s, [](int i) { return 0; });
  nodes[2] = CycleTree::stack_node(y_s, &nodes[0], &nodes[1]);
  nodes[3] = CycleTree::range_node(d * d - d * y_s, [](int i) { return 0; });
  nodes[4] = CycleTree::stack_node(z_s, &nodes[2], &nodes[3]);

  auto i0 = x0 + y0 * d + z0 * d * d;
  auto i1 = i0 + x_s + d * (y_s - 1) + d * d * (z_s - 1);

  // Create the cycle fn.
  CycleFn fn(i0, i1, CycleTree(std::move(nodes), 4));
  {
    int64_t cnt = 0;
    int64_t sum = 0;
    auto prev_end = 0;
    for (int i = i0; i <= i1; i += 1) {
      auto end = fn(i);
      if (prev_end != end) {
        sum += v[end - 1];
        prev_end = end;
        cnt++;
      }
    }
    REQUIRE(cnt == x_s * y_s * z_s);
    REQUIRE(sum == 28);
  }
}

TEST_CASE("Benchmark static lower bound", "[step_static_lower_bound]") {
  constexpr auto d = 256;
  constexpr auto n = d * d * d;

  std::vector<int> v(n);
  for (int i = 0; i < n; i += 1) {
    v[i] = i % 10;
  }

  auto x0 = 2, x1 = 254, xs = 1;
  auto y0 = 2, y1 = 254, ys = 1;
  auto z0 = 2, z1 = 254, zs = 1;

  BENCHMARK("v[2:254, 2:254, 2:254].sum(); d=256") {
    int64_t cnt = 0;
    int64_t sum = 0;
    for (int z = z0; z < z1; z += zs) {
      for (int y = y0; y < y1; y += ys) {
        for (int x = x0; x < x1; x += xs) {
          sum += v[cnt++];
        }
      }
    }
    REQUIRE(cnt == (x1 - x0) * (y1 - y0) * (z1 - z0));
    REQUIRE(sum == 72013528);
  };
}

TEST_CASE("Benchmark cycle fn", "[step_cycle_fn]") {
  constexpr auto d = 256;
  constexpr auto n = d * d * d;

  std::vector<int> v(n);
  for (int i = 0; i < n; i += 1) {
    v[i] = i % 10;
  }

  auto x0 = 2, x1 = 254;
  auto y0 = 2, y1 = 254;
  auto z0 = 2, z1 = 254;

  auto x_s = x1 - x0;
  auto y_s = y1 - y0;
  auto z_s = z1 - z0;

  // Create the nodes defining the cycle tree.
  CycleTree::NodeStore nodes(5);
  nodes[0] = CycleTree::range_node(x_s, [](int i) { return i; });
  nodes[1] = CycleTree::range_node(d - x_s, [](int i) { return 0; });
  nodes[2] = CycleTree::stack_node(y_s, &nodes[0], &nodes[1]);
  nodes[3] = CycleTree::range_node(d * d - d * y_s, [](int i) { return 0; });
  nodes[4] = CycleTree::stack_node(z_s, &nodes[2], &nodes[3]);

  auto i0 = x0 + y0 * d + z0 * d * d;
  auto i1 = i0 + x_s + d * (y_s - 1) + d * d * (z_s - 1);

  // Create the cycle fn.
  CycleFn fn(i0, i1, CycleTree(std::move(nodes), 4));
  BENCHMARK("v[2:254, 2:254, 2:254].sum(); d=256") {
    int64_t cnt = 0;
    int64_t sum = 0;
    auto prev_end = 0;
    for (int i = i0; i <= i1; i += 1) {
      auto end = fn(i);
      if (prev_end != end) {
        sum += v[end - 1];
        prev_end = end;
        cnt++;
      }
    }
    REQUIRE(cnt == x_s * y_s * z_s);
    REQUIRE(sum == 72013528);
  };
}
