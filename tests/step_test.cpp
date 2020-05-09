#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <fmt/format.h>

#include <catch2/catch.hpp>
#include <skimpy/detail/core.hpp>
#include <skimpy/detail/step.hpp>

using namespace skimpy::detail::step;

template <typename Fn>
auto gen(Fn&& fn, std::initializer_list<Pos> args) {
  std::vector<Pos> ret;
  for (auto arg : args) {
    ret.push_back(fn(arg));
  }
  return ret;
}

TEST_CASE("Test simple step functions", "[step_simple]") {
  REQUIRE_THAT(
      gen(identity(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      Catch::Equals<Pos>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

  REQUIRE_THAT(
      gen(constant<0>(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      Catch::Equals<Pos>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

  REQUIRE_THAT(
      gen(constant<2>(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      Catch::Equals<Pos>({0, 2, 2, 2, 2, 2, 2, 2, 2, 2}));

  REQUIRE_THAT(
      gen(stride_fn<2>(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      Catch::Equals<Pos>({0, 1, 1, 2, 2, 3, 3, 4, 4, 5}));

  REQUIRE_THAT(
      gen(stride_fn<3>(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      Catch::Equals<Pos>({0, 1, 1, 1, 2, 2, 2, 3, 3, 3}));

  REQUIRE_THAT(
      gen(stride_fn<4>(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      Catch::Equals<Pos>({0, 1, 1, 1, 1, 2, 2, 2, 2, 3}));
}

TEST_CASE("Test cyclic step functions", "[step_cyclic]") {
  using namespace cyclic;
  auto steps = stack(1, range(8, identity()), range(2, zero()));
  auto cyclic_fn = cyclic::build(2, 12, steps);

  REQUIRE_THAT(
      gen(cyclic_fn, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}),
      Catch::Equals<Pos>({0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8}));
}

TEST_CASE("Test cyclic tree", "[step_cyclic_tree]") {
  CyclicTree::NodeStore nodes(3);
  nodes[0] = CyclicTree::range_node(8, [](int i) { return i; });
  nodes[1] = CyclicTree::range_node(2, [](int i) { return 0; });
  nodes[2] = CyclicTree::stack_node(4, &nodes[0], &nodes[1]);

  CyclicTree tree(std::move(nodes), 2);

  // Validate the tree.
  REQUIRE(tree.depth() == 2);
  REQUIRE(tree.root()->span == 40);
  REQUIRE(tree.root()->kind == CyclicTree::Node::STACK);
  REQUIRE(tree.root()->stack.loop_span == 10);
  REQUIRE(tree.root()->stack.loop_step == 8);
  REQUIRE(tree.root()->stack.l_child->kind == CyclicTree::Node::RANGE);
  REQUIRE(tree.root()->stack.l_child->span == 8);
  REQUIRE(tree.root()->stack.l_child->step == 8);
  REQUIRE(tree.root()->stack.r_child->kind == CyclicTree::Node::RANGE);
  REQUIRE(tree.root()->stack.r_child->span == 2);
  REQUIRE(tree.root()->stack.r_child->step == 0);

  // Create a cycle fn for this tree.
  CyclicStepFn fn(0, 40, std::move(tree));
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

TEST_CASE("Test cycle fn multi-dimensional slicing", "[step_3d_cyclic]") {
  constexpr auto d = 4;
  constexpr auto n = d * d * d;

  auto x0 = 1, x1 = 3;
  auto y0 = 1, y1 = 3;
  auto z0 = 1, z1 = 3;

  auto x_s = x1 - x0;
  auto y_s = y1 - y0;
  auto z_s = z1 - z0;

  auto i0 = x0 + y0 * d + z0 * d * d;
  auto i1 = i0 + x_s + d * (y_s - 1) + d * d * (z_s - 1);

  // Create the cycle step functino.
  using namespace cyclic;
  auto steps = stack(
      z_s,
      stack(y_s, range(x_s, identity()), range(d - x_s, zero())),
      range(d * d - d * y_s, zero()));
  auto fn = cyclic::build(i0, i1, steps);

  // Initialize some indexable data to access by the step function.
  std::vector<int> v(n);
  for (int i = 0; i < n; i += 1) {
    v[i] = i % 10;
  }

  // Run the test.
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

TEST_CASE("Test step function spans", "[step_spans]") {
  // Test with a simple function.
  REQUIRE(span(0, 10, identity()) == 10);
  REQUIRE(span(2, 10, identity()) == 8);
  REQUIRE(span(2, 2, identity()) == 0);
  REQUIRE(span(3, 10, zero()) == 0);
  REQUIRE(span(0, 10, constant<3>()) == 3);
  REQUIRE(span(3, 10, constant<3>()) == 0);
  REQUIRE(span(0, 1, constant<3>()) == 3);
  REQUIRE(span(0, 0, constant<3>()) == 0);
  REQUIRE(span(5, 5, constant<3>()) == 0);

  // Test with a cyclic function.
  {
    using namespace cyclic;
    auto steps = stack(1, range(8, identity()), range(2, zero()));
    auto cyclic_fn = cyclic::build(2, 12, steps);
    REQUIRE(span(0, 12, cyclic_fn) == 8);
    REQUIRE(span(2, 12, cyclic_fn) == 8);
    REQUIRE(span(2, 14, cyclic_fn) == 8);
  }
}

TEST_CASE("Test step function inverse", "[step_invert]") {
  // TODO: Add tests.
}

TEST_CASE("Test step composition", "[step_compose]") {
  // TODO: Add tests.
}
