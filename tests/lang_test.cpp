#define CATCH_CONFIG_MAIN

#include <fmt/format.h>

#include <catch2/catch.hpp>
#include <skimpy/detail/conv.hpp>
#include <skimpy/detail/core.hpp>
#include <skimpy/detail/lang.hpp>
#include <skimpy/detail/step.hpp>

using namespace skimpy::detail;
using namespace skimpy::detail::lang;

std::string join() {
  return "";
}

template <typename Head, typename... Tail>
std::string join(Head&& head, Tail&&... tail) {
  if constexpr (sizeof...(tail) == 0) {
    return head;
  } else {
    return head + std::string(";\n") + join(std::forward<Tail>(tail)...);
  }
}

static constexpr auto add = [](int a, int b) { return a + b; };
static constexpr auto mul = [](int a, int b) { return a * b; };
static constexpr auto min = [](char a, char b) { return a < b ? a : b; };

TEST_CASE("Test building up expressions", "[lang]") {
  {
    auto x = store(10, 1);
    x = merge(slice(x, 5), slice(x, 5, 10), add);

    auto x_s = join(
        "x0 = store(span=10)",
        "x1 = slice(x0)",
        "x2 = slice(x0)",
        "x3 = merge(x1, x2)",
        "x3 : int");
    REQUIRE(debug_str(x) == x_s);
  }

  {
    auto q = store(10, 'q');
    auto r = store(10, 'r');
    q = merge(slice(q, 2, 8), slice(r, 6), min);

    auto q_s = join(
        "x0 = store(span=10)",
        "x1 = slice(x0)",
        "x2 = store(span=10)",
        "x3 = slice(x2)",
        "x4 = merge(x1, x3)",
        "x4 : char");
    REQUIRE(debug_str(q) == q_s);
  }
}

TEST_CASE("Test normalizing expressions", "[lang]") {
  auto q = store(10, 1);
  auto r = store(10, 2);
  q = slice(merge(slice(q, 2, 8), slice(r, 6), mul), 2, 4);

  {
    ExprGraph graph;
    auto q_node = dagify(graph, q.expr);
    normalize(graph, q_node);

    // Validate the normalized expression graph.
    REQUIRE(q_node->data.kind == ExprData::MERGE_2);
    REQUIRE(q_node->deps[0]->data.kind == ExprData::SLICE);
    REQUIRE(q_node->deps[0]->deps[0]->data.kind == ExprData::STORE);
    REQUIRE(!q_node->deps[0]->deps[1]);
    REQUIRE(q_node->deps[1]->data.kind == ExprData::SLICE);
    REQUIRE(q_node->deps[1]->deps[0]->data.kind == ExprData::STORE);
    REQUIRE(!q_node->deps[1]->deps[1]);

    // Validate the string-format of the normalized expression.
    auto q_s = join(
        "x0 = store(span=10)",
        "x1 = slice(x0)",
        "x2 = store(span=10)",
        "x3 = slice(x2)",
        "x4 = merge(x1, x3)",
        "x4 : int");
    REQUIRE(debug_str(expressify<int>(q_node)) == q_s);
  }
}

TEST_CASE("Test building evaluation plans", "[lang]") {
  auto q = store(10, 1);
  auto r = store(10, 2);
  q = slice(merge(slice(q, 2, 8), slice(r, 6), mul), 2, 4);

  ExprGraph graph;
  auto root = dagify(graph, q.expr);
  normalize(graph, root);

  auto plan = build_plan(root);
  REQUIRE(plan.depth == 2);

  // Validate sources.
  REQUIRE(plan.sources.size() == 2);
  REQUIRE(plan.sources[0]->data.kind == ExprData::SLICE);
  REQUIRE(plan.sources[0]->deps[0]->data.kind == ExprData::STORE);
  REQUIRE(plan.sources[1]->data.kind == ExprData::SLICE);
  REQUIRE(plan.sources[1]->deps[0]->data.kind == ExprData::STORE);

  // Validate eval nodes.
  REQUIRE(plan.nodes.size() == 3);
  REQUIRE(plan.nodes[0].kind == EvalPlan::Node::SOURCE);
  REQUIRE(plan.nodes[0].index == 0);
  REQUIRE(plan.nodes[1].kind == EvalPlan::Node::SOURCE);
  REQUIRE(plan.nodes[1].index == 1);
  REQUIRE(plan.nodes[2].kind == EvalPlan::Node::MERGE_2);
  REQUIRE(plan.nodes[2].fn);
}

TEST_CASE("Test evaluation with a _really_ simple case", "[lang]") {
  auto result = materialize(slice(store(10, 1), 2, 4));
  REQUIRE(conv::to_string(*result) == "2=>1");
}

TEST_CASE("Test evaluation with a _fairly_ simple case", "[lang]") {
  auto q = store(10, 2);
  auto r = store(10, 3);
  q = slice(merge(slice(q, 2, 8), slice(r, 6), mul), 2, 4);

  auto result = materialize(q);
  REQUIRE(conv::to_string(*result) == "2=>6");
}

TEST_CASE("Test evaluation with a multi-type case", "[lang]") {
  auto m = conv::to_store({true, false, true, false, true, false});
  auto x = conv::to_store({'a', 'b', 'c', 'd', 'e', 'f'});
  auto y = conv::to_store({'A', 'B', 'C', 'D', 'E', 'F'});

  auto z = materialize(
      merge(store(m), store(x), store(y), [](bool m, char a, char b) {
        return m ? a : b;
      }));
  REQUIRE_THAT(
      conv::to_vector(*z), Catch::Equals<char>({'a', 'B', 'c', 'D', 'e', 'F'}));
}

TEST_CASE("Test evaluation with a complex step function", "[lang]") {
  using namespace skimpy::detail::step;

  static constexpr auto d = 8;

  // Generate an 8x8 range of values as a store.
  auto x = [] {
    std::vector<int> range;
    for (int i = 0; i < d * d; i += 1) {
      range.push_back(i);
    }
    return conv::to_store(range);
  }();

  // Slice the 8x8 range of values into the middle 4x4 sub-rect.
  auto i_0 = 2 + 2 * d, i_1 = 6 + 5 * d;
  auto i_s = cyclic::build(
      i_0, i_1, cyclic::stack(4, cyclic::scaled<1>(4), cyclic::fixed<0>(4)));
  auto y = materialize(slice(store(x), i_s));

  REQUIRE_THAT(
      conv::to_vector(*y),
      Catch::Equals<int>(
          {18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45}));
}
