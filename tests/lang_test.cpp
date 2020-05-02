#define CATCH_CONFIG_MAIN

#include "skimpy/detail/lang.hpp"

#include <fmt/core.h>

#include <catch2/catch.hpp>
#include <typeindex>
#include <typeinfo>

#include "skimpy/detail/conv.hpp"
#include "skimpy/detail/util.hpp"

using Catch::Equals;
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
    return head + join(std::forward<Tail>(tail)...);
  }
}

TEST_CASE("Test building an ops graph", "[ops_build]") {
  auto x = store(5, 'a');
  x = stack(stack(slice(x, 0, 2), store(1, 'b')), slice(x, 3, 5));

  auto x_s = join(
      "x0 = store(5=>a); ",
      "x1 = slice(x0, 0:2); ",
      "x2 = store(1=>b); ",
      "x3 = stack(x1, x2); ",
      "x4 = slice(x0, 3:5); ",
      "x5 = stack(x3, x4); ",
      "x5");
  REQUIRE(str(x) == x_s);

  auto mul = [](int x, int y) { return x * y; };
  auto neg = [](int x) { return -x; };
  auto y = apply(merge(store(2, 1), slice(store(8, 2), 6, 8), mul), neg);

  auto y_s = join(
      "x0 = store(2=>1); ",
      "x1 = store(8=>2); ",
      "x2 = slice(x1, 6:8); ",
      "x3 = merge(x0, x2); ",
      "x4 = apply(x3); ",
      "x4");
  REQUIRE(str(y) == y_s);
}

TEST_CASE("Test linearizing an ops graph", "[ops_build]") {
  auto x_0 = store(5, 'a');
  auto x_1 = slice(x_0, 0, 2);
  auto x_2 = store(1, 'b');
  auto x_3 = stack(x_1, x_2);
  auto x_4 = slice(x_0, 3, 5);
  auto x_5 = stack(x_3, x_4);

  auto l = linearize(x_5);
  REQUIRE(l.size() == 6);
  REQUIRE(l[0] == x_0);
  REQUIRE(l[1] == x_1);
  REQUIRE(l[2] == x_2);
  REQUIRE(l[3] == x_3);
  REQUIRE(l[4] == x_4);
  REQUIRE(l[5] == x_5);
}

TEST_CASE("Test normalizing an ops graph", "[ops_normalize]") {
  // Normalize an example with some stack and slice operations.
  auto x = store(5, 'a');
  x = slice(stack(stack(slice(x, 0, 2), store(1, 'b')), slice(x, 3, 5)), 2, 4);
  x = normalize(x);

  auto x_s = join(
      "x0 = store(1=>b); ",
      "x1 = slice(x0, 0:1); ",
      "x2 = store(5=>a); ",
      "x3 = slice(x2, 3:4); ",
      "x4 = stack(x1, x3); ",
      "x4");
  REQUIRE(str(x) == x_s);

  // Normalize an example with a merge and apply operation.
  auto mul = [](int x, int y) { return x * y; };
  auto neg = [](int x) { return -x; };
  auto y = apply(merge(store(2, 1), slice(store(8, 2), 6, 8), mul), neg);
  y = normalize(y);

  auto y_s = join(
      "x0 = store(2=>1); ",
      "x1 = slice(x0, 0:2); ",
      "x2 = store(8=>2); ",
      "x3 = slice(x2, 6:8); ",
      "x4 = merge(x1, x3); ",
      "x5 = apply(x4); ",
      "x6 = stack(x5); ",
      "x6");
  REQUIRE(str(y) == y_s);

  // Normalize an example with a stack of merge and apply operation.
  auto s = store(5, 3);
  auto z = merge(
      stack(slice(s, 0, 2), slice(s, 3, 5)),
      stack(slice(s, 1, 2), slice(s, 0, 2), slice(s, 3, 4)),
      mul);
  z = normalize(z);

  auto z_s = join(
      "x0 = store(5=>3); ",
      "x1 = slice(x0, 0:1); ",
      "x2 = slice(x0, 1:2); ",
      "x3 = merge(x1, x2); ",
      "x4 = merge(x2, x1); ",
      "x5 = slice(x0, 3:4); ",
      "x6 = merge(x5, x2); ",
      "x7 = slice(x0, 4:5); ",
      "x8 = merge(x7, x5); ",
      "x9 = stack(x3, x4, x6, x8); "
      "x9");
  REQUIRE(str(z) == z_s);
}

TEST_CASE("Test computing depth of an ops graph", "[ops_depth]") {
  auto x = store(5, 'a');
  x = stack(stack(slice(x, 0, 2), store(1, 'b')), slice(x, 3, 5));
  REQUIRE(depth(x) == 4);
}

TEST_CASE("Test evaluating an ops graph", "[ops_eval]") {
  auto x = stack(store(2, 0), store(1, 1), store(2, 2), store(1, 3));
  auto y = apply(x, [](int a) { return 4 - a; });
  auto z = merge(x, y, [](int a, int b) { return a * b; });
  auto result = materialize(z);
  REQUIRE_THAT(conv::to_vector(*result), Equals<int>({0, 0, 3, 4, 4, 3}));
}
