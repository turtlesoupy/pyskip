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
    return head + std::string(", ") + join(std::forward<Tail>(tail)...);
  }
}

template <typename First, typename... Args>
std::string lang_s(First&& first, Args&&... args) {
  return fmt::format("{}({})", first, join(std::forward<Args>(args)...));
}

template <typename... Args>
std::string store_s(Args&&... args) {
  return lang_s("store", std::forward<Args>(args)...);
}

template <typename... Args>
std::string slice_s(Args&&... args) {
  return lang_s("slice", std::forward<Args>(args)...);
}

template <typename... Args>
std::string stack_s(Args&&... args) {
  return lang_s("stack", std::forward<Args>(args)...);
}

template <typename... Args>
std::string merge_s(Args&&... args) {
  return lang_s("merge", std::forward<Args>(args)...);
}

template <typename... Args>
std::string apply_s(Args&&... args) {
  return lang_s("apply", std::forward<Args>(args)...);
}

TEST_CASE("Test building an ops graph", "[ops_build]") {
  auto x = store(5, 'a');
  x = stack(stack(slice(x, 0, 2), store(1, 'b')), slice(x, 3, 5));

  auto x_s = stack_s(
      stack_s(slice_s(store_s("5=>a"), "0:2:1"), store_s("1=>b")),
      slice_s(store_s("5=>a"), "3:5:1"));
  REQUIRE(str(x) == x_s);

  auto mul = [](int x, int y) { return x * y; };
  auto neg = [](int x) { return -x; };
  auto y = apply(merge(store(2, 1), slice(store(8, 2), 6, 8), mul), neg);

  auto y_s =
      apply_s(merge_s(store_s("2=>1"), slice_s(store_s("8=>2"), "6:8:1")));
  REQUIRE(str(y) == y_s);
}

TEST_CASE("Test linearizing an ops graph", "[ops_build]") {
  auto x_1 = store(5, 'a');
  auto x_2 = slice(x_1, 0, 2);
  auto x_3 = store(1, 'b');
  auto x_4 = stack(x_2, x_3);
  auto x_5 = slice(x_1, 3, 5);
  auto x_6 = stack(x_4, x_5);

  auto x_s = stack_s(
      stack_s(slice_s(store_s("5=>a"), "0:2:1"), store_s("1=>b")),
      slice_s(store_s("5=>a"), "3:5:1"));
  REQUIRE(str(x_6) == x_s);

  auto l = linearize(x_6);
  REQUIRE(l.size() == 6);
  REQUIRE(l[0] == x_1);
  REQUIRE(l[1] == x_2);
  REQUIRE(l[2] == x_3);
  REQUIRE(l[3] == x_4);
  REQUIRE(l[4] == x_5);
  REQUIRE(l[5] == x_6);
}

TEST_CASE("Test normalizing an ops graph", "[ops_normalize]") {
  // Normalize an example with some stack and slice operations.
  auto x = store(5, 'a');
  x = slice(stack(stack(slice(x, 0, 2), store(1, 'b')), slice(x, 3, 5)), 2, 4);
  x = normalize(x);

  auto x_s = stack_s(
      slice_s(store_s("1=>b"), "0:1:1"), slice_s(store_s("5=>a"), "3:4:1"));
  REQUIRE(str(x) == x_s);

  // Normalize an example with a merge and apply operation.
  auto mul = [](int x, int y) { return x * y; };
  auto neg = [](int x) { return -x; };
  auto y = apply(merge(store(2, 1), slice(store(8, 2), 6, 8), mul), neg);
  y = normalize(y);

  auto y_s = stack_s(apply_s(merge_s(
      slice_s(store_s("2=>1"), "0:2:1"), slice_s(store_s("8=>2"), "6:8:1"))));
  REQUIRE(str(y) == y_s);

  // Normalize an example with a stack of merge and apply operation.
  auto s = store(5, 3);
  auto z = merge(
      stack(slice(s, 0, 2), slice(s, 3, 5)),
      stack(slice(s, 1, 2), slice(s, 0, 2), slice(s, 3, 4)),
      mul);
  z = normalize(z);

  auto s_s = store_s("5=>3");
  auto z_s = stack_s(
      merge_s(slice_s(s_s, "0:1:1"), slice_s(s_s, "1:2:1")),
      merge_s(slice_s(s_s, "1:2:1"), slice_s(s_s, "0:1:1")),
      merge_s(slice_s(s_s, "3:4:1"), slice_s(s_s, "1:2:1")),
      merge_s(slice_s(s_s, "4:5:1"), slice_s(s_s, "3:4:1")));
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
