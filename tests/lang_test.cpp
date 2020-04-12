
#define CATCH_CONFIG_MAIN

#include "skimpy/detail/lang.hpp"

#include <fmt/core.h>

#include <catch2/catch.hpp>
#include <typeindex>
#include <typeinfo>

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
  REQUIRE(x->str() == x_s);
}

TEST_CASE("Test normalizing an ops graph", "[ops_normalize]") {
  auto x = store(5, 'a');
  x = slice(stack(stack(slice(x, 0, 2), store(1, 'b')), slice(x, 3, 5)), 2, 4);
  x = normalize(x);

  auto x_s = stack_s(
      slice_s(store_s("1=>b"), "0:1:1"), slice_s(store_s("5=>a"), "3:4:1"));
  REQUIRE(x->str() == x_s);
}

TEST_CASE("Test visiting an ops graph", "[ops_build]") {
  auto x = store(5, 'a');
  x = stack(stack(slice(x, 0, 2), store(1, 'b')), slice(x, 3, 5));

  std::vector<std::type_index> op_types;
  Fix([&](auto visitor, const OpPtr<char>& op) -> void {
    op_types.emplace_back(op->type());
    recurse(op, visitor);
  })(x);

  REQUIRE_THAT(
      op_types,
      Equals<std::type_index>({
          typeid(Stack<char>),
          typeid(Stack<char>),
          typeid(Slice<char>),
          typeid(Store<char>),
          typeid(Store<char>),
          typeid(Slice<char>),
          typeid(Store<char>),
      }));
}
