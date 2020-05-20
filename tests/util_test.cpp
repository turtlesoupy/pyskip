#define CATCH_CONFIG_MAIN

#include "skimpy/detail/util.hpp"

#include <fmt/core.h>

#include <catch2/catch.hpp>
#include <unordered_map>

using namespace skimpy::detail;

TEST_CASE("Test power of two util functions", "[util_power_of_two]") {
  for (int i = 0; i < 31; i += 1) {
    REQUIRE(is_power_of_two(1 << i));
  }

  REQUIRE(round_up_to_power_of_two(14) == 16);
  REQUIRE(round_up_to_power_of_two(16) == 16);
  REQUIRE(round_up_to_power_of_two(17) == 32);
  REQUIRE(round_up_to_power_of_two(1) == 1);
  REQUIRE(round_up_to_power_of_two(2) == 2);
  REQUIRE(round_up_to_power_of_two(3) == 4);
  REQUIRE(round_up_to_power_of_two(4) == 4);

  REQUIRE(lg2(1) == 0);
  REQUIRE(lg2(2) == 1);
  REQUIRE(lg2(3) == 1);
  REQUIRE(lg2(4) == 2);
  REQUIRE(lg2(15) == 3);
  REQUIRE(lg2(16) == 4);
  REQUIRE(lg2(17) == 4);
  REQUIRE(lg2(31) == 4);
  REQUIRE(lg2(32) == 5);
}

TEST_CASE("Test fixed-point combinator pattern", "[y_combinator]") {
  Fix fib([](auto f, size_t n) {
    if (n == 0 || n == 1) {
      return n;
    } else {
      return f(n - 1) + f(n - 2);
    }
  });

  REQUIRE(fib(0) == 0);
  REQUIRE(fib(1) == 1);
  REQUIRE(fib(2) == 1);
  REQUIRE(fib(3) == 2);
  REQUIRE(fib(4) == 3);
  REQUIRE(fib(5) == 5);
  REQUIRE(fib(6) == 8);

  Fix const_fib([](const auto fib, size_t n) {
    if (n == 0 || n == 1) {
      return n;
    } else {
      return fib(n - 1) + fib(n - 2);
    }
  });

  REQUIRE(const_fib(0) == 0);
  REQUIRE(const_fib(1) == 1);
  REQUIRE(const_fib(2) == 1);
  REQUIRE(const_fib(3) == 2);
  REQUIRE(const_fib(4) == 3);
  REQUIRE(const_fib(5) == 5);
  REQUIRE(const_fib(6) == 8);

  Fix const_ref_fib([](const auto& fib, size_t n) {
    if (n == 0 || n == 1) {
      return n;
    } else {
      return fib(n - 1) + fib(n - 2);
    }
  });

  REQUIRE(const_ref_fib(0) == 0);
  REQUIRE(const_ref_fib(1) == 1);
  REQUIRE(const_ref_fib(2) == 1);
  REQUIRE(const_ref_fib(3) == 2);
  REQUIRE(const_ref_fib(4) == 3);
  REQUIRE(const_ref_fib(5) == 5);
  REQUIRE(const_ref_fib(6) == 8);

  std::unordered_map<size_t, size_t> memo;
  Fix mutable_fib([memo](auto fib, size_t n) mutable {
    if (n == 0 || n == 1) {
      return n;
    } else {
      if (!memo.count(n)) {
        memo[n] = fib(n - 1) + fib(n - 2);
      }
      return memo[n];
    }
  });

  REQUIRE(mutable_fib(0) == 0);
  REQUIRE(mutable_fib(1) == 1);
  REQUIRE(mutable_fib(2) == 1);
  REQUIRE(mutable_fib(3) == 2);
  REQUIRE(mutable_fib(4) == 3);
  REQUIRE(mutable_fib(5) == 5);
  REQUIRE(mutable_fib(6) == 8);
}
