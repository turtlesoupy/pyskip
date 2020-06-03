#define CATCH_CONFIG_MAIN

#include <fmt/ranges.h>

#include <catch2/catch.hpp>
#include <skimpy/skimpy.hpp>
#include <utils.hpp>

using namespace skimpy_3d;

TEST_CASE("Test tensor walk", "[utils]") {
  auto m = skimpy::from_vector<bool>({1, 1, 0, 1, 1, 0});
  auto s = skimpy::from_vector<int>({1, 2, 3, 4, 5, 6});
  auto t = skimpy::from_vector<char>({'a', 'b', 'b', 'b', 'c', 'c'});

  std::vector<std::tuple<int, char>> results;
  array_walk(
      [&](int pos, int s_val, char t_val) {
        results.emplace_back(s_val, t_val);
      },
      m,
      s,
      t);

  REQUIRE(std::tuple(1, 'a') == results[0]);
  REQUIRE(std::tuple(2, 'b') == results[1]);
  REQUIRE(std::tuple(4, 'b') == results[2]);
  REQUIRE(std::tuple(5, 'c') == results[3]);
}
