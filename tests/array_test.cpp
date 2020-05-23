#define CATCH_CONFIG_MAIN

#include "skimpy/array.hpp"

#include <catch2/catch.hpp>

#include "skimpy/detail/conv.hpp"

TEST_CASE("Test arrays", "[arrays]") {
  auto x = skimpy::make_array(5, 0);

  // Test initialization
  REQUIRE(x.get(0) == 0);
  REQUIRE(x.get(1) == 0);
  REQUIRE(x.get(2) == 0);
  REQUIRE(x.get(3) == 0);
  REQUIRE(x.get(4) == 0);

  // Test value assignment
  x.set(2, 5);
  x.set(1, 3);
  x.set(4, 2);
  REQUIRE(x.get(0) == 0);
  REQUIRE(x.get(1) == 3);
  REQUIRE(x.get(2) == 5);
  REQUIRE(x.get(3) == 0);
  REQUIRE(x.get(4) == 2);
  REQUIRE(x.get(1) * x.get(2) == 15);

  // Test slice retrieval
  REQUIRE(x.get(skimpy::Slice(0, 3)).len() == 3);
  REQUIRE(x.get(skimpy::Slice(0, 3)).get(0) == 0);
  REQUIRE(x.get(skimpy::Slice(0, 3)).get(1) == 3);
  REQUIRE(x.get(skimpy::Slice(0, 3)).get(2) == 5);

  REQUIRE(x.get(skimpy::Slice(2, 4)).len() == 2);
  REQUIRE(x.get(skimpy::Slice(2, 4)).get(0) == 5);
  REQUIRE(x.get(skimpy::Slice(2, 4)).get(1) == 0);

  REQUIRE(x.get(skimpy::Slice(4, 5)).len() == 1);
  REQUIRE(x.get(skimpy::Slice(4, 5)).get(0) == 2);

  REQUIRE(x.get(skimpy::Slice(2, 5, 2)).len() == 2);
  REQUIRE(x.get(skimpy::Slice(2, 5, 2)).get(0) == 5);
  REQUIRE(x.get(skimpy::Slice(2, 5, 2)).get(1) == 2);

  REQUIRE(x.get(skimpy::Slice(0, 4, 3)).len() == 2);
  REQUIRE(x.get(skimpy::Slice(0, 4, 3)).get(0) == 0);
  REQUIRE(x.get(skimpy::Slice(0, 4, 3)).get(1) == 0);

  // Test slice assignment
  x.set(skimpy::Slice(0, 3), 2);
  x.set(skimpy::Slice(2, 5), 3);
  x.set(skimpy::Slice(1, 3), -x.get(skimpy::Slice(2, 4)));
  x.set(skimpy::Slice(4, 5), x.get(skimpy::Slice(0, 1)) * 2);
  REQUIRE(x.get(0) == 2);
  REQUIRE(x.get(1) == -3);
  REQUIRE(x.get(2) == -3);
  REQUIRE(x.get(3) == 3);
  REQUIRE(x.get(4) == 4);
}

TEST_CASE("Test arrays merge routines", "[arrays]") {
  auto x = skimpy::make_array(5, 1);
  auto y = skimpy::make_array(5, 2);

  REQUIRE_THAT(skimpy::to_vector(x + y), Catch::Equals<int>({3, 3, 3, 3, 3}));
  REQUIRE_THAT(
      skimpy::to_vector(2 * x + 3 * y), Catch::Equals<int>({8, 8, 8, 8, 8}));

  x.set(skimpy::Slice(1, 2), 3);
  x.set(skimpy::Slice(2, 4), 2);
  y.set(1, 3);
  y.set(skimpy::Slice(2, 5, 2), 4);

  REQUIRE_THAT(skimpy::to_vector(x), Catch::Equals<int>({1, 3, 2, 2, 1}));
  REQUIRE_THAT(skimpy::to_vector(y), Catch::Equals<int>({2, 3, 4, 2, 4}));

  x.set(skimpy::Slice(2, 5, 2), y.get(skimpy::Slice(1, 4, 2)));
  y.set(
      skimpy::Slice(2, 5, 2),
      x.get(skimpy::Slice(0, 2)) * y.get(skimpy::Slice(1, 4, 2)));

  REQUIRE_THAT(skimpy::to_vector(x), Catch::Equals<int>({1, 3, 3, 2, 2}));
  REQUIRE_THAT(skimpy::to_vector(y), Catch::Equals<int>({2, 3, 3, 2, 6}));
  REQUIRE(x.str() == "1=>1, 3=>3, 5=>2");
  REQUIRE(y.str() == "1=>2, 3=>3, 4=>2, 5=>6");
}