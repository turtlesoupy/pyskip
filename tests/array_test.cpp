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
  REQUIRE(x.get({0, 3}).len() == 3);
  REQUIRE(x.get({0, 3}).get(0) == 0);
  REQUIRE(x.get({0, 3}).get(1) == 3);
  REQUIRE(x.get({0, 3}).get(2) == 5);

  REQUIRE(x.get({2, 4}).len() == 2);
  REQUIRE(x.get({2, 4}).get(0) == 5);
  REQUIRE(x.get({2, 4}).get(1) == 0);

  REQUIRE(x.get({4, 5}).len() == 1);
  REQUIRE(x.get({4, 5}).get(0) == 2);

  REQUIRE(x.get({2, 5, 2}).len() == 2);
  REQUIRE(x.get({2, 5, 2}).get(0) == 5);
  REQUIRE(x.get({2, 5, 2}).get(1) == 2);

  REQUIRE(x.get({0, 4, 3}).len() == 2);
  REQUIRE(x.get({0, 4, 3}).get(0) == 0);
  REQUIRE(x.get({0, 4, 3}).get(1) == 0);

  // Test slice assignment
  x.set({0, 3}, 2);
  x.set({2, 5}, 3);
  x.set({1, 3}, -x.get({2, 4}));
  x.set({4, 5}, 2 * x.get({0, 1}));
  REQUIRE(x.get(0) == 2);
  REQUIRE(x.get(1) == -3);
  REQUIRE(x.get(2) == -3);
  REQUIRE(x.get(3) == 3);
  REQUIRE(x.get(4) == 4);

  // Test strided slices
  x.set({0, 3, 2}, 1);
  x.set({1, 5, 2}, 2 * x.get({0, 3, 2}));
  REQUIRE(x.get(0) == 1);
  REQUIRE(x.get(1) == 2);
  REQUIRE(x.get(2) == 1);
  REQUIRE(x.get(3) == 2);
  REQUIRE(x.get(4) == 4);
}

TEST_CASE("Test arrays merge routines", "[arrays]") {
  auto x = skimpy::make_array(5, 1);
  auto y = skimpy::make_array(5, 2);

  REQUIRE_THAT(skimpy::to_vector(x + y), Catch::Equals<int>({3, 3, 3, 3, 3}));
  REQUIRE_THAT(
      skimpy::to_vector(2 * x + 3 * y), Catch::Equals<int>({8, 8, 8, 8, 8}));

  x.set({1, 2}, 3);
  x.set({2, 4}, 2);
  y.set(1, 3);
  y.set({2, 5, 2}, 4);

  REQUIRE_THAT(skimpy::to_vector(x), Catch::Equals<int>({1, 3, 2, 2, 1}));
  REQUIRE_THAT(skimpy::to_vector(y), Catch::Equals<int>({2, 3, 4, 2, 4}));

  x.set({2, 5, 2}, y.get({1, 4, 2}));
  y.set({2, 5, 2}, x.get({0, 2}) * y.get({1, 4, 2}));

  REQUIRE_THAT(skimpy::to_vector(x), Catch::Equals<int>({1, 3, 3, 2, 2}));
  REQUIRE_THAT(skimpy::to_vector(y), Catch::Equals<int>({2, 3, 3, 2, 6}));
  REQUIRE(x.str() == "1=>1, 3=>3, 5=>2");
  REQUIRE(y.str() == "1=>2, 3=>3, 4=>2, 5=>6");
}

TEST_CASE("Test float arrays", "[arrays]") {
  auto x = skimpy::make_array(5, 1.0f);
  auto y = skimpy::make_array(5, 2.0f);

  REQUIRE_THAT(
      skimpy::to_vector(x + y),
      Catch::Equals<float>({3.0f, 3.0f, 3.0f, 3.0f, 3.0f}));

  REQUIRE_THAT(
      skimpy::to_vector((3.0f * x) % y),
      Catch::Equals<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
}
