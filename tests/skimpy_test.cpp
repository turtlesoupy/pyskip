#define CATCH_CONFIG_MAIN

#include "skimpy/skimpy.hpp"

#include <fmt/core.h>

#include <catch2/catch.hpp>

#include "skimpy/detail/conv.hpp"

TEST_CASE("Test arrays", "[arrays]") {
  skimpy::Array<int> x(5, 0);

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
