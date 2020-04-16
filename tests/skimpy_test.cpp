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

TEST_CASE("Test array builders", "[array_builders]") {
  skimpy::Array<char> x(5, 'x');

  x.set(0, 'a');
  x.set(1, 'b');
  x.set(2, 'c');
  x.set(3, 'd');
  x.set(4, 'e');

  {
    skimpy::ArrayBuilder<char> b(x);
    b.set(4, 'f');
    b.set(3, 'g');
    b.set(2, 'h');
    b.set(1, 'i');
    b.set(0, 'j');
    x = b.build();
  }

  REQUIRE(x.get(0) == 'j');
  REQUIRE(x.get(1) == 'i');
  REQUIRE(x.get(2) == 'h');
  REQUIRE(x.get(3) == 'g');
  REQUIRE(x.get(4) == 'f');

  skimpy::Array<int> y(10, 1);
  y = skimpy::ArrayBuilder<int>(y).set(1, 3).set(3, 4).build();
  REQUIRE(y.get(1) == 3);
  REQUIRE(y.get(3) == 4);

  skimpy::Array<int> z(y.get(skimpy::Slice(0, 10, 2)));
  REQUIRE(z.get(0) == 1);
  REQUIRE(z.get(1) == 1);
  REQUIRE(z.get(2) == 1);
  REQUIRE(z.get(3) == 1);
  REQUIRE(z.get(4) == 1);
  REQUIRE(z.str() == "[1, 1, 1, 1, 1]");
}
