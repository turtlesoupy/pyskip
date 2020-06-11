#define CATCH_CONFIG_MAIN

#include "pyskip/builder.hpp"

#include <catch2/catch.hpp>

#include "pyskip/array.hpp"

TEST_CASE("Test array builders", "[builders]") {
  auto x = pyskip::make_array(5, 'x');
  x.set(0, 'a');
  x.set(1, 'b');
  x.set(2, 'c');
  x.set(3, 'd');
  x.set(4, 'e');

  x = [&] {
    auto b = pyskip::make_builder(x);
    b.set(4, 'f');
    b.set(3, 'g');
    b.set(2, 'h');
    b.set(1, 'i');
    b.set(0, 'j');
    return b.build();
  }();

  REQUIRE(x.get(0) == 'j');
  REQUIRE(x.get(1) == 'i');
  REQUIRE(x.get(2) == 'h');
  REQUIRE(x.get(3) == 'g');
  REQUIRE(x.get(4) == 'f');

  auto y = [] {
    return pyskip::make_builder(10, 1).set(1, 3).set(3, 4).build();
  }();
  REQUIRE(y.get(1) == 3);
  REQUIRE(y.get(3) == 4);

  auto z = y.get(pyskip::Slice(0, 10, 2));
  REQUIRE(z.get(0) == 1);
  REQUIRE(z.get(1) == 1);
  REQUIRE(z.get(2) == 1);
  REQUIRE(z.get(3) == 1);
  REQUIRE(z.get(4) == 1);
  REQUIRE(z.str() == "5=>1");

  auto big = [] {
    auto b = pyskip::make_builder(1024 * 1024, 1);
    for (auto i = 0; i < 30; i += 1) {
      b.set(i, i);
    }
    return b.build();
  }();
  REQUIRE(big.get(23) == 23);
}
