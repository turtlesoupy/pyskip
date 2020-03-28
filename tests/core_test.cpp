#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <fmt/format.h>
#include <catch2/catch.hpp>
#include <cmath>

#include <skimpy/detail/core.hpp>

TEST_CASE("Test range map scanning", "[range_map_scan]") {
  auto rm = skimpy::detail::make_range_map(9, 'a');

  {
    auto gen = rm.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 9, 'a'));
    REQUIRE(gen.done());
  }

  {
    auto gen = rm.scan(1);
    REQUIRE(gen.next() == std::tuple(0, 9, 'a'));
    REQUIRE(gen.done());
  }

  {
    auto gen = rm.scan(8);
    REQUIRE(gen.next() == std::tuple(0, 9, 'a'));
    REQUIRE(gen.done());
  }

  REQUIRE_THROWS(rm.scan(9));
}

TEST_CASE("Test range map sets", "[range_map_sets]") {
  auto rm = skimpy::detail::make_range_map(9, 'a');
  rm.set(0, 'b');
  rm.set(3, 'b');
  rm.set(4, 'c');
  rm.set(8, 'a');

  {
    auto gen = rm.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 1, 'b'));
    REQUIRE(gen.next() == std::tuple(1, 3, 'a'));
    REQUIRE(gen.next() == std::tuple(3, 4, 'b'));
    REQUIRE(gen.next() == std::tuple(4, 5, 'c'));
    REQUIRE(gen.next() == std::tuple(5, 9, 'a'));
    REQUIRE(gen.done());
  }

  for (int i = 8; i >= 0; i -= 1) {
    rm.set(i, 'X');
  }

  {
    auto gen = rm.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 9, 'X'));
    REQUIRE(gen.done());
  }

  rm.set(8, 'c');
  rm.set(6, 'a');
  rm.set(7, 'b');

  {
    auto gen = rm.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 6, 'X'));
    REQUIRE(gen.next() == std::tuple(6, 7, 'a'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'b'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'c'));
    REQUIRE(gen.done());
  }

  REQUIRE_THROWS(rm.set(9, '?'));

  rm.slice(3, 5).set(0, 'q');
  rm.slice(3, 5).set(1, 'q');
  rm.slice(4, 9).set(0, 'r');
  rm.slice(0, 9).set(5, 's');

  {
    auto gen = rm.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 3, 'X'));
    REQUIRE(gen.next() == std::tuple(3, 4, 'q'));
    REQUIRE(gen.next() == std::tuple(4, 5, 'r'));
    REQUIRE(gen.next() == std::tuple(5, 6, 's'));
    REQUIRE(gen.next() == std::tuple(6, 7, 'a'));
    REQUIRE(gen.next() == std::tuple(7, 8, 'b'));
    REQUIRE(gen.next() == std::tuple(8, 9, 'c'));
    REQUIRE(gen.done());
  }
}

TEST_CASE("Test range map slices", "[range_map_slices]") {
  auto rm = skimpy::detail::make_range_map(5, 'a');

  rm.set(0, 'a');
  rm.set(1, 'b');
  rm.set(2, 'c');
  rm.set(3, 'd');
  rm.set(4, 'e');

  {
    auto gen = rm.slice(0, 5).scan(0);
    REQUIRE(gen.next() == std::tuple(0, 1, 'a'));
    REQUIRE(gen.next() == std::tuple(1, 2, 'b'));
    REQUIRE(gen.next() == std::tuple(2, 3, 'c'));
    REQUIRE(gen.next() == std::tuple(3, 4, 'd'));
    REQUIRE(gen.next() == std::tuple(4, 5, 'e'));
    REQUIRE(gen.done());
  }

  {
    auto gen = rm.slice(1, 5).scan(0);
    REQUIRE(gen.next() == std::tuple(0, 1, 'b'));
    REQUIRE(gen.next() == std::tuple(1, 2, 'c'));
    REQUIRE(gen.next() == std::tuple(2, 3, 'd'));
    REQUIRE(gen.next() == std::tuple(3, 4, 'e'));
    REQUIRE(gen.done());
  }

  {
    auto gen = rm.slice(1, 3).scan(0);
    REQUIRE(gen.next() == std::tuple(0, 1, 'b'));
    REQUIRE(gen.next() == std::tuple(1, 2, 'c'));
    REQUIRE(gen.done());
  }
}

TEST_CASE("Test range map assigns", "[range_map_assigns]") {
  auto rm = skimpy::detail::make_range_map(9, 0);

  rm.slice(3, 7).assign(1);

  {
    auto gen = rm.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 3, 0));
    REQUIRE(gen.next() == std::tuple(3, 7, 1));
    REQUIRE(gen.next() == std::tuple(7, 9, 0));
    REQUIRE(gen.done());
  }

  for (int i = 0; i < 9; i += 1) {
    rm.set(i, i);
  }

  rm.slice(0, 8).assign(rm.slice(1, 9));

  {
    auto gen = rm.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 1, 1));
    REQUIRE(gen.next() == std::tuple(1, 2, 2));
    REQUIRE(gen.next() == std::tuple(2, 3, 3));
    REQUIRE(gen.next() == std::tuple(3, 4, 4));
    REQUIRE(gen.next() == std::tuple(4, 5, 5));
    REQUIRE(gen.next() == std::tuple(5, 6, 6));
    REQUIRE(gen.next() == std::tuple(6, 7, 7));
    REQUIRE(gen.next() == std::tuple(7, 9, 8));
    REQUIRE(gen.done());
  }

  rm.slice(4, 9).assign(rm.slice(0, 5));

  {
    auto gen = rm.scan(0);
    REQUIRE(gen.next() == std::tuple(0, 1, 1));
    REQUIRE(gen.next() == std::tuple(1, 2, 2));
    REQUIRE(gen.next() == std::tuple(2, 3, 3));
    REQUIRE(gen.next() == std::tuple(3, 4, 4));
    REQUIRE(gen.next() == std::tuple(4, 5, 1));
    REQUIRE(gen.next() == std::tuple(5, 6, 2));
    REQUIRE(gen.next() == std::tuple(6, 7, 3));
    REQUIRE(gen.next() == std::tuple(7, 8, 4));
    REQUIRE(gen.next() == std::tuple(8, 9, 5));
    REQUIRE(gen.done());
  }
}
