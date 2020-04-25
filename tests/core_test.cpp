#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <fmt/format.h>

#include <catch2/catch.hpp>
#include <cmath>
#include <skimpy/detail/conv.hpp>
#include <skimpy/detail/core.hpp>

using namespace skimpy::detail::core;
using namespace skimpy::detail::conv;

TEST_CASE("Test point sets into stores", "[core_set]") {
  auto store = make_store(9, 'a');

  set(store, 2, 'c');
  set(store, 4, 'd');

  REQUIRE(store.size == 5);
  REQUIRE(store.ends[0] == 2);
  REQUIRE(store.ends[1] == 3);
  REQUIRE(store.ends[2] == 4);
  REQUIRE(store.ends[3] == 5);
  REQUIRE(store.ends[4] == 9);
  REQUIRE(store.vals[0] == 'a');
  REQUIRE(store.vals[1] == 'c');
  REQUIRE(store.vals[2] == 'a');
  REQUIRE(store.vals[3] == 'd');
  REQUIRE(store.vals[4] == 'a');

  set(store, 2, 'a');
  set(store, 5, 'b');
  set(store, 0, 'b');
  set(store, 1, 'b');
  set(store, 4, 'c');

  REQUIRE(store.size == 5);
  REQUIRE(store.ends[0] == 2);
  REQUIRE(store.ends[1] == 4);
  REQUIRE(store.ends[2] == 5);
  REQUIRE(store.ends[3] == 6);
  REQUIRE(store.ends[4] == 9);
  REQUIRE(store.vals[0] == 'b');
  REQUIRE(store.vals[1] == 'a');
  REQUIRE(store.vals[2] == 'c');
  REQUIRE(store.vals[3] == 'b');
  REQUIRE(store.vals[4] == 'a');

  set(store, 4, 'a');
  set(store, 3, 'c');
  set(store, 2, 'c');
  set(store, 3, 'a');
  set(store, 4, 'd');
  set(store, 8, 'b');

  REQUIRE(store.size == 7);
  REQUIRE(store.ends[0] == 2);
  REQUIRE(store.ends[1] == 3);
  REQUIRE(store.ends[2] == 4);
  REQUIRE(store.ends[3] == 5);
  REQUIRE(store.ends[4] == 6);
  REQUIRE(store.ends[5] == 8);
  REQUIRE(store.ends[6] == 9);
  REQUIRE(store.vals[0] == 'b');
  REQUIRE(store.vals[1] == 'c');
  REQUIRE(store.vals[2] == 'a');
  REQUIRE(store.vals[3] == 'd');
  REQUIRE(store.vals[4] == 'b');
  REQUIRE(store.vals[5] == 'a');
  REQUIRE(store.vals[6] == 'b');
}

TEST_CASE("Test inserting into stores", "[core_insert]") {
  auto store = make_store(9, 'a');

  {
    auto input = make_store(5, 'b');
    insert(store, input, 3);
  }

  REQUIRE(store.size == 3);
  REQUIRE(store.ends[0] == 3);
  REQUIRE(store.ends[1] == 8);
  REQUIRE(store.ends[2] == 9);
  REQUIRE(store.vals[0] == 'a');
  REQUIRE(store.vals[1] == 'b');
  REQUIRE(store.vals[2] == 'a');

  {
    auto input = make_store(3, 'c');
    insert(store, input, 1);
  }

  REQUIRE(store.size == 4);
  REQUIRE(store.ends[0] == 1);
  REQUIRE(store.ends[1] == 4);
  REQUIRE(store.ends[2] == 8);
  REQUIRE(store.ends[3] == 9);
  REQUIRE(store.vals[0] == 'a');
  REQUIRE(store.vals[1] == 'c');
  REQUIRE(store.vals[2] == 'b');
  REQUIRE(store.vals[3] == 'a');

  {
    auto input = make_store(2, 'c');
    insert(store, input, 0);
  }

  REQUIRE(store.size == 3);
  REQUIRE(store.ends[0] == 4);
  REQUIRE(store.ends[1] == 8);
  REQUIRE(store.ends[2] == 9);
  REQUIRE(store.vals[0] == 'c');
  REQUIRE(store.vals[1] == 'b');
  REQUIRE(store.vals[2] == 'a');

  {
    auto input = make_store(3, 'd');
    insert(store, input, 6);
  }

  REQUIRE(store.size == 3);
  REQUIRE(store.ends[0] == 4);
  REQUIRE(store.ends[1] == 6);
  REQUIRE(store.ends[2] == 9);
  REQUIRE(store.vals[0] == 'c');
  REQUIRE(store.vals[1] == 'b');
  REQUIRE(store.vals[2] == 'd');
}

TEST_CASE("Test inserting into stores with ranges", "[core_range_insert]") {
  auto store = make_store(9, 'a');

  {
    auto input = make_store(9, 'b');
    insert(store, Range(input, 0, 5), 3);
  }

  REQUIRE(store.size == 3);
  REQUIRE(store.ends[0] == 3);
  REQUIRE(store.ends[1] == 8);
  REQUIRE(store.ends[2] == 9);
  REQUIRE(store.vals[0] == 'a');
  REQUIRE(store.vals[1] == 'b');
  REQUIRE(store.vals[2] == 'a');

  {
    auto input = make_store(9, 'c');
    insert(store, Range(input, 6, 9), 1);
  }

  REQUIRE(store.size == 4);
  REQUIRE(store.ends[0] == 1);
  REQUIRE(store.ends[1] == 4);
  REQUIRE(store.ends[2] == 8);
  REQUIRE(store.ends[3] == 9);
  REQUIRE(store.vals[0] == 'a');
  REQUIRE(store.vals[1] == 'c');
  REQUIRE(store.vals[2] == 'b');
  REQUIRE(store.vals[3] == 'a');

  {
    auto input = make_store(2, 'c');
    insert(store, Range(input, 0, 2), 0);
  }

  REQUIRE(store.size == 3);
  REQUIRE(store.ends[0] == 4);
  REQUIRE(store.ends[1] == 8);
  REQUIRE(store.ends[2] == 9);
  REQUIRE(store.vals[0] == 'c');
  REQUIRE(store.vals[1] == 'b');
  REQUIRE(store.vals[2] == 'a');

  {
    auto input = make_store(11, 'd');
    insert(store, Range(input, 5, 8), 6);
  }

  REQUIRE(store.size == 3);
  REQUIRE(store.ends[0] == 4);
  REQUIRE(store.ends[1] == 6);
  REQUIRE(store.ends[2] == 9);
  REQUIRE(store.vals[0] == 'c');
  REQUIRE(store.vals[1] == 'b');
  REQUIRE(store.vals[2] == 'd');
}
