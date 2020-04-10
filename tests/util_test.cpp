#define CATCH_CONFIG_MAIN

#include "skimpy/detail/util.hpp"

#include <catch2/catch.hpp>

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
