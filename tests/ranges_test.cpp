
#define CATCH_CONFIG_MAIN

#include <fmt/format.h>

#include <catch2/catch.hpp>
#include <cmath>
#include <skimpy/detail/ranges.hpp>

TEST_CASE("Test range map scanning", "[range_map_scan]") {
  using namespace skimpy::detail::ranges;
  /*

  auto x = store(5, 'a');
  x = stack(stack(slice(x, 0, 2), store(1, 'b')), slice(x, 3, 5));
  auto results = emit(x);
  */
}
