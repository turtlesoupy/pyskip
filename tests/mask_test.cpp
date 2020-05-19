#define CATCH_CONFIG_MAIN

#include <fmt/format.h>

#include <catch2/catch.hpp>
#include <skimpy/detail/conv.hpp>
#include <skimpy/detail/mask.hpp>

using namespace skimpy::detail::conv;
using namespace skimpy::detail::mask;

TEST_CASE("Test basic generation of a mask", "[mask]") {
  // Test all exclude case
  REQUIRE(to_string(*build(range(10, 0))) == "10=>0");

  // Test all include case
  REQUIRE(to_string(*build(range(10, 1))) == "10=>1");

  // Test a more complex mask expression
  {
    // 0000100100100100
    auto m = build(stack(1, range(4, 0), stack(4, range(1, 1), range(2, 0))));
    auto m_s = "4=>0, 5=>1, 7=>0, 8=>1, 10=>0, 11=>1, 13=>0, 14=>1, 16=>0";
    REQUIRE(to_string(*m) == m_s);
  }

  // Test a more complex mask expression with compression
  {
    // 111111111111
    auto m = build(stack(2, range(2, 1), stack(1, range(1, 1), range(3, 1))));
    REQUIRE(to_string(*m) == "12=>1");
  }
}

TEST_CASE("Test strided mask utility", "[mask]") {
  auto mask = stride_mask(20, 4, 14, 3);
  fmt::print("{}\n", to_string(*mask));

  auto m2 = stride_mask<int>(20, 4, 14, 3);
  auto v = to_vector(*m2);
  fmt::print("{}\n", fmt::join(v, ""));
}
