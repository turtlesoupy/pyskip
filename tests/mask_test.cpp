#define CATCH_CONFIG_MAIN

#include <fmt/format.h>

#include <array>
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
  {
    auto m_s = "4=>0, 5=>1, 7=>0, 8=>1, 10=>0, 11=>1, 13=>0, 14=>1, 20=>0";
    REQUIRE(to_string(*stride_mask<int>(20, 4, 14, 3)) == m_s);
  }

  {
    auto m_s = "4=>0, 5=>1, 7=>0, 8=>1, 10=>0, 11=>1, 20=>0";
    REQUIRE(to_string(*stride_mask<int>(20, 4, 13, 3)) == m_s);
  }

  {
    std::array<int, 2> shape = {4, 4};
    std::array<std::array<int, 3>, 2> components = {0, 4, 3, 1, 4, 2};

    auto scale = 1;
    auto i0 = 0, i1 = 1;
    Expr<int> body{nullptr};
    for (int i = 0; i < 2; i += 1) {
      auto [c_0, c_1, c_s] = components[i];
      i0 += c_0 * scale;
      i1 += (c_1 - 1) * scale;
      if (i == 0) {
        body = strided<int>(c_1 - c_0, c_s);
      } else {
        auto reps = 1 + (c_1 - c_0 - 1) / c_s;
        auto tail = scale - body->data.span + (c_s - 1) * scale;
        auto s = i1 - i0;
        if (tail > 0) {
          body = clamp(s, stack(reps, body, range(tail, 0)));
        } else {
          body = clamp(s, stack(reps, body));
        }
      }
      scale *= shape[i];
    }

    auto head = range(i0, 0);
    auto tail = range(4 * 4 - i1, 0);
    auto mask = build(stack(head, stack(body, tail)));

    auto m_s = "4=>0, 5=>1, 7=>0, 8=>1, 12=>0, 13=>1, 15=>0, 16=>1";
    REQUIRE(to_string(*mask) == m_s);
  }
}
