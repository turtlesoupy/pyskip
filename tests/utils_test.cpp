#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <optional>

#include <skimpy/detail/utils.hpp>

using namespace skimpy::detail;

/*
TEST_CASE("Test generator pattern with no elements", "[generator_empty]") {
  auto gen = make_generator<int>([]() -> std::optional<int> { return {}; });

  REQUIRE(gen.done());
  REQUIRE_THROWS(gen.get());
  REQUIRE_THROWS(gen.next());
}

TEST_CASE("Test generator pattern with finite elements", "[generator_empty]") {
  auto gen = make_generator<int>([i = 0]() mutable -> std::optional<int> {
    if (i < 4) {
      return i++;
    } else {
      return {};
    }
  });

  REQUIRE(!gen.done());
  REQUIRE(gen.get() == 0);
  REQUIRE(gen.get() == 0);
  gen.next();
  REQUIRE(!gen.done());
  REQUIRE(gen.get() == 1);
  gen.next();
  REQUIRE(!gen.done());
  REQUIRE(gen.get() == 2);
  REQUIRE(gen.get() == 2);
  REQUIRE(gen.get() == 2);
  gen.next();
  REQUIRE(!gen.done());
  REQUIRE(gen.get() == 3);
  gen.next();
  REQUIRE(gen.done());
  REQUIRE_THROWS(gen.get());
  REQUIRE_THROWS(gen.next());
}
*/
