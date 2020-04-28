#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <fmt/format.h>

#include <catch2/catch.hpp>
#include <skimpy/detail/core.hpp>
#include <skimpy/detail/step.hpp>

using namespace skimpy::detail::core;
using namespace skimpy::detail::step;

template <typename Fn>
auto gen(Fn&& fn, std::initializer_list<Pos> args) {
  std::vector<Pos> ret;
  for (auto arg : args) {
    ret.push_back(fn(arg));
  }
  return ret;
}

TEST_CASE("Test run-skip step functinos", "[step_run_skip]") {
  REQUIRE_THAT(
      gen(run_skip_fn(1, 0), {1, 2, 3, 4, 5, 6, 7, 8}),
      Catch::Equals<Pos>({1, 2, 3, 4, 5, 6, 7, 8}));

  REQUIRE_THAT(
      gen(run_skip_fn(1, 1), {1, 2, 3, 4, 5, 6, 7, 8}),
      Catch::Equals<Pos>({1, 1, 2, 2, 3, 3, 4, 4}));

  REQUIRE_THAT(
      gen(run_skip_fn(2, 1), {1, 2, 3, 4, 5, 6, 7, 8}),
      Catch::Equals<Pos>({1, 2, 2, 3, 4, 4, 5, 6}));

  REQUIRE_THAT(
      gen(run_skip_fn(2, 2), {1, 2, 3, 4, 5, 6, 7, 8}),
      Catch::Equals<Pos>({1, 2, 2, 2, 3, 4, 4, 4}));

  REQUIRE_THAT(
      gen(run_skip_fn(1, 7), {1, 4, 7, 10, 13, 16, 19, 21}),
      Catch::Equals<Pos>({1, 1, 1, 2, 2, 2, 3, 3}));
}
