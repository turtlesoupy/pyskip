#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <fmt/format.h>

#include <catch2/catch.hpp>
#include <skimpy/detail/conv.hpp>
#include <skimpy/detail/eval2.hpp>
#include <skimpy/detail/step.hpp>

using namespace skimpy::detail::conv;
using namespace skimpy::detail::eval;
using namespace skimpy::detail::step;

TEST_CASE("Test simple eval routine", "[eval_simple]") {
  auto s = to_store({1, 2, 3});
  auto t = to_store({4, 5, 6});

  auto result = eval_simple<int, int>(
      [](const int* v) { return v[0] * v[1]; },
      SimpleSource<int>(s),
      SimpleSource<int>(t));

  REQUIRE(to_string(*result) == "1=>4, 2=>10, 3=>18");
}

TEST_CASE("Test mixed eval routine", "[eval_mixed]") {
  auto m = to_store({false, true, false});
  auto s = to_store({'a', 'x', 'c'});
  auto t = to_store({'x', 'b', 'x'});

  auto result = eval_mixed<char>(
      [](const Mix* args) {
        auto m = std::get<bool>(args[0]);
        auto s = std::get<char>(args[1]);
        auto t = std::get<char>(args[2]);
        return m ? t : s;
      },
      MixSource<bool>(m),
      MixSource<char>(s),
      MixSource<char>(t));

  REQUIRE(to_string(*result) == "1=>a, 2=>b, 3=>c");
}

TEST_CASE("Test eval routine to weave two stores", "[eval_weave]") {
  auto m = to_store({0, 1, 0, 1, 0, 1});
  auto s = to_store({'a', 'c', 'e'});
  auto t = to_store({'b', 'd', 'f'});

  auto result = eval_mixed<char>(
      [](const Mix* args) {
        auto m = std::get<int>(args[0]);
        auto s = std::get<char>(args[1]);
        auto t = std::get<char>(args[2]);
        return m ? t : s;
      },
      MixSource<int>(m),
      MixSource<char>(s, 0, 3, shift(step_fn(1, 0, 1), 1)),
      MixSource<char>(t, 0, 3, shift(step_fn(1, 0, 1), 1)));

  REQUIRE(to_string(*result) == "1=>a, 2=>b, 3=>c, 4=>d, 5=>e, 6=>f");
}

TEST_CASE("Test eval routine to stack two stores", "[eval_stack]") {
  auto m = to_store({0, 0, 0, 1, 1, 1, 1});
  auto s = to_store({'a', 'b', 'c'});
  auto t = to_store({'d', 'e', 'f', 'g'});

  auto result = eval_mixed<char>(
      [](const Mix* args) {
        auto m = std::get<int>(args[0]);
        auto s = std::get<char>(args[1]);
        auto t = std::get<char>(args[2]);
        return m ? t : s;
      },
      MixSource<int>(m),
      MixSource<char>(s, 0, 3, shift(step_fn(3, 0, 4, 1), -1)),
      MixSource<char>(t, 0, 4, shift(step_fn(4, 0, 3, 0), 3)));

  REQUIRE(to_string(*result) == "1=>a, 2=>b, 3=>c, 4=>d, 5=>e, 6=>f, 7=>g");
}
