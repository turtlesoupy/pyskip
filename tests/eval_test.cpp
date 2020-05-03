#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <fmt/format.h>

#include <catch2/catch.hpp>
#include <skimpy/detail/conv.hpp>
#include <skimpy/detail/eval.hpp>
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
      MixSource<char, StepFn>(s, 0, 3, shift(step_fn(1, 0, 1), 1)),
      MixSource<char, StepFn>(t, 0, 3, shift(step_fn(1, 0, 1), 1)));

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
      MixSource<char, StepFn>(s, 0, 3, shift(step_fn(3, 0, 4, 1), -1)),
      MixSource<char, StepFn>(t, 0, 4, shift(step_fn(4, 0, 3, 0), 3)));

  REQUIRE(to_string(*result) == "1=>a, 2=>b, 3=>c, 4=>d, 5=>e, 6=>f, 7=>g");
}

TEST_CASE("Test pool partitioning", "[eval_partition]") {
  auto s = to_store({'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'});
  auto t = to_store({1, 2, 3, 4, 5, 6, 7, 8});

  auto p = partition_pool(
      make_pool<MixSourceBase>(MixSource<char>(s), MixSource<int>(t)), 4);

  auto get_end = [&](int part, int src, int idx) {
    return p[part][src].end(p[part][src].iter() + idx);
  };

  auto get_char = [&](int part, int src, int idx) {
    return std::get<char>(p[part][src].val(p[part][src].iter() + idx));
  };

  auto get_int = [&](int part, int src, int idx) {
    return std::get<int>(p[part][src].val(p[part][src].iter() + idx));
  };

  REQUIRE(p.size() == 4);

  // Validate partition ends.
  REQUIRE(p[0].span() == 2);
  REQUIRE(get_end(0, 0, 0) == 1);
  REQUIRE(get_end(0, 0, 1) == 2);
  REQUIRE(get_end(0, 1, 0) == 1);
  REQUIRE(get_end(0, 1, 1) == 2);

  REQUIRE(p[1].span() == 2);
  REQUIRE(get_end(1, 0, 0) == 3);
  REQUIRE(get_end(1, 0, 1) == 4);
  REQUIRE(get_end(1, 1, 0) == 3);
  REQUIRE(get_end(1, 1, 1) == 4);

  REQUIRE(p[2].span() == 2);
  REQUIRE(get_end(2, 0, 0) == 5);
  REQUIRE(get_end(2, 0, 1) == 6);
  REQUIRE(get_end(2, 1, 0) == 5);
  REQUIRE(get_end(2, 1, 1) == 6);

  REQUIRE(p[3].span() == 2);
  REQUIRE(get_end(3, 0, 0) == 7);
  REQUIRE(get_end(3, 0, 1) == 8);
  REQUIRE(get_end(3, 1, 0) == 7);
  REQUIRE(get_end(3, 1, 1) == 8);

  // Validate partition vals.
  REQUIRE(get_char(0, 0, 0) == 'a');
  REQUIRE(get_char(0, 0, 1) == 'b');
  REQUIRE(get_int(0, 1, 0) == 1);
  REQUIRE(get_int(0, 1, 1) == 2);

  REQUIRE(get_char(1, 0, 0) == 'c');
  REQUIRE(get_char(1, 0, 1) == 'd');
  REQUIRE(get_int(1, 1, 0) == 3);
  REQUIRE(get_int(1, 1, 1) == 4);

  REQUIRE(get_char(2, 0, 0) == 'e');
  REQUIRE(get_char(2, 0, 1) == 'f');
  REQUIRE(get_int(2, 1, 0) == 5);
  REQUIRE(get_int(2, 1, 1) == 6);

  REQUIRE(get_char(3, 0, 0) == 'g');
  REQUIRE(get_char(3, 0, 1) == 'h');
  REQUIRE(get_int(3, 1, 0) == 7);
  REQUIRE(get_int(3, 1, 1) == 8);
}
