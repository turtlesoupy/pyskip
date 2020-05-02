#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <fmt/format.h>

#include <catch2/catch.hpp>
#include <skimpy/detail/core.hpp>
#include <skimpy/detail/step.hpp>

using namespace skimpy::detail::step;

template <typename Fn>
auto gen(Fn&& fn, std::initializer_list<Pos> args) {
  std::vector<Pos> ret;
  for (auto arg : args) {
    ret.push_back(fn(arg));
  }
  return ret;
}

TEST_CASE("Test step function sequences", "[step_sequences]") {
  REQUIRE_THAT(
      gen(step_fn(1, 0), {1, 2, 3, 4, 5, 6, 7, 8}),
      Catch::Equals<Pos>({1, 2, 3, 4, 5, 6, 7, 8}));

  REQUIRE_THAT(
      gen(step_fn(1, 1), {1, 2, 3, 4, 5, 6, 7, 8}),
      Catch::Equals<Pos>({1, 1, 2, 2, 3, 3, 4, 4}));

  REQUIRE_THAT(
      gen(step_fn(2, 1), {1, 2, 3, 4, 5, 6, 7, 8}),
      Catch::Equals<Pos>({1, 2, 2, 3, 4, 4, 5, 6}));

  REQUIRE_THAT(
      gen(step_fn(2, 2), {1, 2, 3, 4, 5, 6, 7, 8}),
      Catch::Equals<Pos>({1, 2, 2, 2, 3, 4, 4, 4}));

  REQUIRE_THAT(
      gen(step_fn(1, 7), {1, 4, 7, 10, 13, 16, 19, 22}),
      Catch::Equals<Pos>({1, 1, 1, 2, 2, 2, 3, 3}));

  REQUIRE_THAT(
      gen(step_fn(1, 6), {1, 4, 7, 10, 13, 16, 19, 22}),
      Catch::Equals<Pos>({1, 1, 1, 2, 2, 3, 3, 4}));

  REQUIRE_THAT(
      gen(step_fn(1, 0, 1), {1, 2, 3, 4, 5, 6, 7, 8}),
      Catch::Equals<Pos>({1, 3, 5, 7, 9, 11, 13, 15}));

  REQUIRE_THAT(
      gen(step_fn(1, 1, 2), {1, 2, 3, 4, 5, 6, 7, 8}),
      Catch::Equals<Pos>({1, 1, 4, 4, 7, 7, 10, 10}));

  REQUIRE_THAT(
      gen(step_fn(2, 2, 3, 1), {1, 2, 3, 4, 5, 6, 7, 8}),
      Catch::Equals<Pos>({2, 2, 2, 6, 7, 7, 7, 11}));
}

TEST_CASE("Test step function spans", "[step_spans]") {
  REQUIRE(span(2, 7, step_fn(1, 0)) == 5);
  REQUIRE(span(2, 7, step_fn(1, 1)) == 3);
  REQUIRE(span(2, 7, step_fn(1, 1, 0, 1)) == 2);
  REQUIRE(span(2, 7, step_fn(1, 1, 0, 1)) == 2);
  REQUIRE(span(2, 7, step_fn(2, 2, 3, 1)) == 5);
}

TEST_CASE("Test step function inverse", "[step_invert]") {
  REQUIRE(invert(1, 3, 10, step_fn(1, 1, 0, -3)) == 4);
  REQUIRE(invert(1, 3, 10, step_fn(1, 1, 0, -3)) == 4);
  REQUIRE(invert(3, 3, 10, step_fn(1, 1, 0, -3)) == 8);
}

TEST_CASE("Test step composition", "[step_compose]") {
  char vals[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'};

  auto take = [&](int start, int stop, StepFn step) {
    std::vector<char> ret;
    auto prev_end = step(start);
    for (int i = start; i < stop; i += 1) {
      auto next_end = step(i + 1);
      if (next_end != prev_end) {
        ret.push_back(vals[i]);
        prev_end = next_end;
      }
    }
    return ret;
  };

  auto slice = [&](int start, int stop, int step = 1) {
    return take(start, stop, step_fn(1, step - 1, 0, -start));
  };

  auto start_invert = [](int pos, int start, int stop, int step = 1) {
    return invert(1 + pos, start, stop, step_fn(1, step - 1, 0, -start)) - 1;
  };

  auto stop_invert = [](int pos, int start, int stop, int step = 1) {
    return invert(pos, start, stop, step_fn(1, step - 1, 0, -start));
  };

  // Test basic slicing.
  REQUIRE_THAT(slice(2, 5), Catch::Equals<char>({'c', 'd', 'e'}));
  REQUIRE_THAT(slice(2, 7, 2), Catch::Equals<char>({'c', 'e', 'g'}));
  REQUIRE_THAT(slice(2, 7, 3), Catch::Equals<char>({'c', 'f'}));
  REQUIRE_THAT(slice(2, 10, 3), Catch::Equals<char>({'c', 'f', 'i'}));
  REQUIRE_THAT(slice(2, 12, 3), Catch::Equals<char>({'c', 'f', 'i', 'l'}));

  // Test slice position inversion.
  REQUIRE_THAT(
      slice(start_invert(0, 2, 10, 3), stop_invert(3, 2, 10, 3), 3),
      Catch::Equals(slice(2, 10, 3)));
  REQUIRE_THAT(
      slice(start_invert(0, 2, 12, 3), stop_invert(4, 2, 12, 3), 3),
      Catch::Equals(slice(2, 12, 3)));
  REQUIRE_THAT(
      slice(start_invert(2, 2, 12, 3), stop_invert(3, 2, 12, 3), 3),
      Catch::Equals<char>({'i'}));

  // Test slice composition.
  REQUIRE_THAT(slice(2, 12, 2), Catch::Equals<char>({'c', 'e', 'g', 'i', 'k'}));
  {
    Pos p_slice[] = {2, 12, 2};
    Pos c_slice[] = {1, 5, 2};
    StepFn p_step_fn = stride_fn(p_slice[0], p_slice[2]);
    StepFn c_step_fn = stride_fn(c_slice[0], c_slice[2]);
    Pos pc_slice[] = {
        start_invert(c_slice[0], p_slice[0], p_slice[1], p_slice[2]),
        stop_invert(c_slice[1], p_slice[0], p_slice[1], p_slice[2]),
    };
    REQUIRE_THAT(
        take(pc_slice[0], pc_slice[1], c_step_fn.compose(p_step_fn)),
        Catch::Equals<char>({'e', 'i'}));
  }
  {
    Pos p_slice[] = {2, 12, 2};
    Pos c_slice[] = {2, 5, 1};
    StepFn p_step_fn = stride_fn(p_slice[0], p_slice[2]);
    StepFn c_step_fn = stride_fn(c_slice[0], c_slice[2]);
    Pos pc_slice[] = {
        start_invert(c_slice[0], p_slice[0], p_slice[1], p_slice[2]),
        stop_invert(c_slice[1], p_slice[0], p_slice[1], p_slice[2]),
    };
    REQUIRE_THAT(
        take(pc_slice[0], pc_slice[1], c_step_fn.compose(p_step_fn)),
        Catch::Equals<char>({'g', 'i', 'k'}));
  }
  {
    Pos p_slice[] = {2, 12, 2};
    Pos c_slice[] = {1, 5, 3};
    StepFn p_step_fn = stride_fn(p_slice[0], p_slice[2]);
    StepFn c_step_fn = stride_fn(c_slice[0], c_slice[2]);
    Pos pc_slice[] = {
        start_invert(c_slice[0], p_slice[0], p_slice[1], p_slice[2]),
        stop_invert(c_slice[1], p_slice[0], p_slice[1], p_slice[2]),
    };
    REQUIRE_THAT(
        take(pc_slice[0], pc_slice[1], c_step_fn.compose(p_step_fn)),
        Catch::Equals<char>({'e', 'k'}));
  }
}
