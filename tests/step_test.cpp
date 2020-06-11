
#define CATCH_CONFIG_MAIN

#include <fmt/format.h>

#include <array>
#include <catch2/catch.hpp>
#include <pyskip/detail/step.hpp>
#include <vector>

using namespace pyskip::detail::step;

template <typename Fn>
auto gen(Fn&& fn, std::initializer_list<Pos> args) {
  std::vector<Pos> ret;
  for (auto arg : args) {
    ret.push_back(fn(arg));
  }
  return ret;
}

TEST_CASE("Test cyclic step functions", "[step]") {
  using namespace cyclic;

  // Strided step function
  {
    auto step_fn = build(strided<4>(10));
    REQUIRE_THAT(
        gen(step_fn, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
        Catch::Equals<Pos>({0, 1, 1, 1, 1, 2, 2, 2, 2, 3}));
  }

  // Strided step function with start offset
  {
    auto step_fn = build(stack(fixed<0>(2), strided<2>(8)));
    REQUIRE_THAT(
        gen(step_fn, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}),
        Catch::Equals<Pos>({0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4}));
  }

  // Scaled step function
  {
    auto step_fn = build(scaled<4>(10));
    REQUIRE_THAT(
        gen(step_fn, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
        Catch::Equals<Pos>({0, 4, 8, 12, 16, 20, 24, 28, 32, 36}));
  }

  // Fixed step function
  {
    auto step_fn = build(fixed<4>(10));
    REQUIRE_THAT(
        gen(step_fn, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
        Catch::Equals<Pos>({0, 4, 4, 4, 4, 4, 4, 4, 4, 4}));
  }

  // Step function with slicing offset
  {
    auto step_fn = build(3, 7, scaled<1>(10));
    REQUIRE_THAT(
        gen(step_fn, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
        Catch::Equals<Pos>({0, 0, 0, 0, 1, 2, 3, 4, 4, 4}));
  }
}

TEST_CASE("Test multi-dimensional cyclic step function", "[step]") {
  constexpr auto d = 4;
  constexpr auto n = d * d * d;

  auto x0 = 1, x1 = 3;
  auto y0 = 1, y1 = 3;
  auto z0 = 1, z1 = 3;

  auto x_s = x1 - x0;
  auto y_s = y1 - y0;
  auto z_s = z1 - z0;

  auto i0 = x0 + y0 * d + z0 * d * d;
  auto i1 = i0 + x_s + d * (y_s - 1) + d * d * (z_s - 1);

  // Create the cycle step functino.
  auto fn = [&] {
    using namespace cyclic;
    auto steps = stack(
        z_s,
        stack(y_s, scaled<1>(x_s), fixed<0>(d - x_s)),
        fixed<0>(d * d - d * y_s));
    return cyclic::build(i0, i1, steps);
  }();

  // Initialize some indexable data to access by the step function.
  std::vector<int> v(n);
  for (int i = 0; i < n; i += 1) {
    v[i] = i % 10;
  }

  // Run the test.
  {
    int64_t cnt = 0;
    int64_t sum = 0;
    auto prev_end = 0;
    for (int i = i0; i <= i1; i += 1) {
      auto end = fn(i);
      if (prev_end != end) {
        sum += v[end - 1];
        prev_end = end;
        cnt++;
      }
    }
    REQUIRE(cnt == x_s * y_s * z_s);
    REQUIRE(sum == 28);
  }
}

TEST_CASE("Test large multi-dimensional cyclic step function", "[step]") {
  constexpr auto x_d = 512, y_d = 512;
  constexpr auto n = x_d * y_d;

  std::array<int, 2> shape = {x_d, y_d};
  std::array<std::array<int, 2>, 2> components = {1, x_d - 1, 1, y_d - 1};

  // Create the cycle step functino.
  auto fn = [&] {
    namespace sc = cyclic;

    auto scale = 1;
    auto i_0 = 0, i_1 = 1;
    sc::ExprNode::Ptr expr = nullptr;
    for (int i = 0; i < 2; i += 1) {
      auto [c_0, c_1] = components[i];
      i_0 += c_0 * scale;
      i_1 += (c_1 - 1) * scale;
      if (i == 0) {
        expr = sc::strided(c_1 - c_0, 1);
      } else {
        auto reps = c_1 - c_0;
        auto tail = scale - expr->data.span;
        auto span = i_1 - i_0;
        if (tail > 0) {
          expr = sc::clamp(span, sc::stack(reps, expr, sc::fixed<0>(tail)));
        } else if (reps > 1) {
          expr = sc::clamp(span, sc::stack(reps, expr));
        }
      }
      scale *= shape[i];
    }

    return sc::build(i_0, i_1, expr);
  }();

  // Validate the size of the step function.
  REQUIRE(span(0, n, fn) == (x_d - 2) * (y_d - 2));
}

TEST_CASE("Test step function spans", "[step]") {
  using namespace cyclic;

  REQUIRE(span(0, 10, identity()) == 10);
  REQUIRE(span(2, 10, identity()) == 8);
  REQUIRE(span(2, 2, identity()) == 0);
  REQUIRE(span(3, 10, zero()) == 0);
  REQUIRE(span(0, 10, constant<3>()) == 3);
  REQUIRE(span(3, 10, constant<3>()) == 0);
  REQUIRE(span(0, 1, constant<3>()) == 3);
  REQUIRE(span(0, 0, constant<3>()) == 0);
  REQUIRE(span(5, 5, constant<3>()) == 0);

  {
    auto cyclic_fn = build(2, 12, stack(1, scaled<1>(8), fixed<0>(2)));
    REQUIRE(span(0, 12, cyclic_fn) == 8);
    REQUIRE(span(2, 12, cyclic_fn) == 8);
    REQUIRE(span(2, 14, cyclic_fn) == 8);
  }
}

TEST_CASE("Test corner-case cyclic step function", "[step]") {
  using namespace cyclic;

  {
    StepFn step_fn;
    REQUIRE_THAT(
        gen(step_fn, {-2, -1, 0, 1, 2, 3}),
        Catch::Equals<Pos>({0, 0, 0, 0, 0, 0}));
  }

  {
    auto step_fn = build(shift(2));
    REQUIRE_THAT(
        gen(step_fn, {-2, -1, 0, 1, 2, 3}),
        Catch::Equals<Pos>({0, 0, 0, 0, 0, 0}));
  }

  {
    auto step_fn = build(stack(shift(2), scaled<1>(3)));
    REQUIRE_THAT(
        gen(step_fn, {0, 1, 2, 3, 4, 5, 6}),
        Catch::Equals<Pos>({0, 3, 4, 5, 5, 5, 5}));
  }
}

TEST_CASE("Test step function inverse", "[step]") {
  // TODO: Add tests.
}

TEST_CASE("Test step composition", "[step]") {
  // TODO: Add tests.
}
