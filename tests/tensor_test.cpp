#define CATCH_CONFIG_MAIN

#include "skimpy/tensor.hpp"

#include <fmt/core.h>

#include <catch2/catch.hpp>

#include "skimpy/detail/box.hpp"
#include "skimpy/detail/conv.hpp"
#include "skimpy/detail/core.hpp"

using namespace skimpy::detail;

TEST_CASE("Test basic tensor access", "[tensors]") {
  auto store = std::make_shared<core::Store<box::Box>>(9);
  for (int i = 0; i < 9; i += 1) {
    store->ends[i] = i + 1;
    store->vals[i] = i;
  }

  auto x = skimpy::Tensor<2, int>({3, 3}, store);

  // Test position access
  REQUIRE(x.get({0, 0}) == 0);
  REQUIRE(x.get({1, 0}) == 1);
  REQUIRE(x.get({2, 0}) == 2);
  REQUIRE(x.get({0, 1}) == 3);
  REQUIRE(x.get({1, 1}) == 4);
  REQUIRE(x.get({2, 1}) == 5);
  REQUIRE(x.get({0, 2}) == 6);
  REQUIRE(x.get({1, 2}) == 7);
  REQUIRE(x.get({2, 2}) == 8);

  // Test slice access
  REQUIRE(x.get({{0, 0, 1}, {1, 1, 1}}).len() == 0);
  REQUIRE(x.get({{1, 1, 2}, {1, 3, 1}}).len() == 0);
  REQUIRE(x.get({{1, 3, 1}, {0, 2, 1}}).get({0, 0}) == 1);
  REQUIRE(x.get({{1, 3, 1}, {0, 2, 1}}).get({1, 0}) == 2);
  REQUIRE(x.get({{1, 3, 1}, {0, 2, 1}}).get({0, 1}) == 4);
  REQUIRE(x.get({{1, 3, 1}, {0, 2, 1}}).get({1, 1}) == 5);

  // Test slice access with striding
  REQUIRE(x.get({{0, 3, 2}, {1, 3, 1}}).get({0, 0}) == 3);
  REQUIRE(x.get({{0, 3, 2}, {1, 3, 1}}).get({1, 0}) == 5);
  REQUIRE(x.get({{0, 3, 2}, {1, 3, 1}}).get({0, 1}) == 6);
  REQUIRE(x.get({{0, 3, 2}, {1, 3, 1}}).get({1, 1}) == 8);
}

TEST_CASE("Test basic tensor assignments", "[tensors]") {
  auto x = skimpy::make_tensor<2>({3, 3}, 0);

  // Do a bunch of point assignments
  x.set({0, 0}, 1);
  x.set({1, 0}, 2);
  x.set({2, 0}, 3);
  x.set({0, 1}, 4);
  x.set({1, 1}, 5);
  x.set({2, 1}, 6);
  x.set({0, 2}, 7);
  x.set({1, 2}, 8);
  x.set({2, 2}, 9);
  REQUIRE(x.get({0, 0}) == 1);
  REQUIRE(x.get({1, 0}) == 2);
  REQUIRE(x.get({2, 0}) == 3);
  REQUIRE(x.get({0, 1}) == 4);
  REQUIRE(x.get({1, 1}) == 5);
  REQUIRE(x.get({2, 1}) == 6);
  REQUIRE(x.get({0, 2}) == 7);
  REQUIRE(x.get({1, 2}) == 8);
  REQUIRE(x.get({2, 2}) == 9);

  // Do some tensor assignments
  x.set({{1, 2, 1}, {1, 3, 1}}, x.get({{2, 3, 1}, {0, 2, 1}}));
  x.set({{1, 3, 1}, {0, 1, 1}}, x.get({{0, 2, 1}, {1, 2, 1}}));
  REQUIRE(x.get({0, 0}) == 1);
  REQUIRE(x.get({1, 0}) == 4);
  REQUIRE(x.get({2, 0}) == 3);
  REQUIRE(x.get({0, 1}) == 4);
  REQUIRE(x.get({1, 1}) == 3);
  REQUIRE(x.get({2, 1}) == 6);
  REQUIRE(x.get({0, 2}) == 7);
  REQUIRE(x.get({1, 2}) == 6);
  REQUIRE(x.get({2, 2}) == 9);
  x.set({{0, 2, 1}, {0, 2, 1}}, x.get({{1, 3, 1}, {0, 2, 1}}));
  x.set({{1, 3, 1}, {0, 3, 2}}, x.get({{0, 3, 2}, {1, 3, 1}}));
  REQUIRE(x.get({0, 0}) == 4);
  REQUIRE(x.get({1, 0}) == 3);
  REQUIRE(x.get({2, 0}) == 6);
  REQUIRE(x.get({0, 1}) == 3);
  REQUIRE(x.get({1, 1}) == 6);
  REQUIRE(x.get({2, 1}) == 6);
  REQUIRE(x.get({0, 2}) == 7);
  REQUIRE(x.get({1, 2}) == 7);
  REQUIRE(x.get({2, 2}) == 9);

  // Test value broadcasting
  x.set({{0, 2, 1}, {0, 2, 1}}, 1);
  x.set({{0, 3, 2}, {1, 2, 1}}, 2);
  x.set({{1, 3, 1}, {1, 3, 2}}, 3);
  REQUIRE(x.get({0, 0}) == 1);
  REQUIRE(x.get({1, 0}) == 1);
  REQUIRE(x.get({2, 0}) == 6);
  REQUIRE(x.get({0, 1}) == 2);
  REQUIRE(x.get({1, 1}) == 3);
  REQUIRE(x.get({2, 1}) == 3);
  REQUIRE(x.get({0, 2}) == 7);
  REQUIRE(x.get({1, 2}) == 7);
  REQUIRE(x.get({2, 2}) == 9);
}

TEST_CASE("Test complex tensor assignment", "[tensors]") {
  auto t = skimpy::make_tensor<2>({4, 4}, 0);
  t.set({{0, 4, 2}, {0, 4, 2}}, 1);
  t.set({{1, 4, 2}, {0, 4, 2}}, 2);
  t.set({{0, 4, 2}, {1, 4, 2}}, 3);
  t.set({{1, 4, 2}, {1, 4, 2}}, 4);

  REQUIRE_THAT(
      skimpy::to_vector(t),
      Catch::Equals<int>({1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4}));

  REQUIRE_THAT(
      skimpy::to_vector(t.get({{1, 4, 2}, {0, 4, 2}})),
      Catch::Equals<int>({2, 2, 2, 2}));
}

TEST_CASE("Test tensor set functions", "[tensors]") {
  // Test metadata
  {
    skimpy::TensorSlice<2> slice({{1, 3, 1}, {1, 3, 1}});
    REQUIRE(slice.len() == 4);
    REQUIRE(slice.shape() == skimpy::TensorShape<2>({2, 2}));
    REQUIRE(slice.valid({4, 4}));
    REQUIRE(slice.valid({3, 4}));
    REQUIRE(slice.valid({4, 3}));
    REQUIRE(!slice.valid({2, 4}));
    REQUIRE(!slice.valid({3, 2}));
  }

  // Test set fn generator
  {
    skimpy::TensorSlice<2> slice({{1, 3, 1}, {1, 3, 1}});
    auto step_fn = slice.set_fn({4, 4});
    REQUIRE(step_fn(0) == 0);
    REQUIRE(step_fn(1) == 6);
    REQUIRE(step_fn(2) == 9);
    REQUIRE(step_fn(3) == 10);
    REQUIRE(step_fn(4) == 16);
    REQUIRE(step_fn(5) == 16);
  }
  {
    skimpy::TensorSlice<2> slice({{1, 3, 1}, {1, 4, 2}});
    auto step_fn = slice.set_fn({4, 4});
    REQUIRE(step_fn(0) == 0);
    REQUIRE(step_fn(1) == 6);
    REQUIRE(step_fn(2) == 13);
    REQUIRE(step_fn(3) == 14);
    REQUIRE(step_fn(4) == 16);
    REQUIRE(step_fn(5) == 16);
  }
}

TEST_CASE("Test large multi-dimensional tensor slice", "[tensors]") {
  constexpr auto d = 512, r = d / 2;
  auto disc = skimpy::make_tensor<2>({d, d}, 0);
  for (int y = 0; y < d; y += 1) {
    auto distance = (y - d / 2 + 0.5);
    auto discriminant = r * r - distance * distance;
    if (discriminant < 0) {
      continue;
    }
    auto x_0 = int(d / 2 - 0.5 - std::sqrt(discriminant));
    auto x_1 = int(d / 2 - 0.5 + std::sqrt(discriminant));
    disc.set({{x_0, x_1, 1}, {y, y + 1, 1}}, 1);
  }

  REQUIRE(disc.shape() == skimpy::make_shape<2>({d, d}));
  REQUIRE(
      disc.get({{1, d - 1, 1}, {1, d - 1, 1}}).shape() ==
      skimpy::make_shape<2>({d - 2, d - 2}));
  REQUIRE(
      disc.get({{1, d - 1, 1}, {1, d - 1, 1}}).eval().array().len() ==
      (d - 2) * (d - 2));
}
