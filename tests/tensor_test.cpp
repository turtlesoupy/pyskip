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
  x.set({{0, 2, 1}, {0, 2, 1}}, x.get({{1, 3, 1}, {0, 2, 1}}));
  x.set({{1, 3, 1}, {0, 3, 2}}, x.get({{0, 3, 2}, {1, 3, 1}}));
  REQUIRE(x.get({0, 0}) == 2);
  REQUIRE(x.get({1, 0}) == 5);
  REQUIRE(x.get({2, 0}) == 6);
  REQUIRE(x.get({0, 1}) == 5);
  REQUIRE(x.get({1, 1}) == 6);
  REQUIRE(x.get({2, 1}) == 6);
  REQUIRE(x.get({0, 2}) == 7);
  REQUIRE(x.get({1, 2}) == 7);
  REQUIRE(x.get({2, 2}) == 9);
}

TEST_CASE("Test tensor slices", "[tensors]") {
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
