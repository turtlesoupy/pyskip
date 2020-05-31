#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <array>
#include <catch2/catch.hpp>
#include <thread>
#include <vector>

#include "skimpy/detail/box.hpp"
#include "skimpy/detail/config.hpp"
#include "skimpy/skimpy.hpp"

namespace box = skimpy::detail::box;

TEST_CASE("Benchmark tensor assignments", "[tensors]") {
  // Assign to a random entry in a univariate 2D tensor.
  BENCHMARK("2d_univariate_assign_1") {
    auto x = skimpy::make_tensor<2>({1000, 1000}, 0);
    x.set({892, 643}, 1);
    x.eval();
  };

  // Assign 10 random entries in a univariate array.
  BENCHMARK("2d_univariate_assign_10") {
    auto x = skimpy::make_tensor<2>({1000, 1000}, 0);
    x.set({{872, 882, 1}, {543, 553, 1}}, 1);
    x.eval();
  };

  // Assign 10 random entries in a univariate array.
  BENCHMARK("2d_univariate_assign_100") {
    auto x = skimpy::make_tensor<2>({1000, 1000}, 0);
    x.set({{872, 972, 1}, {543, 643, 1}}, 1);
    x.eval();
  };
}

TEST_CASE("Benchmark tensor merge routines", "[tensors]") {
  // Populate a dense store for testing large batch operations.
  auto store = std::make_shared<box::BoxStore>(1000 * 1000);
  for (int i = 0; i < 1000 * 1000; i += 1) {
    store->ends[i] = i + 1;
    store->vals[i] = i;
  }

  // Merge two arrays with a batch operation.
  BENCHMARK("multiply_2") {
    auto a = skimpy::Tensor<2, int>({1000, 1000}, store);
    auto b = skimpy::Tensor<2, int>({1000, 1000}, store);
    auto z = a.array() * b.array();
    z.eval();
  };

  // Merge four arrays with a batch operation.
  BENCHMARK("multiply_4") {
    auto a = skimpy::Tensor<2, int>({1000, 1000}, store);
    auto b = skimpy::Tensor<2, int>({1000, 1000}, store);
    auto c = skimpy::Tensor<2, int>({1000, 1000}, store);
    auto d = skimpy::Tensor<2, int>({1000, 1000}, store);
    auto z = a.array() * b.array() * c.array() * d.array();
    z.eval();
  };

  // Merge four arrays with a batch operation.
  BENCHMARK("multiply_8") {
    std::array<skimpy::Tensor<2, int>, 8> t = {
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
    };
    auto z = t[0].array();
    for (int i = 1; i < 8; i += 1) {
      z = z * t[i].array();
    }
    z.eval();
  };

  BENCHMARK("multiply_16") {
    std::array<skimpy::Tensor<2, int>, 16> t = {
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
        skimpy::Tensor<2, int>({1000, 1000}, store),
    };
    auto z = t[0].array();
    for (int i = 1; i < 16; i += 1) {
      z = z * t[i].array();
    }
    z.eval();
  };
}

TEST_CASE("Benchmark tensor 2D convolutions", "[tensors][conv]") {
  // Populate a dense store for testing large batch operations.
  auto store = std::make_shared<box::BoxStore>(1000 * 1000);
  for (int i = 0; i < 1000 * 1000; i += 1) {
    store->ends[i] = i + 1;
    store->vals[i] = i;
  }

  auto conv_2d = [](auto& tensor, auto& kernel, int padd = 0, int fill = 0) {
    auto tw = tensor.shape()[0], th = tensor.shape()[1];
    auto kw = kernel.shape()[0], kh = kernel.shape()[1];
    auto pw = tw + 2 * padd, ph = th + 2 * padd;

    auto p_shape = skimpy::make_shape<2>({tw + 2 * padd, th + 2 * padd});
    auto p = skimpy::make_tensor<2>(p_shape, fill);
    p.set({{padd, pw - padd, 1}, {padd, ph - padd, 1}}, tensor);
    p = p.eval();

    auto out_shape = skimpy::make_shape<2>({pw - kw + 1, ph - kh + 1});
    auto out = skimpy::make_tensor<2>(out_shape, 0);
    for (int y = 0; y < kh; y += 1) {
      for (int x = 0; x < kw; x += 1) {
        auto stop_x = pw - kw + x + 1;
        auto stop_y = ph - kh + y + 1;
        auto k_scl = kernel.get({x, y});
        auto p_arr = p.get({{x, stop_x, 1}, {y, stop_y, 1}}).array();
        out = skimpy::make_tensor<2>(out_shape, out.array() + k_scl * p_arr);
      }
    }
    return out;
  };

  BENCHMARK("conv_1x1") {
    auto tensor = skimpy::Tensor<2, int>({1000, 1000}, store);
    auto kernel = skimpy::make_tensor<2>({1, 1}, 2);
    volatile auto ret = conv_2d(tensor, kernel).eval();
  };

  BENCHMARK("conv_3x3") {
    std::array<int, 9> sobel = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    auto tensor = skimpy::Tensor<2, int>({1000, 1000}, store);
    auto kernel = skimpy::from_buffer<2>({3, 3}, 9, &sobel[0]);
    volatile auto ret = conv_2d(tensor, kernel, 1).eval();
  };

  BENCHMARK("conv_5x5") {
    std::array<std::array<int, 5>, 5> sobel = {{
        {-5, -4, 0, 4, 5},
        {-8, -10, 0, 8, 10},
        {-10, -20, 0, 20, 10},
        {-8, -10, 0, 8, 10},
        {-5, -4, 0, 4, 5},
    }};
    auto tensor = skimpy::Tensor<2, int>({1000, 1000}, store);
    auto kernel = skimpy::from_buffer<2>({5, 5}, 25, &sobel[0][0]);
    volatile auto ret = conv_2d(tensor, kernel, 2).eval();
  };
}

TEST_CASE("Benchmark evaluation options", "[tensors][config]") {
  using skimpy::detail::config::GlobalConfig;
  static constexpr auto kTensorNonZeroCount = 1000000;

  auto make_tensors = [](size_t k) {
    std::vector<skimpy::Tensor<1, int>> tensors;
    for (int i = 1; i <= k, i += 1) {
      auto t = skimpy::make_tensor<1>(k * kTensorNonZeroCount, 0);
      t.set({{0, k * kTensorNonZeroCount, k}}) = k;
      tensors.push_back(std::move(t));
    }
    return tensors;
  };

  BENCHMARK("mul(k=32); accelerated_eval=false") {
    auto tensors = make_tensors(32);
  }
}
