#include <immintrin.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include "taco.h"

namespace py = pybind11;

taco::Tensor<int32_t> randTacoDense(const int32_t size, const int32_t seed) {
  taco::Format dense({taco::Dense});
  taco::Tensor<int32_t> out({size}, dense);
  std::default_random_engine dre(seed);
  std::uniform_int_distribution<int32_t> di(0, INT16_MAX);
  for (int i = 0; i < size; i++) {
    out.insert({i}, di(dre));
  }
  out.pack();
  return out;
}

taco::Tensor<int32_t> randTacoSparse(
    const int32_t size,
    const int32_t numNonZero,
    const int32_t maxRunLength,
    const int32_t randomSeed) {
  std::default_random_engine dre(randomSeed);
  std::uniform_int_distribution<int32_t> runLengthRandom(1, maxRunLength);
  std::uniform_int_distribution<int32_t> insertionRandom(0, INT16_MAX);
  std::uniform_int_distribution<int32_t> valueRandom(0, INT16_MAX);

  taco::Format sparse({taco::Sparse});
  taco::Tensor<int32_t> out({size}, sparse);
  for (int i = 0; i < numNonZero; i++) {
    auto insertPosition = insertionRandom(dre);
    auto value = valueRandom(dre);
    out.insert({insertPosition}, value);
    auto runLength = maxRunLength <= 1 ? 1 : runLengthRandom(dre);
    for (int j = 1; j < runLength; j++) {
      out.insert({insertPosition + j}, value);
    }
  }
  out.pack();

  return out;
}

auto tacoSparseSum(
    const int32_t size,
    const int32_t numNonZero,
    const int32_t maxRunLength,
    const bool alignInputs,
    const int numInputs,
    const bool includeCompile) {
  int seed = 42;

  taco::Format sparse({taco::Sparse});
  taco::Tensor<int32_t> out({size}, sparse);

  std::vector<taco::Tensor<int32_t>> inputs;
  for (int i = 0; i < numInputs; i++) {
    inputs.push_back(randTacoSparse(size, numNonZero, maxRunLength, seed));
    if (!alignInputs) {
      seed++;
    }
  }

  taco::IndexVar taco_i;
  if (numInputs == 1) {
    out(taco_i) = inputs[0](taco_i);
  } else {
    auto tmp = inputs[0](taco_i) + inputs[1](taco_i);
    for (int i = 2; i < numInputs; i++) {
      tmp = tmp + inputs[i](taco_i);
    }
    out(taco_i) = tmp;
  }

  auto startTime = std::chrono::high_resolution_clock::now();
  out.compile();
  out.assemble();
  if (!includeCompile) {
    startTime = std::chrono::high_resolution_clock::now();
  }
  out.compute();
  auto duration = std::chrono::high_resolution_clock::now() - startTime;

  auto element = out.beginTyped<int32_t>()->second;
  return std::make_tuple(
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count(),
      element);
}

auto tacoDenseSum(
    const int32_t size, const int numInputs, const bool includeCompile) {
  taco::Format dense({taco::Dense});
  taco::Tensor<int32_t> out({size}, dense);

  int seed = 42;

  std::vector<taco::Tensor<int32_t>> inputs;
  for (int i = 0; i < numInputs; i++) {
    inputs.push_back(randTacoDense(size, seed++));
  }

  taco::IndexVar taco_i;
  if (numInputs == 1) {
    out(taco_i) = inputs[0](taco_i);
  } else {
    auto tmp = inputs[0](taco_i) + inputs[1](taco_i);
    for (int i = 2; i < numInputs; i++) {
      tmp = tmp + inputs[i](taco_i);
    }
    out(taco_i) = tmp;
  }

  auto startTime = std::chrono::high_resolution_clock::now();
  out.compile();
  out.assemble();
  if (!includeCompile) {
    startTime = std::chrono::high_resolution_clock::now();
  }
  out.compute();
  auto duration = std::chrono::high_resolution_clock::now() - startTime;

  auto element = out.beginTyped<int32_t>()->second;
  return std::make_tuple(
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count(),
      element);
}

void addTacoBindings(py::module &m) {
  using namespace pybind11::literals;
  m.def(
      "dense_sum",
      [](int32_t size, int numInputs, bool includeCompile) {
        auto result = tacoDenseSum(size, numInputs, includeCompile);
        return std::get<0>(result);
      },
      "Dense add arrays in taco",
      "num_elements"_a,
      "num_input_arrays"_a,
      "include_compile_time"_a = true);
  m.def(
      "sparse_sum",
      [](int32_t size,
         int numNonZero,
         int maxRunLength,
         bool alignInputs,
         int numInputs,
         bool includeCompile) {
        auto result = tacoSparseSum(
            size,
            numNonZero,
            maxRunLength,
            alignInputs,
            numInputs,
            includeCompile);
        return std::get<0>(result);
      },
      "Sparse add arrays in taco",
      "num_elements"_a,
      "num_non_zero"_a,
      "max_run_length"_a,
      "align_inputs"_a,
      "num_input_arrays"_a,
      "include_compile_time"_a = true);
}