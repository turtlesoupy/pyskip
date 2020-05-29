#include <immintrin.h>
#include <omp.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>

#include "taco.h"

namespace py = pybind11;

#if defined(WIN32)
using Pos = int32_t;
#else
using Pos = size_t;
#endif

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

std::unique_ptr<int32_t[]> newRandIntArray(const Pos size) {
  std::unique_ptr<int32_t[]> space(new int32_t[size]);
  int32_t* spacePtr = space.get();

#pragma omp parallel
  {
    std::default_random_engine dre(42);
    std::uniform_int_distribution<int32_t> di(0, INT16_MAX);

#pragma omp for
    for (Pos i = 0; i < size; i++) {
      spacePtr[i] = di(dre);
    }
  }

  return space;
}

#if !defined(WIN32)
__attribute__((optimize("no-tree-vectorize")))
#endif
auto noSIMDIntCumSumWrite(const Pos num, const int numInputs, const int numThreads) {
  std::vector<std::unique_ptr<int32_t[]>> spaces;
  assert(numInputs < 1024);
  int32_t* spacePtrs[1024];
  for (int i = 0; i < numInputs; i++) {
    spaces.push_back(newRandIntArray(num));
    spacePtrs[i] = spaces[i].get();
  }

  std::unique_ptr<int32_t[]> output(newRandIntArray(num));
  int32_t* outputPtr = output.get();

  auto startTime = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(numThreads)
  {
    assert(omp_get_num_threads() == numThreads);
    int cumSum = 0;
#pragma omp for
    for (Pos i = 0; i < num; i++) {
      for (int j = 0; j < numInputs; j++) {
        cumSum += spacePtrs[j][i];
      }
      outputPtr[i] = cumSum;
    }
  }

  auto duration = std::chrono::high_resolution_clock::now() - startTime;
  return std::make_tuple(
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count(),
      outputPtr[0]);
}

#if !defined(WIN32)
__attribute__((optimize("no-tree-vectorize")))
#endif
auto noSIMDIntSumMultiInput(const Pos num, const int numInputs, const int numThreads) {
  std::vector<std::unique_ptr<int32_t[]>> spaces;
  assert(numInputs < 1024);
  int32_t* spacePtrs[1024];
  for (int i = 0; i < numInputs; i++) {
    spaces.push_back(newRandIntArray(num));
    spacePtrs[i] = spaces[i].get();
  }

  std::atomic<int> bigSum;
  auto startTime = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(numThreads)
  {
    assert(omp_get_num_threads() == numThreads);
    int reduction = 0;
#pragma omp for
    for (Pos i = 0; i < num; i++) {
      for (int j = 0; j < numInputs; j++) {
        reduction += spacePtrs[j][i];
      }
    }
    bigSum += reduction;
  }

  auto duration = std::chrono::high_resolution_clock::now() - startTime;

  return std::make_tuple(
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count(),
      bigSum.load());
}

#if !defined(WIN32)
__attribute__((optimize("no-tree-vectorize")))
#endif
auto SIMDIntSumMultiInput(const Pos num, const int numInputs, const int numThreads) {
  std::vector<std::unique_ptr<int32_t[]>> spaces;
  assert(numInputs < 1024);
  int32_t* spacePtrs[1024];
  for (int i = 0; i < numInputs; i++) {
    spaces.push_back(newRandIntArray(num));
    spacePtrs[i] = spaces[i].get();
  }

  const int simd_width = 8;
  const int reduction_scope = num - num % (numThreads * simd_width);

  std::atomic<int> bigSum;
  auto startTime = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(numThreads)
  {
    assert(omp_get_num_threads() == numThreads);
    __m256i simd_reduction{0};
#pragma omp for
    for (Pos i = 0; i < reduction_scope; i += simd_width) {
      for (int j = 0; j < numInputs; j++) {
        simd_reduction = _mm256_add_epi32(
            simd_reduction,
            _mm256_load_si256(
                reinterpret_cast<__m256i const*>(&spacePtrs[j][i])));
      }
    }

    int accum = 0;
    accum += _mm256_extract_epi32(simd_reduction, 0);
    accum += _mm256_extract_epi32(simd_reduction, 1);
    accum += _mm256_extract_epi32(simd_reduction, 2);
    accum += _mm256_extract_epi32(simd_reduction, 3);
    accum += _mm256_extract_epi32(simd_reduction, 4);
    accum += _mm256_extract_epi32(simd_reduction, 5);
    accum += _mm256_extract_epi32(simd_reduction, 6);
    accum += _mm256_extract_epi32(simd_reduction, 7);
    bigSum += accum;
  }

  // Remainder
  int accum = 0;
  for (Pos i = reduction_scope; i < num; i++) {
    for (int j = 0; j < numInputs; j++) {
      accum += spacePtrs[j][i];
    }
  }
  bigSum += accum;

  auto duration = std::chrono::high_resolution_clock::now() - startTime;

  return std::make_tuple(
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count(),
      bigSum.load());
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
  std::cout << element << std::endl;
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
  std::cout << element << std::endl;
  return std::make_tuple(
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count(),
      element);
}

PYBIND11_MODULE(_skimpy_bench_cpp_ext, m) {
  using namespace pybind11::literals;
  m.doc() = "Benchmarks for skimpy";
  m.attr("__version__") = "0.2";

  py::module taco = m.def_submodule("taco", "Taco Comparison Benchmarks");
  taco.def(
      "dense_sum",
      [](int32_t size, int numInputs, bool includeCompile) {
        auto result = tacoDenseSum(size, numInputs, includeCompile);
        return std::get<0>(result);
      },
      "Dense add arrays in taco",
      "num_elements"_a,
      "num_input_arrays"_a,
      "include_compile_time"_a = true);
  taco.def(
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

  py::module memory = m.def_submodule("memory", "Memory benchmarks");
  memory.def(
      "no_simd_int_sum",
      [](Pos num, int numInputs, int numThreads) {
        auto result = noSIMDIntSumMultiInput(num, numInputs, numThreads);
        return std::get<0>(result);
      },
      "num_elements"_a,
      "num_input_arrays"_a,
      "num_threads"_a);
  memory.def(
      "simd_int_sum",
      [](Pos num, int numInputs, int numThreads) {
        auto result = SIMDIntSumMultiInput(num, numInputs, numThreads);
        return std::get<0>(result);
      },
      "num_elements"_a,
      "num_input_arrays"_a,
      "num_threads"_a);
  memory.def(
      "no_simd_int_cum_sum_write",
      [](Pos num, int numInputs, int numThreads) {
        auto result = noSIMDIntCumSumWrite(num, numInputs, numThreads);
        return std::get<0>(result);
      },
      "num_elements"_a,
      "num_input_arrays"_a,
      "num_threads"_a);
}
