#include <immintrin.h>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>

namespace py = pybind11;

#ifdef _MSC_VER
#define NO_TREE_VECTORIZE
#define NO_INLINE __declspec(noinline)
using Pos = int32_t;
#else
using Pos = size_t;
#define NO_TREE_VECTORIZE __attribute__((optimize("no-tree-vectorize")))
#define NO_INLINE __attribute__ ((noinline))
#endif

static constexpr int kMaxInputs = 1024;

NO_INLINE std::unique_ptr<int32_t[]> newRandIntArray(const Pos size) {
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

NO_INLINE std::vector<int32_t> randRunLengthArray(
    const Pos size,
    const int32_t numNonZero,
    const int32_t runLength,
    const bool deterministicRunLength,
    const int32_t randomSeed) {
  std::default_random_engine dre(randomSeed);
  std::uniform_int_distribution<int32_t> runLengthRandom(1, runLength);
  std::uniform_int_distribution<int32_t> insertionRandom(0, size - 1);
  std::uniform_int_distribution<int32_t> valueRandom(0, INT16_MAX);

  std::vector<int32_t> ret;
  ret.resize(size);

  for (int i = 0; i < numNonZero; i++) {
    auto insertPosition = insertionRandom(dre);
    auto value = valueRandom(dre);
    ret[insertPosition] = value;
    int32_t myRunLength = runLength;
    if (!deterministicRunLength) {
      myRunLength = runLength <= 1 ? 1 : runLengthRandom(dre);
    }

    for (int j = 1; j < myRunLength && (insertPosition + j) < size; j++) {
      ret[insertPosition + j] = value;
    }
  }

  return ret;
}

NO_TREE_VECTORIZE auto noSIMDIntCumSumWrite(
    const Pos num, const int numInputs, const int numThreads) {
  assert(numInputs <= kMaxInputs);
  int32_t* spacePtrs[kMaxInputs];
  std::vector<std::unique_ptr<int32_t[]>> spaces;
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

NO_TREE_VECTORIZE auto noSIMDIntSumMultiInput(
    const Pos num, const int numInputs, const int numThreads) {
  assert(numInputs <= kMaxInputs);
  int32_t* spacePtrs[kMaxInputs];
  std::vector<std::unique_ptr<int32_t[]>> spaces;
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

NO_TREE_VECTORIZE auto SIMDIntSumMultiInput(
    const Pos num, const int numInputs, const int numThreads) {
  assert(numInputs <= kMaxInputs);
  int32_t* spacePtrs[kMaxInputs];
  std::vector<std::unique_ptr<int32_t[]>> spaces;
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

#ifdef SKIMPY_BENCH_ENABLE_TACO
void addTacoBindings(py::module& m);
#endif

PYBIND11_MODULE(_skimpy_bench_cpp_ext, m) {
  using namespace pybind11::literals;
  m.doc() = "Benchmarks for skimpy";
  m.attr("__version__") = "0.2";

  m.def(
      "run_length_array",
      [](Pos num,
         int numNonZero,
         int runLength,
         bool determinsticRunLength,
         int randomSeed) {
        auto vec = randRunLengthArray(
            num, numNonZero, runLength, determinsticRunLength, randomSeed);
        return py::array(vec.size(), vec.data());
      },
      "num_elements"_a,
      "num_non_zero"_a,
      "run_length"_a,
      "deterministic_run_length"_a,
      "random_seed"_a = 42);

#ifdef SKIMPY_BENCH_ENABLE_TACO
  py::module taco = m.def_submodule("taco", "Taco Comparison Benchmarks");
  addTacoBindings(taco);
#endif

  py::module memory = m.def_submodule("memory", "Memory benchmarks");
  memory.def(
      "no_simd_int_sum",
      [](Pos num, int numInputs, int numThreads) {
        auto res = noSIMDIntSumMultiInput(num, numInputs, numThreads);
        return std::get<0>(res);
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
