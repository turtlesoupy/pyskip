#include <pybind11/pybind11.h>
#include <chrono>
#include <random>
#include <atomic>
#include <iostream>
#include <omp.h>

namespace py = pybind11;

#if defined(WIN32)
using Pos = int32_t;
#else 
using Pos = size_t;
#endif

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
    outputPtr[0]
  );
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
    bigSum.load()
  );
}

PYBIND11_MODULE(_skimpy_bench_cpp_ext, m) {
  m.doc() = "Benchmarks for skimpy";
  m.attr("__version__") = "0.1";

  py::module memory = m.def_submodule("memory", "Memory benchmarks");
  memory.def("no_simd_int_sum", [](Pos num, int numInputs, int numThreads) {
    auto result = noSIMDIntSumMultiInput(num, numInputs, numThreads); 
    return std::get<0>(result);
  });
  memory.def("no_simd_int_cum_sum_write", [](Pos num, int numInputs, int numThreads) {
    auto result = noSIMDIntCumSumWrite(num, numInputs, numThreads); 
    return std::get<0>(result);
  });
}
