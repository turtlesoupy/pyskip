#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <algorithm>
#include <catch2/catch.hpp>
#include <thread>

__declspec(noinline) static void stl_merge(
    const int n1, const int* in1, const int n2, const int* in2, int* out) {
  std::merge(in1, in1 + n1, in2, in2 + n2, out);
}

__declspec(noinline) static void index_merge(
    const int n1, const int* in1, const int n2, const int* in2, int* out) {
  int i = 0;
  int i1 = 0;
  int i2 = 0;
  if (i1 < n1 && i2 < n2) {
    for (;;) {
      auto el1 = in1[i1];
      auto el2 = in2[i2];
      if (el1 < el2) {
        out[i] = el1;
        ++i;
        ++i1;
        if (i1 == n1) {
          break;
        }
      } else {
        out[i] = el2;
        ++i;
        ++i2;
        if (i2 == n2) {
          break;
        }
      }
    }
  }
  while (i1 < n1) {
    out[i++] = in1[i1++];
  }
  while (i2 < n2) {
    out[i++] = in2[i2++];
  }
}

__declspec(noinline) static void ptr_merge(
    const int n1, const int* in1, const int n2, const int* in2, int* out) {
  auto p1 = in1;
  auto p2 = in2;
  auto p3 = out;
  auto e1 = p1 + n1;
  auto e2 = p2 + n2;
  auto e3 = p3 + n1 + n2;
  if (p1 != e1 && p2 != e2) {
    for (;;) {
      if (*p1 < *p2) {
        *p3 = *p1;
        ++p3;
        ++p1;
        if (p1 == e1) {
          break;
        }
      } else {
        *p3 = *p2;
        ++p3;
        ++p2;
        if (p2 == e2) {
          break;
        }
      }
    }
  }
  while (p1 != e1) {
    *p3++ = *p1++;
  }
  while (p2 != e2) {
    *p3++ = *p2++;
  }
}

__declspec(noinline) static void parallel_merge(
    int tc,
    const int n1,
    const int* in1,
    const int n2,
    const int* in2,
    int* out) {
  auto b1 = in1;
  auto e1 = in1 + n1;
  auto b2 = in2;
  auto e2 = in2 + n2;
  auto bound = std::max(in1[n1 - 1], in2[n2 - 1]);

  std::vector<std::thread> threads;
  for (int i = 0; i < tc; i += 1) {
    int pivot = (i + 1) * bound / tc;
    auto m1 = std::upper_bound(b1, e1, pivot);
    auto m2 = std::upper_bound(b2, e2, pivot);
    auto c = (b1 - in1) + (b2 - in2);
    threads.emplace_back(
        [b1, b2, m1, m2, out = out + c] { std::merge(b1, m1, b2, m2, out); });
    b1 = m1;
    b2 = m2;
  }
  for (auto& t : threads) {
    t.join();
  }
}

__declspec(noinline) static auto pivot(
    const int n1, const int* in1, const int n2, const int* in2) {
  auto b1 = in1;
  auto e1 = in1 + n1;
  auto b2 = in2;
  auto e2 = in2 + n2;
  auto bound = std::max(in1[n1 - 1], in2[n2 - 1]);
  auto m1 = std::upper_bound(b1, e1, bound / 2);
  auto m2 = std::upper_bound(b2, e2, bound / 2);
  return std::pair(m1, m2);
}

TEST_CASE("Benchmark merge on alternating data", "[merge_alternating]") {
  constexpr auto n = 128 * 1024 * 1024;

  std::unique_ptr<int[]> in1(new int[n]);
  std::unique_ptr<int[]> in2(new int[n]);
  BENCHMARK("init_input") {
#pragma omp parallel for
    for (int i = 0; i < n; i += 1) {
      in1[i] = 2 * i + 1;
      in2[i] = 2 * i + 2;
    }
  };

  std::unique_ptr<int[]> stl_out(new int[2 * n]);
  BENCHMARK("stl_merge") {
    stl_merge(n, in1.get(), n, in2.get(), stl_out.get());
  };

  std::unique_ptr<int[]> index_out(new int[2 * n]);
  BENCHMARK("index_merge") {
    index_merge(n, in1.get(), n, in2.get(), index_out.get());
  };

  std::unique_ptr<int[]> ptr_out(new int[2 * n]);
  BENCHMARK("ptr_merge") {
    ptr_merge(n, in1.get(), n, in2.get(), ptr_out.get());
  };

  std::unique_ptr<int[]> parallel_out_tc1(new int[2 * n]);
  BENCHMARK("parallel_merge[tc=1]") {
    parallel_merge(1, n, in1.get(), n, in2.get(), parallel_out_tc1.get());
  };

  std::unique_ptr<int[]> parallel_out_tc2(new int[2 * n]);
  BENCHMARK("parallel_merge[tc=2]") {
    parallel_merge(2, n, in1.get(), n, in2.get(), parallel_out_tc2.get());
  };

  std::unique_ptr<int[]> parallel_out_tc4(new int[2 * n]);
  BENCHMARK("parallel_merge[tc=4]") {
    parallel_merge(4, n, in1.get(), n, in2.get(), parallel_out_tc4.get());
  };

  std::unique_ptr<int[]> parallel_out_tc8(new int[2 * n]);
  BENCHMARK("parallel_merge[tc=8]") {
    parallel_merge(8, n, in1.get(), n, in2.get(), parallel_out_tc8.get());
  };

  {
    int pivot_1, pivot_2;
    BENCHMARK("pivot") {
      auto pivot_out = pivot(n, in1.get(), n, in2.get());
      pivot_1 = pivot_out.first - in1.get();
      pivot_2 = pivot_out.second - in2.get();
    };
    std::cout << "\npivot = [" << pivot_1 << "," << pivot_2 << "]" << std::endl;
  }

  if (false) {
    for (int i = 0; i < 2 * n; i += 1) {
      REQUIRE(stl_out[i] == index_out[i]);
    }

    for (int i = 0; i < 2 * n; i += 1) {
      REQUIRE(stl_out[i] == ptr_out[i]);
    }

    for (int i = 0; i < 2 * n; i += 1) {
      REQUIRE(stl_out[i] == parallel_out_tc8[i]);
    }

    std::cout << "\nNumber of virtual cores: "
              << std::thread::hardware_concurrency() << std::endl;
  }
}

TEST_CASE("Benchmark merge on random data", "[merge_random]") {
  constexpr auto n = 128 * 1024 * 1024;

  std::unique_ptr<int[]> in1(new int[n]);
  std::unique_ptr<int[]> in2(new int[n]);
  BENCHMARK("init_input") {
    constexpr int T = 4;
    std::vector<std::thread> threads;
    for (int t = 0; t < T; t += 1) {
      threads.emplace_back([n, T, &in1, &in2, t] {
        auto s = t * n / T;
        auto e = (t + 1) * n / T;
        for (int i = s; i < e; i += 1) {
          in1[i] = 10 * i + (std::rand() % 10);
          in2[i] = 10 * i + (std::rand() % 10);
        }
      });
    }
    for (auto& t : threads) {
      t.join();
    }
  };

  std::unique_ptr<int[]> stl_out(new int[2 * n]);
  BENCHMARK("stl_merge") {
    stl_merge(n, in1.get(), n, in2.get(), stl_out.get());
  };

  std::unique_ptr<int[]> index_out(new int[2 * n]);
  BENCHMARK("index_merge") {
    index_merge(n, in1.get(), n, in2.get(), index_out.get());
  };

  std::unique_ptr<int[]> ptr_out(new int[2 * n]);
  BENCHMARK("ptr_merge") {
    ptr_merge(n, in1.get(), n, in2.get(), ptr_out.get());
  };

  std::unique_ptr<int[]> parallel_out_tc1(new int[2 * n]);
  BENCHMARK("parallel_merge[tc=1]") {
    parallel_merge(1, n, in1.get(), n, in2.get(), parallel_out_tc1.get());
  };

  std::unique_ptr<int[]> parallel_out_tc2(new int[2 * n]);
  BENCHMARK("parallel_merge[tc=2]") {
    parallel_merge(2, n, in1.get(), n, in2.get(), parallel_out_tc2.get());
  };

  std::unique_ptr<int[]> parallel_out_tc4(new int[2 * n]);
  BENCHMARK("parallel_merge[tc=4]") {
    parallel_merge(4, n, in1.get(), n, in2.get(), parallel_out_tc4.get());
  };

  std::unique_ptr<int[]> parallel_out_tc8(new int[2 * n]);
  BENCHMARK("parallel_merge[tc=8]") {
    parallel_merge(8, n, in1.get(), n, in2.get(), parallel_out_tc8.get());
  };

  {
    int pivot_1, pivot_2;
    BENCHMARK("pivot") {
      auto pivot_out = pivot(n, in1.get(), n, in2.get());
      pivot_1 = pivot_out.first - in1.get();
      pivot_2 = pivot_out.second - in2.get();
    };
    std::cout << "\npivot = [" << pivot_1 << "," << pivot_2 << "]" << std::endl;
  }

  if (false) {
    for (int i = 0; i < 2 * n; i += 1) {
      REQUIRE(stl_out[i] == index_out[i]);
    }

    for (int i = 0; i < 2 * n; i += 1) {
      REQUIRE(stl_out[i] == ptr_out[i]);
    }

    for (int i = 0; i < 2 * n; i += 1) {
      if (stl_out[i] != parallel_out_tc8[i]) {
        std::cout << "i=" << i << std::endl;
      }
      REQUIRE(stl_out[i] == parallel_out_tc8[i]);
    }
  }

  std::cout << "\nNumber of virtual cores: "
            << std::thread::hardware_concurrency() << std::endl;
}
