#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>

#include <skimpy/detail/utils.hpp>

TEST_CASE("Benchmark range operations", "[range_ops_benchmark]") {
  constexpr auto n = 128 * 1024 * 1024;

  std::unique_ptr<int[]> ends_1(new int[n]);
  std::unique_ptr<int[]> vals_1(new int[n]);
  BENCHMARK("init_ranges_1") {
    for (int i = 0; i < n - 1; i += 1) {
      ends_1[i] = 2 * i + 1;
      vals_1[i] = 1;
    }
    ends_1[n - 1] = 2 * n + 2;
    vals_1[n - 1] = 1;
  };

  std::unique_ptr<int[]> ends_2(new int[n]);
  std::unique_ptr<int[]> vals_2(new int[n]);
  BENCHMARK("init_ranges_2") {
    for (int i = 0; i < n - 1; i += 1) {
      ends_2[i] = 2 * i + 2;
      vals_2[i] = i;
    }
    ends_2[n - 1] = 2 * n + 2;
    vals_2[n - 1] = n;
  };

  std::unique_ptr<int[]> ends_3;
  std::unique_ptr<int[]> vals_3;
  BENCHMARK("merge_ranges_v1") {
    ends_3.reset(new int[2 * n]);
    vals_3.reset(new int[2 * n]);

    int* e_ptr_1 = ends_1.get();
    int* v_ptr_1 = vals_1.get();
    int* e_ptr_2 = ends_2.get();
    int* v_ptr_2 = vals_2.get();
    int* e_ptr_3 = ends_3.get();
    int* v_ptr_3 = vals_3.get();

    // Merge in remaining ranges of output.
    int prev_value = 0;
    for (int i = 0; i < 2 * n - 1; i += 1) {
      int end_1 = *e_ptr_1;
      int val_1 = *v_ptr_1;

      int end_2 = *e_ptr_2;
      int val_2 = *v_ptr_2;

      int end_3 = std::min(end_1, end_2);
      int val_3 = val_1 * val_2;

      if (i == 0 || val_3 != prev_value) {
        *e_ptr_3++ = end_3;
        *v_ptr_3++ = val_3;
        prev_value = val_3;
      } else {
        *(e_ptr_3 - 1) = end_3;
      }

      if (end_1 == end_3) {
        ++e_ptr_1;
        ++v_ptr_1;
      } else {
        ++e_ptr_2;
        ++v_ptr_2;
      }
    }
  };

  BENCHMARK("merge_ranges_v2") {
    ends_3.reset(new int[2 * n]);
    vals_3.reset(new int[2 * n]);

    auto gen_1 = skimpy::detail::make_inf_generator<std::pair<int, int>>(
        [e_ptr = ends_1.get(), v_ptr = vals_1.get()]() mutable {
          return std::pair(*e_ptr++, *v_ptr++);
        });

    auto gen_2 = skimpy::detail::make_inf_generator<std::pair<int, int>>(
        [e_ptr = ends_2.get(), v_ptr = vals_2.get()]() mutable {
          return std::pair(*e_ptr++, *v_ptr++);
        });

    auto gen_3 = skimpy::detail::make_inf_generator<std::pair<int, int>>(
        [gen_1 = std::move(gen_1), gen_2 = std::move(gen_2)]() mutable {
          auto end_1 = gen_1.get().first;
          auto val_1 = gen_1.get().second;

          auto end_2 = gen_2.get().first;
          auto val_2 = gen_2.get().second;

          auto end_3 = std::min(end_1, end_2);
          auto val_3 = val_1 * val_2;

          if (end_1 == end_3) {
            gen_1.next();
          } else {
            gen_2.next();
          }

          return std::pair(end_3, val_3);
        });

    auto prev_value = 0;
    auto e_ptr_3 = ends_3.get();
    auto v_ptr_3 = vals_3.get();
    for (int i = 0; i < 2 * n - 1; i += 1) {
      auto end_3 = gen_3.get().first;
      auto val_3 = gen_3.get().second;

      if (i == 0 || val_3 != prev_value) {
        *e_ptr_3++ = end_3;
        *v_ptr_3++ = val_3;
        prev_value = val_3;
      } else {
        *(e_ptr_3 - 1) = end_3;
      }

      gen_3.next();
    }
  };

  BENCHMARK("array_product_v1") {
    std::unique_ptr<int[]> vals_3(new int[n]);
    auto v_ptr_1 = vals_1.get();
    auto v_ptr_2 = vals_2.get();
    auto v_ptr_3 = vals_3.get();
    for (int i = 0; i < n; i += 1) {
      *v_ptr_3++ = (*v_ptr_1++) * (*v_ptr_2++);
    }
  };

  BENCHMARK("array_product_v2") {
    std::unique_ptr<int[]> vals_3(new int[n]);
    for (int i = 0; i < n; i += 1) {
      vals_3[i] = vals_1[i] * vals_2[i];
    }
  };

  auto print_range = [](auto& ends, auto& vals) {
    std::cout << "range: ";
    for (int i = 0; i < 10; i += 1) {
      std::cout << ends[i] << "," << vals[i] << "; ";
    }
    std::cout << std::endl;
  };

  std::cout << std::endl;
  print_range(ends_1, vals_1);
  print_range(ends_2, vals_2);
  print_range(ends_3, vals_3);
}

TEST_CASE("Benchmark range operations using STL", "[range_ops_stl_benchmark]") {
  constexpr auto n = 128 * 1024 * 1024;

  std::vector<std::pair<int, int>> ranges_1;
  BENCHMARK("init_ranges_1") {
    ranges_1.reserve(n);
    for (int i = 0; i < n - 1; i += 1) {
      ranges_1.emplace_back(2 * i + 1, 1);
    }
    ranges_1.emplace_back(2 * n + 2, 1);
  };

  std::vector<std::pair<int, int>> ranges_2;
  BENCHMARK("init_ranges_2") {
    ranges_2.reserve(n);
    for (int i = 0; i < n - 1; i += 1) {
      ranges_2.emplace_back(2 * i + 2, i);
    }
    ranges_2.emplace_back(2 * n + 2, n - 1);
  };

  std::vector<std::pair<int, int>> ranges_3;
  BENCHMARK("merge_ranges_v1") {
    auto gen_3 = skimpy::detail::make_inf_generator<std::pair<int, int>>(
        [iter_1 = ranges_1.begin(), iter_2 = ranges_2.begin()]() mutable {
          auto end_1 = iter_1->first;
          auto val_1 = iter_1->second;

          auto end_2 = iter_2->first;
          auto val_2 = iter_2->second;

          auto end_3 = std::min(end_1, end_2);
          auto val_3 = val_1 * val_2;

          if (end_1 == end_3) {
            ++iter_1;
          } else {
            ++iter_2;
          }

          return std::pair(end_3, val_3);
        });

    ranges_3.reserve(2 * n - 1);

    int prev_value = 0;
    for (int i = 0; i < 2 * n - 1; i += 1) {
      int end_3 = gen_3.get().first;
      int val_3 = gen_3.get().second;

      if (i == 0 || val_3 != prev_value) {
        ranges_3.emplace_back(end_3, val_3);
        prev_value = val_3;
      } else {
        ranges_3.back().first = end_3;
      }

      gen_3.next();
    }
  };

  auto print_range = [](auto& range) {
    std::cout << "range: ";
    for (int i = 0; i < 10; i += 1) {
      std::cout << range[i].first << "," << range[i].second << "; ";
    }
    std::cout << std::endl;
  };

  std::cout << std::endl;
  print_range(ranges_1);
  print_range(ranges_2);
  print_range(ranges_3);
}

TEST_CASE("Benchmark range ops on pairs", "[range_ops_pairs_benchmark]") {
  constexpr auto n = 128 * 1024 * 1024;

  struct Range {};

  std::unique_ptr<std::pair<int, int>[]> ranges_1;
  BENCHMARK("init_ranges_1") {
    ranges_1.reset(new std::pair<int, int>[n]);
    for (int i = 0; i < n - 1; i += 1) {
      ranges_1[i] = std::pair(2 * i + 1, 1);
    }
    ranges_1[n - 1] = std::pair(2 * n + 1, 1);
  };

  std::unique_ptr<std::pair<int, int>[]> ranges_2;
  BENCHMARK("init_ranges_2") {
    ranges_2.reset(new std::pair<int, int>[n]);
    for (int i = 0; i < n - 1; i += 1) {
      ranges_2[i] = std::pair(2 * i + 2, i);
    }
    ranges_2[n - 1] = std::pair(2 * n + 2, n - 1);
  };

  std::unique_ptr<std::pair<int, int>[]> ranges_3;
  BENCHMARK("merge_ranges_v1") {
    ranges_3.reset(new std::pair<int, int>[2 * n - 1]);

    auto gen_3 = skimpy::detail::make_inf_generator<std::pair<int, int>>(
        [iter_1 = ranges_1.get(), iter_2 = ranges_2.get()]() mutable {
          auto end_1 = iter_1->first;
          auto val_1 = iter_1->second;

          auto end_2 = iter_2->first;
          auto val_2 = iter_2->second;

          auto end_3 = std::min(end_1, end_2);
          auto val_3 = val_1 * val_2;

          if (end_1 == end_3) {
            ++iter_1;
          } else {
            ++iter_2;
          }

          return std::pair(end_3, val_3);
        });

    auto prev_value = 0;
    auto r_ptr_3 = ranges_3.get();
    for (int i = 0; i < 2 * n - 1; i += 1) {
      auto end_3 = gen_3.get().first;
      auto val_3 = gen_3.get().second;

      if (i == 0 || val_3 != prev_value) {
        *r_ptr_3++ = std::pair(end_3, val_3);
        prev_value = val_3;
      } else {
        (r_ptr_3 - 1)->first = end_3;
      }

      gen_3.next();
    }
  };

  BENCHMARK("merge_ranges_v2") {
    ranges_3.reset(new std::pair<int, int>[2 * n - 1]);

    auto gen_1 = skimpy::detail::make_inf_generator<std::pair<int, int>>(
        [r_ptr_1 = ranges_1.get()]() mutable { return *r_ptr_1++; });

    auto gen_2 = skimpy::detail::make_inf_generator<std::pair<int, int>>(
        [r_ptr_2 = ranges_2.get()]() mutable { return *r_ptr_2++; });

    auto gen_3 = skimpy::detail::make_inf_generator<std::pair<int, int>>(
        [gen_1 = std::move(gen_1), gen_2 = std::move(gen_2)]() mutable {
          auto end_1 = gen_1.get().first;
          auto val_1 = gen_1.get().second;

          auto end_2 = gen_2.get().first;
          auto val_2 = gen_2.get().second;

          auto end_3 = std::min(end_1, end_2);
          auto val_3 = val_1 * val_2;

          if (end_1 == end_3) {
            gen_1.next();
          } else {
            gen_2.next();
          }

          return std::pair(end_3, val_3);
        });

    auto prev_value = 0;
    auto r_ptr_3 = ranges_3.get();
    for (int i = 0; i < 2 * n - 1; i += 1) {
      auto end_3 = gen_3.get().first;
      auto val_3 = gen_3.get().second;

      if (i == 0 || val_3 != prev_value) {
        *r_ptr_3++ = std::pair(end_3, val_3);
        prev_value = val_3;
      } else {
        (r_ptr_3 - 1)->first = end_3;
      }

      gen_3.next();
    }
  };

  auto print_range = [](auto& range) {
    std::cout << "range: ";
    for (int i = 0; i < 10; i += 1) {
      std::cout << range[i].first << "," << range[i].second << "; ";
    }
    std::cout << std::endl;
  };

  std::cout << std::endl;
  print_range(ranges_1);
  print_range(ranges_2);
  print_range(ranges_3);
}
