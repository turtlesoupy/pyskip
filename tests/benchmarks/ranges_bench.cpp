#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>
#include <cmath>

#include <skimpy/detail/utils.hpp>

TEST_CASE("Benchmark range map assigns", "[range_map_benchmark]") {
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

  BENCHMARK("merge_ranges_v1") {
    std::unique_ptr<int[]> ends_3(new int[2 * n]);
    std::unique_ptr<int[]> vals_3(new int[2 * n]);

    auto e_ptr_1 = ends_1.get();
    auto v_ptr_1 = vals_1.get();
    auto e_ptr_2 = ends_2.get();
    auto v_ptr_2 = vals_2.get();
    auto e_ptr_3 = ends_3.get();
    auto v_ptr_3 = vals_3.get();

    // Merge in remaining ranges of output.
    auto prev_value = 0;
    for (int i = 0; i < 2 * n - 1; i += 1) {
      auto end_1 = *e_ptr_1;
      auto val_1 = *v_ptr_1;

      auto end_2 = *e_ptr_2;
      auto val_2 = *v_ptr_2;

      auto end_3 = std::min(end_1, end_2);
      auto val_3 = val_1 * val_2;

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
    std::unique_ptr<int[]> ends_3(new int[2 * n]);
    std::unique_ptr<int[]> vals_3(new int[2 * n]);

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

    int* e_ptr_3 = ends_3.get();
    int* v_ptr_3 = vals_3.get();
    int prev_value = 0;
    for (int i = 0; i < 2 * n - 1; i += 1) {
      int end_3 = gen_3.get().first;
      int val_3 = gen_3.get().second;

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
}
