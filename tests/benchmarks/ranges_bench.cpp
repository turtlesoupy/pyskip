#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <algorithm>
#include <catch2/catch.hpp>

#include <skimpy/detail/utils.hpp>

static void exec_plan_1(
    int n,
    std::unique_ptr<int[]>& ends_1,
    std::unique_ptr<int[]>& vals_1,
    std::unique_ptr<int[]>& ends_2,
    std::unique_ptr<int[]>& vals_2,
    std::unique_ptr<int[]>& ends_3,
    std::unique_ptr<int[]>& vals_3) {
  // TODO: Figure out how to handle the source sentinels. One option is to add
  // a range to the end of every array that always loses (max_array_size + 1).

  // Initialize source op 1.
  auto s1_end_ptr = ends_1.get();
  auto s1_val_ptr = vals_1.get();
  auto s1_end = ends_1[0];
  auto s1_val = vals_1[0];

  // Initialize source op 2.
  auto s2_end_ptr = ends_2.get();
  auto s2_val_ptr = vals_2.get();
  auto s2_end = ends_2[0];
  auto s2_val = vals_2[0];

  // Initialize binary op 1.
  auto b1_val = 0;
  auto b1_end = 0;

  // Initialize binary op 1.
  auto out_val = 0;
  auto out_end = 0;
  auto out_end_ptr = ends_3.get();
  auto out_val_ptr = vals_3.get();

  for (int i = 0; i < 2 * n - 1; i += 1) {
    // Compute the output value and range.
    b1_val = s1_val * s2_val;
    if (s1_end <= s2_end) {
      b1_end = s1_end;
      s1_end = *s1_end_ptr++;
      s1_val = *s1_val_ptr++;
    } else {
      b1_end = s2_end;
      s2_end = *s2_end_ptr++;
      s2_val = *s2_val_ptr++;
    }

    // Emit the result with compression.
    if (i != 0 && b1_end == out_end) {
      continue;
    } else if (i != 0 && b1_val == out_val) {
      out_end = b1_end;
      *(out_end_ptr - 1) = out_end;
    } else {
      out_val = b1_val;
      out_end = b1_end;
      *out_end_ptr++ = out_end;
      *out_val_ptr++ = out_val;
    }
  }
}

static void exec_plan_2(
    int n,
    std::unique_ptr<int[]>& ends_1,
    std::unique_ptr<int[]>& vals_1,
    std::unique_ptr<int[]>& ends_2,
    std::unique_ptr<int[]>& vals_2,
    std::unique_ptr<int[]>& ends_3,
    std::unique_ptr<int[]>& vals_3) {
  // TODO: Figure out how to handle the source sentinels. One option is to add
  // a range to the end of every array that always loses (max_array_size + 1).

  // Initialize source op 1.
  auto s1_end_ptr = ends_1.get();
  auto s1_val_ptr = vals_1.get();
  auto s1_end = ends_1[0];
  auto s1_val = vals_1[0];

  // Initialize source op 2.
  auto s2_end_ptr = ends_2.get();
  auto s2_val_ptr = vals_2.get();
  auto s2_end = ends_2[0];
  auto s2_val = vals_2[0];

  // Initialize binary op.
  auto b1_val = 0;
  auto b1_end = 0;

  // Initialize output op.
  auto out_val = 0;
  auto out_end = 0;
  auto out_end_ptr = ends_3.get();
  auto out_val_ptr = vals_3.get();

  // Special-case first past
  out_end = std::min(s1_end, s2_end);
  out_val = s1_val * s2_val;
  if (s1_end == out_end) {
    s1_end = *s1_end_ptr++;
    s1_val = *s1_val_ptr++;
  }
  if (s2_end == out_end) {
    s2_end = *s2_end_ptr++;
    s2_val = *s2_val_ptr++;
  }
  *out_end_ptr++ = out_end;
  *out_val_ptr++ = out_val;

  for (int i = 0; i < 2 * n - 2; i += 1) {
    // Evaluate the binary operation.
    if (s1_end <= s2_end) {
      b1_end = s1_end;
      s1_end = *s1_end_ptr++;
      s1_val = *s1_val_ptr++;
    } else {
      b1_end = s2_end;
      s2_end = *s2_end_ptr++;
      s2_val = *s2_val_ptr++;
    }
    b1_val = s1_val * s2_val;

    // Emit the result with compression.
    if (b1_end == out_end) {
      continue;
    } else if (b1_val == out_val) {
      out_end = b1_end;
      *(out_end_ptr - 1) = out_end;
    } else {
      out_end = b1_end;
      out_val = b1_val;
      *out_end_ptr++ = out_end;
      *out_val_ptr++ = out_val;
    }
  }
}

__declspec(noalias) static void exec_plan_3(
    int n,
    int* s1_end_ptr,
    int* s1_val_ptr,
    int* s2_end_ptr,
    int* s2_val_ptr,
    int* out_end_ptr,
    int* out_val_ptr) {
  // Special-case first range.
  auto out_end = std::min(*s1_end_ptr, *s2_end_ptr);
  auto out_val = *s1_val_ptr * *s2_val_ptr;
  if (*s1_end_ptr == out_end) {
    ++s1_end_ptr;
    ++s1_val_ptr;
  }
  if (*s2_end_ptr == out_end) {
    ++s2_end_ptr;
    ++s2_val_ptr;
  }
  *out_end_ptr++ = out_end;
  *out_val_ptr++ = out_val;

  // Merge in remaining ranges.
  for (int i = 0; i < 2 * n - 2; i += 1) {
    int s1_end = *s1_end_ptr;
    int s2_end = *s2_end_ptr;
    int b1_val = *s1_val_ptr * *s2_val_ptr;
    if (s1_end <= s2_end) {
      ++s1_end_ptr;
      ++s1_val_ptr;
      if (s1_end == out_end) {
        continue;
      } else if (b1_val != out_val) {
        out_end = s1_end;
        out_val = b1_val;
        *out_end_ptr++ = out_end;
        *out_val_ptr++ = out_val;
      } else {
        out_end = s1_end;
        *(out_end_ptr - 1) = out_end;
      }
    } else {
      ++s2_end_ptr;
      ++s2_val_ptr;
      if (s2_end == out_end) {
        continue;
      } else if (b1_val != out_val) {
        out_end = s2_end;
        out_val = b1_val;
        *out_end_ptr++ = out_end;
        *out_val_ptr++ = out_val;
      } else {
        out_end = s2_end;
        *(out_end_ptr - 1) = out_end;
      }
    }
  }
}

static void just_merge_1(
    int n,
    std::unique_ptr<int[]>& ends_1,
    std::unique_ptr<int[]>& vals_1,
    std::unique_ptr<int[]>& ends_2,
    std::unique_ptr<int[]>& vals_2,
    std::unique_ptr<int[]>& ends_3,
    std::unique_ptr<int[]>& vals_3) {
  auto ends_ptr_1 = ends_1.get();
  auto ends_ptr_2 = ends_2.get();
  auto ends_ptr_3 = ends_3.get();
  for (int i = 0; i < 2 * n - 1; i += 1) {
    if (*ends_ptr_1 <= *ends_ptr_2) {
      *ends_ptr_3++ = *ends_ptr_1++;
    } else {
      *ends_ptr_3++ = *ends_ptr_2++;
    }
  }
  /*
  std::merge(
      ends_1.get(),
      ends_1.get() + n,
      ends_2.get(),
      ends_2.get() + n,
      ends_3.get());
      */
}

TEST_CASE("Benchmark range operations", "[range_ops_benchmark]") {
  constexpr auto n = 128 * 1024 * 1024;

  std::unique_ptr<int[]> ends_1(new int[n]);
  std::unique_ptr<int[]> vals_1(new int[n]);
  BENCHMARK("init_ranges_1") {
    ends_1[0] = 1;
    vals_1[0] = 1;
    for (int i = 1; i < n - 1; i += 1) {
      // ends_1[i] = ends_1[i - 1] + 1 + (std::rand() % 3);
      ends_1[i] = ends_1[i - 1] + 2;
      vals_1[i] = 1;
    }
    ends_1[n - 1] = 4 * n + 4;
    vals_1[n - 1] = 1;
  };

  std::unique_ptr<int[]> ends_2(new int[n]);
  std::unique_ptr<int[]> vals_2(new int[n]);
  BENCHMARK("init_ranges_2") {
    ends_2[0] = 2;
    vals_2[0] = 0;
    for (int i = 1; i < n - 1; i += 1) {
      // ends_2[i] = ends_2[i - 1] + 1 + (std::rand() % 3);
      ends_2[i] = ends_2[i - 1] + 2;
      vals_2[i] = i;
    }
    ends_2[n - 1] = 4 * n + 4;
    vals_2[n - 1] = n;
  };

  std::unique_ptr<int[]> ends_3;
  std::unique_ptr<int[]> vals_3;
  BENCHMARK("merge_ranges_1") {
    ends_3.reset(new int[2 * n - 1]);
    vals_3.reset(new int[2 * n - 1]);
    exec_plan_1(n, ends_1, vals_1, ends_2, vals_2, ends_3, vals_3);
  };

  BENCHMARK("merge_ranges_2") {
    ends_3.reset(new int[2 * n - 1]);
    vals_3.reset(new int[2 * n - 1]);
    exec_plan_2(n, ends_1, vals_1, ends_2, vals_2, ends_3, vals_3);
  };

  BENCHMARK("merge_ranges_3") {
    ends_3.reset(new int[2 * n - 1]);
    vals_3.reset(new int[2 * n - 1]);
    exec_plan_3(
        n,
        ends_1.get(),
        vals_1.get(),
        ends_2.get(),
        vals_2.get(),
        ends_3.get(),
        vals_3.get());
  };

  BENCHMARK("just_merge_1") {
    ends_3.reset(new int[2 * n - 1]);
    vals_3.reset(new int[2 * n - 1]);
    just_merge_1(n, ends_1, vals_1, ends_2, vals_2, ends_3, vals_3);
  };

  std::cout << "\nR: ";
  for (int i = 0; i < 10; i += 1) {
    std::cout << ends_3[i] << "," << vals_3[i] << "; ";
  }
  std::cout << std::endl;
}
