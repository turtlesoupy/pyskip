#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <fmt/format.h>

#include <algorithm>
#include <catch2/catch.hpp>
#include <thread>

template <typename Fn>
void parallel(int t, int n, Fn&& fn) {
  std::vector<std::thread> threads;
  for (int i = 0; i < t; i += 1) {
    threads.emplace_back([i, t, n, fn = std::forward<Fn>(fn)] {
      fn(i * n / t, (i + 1) * n / t);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

static int f_mul(int x, int y) {
  return x * y;
}
static int f_add(int x, int y) {
  return x + y;
}
static int f_sub(int x, int y) {
  return x - y;
}
static int f_max(int x, int y) {
  return std::max(x, y);
}

TEST_CASE("Benchmark optree evaluation", "[optree_eval]") {
  constexpr auto n = 128 * 1024 * 1024;

  std::unique_ptr<int[]> in_1(new int[n]);
  std::unique_ptr<int[]> in_2(new int[n]);
  BENCHMARK("init_input") {
#pragma omp parallel for
    for (int i = 0; i < n; i += 1) {
      in_1[i] = std::rand() % 100;
      in_2[i] = std::rand() % 100;
    }
  };

  std::unique_ptr<int[]> out;

  // Begin with a simple test of multiplication.
  BENCHMARK("pointers_1") {
    out.reset(new int[n]);
    parallel(1, n, [&](int start, int end, ...) {
      auto b_1 = in_1.get() + start;
      auto b_2 = in_2.get() + start;
      auto b_3 = out.get() + start;
      auto e_3 = out.get() + end;
      while (b_3 != e_3) {
        *b_3 = *b_1 * *b_2;
        ++b_1;
        ++b_2;
        ++b_3;
      }
    });
  };

  BENCHMARK("pointers_4") {
    out.reset(new int[n]);
    parallel(4, n, [&](int start, int end, ...) {
      auto b_1 = in_1.get() + start;
      auto b_2 = in_2.get() + start;
      auto b_3 = out.get() + start;
      auto e_3 = out.get() + end;
      while (b_3 != e_3) {
        *b_3 = *b_1 * *b_2;
        ++b_1;
        ++b_2;
        ++b_3;
      }
    });
  };

  BENCHMARK("pointers_8") {
    out.reset(new int[n]);
    parallel(8, n, [&](int start, int end, ...) {
      auto b_1 = in_1.get() + start;
      auto b_2 = in_2.get() + start;
      auto b_3 = out.get() + start;
      auto e_3 = out.get() + end;
      while (b_3 != e_3) {
        *b_3 = *b_1 * *b_2;
        ++b_1;
        ++b_2;
        ++b_3;
      }
    });
  };

  BENCHMARK("multi_pointers_8") {
    out.reset(new int[n]);
    parallel(8, n, [&](int start, int end, ...) {
      auto b_1 = in_1.get() + start;
      auto b_2 = in_2.get() + start;
      auto b_3 = out.get() + start;
      auto e_3 = out.get() + end;
      while (b_3 != e_3) {
        auto x = *b_1;
        auto y = *b_1;
        auto z = f_max(f_sub(f_mul(x, y), f_add(x, y)), f_sub(x, y));
        *b_3 = z;
        ++b_1;
        ++b_2;
        ++b_3;
      }
    });
  };

  BENCHMARK("index_1") {
    out.reset(new int[n]);
    parallel(1, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        out[i] = in_1[i] * in_2[i];
      }
    });
  };

  BENCHMARK("index_4") {
    out.reset(new int[n]);
    parallel(4, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        out[i] = in_1[i] * in_2[i];
      }
    });
  };

  BENCHMARK("index_8") {
    out.reset(new int[n]);
    parallel(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        out[i] = in_1[i] * in_2[i];
      }
    });
  };

  auto mul = [](int x, int y, ...) { return f_mul(x, y); };
  auto add = [](int x, int y, ...) { return f_add(x, y); };
  auto sub = [](int x, int y, ...) { return f_sub(x, y); };
  auto max = [](int x, int y, ...) { return f_max(x, y); };

  // Try wrapping the multiplication inside a function call.
  BENCHMARK("wrapped_1[threads=1]") {
    out.reset(new int[n]);
    parallel(1, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        out[i] = mul(in_1[i], in_2[i]);
      }
    });
  };

  BENCHMARK("wrapped_1[threads=4]") {
    out.reset(new int[n]);
    parallel(4, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        out[i] = mul(in_1[i], in_2[i]);
      }
    });
  };

  BENCHMARK("wrapped_1[threads=8]") {
    out.reset(new int[n]);
    parallel(8, n, [&](int start, int end, ...) {
      for (int i = start; i < end; i += 1) {
        out[i] = mul(in_1[i], in_2[i]);
      }
    });
  };

  for (int threads : std::vector<int>{1, 2, 4, 8}) {
    BENCHMARK(fmt::format("multi_wrapped_5[threads={}]", threads)) {
      out.reset(new int[n]);
      parallel(threads, n, [&](int start, int end, ...) {
        for (int i = start; i < end; i += 1) {
          out[i] =
              max(sub(mul(in_1[i], in_2[i]), add(in_1[i], in_2[i])),
                  sub(in_1[i], in_2[i]));
        }
      });
    };
  }

  // Try with switch statements.
  for (int threads : std::vector<int>{1, 2, 4, 8}) {
    BENCHMARK(fmt::format("instruction_5[threads={}]", threads)) {
      out.reset(new int[n]);

      auto populate_instructions = [](auto instructions, ...) {
        instructions[0] = f_mul;
        instructions[1] = f_add;
        instructions[2] = f_sub;
        instructions[3] = f_sub;
        instructions[4] = f_max;
      };

      parallel(threads, n, [&](int start, int end, ...) {
        int (*instructions[5])(int, int);
        populate_instructions(instructions);

        int args[5];
        args[0] = 0;
        args[1] = 0;
        args[2] = 2;
        args[3] = 0;
        args[4] = 4;
        int outs[5];
        outs[0] = 2;
        outs[1] = 3;
        outs[2] = 4;
        outs[3] = 5;
        outs[4] = 6;

        int stack[7];
        for (int i = start; i < end; i += 1) {
          stack[0] = in_1[i];
          stack[1] = in_2[i];
          {
            auto instruction = instructions[0];
            auto arg_1 = args[0];
            auto arg_2 = args[0] + 1;
            auto result = outs[0];
            stack[result] = instruction(stack[arg_1], stack[arg_2]);
          }
          {
            auto instruction = instructions[1];
            auto arg_1 = args[1];
            auto arg_2 = args[1] + 1;
            auto result = outs[1];
            stack[result] = instruction(stack[arg_1], stack[arg_2]);
          }
          {
            auto instruction = instructions[2];
            auto arg_1 = args[2];
            auto arg_2 = args[2] + 1;
            auto result = outs[2];
            stack[result] = instruction(stack[arg_1], stack[arg_2]);
          }
          {
            auto instruction = instructions[3];
            auto arg_1 = args[3];
            auto arg_2 = args[3] + 1;
            auto result = outs[3];
            stack[result] = instruction(stack[arg_1], stack[arg_2]);
          }
          {
            auto instruction = instructions[4];
            auto arg_1 = args[4];
            auto arg_2 = args[4] + 1;
            auto result = outs[4];
            stack[result] = instruction(stack[arg_1], stack[arg_2]);
          }
          out[i] = stack[6];
        }
      });
    };
  }

  for (int threads : std::vector<int>{1, 2, 4, 8}) {
    BENCHMARK(fmt::format("instruction_1[threads={}]", threads)) {
      out.reset(new int[n]);

      auto populate_instructions = [](auto instructions, ...) {
        instructions[0] = f_mul;
      };

      parallel(threads, n, [&](int start, int end, ...) {
        int (*instructions[1])(int, int);
        populate_instructions(instructions);

        int args[1];
        args[0] = 0;
        int outs[1];
        outs[0] = 2;

        int stack[3];

        auto in_1_ptr = &in_1[start];
        auto in_2_ptr = &in_2[start];
        auto out_ptr = &out[start];
        for (int i = start; i < end; i += 1) {
          stack[0] = *in_1_ptr;
          stack[1] = *in_2_ptr;
          ++in_1_ptr;
          ++in_2_ptr;

          {
            auto instruction = instructions[0];
            auto arg_1 = args[0];
            auto arg_2 = args[0] + 1;
            auto result = outs[0];
            stack[result] = instruction(stack[arg_1], stack[arg_2]);
          }

          *out_ptr = stack[6];
          ++out_ptr;
        }
      });
    };
  }
}
