#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <asmjit/asmjit.h>
#include <fmt/format.h>

#include <algorithm>
#include <catch2/catch.hpp>
#include <thread>

template <typename Fn>
void parallel(int t, int n, Fn&& fn) {
  std::vector<std::thread> threads;
  for (int i = 0; i < t; i += 1) {
    threads.emplace_back([i, t, n, fn = std::forward<Fn>(fn)] {
      fn(i * n / t, (i + 1) * n / t, i);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

inline int f_mul(int x, int y) {
  return x * y;
}
inline int f_div(int x, int y) {
  return x / y;
}
inline int f_add(int x, int y) {
  return x + y;
}
inline int f_sub(int x, int y) {
  return x - y;
}
inline int f_max(int x, int y) {
  return std::max(x, y);
}

TEST_CASE("Benchmark optree evaluation", "[optree_eval]") {
  constexpr auto n = 128 * 1024 * 1024;

  std::unique_ptr<int[]> in_1(new int[n]);
  std::unique_ptr<int[]> in_2(new int[n]);
  BENCHMARK("init_input") {
#pragma omp parallel for
    for (int i = 0; i < n; i += 1) {
      in_1[i] = 1 + (std::rand() % 100);
      in_2[i] = 1 + (std::rand() % 100);
    }
  };

  std::unique_ptr<int[]> out;

  for (int threads : std::vector<int>{1, 2, 4, 8}) {
    BENCHMARK(fmt::format("pointers[threads={}]", threads)) {
      out.reset(new int[n]);
      parallel(threads, n, [&](int start, int end, ...) {
        auto b_1 = in_1.get() + start;
        auto b_2 = in_2.get() + start;
        auto b_3 = out.get() + start;
        auto e_3 = out.get() + end;
        while (b_3 != e_3) {
          *b_3++ = *b_1++ * *b_2++;
        }
      });
    };
  }

  for (int threads : std::vector<int>{1, 2, 4, 8}) {
    BENCHMARK(fmt::format("index[threads={}]", threads)) {
      out.reset(new int[n]);
      parallel(threads, n, [&](int start, int end, ...) {
        for (int i = start; i < end; i += 1) {
          out[i] = in_1[i] * in_2[i];
        }
      });
    };
  }

  auto mul = [](int x, int y, ...) { return f_mul(x, y); };
  auto add = [](int x, int y, ...) { return f_add(x, y); };
  auto div = [](int x, int y, ...) { return f_div(x, y); };
  auto sub = [](int x, int y, ...) { return f_sub(x, y); };
  auto max = [](int x, int y, ...) { return f_max(x, y); };

  for (int threads : std::vector<int>{1, 2, 4, 8}) {
    BENCHMARK(fmt::format("wrapped_mul[threads={}]", threads)) {
      out.reset(new int[n]);
      parallel(threads, n, [&](int start, int end, ...) {
        auto b_1 = in_1.get() + start;
        auto b_2 = in_2.get() + start;
        auto b_3 = out.get() + start;
        auto e_3 = out.get() + end;
        while (b_3 != e_3) {
          *b_3++ = mul(*b_1++, *b_2++);
        }
      });
    };
  }

  for (int threads : std::vector<int>{1, 2, 4, 8}) {
    BENCHMARK(fmt::format("wrapped_instructions_5[threads={}]", threads)) {
      out.reset(new int[n]);
      parallel(threads, n, [&](int start, int end, ...) {
        auto b_1 = in_1.get() + start;
        auto b_2 = in_2.get() + start;
        auto b_3 = out.get() + start;
        auto e_3 = out.get() + end;
        while (b_3 != e_3) {
          int x = *b_1++;
          int y = *b_2++;
          *b_3++ = max(sub(mul(x, y), div(x, y)), add(x, y));
        }
      });
    };
  }

  for (int threads : std::vector<int>{1, 2, 4, 8}) {
    BENCHMARK(fmt::format("stack_mul[threads={}]", threads)) {
      out.reset(new int[n]);

      auto populate_instructions = [](auto instructions, ...) {
        instructions[0] = f_mul;
      };

      parallel(threads, n, [&](int start, int end, ...) {
        int (*instructions[1])(int, int);
        populate_instructions(instructions);

        int stack[3];
        int args[1] = {0};
        int outs[1] = {2};

        auto b_1 = &in_1[start];
        auto b_2 = &in_2[start];
        auto b_3 = &out[start];
        auto e_3 = &out[end];
        while (b_3 != e_3) {
          stack[0] = *b_1++;
          stack[1] = *b_2++;

          {
            auto instruction = instructions[0];
            auto arg_1 = args[0];
            auto arg_2 = args[0] + 1;
            auto result = outs[0];
            stack[result] = instruction(stack[arg_1], stack[arg_2]);
          }

          *b_3++ = stack[6];
        }
      });
    };
  }

  for (int threads : std::vector<int>{1, 2, 4, 8}) {
    BENCHMARK(fmt::format("stack_instruction_5[threads={}]", threads)) {
      out.reset(new int[n]);

      auto populate_instructions = [](auto instructions, ...) {
        instructions[0] = f_mul;
        instructions[1] = f_div;
        instructions[2] = f_sub;
        instructions[3] = f_add;
        instructions[4] = f_max;
      };

      parallel(threads, n, [&](int start, int end, ...) {
        int (*instructions[5])(int, int);
        populate_instructions(instructions);

        int stack[7];
        int args[5] = {0, 0, 2, 0, 4};
        int outs[5] = {2, 3, 4, 5, 6};

        auto b_1 = &in_1[start];
        auto b_2 = &in_2[start];
        auto b_3 = &out[start];
        auto e_3 = &out[end];
        while (b_3 != e_3) {
          stack[0] = *b_1++;
          stack[1] = *b_2++;

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

          *b_3++ = stack[6];
        }
      });
    };
  }

  for (int threads : std::vector<int>{1, 2, 4, 8}) {
    BENCHMARK(fmt::format("static_instruction_5[threads={}]", threads)) {
      out.reset(new int[n]);
      parallel(threads, n, [&](int start, int end, ...) {
        auto b_1 = &in_1[start];
        auto b_2 = &in_2[start];
        auto b_3 = &out[start];
        auto e_3 = &out[end];
        while (b_3 != e_3) {
          int x = *b_1++;
          int y = *b_2++;
          int c = f_mul(x, y);
          int d = f_div(x, y);
          int e = f_sub(c, d);
          int f = f_div(y, x);
          int g = f_max(e, f);
          *b_3++ = g;
        }
      });
    };
  }
}
