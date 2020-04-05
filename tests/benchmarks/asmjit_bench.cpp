#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <asmjit/asmjit.h>

#include <catch2/catch.hpp>
#include <iostream>

TEST_CASE("Benchmark asmjit", "[asmjit_bench]") {
  using namespace asmjit;

  JitRuntime rt;

  CodeHolder code;
  code.init(rt.codeInfo());

  int result = 0;
  BENCHMARK("assemble and call") {
    // Create code that sets the eax register to 1 and returns.
    x86::Assembler a(&code);
    a.mov(x86::eax, 1);
    a.ret();

    // Link the function.
    int (*fn)(void);
    Error err = rt.add(&fn, &code);
    REQUIRE(!err);

    // Call the function.
    int result = fn();

    // Release the function from the runtime (not necessary).
    rt.release(fn);
  };

  // Print the result.
  std::cout << result << std::endl;
}
