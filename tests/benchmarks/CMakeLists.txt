cmake_minimum_required(VERSION 3.16)

function(SKIMPY_BENCHMARK TEST_NAME TEST_FILE)
  set(TEST_TARGET ${TEST_NAME}_exe)
  add_executable(${TEST_TARGET} ${TEST_FILE})
  target_link_libraries(
      ${TEST_TARGET}
      PRIVATE
        AsmJit::AsmJit
        Catch2::Catch2
        fmt::fmt
        skimpy
  )
  set_target_properties( ${TEST_TARGET} PROPERTIES OUTPUT_NAME "${TEST_NAME}")
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_TARGET})
endfunction()

# Add each benchmark target and register it for ctest.
skimpy_benchmark(asmjit_bench asmjit_bench.cpp)
skimpy_benchmark(eval_bench eval_bench.cpp)
skimpy_benchmark(merge_bench merge_bench.cpp)
skimpy_benchmark(skimpy_bench skimpy_bench.cpp)
