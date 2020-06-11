cmake_minimum_required(VERSION 3.16)

function(PYSKIP_BENCHMARK TEST_NAME TEST_FILE)
  set(TEST_TARGET ${TEST_NAME}_exe)
  add_executable(${TEST_TARGET} ${TEST_FILE})
  target_link_libraries(
      ${TEST_TARGET}
      PRIVATE
        AsmJit::AsmJit
        Catch2::Catch2
        fmt::fmt
        pyskip
  )
  set_target_properties( ${TEST_TARGET} PROPERTIES OUTPUT_NAME "${TEST_NAME}")
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_TARGET})
endfunction()

# Add each benchmark target and register it for ctest.
pyskip_benchmark(array_bench array_bench.cpp)
pyskip_benchmark(asmjit_bench asmjit_bench.cpp)
pyskip_benchmark(eval_bench eval_bench.cpp)
pyskip_benchmark(merge_bench merge_bench.cpp)
pyskip_benchmark(step_bench step_bench.cpp)
pyskip_benchmark(tensor_bench tensor_bench.cpp)
