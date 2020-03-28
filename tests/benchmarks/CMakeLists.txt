cmake_minimum_required(VERSION 3.16)

function(SKIMPY_BENCHMARK TEST_NAME TEST_FILE)
  set(TEST_TARGET ${TEST_NAME}_exe)
  add_executable(${TEST_TARGET} ${TEST_FILE})
  target_link_libraries(${TEST_TARGET} PRIVATE benchmark::benchmark skimpy)
  set_target_properties( ${TEST_TARGET} PROPERTIES OUTPUT_NAME "${TEST_NAME}")
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_TARGET})
endfunction()

# Add each benchmark target and register it for ctest.
skimpy_benchmark(ranges_bench ranges_bench.cpp)
