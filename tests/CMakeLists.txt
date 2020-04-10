cmake_minimum_required(VERSION 3.16)

function(SKIMPY_TEST TEST_NAME TEST_FILE)
  set(TEST_TARGET ${TEST_NAME}_exe)
  add_executable(${TEST_TARGET} ${TEST_FILE})
  target_link_libraries(${TEST_TARGET} PRIVATE Catch2::Catch2 skimpy)
  set_target_properties( ${TEST_TARGET} PROPERTIES OUTPUT_NAME "${TEST_NAME}")
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_TARGET})
endfunction()

# Add each test target and register it for ctest.
skimpy_test(core_test core_test.cpp)
skimpy_test(lang_test lang_test.cpp)
skimpy_test(util_test util_test.cpp)

# Add benchmark tests.
add_subdirectory(benchmarks)
