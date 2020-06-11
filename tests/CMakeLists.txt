cmake_minimum_required(VERSION 3.16)

function(PYSKIP_TEST TEST_NAME TEST_FILE)
  set(TEST_TARGET ${TEST_NAME}_exe)
  add_executable(${TEST_TARGET} ${TEST_FILE})
  target_link_libraries(${TEST_TARGET} PRIVATE Catch2::Catch2 pyskip)
  set_target_properties( ${TEST_TARGET} PROPERTIES OUTPUT_NAME "${TEST_NAME}")
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_TARGET})
endfunction()

# Add each test target and register it for ctest.
pyskip_test(array_test array_test.cpp)
pyskip_test(box_test box_test.cpp)
pyskip_test(builder_test builder_test.cpp)
pyskip_test(core_test core_test.cpp)
pyskip_test(dags_test dags_test.cpp)
pyskip_test(eval_test eval_test.cpp)
pyskip_test(lang_test lang_test.cpp)
pyskip_test(mask_test mask_test.cpp)
pyskip_test(step_test step_test.cpp)
pyskip_test(tensor_test tensor_test.cpp)
pyskip_test(util_test util_test.cpp)

# Add benchmark tests.
if(PYSKIP_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
