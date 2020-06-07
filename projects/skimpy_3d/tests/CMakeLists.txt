cmake_minimum_required(VERSION 3.16)

function(SKIMPY_TEST TEST_NAME TEST_FILE)
  set(TEST_TARGET ${TEST_NAME}_exe)
  add_executable(${TEST_TARGET} ${TEST_FILE})
  target_include_directories(${TEST_TARGET} PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(${TEST_TARGET} PRIVATE Catch2::Catch2 skimpy)
  set_target_properties( ${TEST_TARGET} PROPERTIES OUTPUT_NAME "${TEST_NAME}")
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_TARGET})
endfunction()

# Add each test target and register it for ctest.
skimpy_test(marching_cubes_test marching_cubes_test.cpp)
skimpy_test(utils_test utils_test.cpp)
skimpy_test(voxels_test voxels_test.cpp)
