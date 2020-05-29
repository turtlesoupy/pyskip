cmake_minimum_required(VERSION 3.16)

find_package(OpenMP REQUIRED)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  SET(AVX_COMPILE_OPTION -mavx2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  SET(AVX_COMPILE_OPTION /QxAVX2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  SET(AVX_COMPILE_OPTION /arch:AVX2)
endif()

add_library(skimpy_bench_cpp_ext MODULE cpp_ext.cpp)
target_compile_options(skimpy_bench_cpp_ext PRIVATE ${OpenMP_CXX_FLAGS} ${AVX_COMPILE_OPTION})
target_link_libraries(skimpy_bench_cpp_ext PRIVATE pybind11::module)

# Enable Taco if we're on GCC.
if(SKIMPY_BENCH_ENABLE_TACO)
  target_link_libraries(skimpy_bench_cpp_ext PRIVATE taco)
  target_include_directories(skimpy_bench_cpp_ext PRIVATE ${CMAKE_SOURCE_DIR}/third_party/taco/include)
endif()

set_target_properties(
  skimpy_bench_cpp_ext
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "_skimpy_bench_cpp_ext"
)
